import logging
import os

import albumentations as A
import cv2
import neptune
import segmentation_models_pytorch as smp
import torch
import torch.utils.model_zoo as model_zoo
from dotenv import load_dotenv
from pretrained_microscopy_models.util import get_pretrained_microscopynet_url
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import config
from dataset import SandGrainsDataset
from metrics import calculate_metrics
from metrics.loss import FocalTverskyLoss
from utils import predict_morphological_feature

logger = logging.getLogger(__name__)


class MicroTextureDetector:

    METRICS_COUNT: int = 3 # TODO move to config

    def __init__(self, mode: str, model_path: str = None):
        """
        Initialize detector.

        :param model_path: optional path to model weights. If None, use DeepLabV3_ResNet50_Weights.
        :param mode: can be 'train', 'eval' and 'infer'. 'eval' is for run model on test dataset, but
        "infer" is for run model on completely new images from user (on production).
        """
        self.mode = mode
        self.model_path = model_path
        logger.warning(f"Model run in {mode} mode!")
        logger.info(f"Device: {config.model.DEVICE}")
        self.model = self.create_model(
            model_name=config.model.MODEL,
            encoder=config.model.ENCODER,
            encoder_weights=config.model.ENCODER_WEIGHTS
        )

        if self.mode in ("train", "eval"):
            self._load_data()
            if self.mode == "train":
                self.run = self._init_neptune()
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.model.LEARNING_RATE)
                self.loss_fn = FocalTverskyLoss()
                self.results_folder_path = self._make_results_folder_path()

    def create_model(self, model_name: str, encoder: str, encoder_weights: str) -> nn.Module:
        """
        Returns segmentation model with the specified encoder backbone.
        Reworked function from https://github.com/nasa/pretrained-microscopy-models/tree/main

        :param model_name: available segmentation model. Can be 'Unet',
        'DeepLabV3Plus' or 'Segformer'.
        :param encoder: available encoder backbones in segmentation_models_pytorch
        such as 'ResNet50' or 'xception'.
        :param encoder_weights: The dataset that the encoder was pre-trained on.
                One of ['micronet', 'image-micronet', 'imagenet', 'None']

        :return: PyTorch model for segmentation
        """
        if model_name not in config.model.AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} is not available.")
        logger.info(f"Model architecture: {model_name}")
        logger.info(f"Encoder backbone: {encoder}")
        logger.info(f"Encoder weights: {encoder_weights}")

        initial_weights = 'imagenet' if encoder_weights == 'imagenet' else None

        # create the model
        model = getattr(smp, model_name)(
            encoder_name=encoder,
            encoder_weights=initial_weights,
            classes=config.data.CLASSES_COUNT
        )

        # load pretrained weights
        if encoder_weights in ['micronet', 'image-micronet']:
            url = get_pretrained_microscopynet_url(encoder, encoder_weights)
            model.encoder.load_state_dict(model_zoo.load_url(url, map_location=config.model.DEVICE))

        # load custom weights
        if self.model_path is not None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"File {self.model_path} does not exists!")
            model.load_state_dict(torch.load(self.model_path, weights_only=True))
            logger.info(f"Model weights loaded from {self.model_path}")

        model.to(config.model.DEVICE)
        return model

    def _load_data(self) -> None:
        """Load train, validation and test datasets and create DataLoaders."""
        if self.mode == "train":
            self.train_dataset = SandGrainsDataset(mode="train")
            self.classes_count = self.train_dataset.dataset_info["num_classes"]
            logger.info(f"Dataset classes count: {self.classes_count}")
            logger.info(f"Train dataset size: {len(self.train_dataset)}")
            self.val_dataset = SandGrainsDataset(mode="val")
            logger.info(f"Val dataset size: {len(self.val_dataset)}")
            self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=config.model.BATCH_SIZE, shuffle=True)
            self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=config.model.BATCH_SIZE, shuffle=True)
        elif self.mode == "eval":
            self.test_dataset = SandGrainsDataset(mode="eval")
            logger.info(f"Test dataset size: {len(self.test_dataset)}")
            self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=config.model.BATCH_SIZE, shuffle=True)
        logger.info(f"Batch size: {config.model.BATCH_SIZE}")

    @staticmethod
    def _init_neptune() -> neptune.Run:
        """Initialize neptune Run."""
        load_dotenv()
        return neptune.init_run(
            project=os.getenv("NEPTUNE_PROJECT"),
            api_token=os.getenv("NEPTUNE_API_KEY"),
            source_files=[]  # TODO add tracked files
        )

    def train(self) -> None:
        """Train and evaluate model on config.model.EPOCH_COUNT epochs."""
        best_loss = float("inf")
        patience = config.model.PATIENCE
        for epoch in tqdm(range(config.model.EPOCH_COUNT), desc="Train model"):
            logger.info(f"EPOCH {epoch}")
            train_loss = self._train_one_epoch()
            val_loss = self._validate_one_epoch()
            logger.info(f"TRAIN loss: {train_loss}   VALIDATION loss: {val_loss}")

            if val_loss is torch.nan or train_loss is torch.nan:
                raise Exception("Loss is nan!!!")

            # early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience = config.model.PATIENCE
                model_path = os.path.join(self.results_folder_path, "model.pt")
                torch.save(self.model.state_dict(), model_path)
                logger.info("Best model saved")
            else:
                patience -= 1
                if patience <= 0:
                    break

    def _make_results_folder_path(self) -> str:
        """
        Make folder for model results (weights, graphs, images, etc.)

        :return: path to results folder
        """
        transforms = config.data.IMAGE_TRAIN_TRANSFORMATION.transforms
        resized = "with_resize" if any(isinstance(t, A.Resize) for t in transforms) else "without_resize"
        normalized = "with_normalization" if any(
            isinstance(t, A.Normalize) for t in transforms) else "without_normalization"
        # assemble name
        folder_name = (
            f"{self.model.__class__.__name__}_{self.loss_fn.__class__.__name__}_{config.model.EPOCH_COUNT}_epochs"
            f"_{config.model.BATCH_SIZE}_batches_{config.model.LEARNING_RATE}_lr"
            f"_{normalized}_{resized}_resnet50_CLAHE_DELETE")
        # make dir for results
        if not os.path.exists(config.model.RESULTS_DIR):
            os.mkdir(config.model.RESULTS_DIR)
        results_folder_path = os.path.join(config.model.RESULTS_DIR, folder_name)
        if not os.path.exists(results_folder_path):
            os.mkdir(results_folder_path)
        logger.info(f"Results saved into: {results_folder_path}")
        return results_folder_path

    def _train_one_epoch(self) -> float:
        """
        Train one epoch.

        :return: average train loss per epoch.
        """
        self.model.train()  # set model in training mode.
        running_cum_loss = 0.0
        for images, masks in self.train_loader:
            images, masks = images.float().to(config.model.DEVICE), masks.float().to(config.model.DEVICE)
            self.optimizer.zero_grad()  # reset the gradients for new batch
            outputs = self.model(images)['out']  # forward
            loss = self.loss_fn(outputs, masks)  # compute loss
            loss.backward()  # backward
            self.optimizer.step()  # step of input optimizer

            # mul on batch size because loss is avg loss for batch, so loss=loss/batch_size
            running_cum_loss += loss.item() * images.shape[0]
        avg_train_loss = running_cum_loss / len(self.train_dataset)
        self.visualizer.train_loss.append(avg_train_loss)
        return avg_train_loss

    def _validate_one_epoch(self, store_metrics: bool = True) -> float:
        """
        Calculate loss and metrics on validation dataset.

        :param store_metrics: if True, store metrics in file.

        :return: average validation loss per epoch.
        """
        self.model.eval()  # set model in evaluation mode.
        running_cum_loss = 0.0
        metrics = torch.zeros((self.METRICS_COUNT, self.classes_count))
        batch_count = 0
        for images, masks in self.val_loader:
            images, masks = images.float().to(config.model.DEVICE), masks.float().to(config.model.DEVICE)
            # disables gradient calculation because we don't call backward prop. It reduces memory consumption.
            with torch.no_grad():
                outputs = self.model(images)['out']
                loss = self.loss_fn(outputs, masks)
            metrics += calculate_metrics(outputs, masks).to("cpu")
            running_cum_loss += loss.item() * images.shape[0]
            batch_count += 1
        # calculate loss and metrics per epoch
        avg_val_loss = running_cum_loss / len(self.val_dataset)
        avg_metrics_per_class = metrics / batch_count
        avg_metrics = torch.mean(avg_metrics_per_class, 1)
        logger.info(f"IOU: {avg_metrics[0]}   Recall: {avg_metrics[1]}   Precision: {avg_metrics[2]}")
        # save data for visualizations
        if store_metrics:
            self.visualizer.store_metrics(avg_metrics.numpy(), avg_metrics_per_class.numpy())
            self.visualizer.validation_loss.append(avg_val_loss)
        return avg_val_loss

    def evaluate_test_data(self, results_folder_path: str) -> None:
        """
        Evaluate test dataset and save results (images with predictions) into results_folder_path.

        :param results_folder_path: path to folder where images with predictions will be saved.
        """
        self.model.eval()  # set model in evaluation mode.
        metrics = torch.zeros((self.METRICS_COUNT, self.classes_count))
        batch_count = 0
        for images, masks in tqdm(self.test_loader, desc="Evaluate test data"):
            images, masks = images.float().to(config.model.DEVICE), masks.float().to(config.model.DEVICE)
            # disables gradient calculation because we don't call backward prop. It reduces memory consumption.
            with torch.no_grad():
                outputs = self.model(images)['out']
            metrics += calculate_metrics(outputs, masks).to("cpu")
            self.visualizer.make_test_images_prediction_visualisations(images, masks, outputs, results_folder_path)
            batch_count += 1
        # calculate metrics
        avg_metrics_per_class = metrics / batch_count
        print(f"Avg metrics: {torch.round(avg_metrics_per_class, decimal=4)}")
        avg_metrics = torch.mean(avg_metrics_per_class, 1)
        logger.info(f"[TEST] IOU: {avg_metrics[0]}   Recall: {avg_metrics[1]}   Precision: {avg_metrics[2]}")

    def predict(self, img_name: str) -> None:
        """
        Generate prediction for image.

        :param img_name: name of image in PREDICTIONS_FOLDER (for example: A2.tif).
        """
        self.model.eval()
        img_path = os.path.join(config.data.PREDICTIONS_FOLDER_PATH, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File with name {img_name} does not found.")

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = config.data.IMAGE_PREDICTION_TRANSFORMATION(image=image)["image"]
        image = image.float().to(config.model.DEVICE).unsqueeze(0)
        img_predictions_folder_path = os.path.join(config.data.PREDICTIONS_FOLDER_PATH, img_name.split('.')[0])
        if not os.path.exists(img_predictions_folder_path):
            os.mkdir(img_predictions_folder_path)

        with torch.no_grad():
            outputs = self.model(image)['out']
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > config.model.THRESHOLD).type(torch.uint8)
            outputs = predict_morphological_feature(outputs)
        self.visualizer.make_prediction_visualisation(image, outputs[0].cpu().numpy(), img_predictions_folder_path)

    def calculate_normalization_std_mean(self):
        """
        Calculate mean and std on train data for normalization.
        Calculation speed depends on the batch size.
        """
        mean = 0
        std = 0
        count = 0
        for images, masks in tqdm(self.train_loader, desc="Calculating std and mean"):
            images, _ = images.float().to(config.model.DEVICE), masks.float().to(config.model.DEVICE)
            std += torch.std(images)
            mean += torch.mean(images)
            count += 1
        std = std / count
        mean = mean / count
        logger.info(f"Mean: {mean}, Std: {std}")
