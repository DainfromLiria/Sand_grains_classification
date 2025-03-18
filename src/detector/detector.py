import json
import logging
import os
from uuid import uuid4
from pathlib import Path

import neptune
import segmentation_models_pytorch as smp
import torch
import torch.utils.model_zoo as model_zoo
from dotenv import load_dotenv
from neptune.integrations.python_logger import NeptuneHandler
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

    def __init__(self, mode: str, experiment_uuid: str = None):
        """
        Initialize detector.

        :param experiment_uuid: uuid of existing experiment. If None, generate new uuid.
        :param mode: can be 'train', 'eval' and 'infer'. 'eval' is for run model on test dataset, but
        "infer" is for run model on completely new images from user (on production).
        """
        self.mode = mode
        self.experiment_uuid: str = str(uuid4()) if self.mode == "train" else experiment_uuid
        if self.mode in ("train", "eval"):
            self.run = self._init_neptune()
        logger.warning(f"Model run in {mode} mode!")
        logger.info(f"Device: {config.model.DEVICE}")
        self.model = self._create_model(
            model_name=config.model.MODEL,
            encoder=config.model.ENCODER,
            encoder_weights=config.model.ENCODER_WEIGHTS
        )

        if self.mode in ("train", "eval"):
            self._load_data()
            if self.mode == "train":
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.model.LEARNING_RATE)
                self.loss_fn = FocalTverskyLoss()
                self.results_folder_path = self._make_results_folder()
                self._save_model_params()

    # TODO check input image size for encoder backbone
    def _create_model(self, model_name: str, encoder: str, encoder_weights: str) -> nn.Module:
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
        if encoder_weights in ['micronet', 'image-micronet'] and model_name != 'Segformer':
            url = get_pretrained_microscopynet_url(encoder, encoder_weights)
            model.encoder.load_state_dict(model_zoo.load_url(url, map_location=config.model.DEVICE))

        # load custom weights
        if self.experiment_uuid and self.mode != "train":
            model_path: Path = config.paths.RESULTS_FOLDER / self.experiment_uuid
            if not model_path.exists():
                raise FileNotFoundError(f"File {model_path} does not exists!")
            model.load_state_dict(torch.load(model_path, weights_only=True))
            logger.info(f"Model weights loaded from {model_path}")
        elif self.mode in ("eval", "infer") and self.experiment_uuid is None:
            raise Exception("experiment_uuid is None. For 'eval' and 'infer' mode experiment_uuid must be provided.")

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

    def _init_neptune(self) -> neptune.Run:
        """Initialize neptune Run."""
        load_dotenv()
        run: neptune.Run = neptune.init_run(
            project=os.getenv("NEPTUNE_PROJECT"),
            api_token=os.getenv("NEPTUNE_API_KEY"),
            name=self.experiment_uuid,
            source_files=[]  # TODO add tracked files
        )
        npt_handler = NeptuneHandler(run=run)
        logger.addHandler(npt_handler)
        return run

    def _make_results_folder(self) -> Path:
        """
        Make folder for model results (weights, setting, etc.)

        :return: path to results folder
        """
        # assemble name
        results_folder_path: Path = config.paths.RESULTS_FOLDER / self.experiment_uuid
        results_folder_path.mkdir(parents=False, exist_ok=False)
        logger.info(f"Results will be save into: {results_folder_path}")
        return results_folder_path

    def _save_model_params(self) -> None:
        """Save main model parameters into json file and send params into neptune."""
        model_params: dict = {
            "mode": self.mode,
            "model": config.model.MODEL,
            "encoder": config.model.ENCODER,
            "encoder_weights": config.model.ENCODER_WEIGHTS,
            "batch_size": config.model.BATCH_SIZE,
            "lr": config.model.LEARNING_RATE,
            "epoch_count": config.model.EPOCH_COUNT,
            "loss_function": self.loss_fn.__class__.__name__,
            "optimizer": self.optimizer.__class__.__name__,
            "patience": config.model.PATIENCE,
            # TODO add another params
        }
        self.run["params"] = model_params
        model_params_path: Path = self.results_folder_path / "model_params.json"
        with open(model_params_path, "w") as f:
            json.dump(model_params, f, indent=4)

    def train(self) -> None:
        """Train and validate model."""
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
                model_path: Path = self.results_folder_path / "model.pt"
                torch.save(self.model.state_dict(), model_path)
                self.run["model/checkpoints"].upload(str(model_path))
                logger.info("Best model saved")
            else:
                patience -= 1
                if patience <= 0:
                    break
        self.run.stop()

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
            outputs = self.model(images)  # forward
            loss = self.loss_fn(outputs, masks)  # compute loss
            loss.backward()  # backward
            self.optimizer.step()  # step of input optimizer

            # mul on batch size because loss is avg loss for batch, so loss=loss/batch_size
            running_cum_loss += loss.item() * images.shape[0]
        avg_train_loss = running_cum_loss / len(self.train_dataset)
        self.run["train/epoch/loss"].append(avg_train_loss)
        return avg_train_loss

    def _validate_one_epoch(self) -> float:
        """
        Calculate loss and metrics on validation dataset.

        :return: average validation loss per epoch.
        """
        self.model.eval()  # set model in evaluation mode.
        running_cum_loss = 0.0
        metrics_per_class = torch.zeros((config.model.METRICS_COUNT, self.classes_count))
        batch_count = 0
        for images, masks in self.val_loader:
            images, masks = images.float().to(config.model.DEVICE), masks.float().to(config.model.DEVICE)
            # disables gradient calculation because we don't call backward prop. It reduces memory consumption.
            with torch.no_grad():
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
            metrics_per_class += calculate_metrics(outputs, masks).to("cpu")
            running_cum_loss += loss.item() * images.shape[0]
            batch_count += 1
        # calculate epoch loss
        avg_val_loss: float = running_cum_loss / len(self.val_dataset)
        self.run["val/epoch/loss"] = avg_val_loss
        # calculate epoch metrics
        self.calculate_metrics(metrics_per_class, batch_count, prefix="val/epoch/metric")
        return avg_val_loss

    def evaluate_test_data(self) -> None:
        """Evaluate test dataset."""
        self.model.eval()  # set model in evaluation mode.
        metrics_per_class = torch.zeros((config.model.METRICS_COUNT, self.classes_count))
        batch_count: int = 0
        for images, masks in tqdm(self.test_loader, desc="Evaluate test data"):
            images, masks = images.float().to(config.model.DEVICE), masks.float().to(config.model.DEVICE)
            # disables gradient calculation because we don't call backward prop. It reduces memory consumption.
            with torch.no_grad():
                outputs = self.model(images)
            metrics_per_class += calculate_metrics(outputs, masks).to("cpu")
            # TODO make visualization on image
            batch_count += 1
        self.calculate_metrics(metrics_per_class, batch_count, prefix="test/metric")

    def calculate_metrics(self, metrics_per_class: torch.Tensor, batch_count: int, prefix: str) -> None:
        """
        Calculate model evaluation metrics.

        :param metrics_per_class: per class metrics in format [classes_count, metrics_count]
        :param batch_count: batch count
        :param prefix: metric prefix for neptune.
        """
        avg_metrics_per_class: torch.Tensor = metrics_per_class / batch_count
        avg_metrics: torch.Tensor = torch.mean(avg_metrics_per_class, 1)
        logger.info(f"IOU: {avg_metrics[0]}   Recall: {avg_metrics[1]}   Precision: {avg_metrics[2]}")
        # send data to neptune
        self.run[f"{prefix}/iou"].append(avg_metrics[0])
        self.run[f"{prefix}/recall"].append(avg_metrics[1])
        self.run[f"{prefix}/precision"].append(avg_metrics[2])
        for i in range(avg_metrics_per_class.shape[1]):
            self.run[f"{prefix}/{i}/iou"].append(avg_metrics_per_class[:, i][0])
            self.run[f"{prefix}/{i}/recall"].append(avg_metrics_per_class[:, i][1])
            self.run[f"{prefix}/{i}/precision"].append(avg_metrics_per_class[:, i][2])

    # TODO method for prediction on completely new data
    # def predict(self, img_name: str) -> None:
    #     """
    #     Generate prediction for image.
    #
    #     :param img_name: name of image in PREDICTIONS_FOLDER (for example: A2.tif).
    #     """
    #     self.model.eval()
    #     img_path = os.path.join(config.data.PREDICTIONS_FOLDER_PATH, img_name)
    #     if not os.path.exists(img_path):
    #         raise FileNotFoundError(f"File with name {img_name} does not found.")
    #
    #     image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #     image = config.data.IMAGE_PREDICTION_TRANSFORMATION(image=image)["image"]
    #     image = image.float().to(config.model.DEVICE).unsqueeze(0)
    #     img_predictions_folder_path = os.path.join(config.data.PREDICTIONS_FOLDER_PATH, img_name.split('.')[0])
    #     if not os.path.exists(img_predictions_folder_path):
    #         os.mkdir(img_predictions_folder_path)
    #
    #     with torch.no_grad():
    #         outputs = self.model(image)
    #         outputs = torch.sigmoid(outputs)
    #         outputs = (outputs > config.model.THRESHOLD).type(torch.uint8)
    #         outputs = predict_morphological_feature(outputs)
    #     self.visualizer.make_prediction_visualisation(image, outputs[0].cpu().numpy(), img_predictions_folder_path)
    #
