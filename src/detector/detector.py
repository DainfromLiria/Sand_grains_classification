import json
import logging
import os
from pathlib import Path
from uuid import uuid4

import cv2
import matplotlib.pyplot as plt
import neptune
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.utils.model_zoo as model_zoo
from dotenv import load_dotenv
from neptune.integrations.python_logger import NeptuneHandler
from pretrained_microscopy_models.util import get_pretrained_microscopynet_url
from torch import nn
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts)
from torch.utils.data import DataLoader
from tqdm import tqdm

import metrics.loss as loss_functions
from configs import config
from dataset import SandGrainsDataset
from metrics import calculate_confusion_matrix
from utils import (calculate_patch_positions, join_and_visualize_patches,
                   visualize_nn_prediction)

logger = logging.getLogger(__name__)


class MicroTextureDetector:

    def __init__(self, mode: str, experiment_uuid: str = None):
        """
        Initialize detector.

        :param experiment_uuid: uuid of existing experiment. If None, generate new uuid.
        :param mode: can be 'train', 'val', 'eval' and 'infer'. 'eval' is for run model on test dataset, but
        "infer" is for run model on completely new images from user.
        """
        self.mode = mode
        self.experiment_uuid: str = str(uuid4()) if self.mode == "train" else experiment_uuid
        if self.mode in ("train", "val"):
            self.run = self._init_neptune()
            if config.transform.USE_TTA:
                raise ValueError("TTA must be used only in infer or eval mode")
        if self.mode in ("infer", "eval"):
            self._read_config()
        logger.warning(f"Model run in {mode} mode!")
        logger.info(f"Device: {config.model.DEVICE}")
        self.model = self._create_model(
            model_name=config.model.MODEL,
            encoder=config.model.ENCODER,
            encoder_weights=config.model.ENCODER_WEIGHTS
        )

        if self.mode in ("train", "val", "eval"):
            self._load_data()
            if self.mode in ("train", "val"):
                # Optimizer
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.model.LEARNING_RATE)

                # Scheduler
                if config.model.USE_CA:
                    if config.model.SCHEDULER == "CosineAnnealingLR":
                        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.model.CA_TMAX)
                    else:
                        self.scheduler = CosineAnnealingWarmRestarts(
                            optimizer=self.optimizer,
                            T_0=config.model.CA_T0,
                            T_mult=config.model.CA_TMULT
                        )
                    logger.info(f"Scheduler: {self.scheduler.__class__.__name__}")

                # Loss
                loss_pkg: object = nn if config.model.LOSS_FUNCTION == "BCEWithLogitsLoss" else loss_functions
                self.loss_fn = getattr(loss_pkg, config.model.LOSS_FUNCTION)()
                logger.info(f"Loss function: {self.loss_fn.__class__.__name__}")
                if self.mode == "train":
                    self.results_folder_path = self._make_results_folder()
                    self._save_model_params()

    def train(self) -> None:
        """Train and validate model."""
        best_loss = float("inf")
        patience = config.model.PATIENCE
        for epoch in tqdm(range(config.model.EPOCH_COUNT), desc="Train model"):
            logger.info(f"EPOCH {epoch}")
            train_loss = self._train_one_epoch()
            val_loss, val_iou = self._validate_one_epoch()
            logger.info(f"TRAIN loss: {train_loss}   VALIDATION loss: {val_loss}")

            if config.model.USE_CA:
                self.scheduler.step()  # step of cosine scheduler
                lr = self.scheduler.get_last_lr()[0]
                self.run['train/epoch/lr'].append(lr)

            if val_loss is torch.nan or train_loss is torch.nan:
                self.run.stop()
                raise Exception("Loss is nan!!!")

            # early stopping
            if val_loss < best_loss:
                self.run["best_val_iou"] = val_iou
                self.run["best_val_loss"] = val_loss
                best_loss = val_loss
                patience = config.model.PATIENCE
                model_path: Path = self.results_folder_path / "model.pt"
                torch.save(self.model.state_dict(), model_path)
                self.run["model/checkpoints"].upload(str(model_path))
                logger.warning("Best model saved")
            else:
                patience -= 1
                if patience <= 0:
                    break
        self.run.stop()

    def validate(self) -> None:
        """Validate model."""
        val_loss, val_iou = self._validate_one_epoch()
        logger.info(f"VALIDATION loss: {val_loss}")

        if val_loss is torch.nan:
            self.run.stop()
            raise Exception("Loss is nan!!!")

        self.run["best_val_iou"] = val_iou
        self.run["best_val_loss"] = val_loss
        self.run.stop()

    def evaluate_test_data(self, show_predictions: bool = False) -> None:
        """
        Evaluate test dataset.

        :param show_predictions: Show model prediction and ground truth mask.
        """
        self.model.eval()  # set model in evaluation mode.
        confusion_matrix: torch.Tensor = torch.zeros((config.model.CONF_MAT_SIZE, config.data.CLASSES_COUNT))
        for images, masks in tqdm(self.test_loader, desc="Evaluate test data"):
            images, masks = images.float().to(config.model.DEVICE), masks.float().to(config.model.DEVICE)
            if config.transform.USE_TTA:
                outputs = self._apply_tta(images)
            else:
                # disables gradient calculation because we don't call backward prop. It reduces memory consumption.
                with torch.no_grad():
                    outputs = self.model(images)

            if show_predictions:
                outputs_for_viz = torch.sigmoid(outputs)
                outputs_for_viz = (outputs_for_viz > config.model.THRESHOLD).type(torch.uint8)
                for i in range(config.model.BATCH_SIZE):
                    for j in range(config.data.CLASSES_COUNT):
                        visualize_nn_prediction(f"{i}_{j}", images[i], masks[i][j], color=(0, 255, 0))
                        visualize_nn_prediction(f"{i}_{j}", images[i], outputs_for_viz[i][j], color=(0, 0, 255))

            confusion_matrix += calculate_confusion_matrix(outputs, masks).to("cpu")
        self._calculate_metrics(confusion_matrix, prefix="test/metric")

    def predict(self, img_name: str) -> None:
        """
        Generate prediction for image.

        :param img_name: name of the image in PREDICTIONS_FOLDER (for example: A2.tif).
        """
        self.model.eval()
        img_path = config.paths.PREDICTIONS_FOLDER / img_name
        if not img_path.exists():
            raise FileNotFoundError(f"File with name {img_name} does not found.")

        image: np.ndarray = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image: torch.Tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)

        if config.transform.TTA_AUGMENTATIONS:
            outputs = self._apply_tta(image)
        else:
            with torch.no_grad():
                outputs = self.model(image)
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > config.model.THRESHOLD).type(torch.uint8)

        for j in range(config.data.CLASSES_COUNT):
            visualize_nn_prediction(f"0_{j}", image[0], outputs[0][j], color=(0, 0, 255))

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
            model_path: Path = config.paths.RESULTS_FOLDER / self.experiment_uuid / "model.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"File {model_path} does not exists!")
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=config.model.DEVICE))
            logger.info(f"Model weights loaded from {model_path}")
        elif self.mode in ("val", "eval", "infer") and self.experiment_uuid is None:
            raise Exception("experiment_uuid is None. For 'val', 'eval' and 'infer' mode experiment_uuid must be provided.")

        model.to(config.model.DEVICE)
        return model

    def _load_data(self) -> None:
        """Load train, validation and test datasets and create DataLoaders."""
        if self.mode in ("train", "val"):
            self.train_dataset = SandGrainsDataset(mode="train")
            self.classes_count = self.train_dataset.dataset_info["num_classes"]
            logger.info(f"Dataset classes count: {self.classes_count}")
            logger.info(f"Train dataset size: {len(self.train_dataset)}")
            self.val_dataset = SandGrainsDataset(mode="val")
            logger.info(f"Val dataset size: {len(self.val_dataset)}")
            self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=config.model.BATCH_SIZE, shuffle=True)
            self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=config.model.BATCH_SIZE, shuffle=False)
        elif self.mode == "eval":
            self.test_dataset = SandGrainsDataset(mode="eval")
            logger.info(f"Test dataset size: {len(self.test_dataset)}")
            self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=config.model.BATCH_SIZE, shuffle=False)
        logger.info(f"Batch size: {config.model.BATCH_SIZE}")

    def _read_config(self) -> None:
        """Read the config file and set config values for the inference run of the existing model."""
        if self.experiment_uuid is None:
            raise Exception("experiment_uuid is None. For 'infer' mode experiment_uuid must be provided.")

        config_path: Path = config.paths.RESULTS_FOLDER / self.experiment_uuid / "model_params.json"
        with open(config_path, "r") as f:
            data = json.load(f)
        # update only configs needed for inference
        config.model.MODEL = data["model"]
        config.model.ENCODER = data["encoder"]
        config.model.ENCODER_WEIGHTS = data["encoder_weights"]
        config.transform.USE_PREPROCESSING = data["use_preprocessing"]
        logger.info("Config file was successfully loaded.")

    def _init_neptune(self) -> neptune.Run:
        """Initialize neptune Run."""
        load_dotenv()
        run: neptune.Run = neptune.init_run(
            project=os.getenv("NEPTUNE_PROJECT"),
            api_token=os.getenv("NEPTUNE_API_KEY"),
            name=self.experiment_uuid,
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
        results_folder_path.mkdir(parents=False, exist_ok=False) # raise exception if the folder with this uuid exists
        logger.info(f"Results will be save into: {results_folder_path}")
        return results_folder_path

    def _save_model_params(self) -> None:
        """Save main model parameters into JSON file and send params into neptune."""
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
            "use_patches": config.model.USE_PATCHES,
            "patch_size": config.model.PATCH_SIZE,
            "overlap_rate": config.model.OVERLAP_RATE,
            "patch_stride": config.model.PATCH_STRIDE,
            "use_augmentations": config.transform.USE_AUGMENTATIONS,
            "use_preprocessing": config.transform.USE_PREPROCESSING,
            "use_ca": config.model.USE_CA,
            "ca_t0": config.model.CA_T0,
            "ca_tmult": config.model.CA_TMULT,
            "classes_count": config.data.CLASSES_COUNT,
            "use_clipping": config.model.USE_CLIPPING,
            "use_resize": config.transform.USE_RESIZE,
            "scheduler": config.model.SCHEDULER,
        }
        logger.info(f"MODEL PARAMETERS: {model_params}")
        self.run["params"] = model_params
        model_params_path: Path = self.results_folder_path / "model_params.json"
        with open(model_params_path, "w") as f:
            json.dump(model_params, f, indent=4)

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

            if config.model.USE_CLIPPING:
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0) # gradient value clipping

            self.optimizer.step()  # optimizer makes one step

            # mul on batch size because loss is avg loss for batch, so loss=loss/batch_size
            running_cum_loss += loss.item() * images.shape[0]
        avg_train_loss = running_cum_loss / len(self.train_dataset)
        self.run["train/epoch/loss"].append(avg_train_loss)
        return avg_train_loss

    def _validate_one_epoch(self) -> (float, torch.Tensor):
        """
        Calculate loss and metrics on validation dataset.

        :return: average validation loss and IoU per epoch.
        """
        self.model.eval()  # set model in evaluation mode.
        running_cum_loss = 0.0
        confusion_matrix: torch.Tensor = torch.zeros((config.model.CONF_MAT_SIZE, config.data.CLASSES_COUNT))
        for images, masks in self.val_loader:
            images, masks = images.float().to(config.model.DEVICE), masks.float().to(config.model.DEVICE)
            # disables gradient calculation because we don't call backward prop. It reduces memory consumption.
            with torch.no_grad():
                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
            confusion_matrix += calculate_confusion_matrix(outputs, masks).to("cpu")
            running_cum_loss += loss.item() * images.shape[0]
        # calculate epoch loss
        avg_val_loss: float = running_cum_loss / len(self.val_dataset)
        self.run["val/epoch/loss"].append(avg_val_loss)
        # calculate epoch metrics
        iou = self._calculate_metrics(confusion_matrix, prefix="val/epoch/metric")
        return avg_val_loss, iou

    def _calculate_metrics(self, confusion_matrix: torch.Tensor, prefix: str) -> torch.Tensor:
        """
        Calculate model evaluation metrics (IoU, recall, precision).

        :param confusion_matrix: per class TP, FP and FN in format [3, classes_count]
        :param prefix: metric prefix for neptune.

        :return: mean IOU across all classes.
        """
        eps: float = config.model.EPS
        tp, fp, fn = confusion_matrix[0], confusion_matrix[1], confusion_matrix[2]
        # per class
        precision_per_class = torch.squeeze((tp + eps) / (tp + fp + eps))
        logger.info(f"precision_per_class: {precision_per_class}")
        recall_per_class = torch.squeeze((tp + eps) / (tp + fn + eps))
        logger.info(f"recall_per_class: {recall_per_class}")
        iou_per_class = torch.squeeze((tp + eps) / (tp + fp + fn + eps))
        logger.info(f"iou_per_class: {iou_per_class}")
        # avg
        avg_iou = torch.mean(iou_per_class)
        avg_recall = torch.mean(recall_per_class)
        avg_precision = torch.mean(precision_per_class)
        logger.info(f"IOU: {avg_iou}   Recall: {avg_recall}   Precision: {avg_precision}")
        # send data to neptune
        if self.mode == "train":
            self.run[f"{prefix}/iou"].append(avg_iou)
            self.run[f"{prefix}/recall"].append(avg_recall)
            self.run[f"{prefix}/precision"].append(avg_precision)
            for i in range(confusion_matrix.shape[1]):
                self.run[f"{prefix}/{i}/iou"].append(iou_per_class[i])
                self.run[f"{prefix}/{i}/recall"].append(recall_per_class[i])
                self.run[f"{prefix}/{i}/precision"].append(precision_per_class[i])
        return avg_iou

    def _apply_tta(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply TTA augmentations on input image or batch of images and return binary prediction mask.
        For now, only vertical and horizontal flips are supported.

        :return: binary prediction mask.
        """
        tta_output: torch.Tensor = torch.zeros(image.shape[0], config.data.CLASSES_COUNT, image.shape[2], image.shape[3])
        tta_transforms: list = config.transform.TTA_AUGMENTATIONS
        for i in range(len(config.transform.TTA_AUGMENTATIONS)):
            # augmentation
            aug_img: torch.Tensor = torch.flip(image, dims=tta_transforms[i]) if tta_transforms[i] else image
           # predict
            with torch.no_grad():
                outputs = self.model(aug_img)
                outputs = torch.sigmoid(outputs)
            # deaugmentation
            outputs = torch.flip(outputs, dims=tta_transforms[i]) if tta_transforms[i] else outputs
            tta_output += outputs
        # final averaged prediction
        tta_output /= len(tta_transforms)
        tta_output = (tta_output > config.model.THRESHOLD).type(torch.uint8)
        return tta_output
