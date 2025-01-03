import logging
import os
import cv2
import numpy as np
from typing import Tuple
import segmentation_models_pytorch as smp

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import (DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights,
                                             deeplabv3_resnet50, deeplabv3_resnet101)
from tqdm import tqdm

from configs import config
from dataset import SandGrainsDataset
from metrics import calculate_metrics
from metrics.loss import FocalLoss, FocalTverskyLoss
from visualizer import Visualizer

import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context

logger = logging.getLogger(__name__)


class MicroTextureDetector:

    def __init__(self, model_path: str = None, train: bool = True):
        logger.info(f"Device: {config.model.DEVICE}")
        if train is True:
            self._load_data()
        self._load_model(model_path)
        self.visualizer = Visualizer()

    def train(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.model.LEARNING_RATE)
        loss_fn = FocalLoss()
        # loss_fn = FocalTverskyLoss()
        # folder_name = (f"{loss_fn.__class__.__name__}_{config.model.EPOCH_COUNT}_epochs_{config.model.BATCH_SIZE}"
        #                f"_batches_{config.model.LEARNING_RATE}"
        #                f"_lr_without_normalization_with_resize_MULTILABELED_aug_only_train_resnet50_only_two_in_alpha")
        folder_name = (f"DeepLabv3+_{loss_fn.__class__.__name__}_{config.model.EPOCH_COUNT}_epochs_{config.model.BATCH_SIZE}"
                       f"_batches_{config.model.LEARNING_RATE}"
                       f"_lr_without_normalization_without_resize_MULTILABELED_aug_only_train_xception_only_two_in_alpha")
        if not os.path.exists(config.model.RESULTS_DIR):
            os.mkdir(config.model.RESULTS_DIR)
        results_folder_path = os.path.join(config.model.RESULTS_DIR, folder_name)
        if not os.path.exists(results_folder_path):
            os.mkdir(results_folder_path)
        logger.info(f"Results saved into: {results_folder_path}")
        self._train_loop(optimizer, loss_fn, results_folder_path)
        self.visualizer.visualize(results_folder_path)

    def _load_data(self):
        """Load train, validation and test datasets and create DataLoaders."""
        self.train_dataset = SandGrainsDataset(path=config.data.TRAIN_SET_PATH)
        self.classes_count = self.train_dataset.info["classes_count"]
        logger.info(f"Dataset classes count: {self.classes_count}")
        logger.info(f"Train dataset size: {len(self.train_dataset)}")
        self.val_dataset = SandGrainsDataset(path=config.data.VAL_SET_PATH)
        logger.info(f"Val dataset size: {len(self.val_dataset)}")
        self.test_dataset = SandGrainsDataset(path=config.data.TEST_SET_PATH)
        logger.info(f"Test dataset size: {len(self.test_dataset)}")
        logger.info(f"Batch size: {config.model.BATCH_SIZE}")
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=config.model.BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=config.model.BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=config.model.BATCH_SIZE, shuffle=True)

    def _load_model(self, model_path: str = None) -> None:
        """
        Load model and weights.

        :param model_path: path to file with model weights.
        """
        # load model architecture with pretrained weights
        # self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
        # self.model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=1,
            classes=19
        )
        # ============================================================================================================
        # change first conv layer of backbone nn (ResNet50) for grayscale images
        # https://discuss.pytorch.org/t/how-to-modify-deeplabv3-and-fcn-models-for-grayscale-images/52688
        self.model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # ============================================================================================================
        # change the last layer output classes
        self.model.classifier[-1] = torch.nn.Conv2d(256, self.classes_count, 1)
        logger.info(f"Model: {self.model.__class__.__name__}")

        # load custom weights
        if model_path is not None:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"File {model_path} does not exists!")
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
            logger.info(f"Loaded model weights from {model_path}")

        self.model.to(config.model.DEVICE)

    def _train_loop(self, optimizer, loss_fn, results_folder_path: str) -> None:
        """
        Train and evaluate model on config.model.EPOCH_COUNT epochs.

        :param optimizer: model optimiser.
        :param loss_fn: loss function.
        """
        best_loss = float("inf")
        patience = config.model.PATIENCE
        for epoch in tqdm(range(config.model.EPOCH_COUNT), desc="Train model"):
            logger.info(f"EPOCH {epoch}")
            train_loss = self._train_one_epoch(optimizer=optimizer, loss_fn=loss_fn)
            val_loss = self._validate_one_epoch(loss_fn=loss_fn)
            logger.info(f"TRAIN loss: {train_loss}   VALIDATION loss: {val_loss}")

            if val_loss is torch.nan or train_loss is torch.nan:
                raise Exception("Loss is nan!!!")

            # early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience = config.model.PATIENCE
                model_path = os.path.join(results_folder_path, "model.pt")
                torch.save(self.model.state_dict(), model_path)
                logger.info("Best model saved")
            else:
                patience -= 1
                if patience <= 0:
                    break

    def _train_one_epoch(self, optimizer, loss_fn) -> float:
        """
        Train one epoch.

        :param optimizer: optimizer
        :param loss_fn: loss function

        :return: average train loss per epoch.
        """
        self.model.train()  # set model in training mode.
        running_cum_loss = 0.0
        for images, masks in self.train_loader:
            images, masks = images.float().to(config.model.DEVICE), masks.float().to(config.model.DEVICE)
            optimizer.zero_grad()  # reset the gradients for new batch
            outputs = self.model(images)['out']  # forward
            loss = loss_fn(outputs, masks)  # compute loss
            loss.backward()  # backward
            optimizer.step()  # step of input optimizer

            # mul on batch size because loss is avg loss for batch, so loss=loss/batch_size
            running_cum_loss += loss.item() * images.shape[0]
        avg_train_loss = running_cum_loss / len(self.train_dataset)
        self.visualizer.train_loss.append(avg_train_loss)
        return avg_train_loss

    def _validate_one_epoch(self, loss_fn) -> float:
        """
        Calculate loss and metrics on validation dataset.

        :param loss_fn: loss function.

        :return: average validation loss per epoch.
        """
        self.model.eval()  # set model in evaluation mode.
        running_cum_loss = 0.0
        metrics = torch.zeros((3, self.classes_count))  # TODO add constant to metric count
        batch_count = 0
        for images, masks in self.val_loader:
            images, masks = images.float().to(config.model.DEVICE), masks.float().to(config.model.DEVICE)
            # disables gradient calculation because we don't call backward prop. It reduces memory consumption.
            with torch.no_grad():
                outputs = self.model(images)['out']
                loss = loss_fn(outputs, masks)
            metrics += calculate_metrics(outputs, masks).to("cpu")
            running_cum_loss += loss.item() * images.shape[0]
            batch_count += 1
        # calculate loss and metrics per epoch
        avg_val_loss = running_cum_loss / len(self.val_dataset)
        avg_metrics_per_class = metrics / batch_count
        avg_metrics = torch.mean(avg_metrics_per_class, 1)
        logger.info(f"IOU: {avg_metrics[0]}   Recall: {avg_metrics[1]}   Precision: {avg_metrics[2]}")
        # save data for visualizations
        self.visualizer.store_metrics(avg_metrics.numpy(), avg_metrics_per_class.numpy())
        self.visualizer.validation_loss.append(avg_val_loss)
        return avg_val_loss

    def evaluate_test_data(self, results_folder_path: str) -> None:
        self.model.eval()  # set model in evaluation mode.
        metrics = torch.zeros((3, self.classes_count))  # TODO add constant to metric count
        batch_count = 0
        make_image_viz = True
        for images, masks in tqdm(self.test_loader, desc="Evaluate test data"):
            images, masks = images.float().to(config.model.DEVICE), masks.float().to(config.model.DEVICE)
            # disables gradient calculation because we don't call backward prop. It reduces memory consumption.
            with torch.no_grad():
                outputs = self.model(images)['out']
            metrics += calculate_metrics(outputs, masks).to("cpu")
            # ==================================================================================
            # create visualization on one test image
            if make_image_viz:
                # outputs = convert_nn_output(outputs=outputs, to_mask=True).type(torch.uint8)
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > 0.85).type(torch.uint8)
                if config.data.USE_NORMALIZATION:
                    mean = torch.tensor([0.485]) * 255
                    std = torch.tensor([0.229]) * 255
                    images[0] = (images[0].cpu() * std) + mean
                    logger.info("Image has been denormalized.")
                if (images[0] < 0).any() and config.data.USE_NORMALIZATION is False:
                    logger.warning(f"Image wasn't denormalize for visualization or denormalization was wrong.")
                self.visualizer.make_test_image_prediction_visualisations(
                    images[0], masks[0], outputs[0], results_folder_path
                )
                make_image_viz = False
            # ==================================================================================
            batch_count += 1
        # calculate metrics
        avg_metrics_per_class = metrics / batch_count
        avg_metrics = torch.mean(avg_metrics_per_class, 1)
        logger.info(f"[TEST] IOU: {avg_metrics[0]}   Recall: {avg_metrics[1]}   Precision: {avg_metrics[2]}")

    def predict(self, img_name: str) -> None:
        self.model.eval()
        img_path = os.path.join(config.data.PREDICTIONS_FOLDER_PATH, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File with name {img_name} does not found.")

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = config.data.IMAGE_TRAIN_TRANSFORMATION(image=image)["image"]  # TODO make it optional
        image = image.float().to(config.model.DEVICE).unsqueeze(0)
        img_predictions_folder_path = os.path.join(config.data.PREDICTIONS_FOLDER_PATH, img_name.split('.')[0])
        if not os.path.exists(img_predictions_folder_path):
            os.mkdir(img_predictions_folder_path)

        with torch.no_grad():
            logger.info(f"Image for model shape: {image.shape}")
            outputs = self.model(image)['out']
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.85).type(torch.uint8)
            print(f"Result shape: {outputs.shape}")

        if config.data.USE_NORMALIZATION:
            mean = torch.tensor([0.485]) * 255
            std = torch.tensor([0.229]) * 255
            image[0] = (image[0].cpu() * std) + mean
            logger.info("Image has been denormalized.")
        if (image[0] < 0).any() and config.data.USE_NORMALIZATION is False:
            logger.warning(f"Image wasn't denormalize for visualization or denormalization was wrong.")

        self.visualizer.make_prediction_visualisation(
            image=image[0], pred=outputs[0].cpu().numpy(), out_folder_path=img_predictions_folder_path
        )

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


