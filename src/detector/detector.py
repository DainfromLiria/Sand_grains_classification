import logging
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from tqdm import tqdm

from configs import config
from metrics import FocalLoss, calculate_metrics
from dataset import SandGrainsDataset

logger = logging.getLogger(__name__)


class MicroTextureDetector:

    def __init__(self):
        torch.hub.set_dir(config.model.MODELS_DIR_PATH)  # TODO move to some setup file
        logger.info(f"Device: {config.model.DEVICE}")
        self._make_data_loader()
        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
        # ============================================================================================================
        # change first conv layer of backbone nn (ResNet101) for grayscale images
        # https://discuss.pytorch.org/t/how-to-modify-deeplabv3-and-fcn-models-for-grayscale-images/52688
        self.model.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # ============================================================================================================
        # change the last layer output classes
        self.model.classifier[-1] = torch.nn.Conv2d(256, self.classes_count, 1)
        self.model.to(config.model.DEVICE)

    def train(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.model.LEARNING_RATE)
        loss_fn = FocalLoss()
        self._train_loop(optimizer, loss_fn)

    def _make_data_loader(self):
        self.train_dataset = SandGrainsDataset(path=config.data.AUG_TRAIN_SET_PATH)
        self.classes_count = self.train_dataset.info["classes_count"]
        logger.info(f"Dataset classes count: {self.classes_count}")
        logger.info(f"Train dataset size: {len(self.train_dataset)}")
        self.val_dataset = SandGrainsDataset(path=config.data.AUG_VAL_SET_PATH)
        logger.info(f"Val dataset size: {len(self.val_dataset)}")
        self.test_dataset = SandGrainsDataset(path=config.data.AUG_TEST_SET_PATH)
        logger.info(f"Test dataset size: {len(self.test_dataset)}")
        # TODO add num_workers
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=config.model.BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=config.model.BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=config.model.BATCH_SIZE, shuffle=True)

    def _train_loop(self, optimizer, loss_fn) -> None:
        best_loss = float("inf")
        patience = config.model.PATIENCE
        for epoch in tqdm(range(config.model.EPOCH_COUNT), desc="Train model"):
            logger.info(f"EPOCH {epoch}")
            train_loss = self._train_one_epoch(optimizer=optimizer, loss_fn=loss_fn)
            val_loss = self._validate_one_epoch(loss_fn=loss_fn)
            logger.info(f"TRAIN loss: {train_loss}   VALIDATION loss: {val_loss}")

            # early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience = config.model.PATIENCE
                torch.save(self.model.state_dict(), config.model.BEST_MODEL_PATH)
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
            running_cum_loss += loss * images.shape[0]
            batch_count += 1
        # calculate loss and metrics per epoch
        avg_val_loss = running_cum_loss / len(self.val_dataset)
        avg_metrics_per_class = metrics / batch_count
        avg_metrics = torch.mean(avg_metrics_per_class, 1)
        # logger.info(f"Avg metrics per class: {avg_metrics_per_class}")
        logger.info(f"IOU: {avg_metrics[0]}   Recall: {avg_metrics[1]}   Precision: {avg_metrics[2]}")
        return avg_val_loss


    # def trainNN(self, model, loss_fn, optimizer, epoch_count, printInfo=False,
    #             acc_arr=None, activation_function=F.relu, onEarlyStop=False, showGraph=False,
    #             train_data_loader=train_loader, val_data_loader=val_loader):
    #
    #     best_vloss = 1000000.  # for early stopping
    #     max_worse = 0
    #     # for graph
    #     train_loss = []
    #     validation_loss = []
    #
    #     for epoch in range(epoch_count):
    #
    #         # Train on (gradient tracking is on)
    #         model.train(True)
    #
    #         # train one epoch
    #         avg_loss = train_epoch(model, loss_fn, optimizer, activation_function, train_data_loader)
    #
    #         # Train off to do reporting
    #         model.train(False)
    #
    #         # Compute validation loss and accuraccy on validation data
    #         running_cum_vloss = 0.0
    #         vcorrect = 0
    #         for vdata in val_data_loader:
    #             vinputs, vlabels = vdata
    #             # get loss on validation data(same computation like in train_epoch for train data)
    #             with torch.no_grad():
    #                 voutputs = model(vinputs)
    #                 vloss = loss_fn(voutputs, vlabels)
    #             running_cum_vloss += vloss * vinputs.shape[0]
    #             # get count the correctly classified samples on val data
    #             vcorrect += (voutputs.argmax(1) == vlabels).float().sum()
    #         # get average loss
    #         avg_vloss = running_cum_vloss / len(Xval)
    #         # get accuraccy
    #         vacc = vcorrect / len(Xval)
    #
    #         # append accuraccy in list for tuning
    #         if (acc_arr != None):
    #             acc_arr.append(torch.round(vacc, decimals=3))
    #
    #         # early stopping
    #         if (onEarlyStop == True):
    #             # If 9 times in a row loss is not higher then existent best loss
    #             if (max_worse >= 9):
    #                 break
    #
    #             # If current loss is lower than best loss, save model
    #             if avg_vloss < best_vloss:
    #                 best_vloss = avg_vloss
    #                 model_path = "saves/best_model.pt"
    #                 torch.save(model.state_dict(), model_path)
    #                 max_worse = 0
    #                 print("Best model saved")
    #             else:
    #                 max_worse += 1
    #
    #                 # append values of loss for train and val data
    #         validation_loss.append(avg_vloss)
    #         train_loss.append(avg_loss)
    #
    #         # print info by one epoch
    #         if (printInfo == True):
    #             print(f"EPOCH {epoch + 1}: ")
    #             print(f"TRAIN loss: {avg_loss:.3f}, VALIDATION loss: {avg_vloss:.3f}, accuraccy: {vacc:.3f}")
    #
    #     # show graph of train and validation loss by epochs
    #     if (showGraph == True):
    #         plt.plot(train_loss, 'or-')
    #         plt.plot(validation_loss, 'ob-')
    #         plt.xlabel('epoch')
    #         plt.ylabel('loss')
    #         plt.legend(['train loss', 'validation loss'])
    #         plt.show()
