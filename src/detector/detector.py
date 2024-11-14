import logging

import torch
from torch.utils.data import DataLoader
from torchvision.models.segmentation import (DeepLabV3_ResNet101_Weights,
                                             deeplabv3_resnet101)
from tqdm import tqdm

from configs import config
from dataset import SandGrainsDataset

logger = logging.getLogger(__name__)


class MicroTextureDetector:

    def __init__(self):
        torch.hub.set_dir(config.model.MODELS_DIR_PATH)  # TODO move to some setup file
        device = config.model.DEVICE
        logger.info(f"Device: {device}")
        self.model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
        self.model.to(device)
        self._make_data_loader()

    def train(self) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        activation_function = torch.nn.ReLU()
        loss_fn = torch.nn.CrossEntropyLoss()
        self._train_loop(optimizer, activation_function, loss_fn)

    def _make_data_loader(self):
        self.train_dataset = SandGrainsDataset(path=config.data.AUG_TRAIN_SET_PATH)
        logger.info(f"Train dataset size: {len(self.train_dataset)}")
        self.val_dataset = SandGrainsDataset(path=config.data.AUG_VAL_SET_PATH)
        logger.info(f"Val dataset size: {len(self.val_dataset)}")
        self.test_dataset = SandGrainsDataset(path=config.data.AUG_TEST_SET_PATH)
        logger.info(f"Test dataset size: {len(self.test_dataset)}")
        # TODO add num_workers
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=config.model.BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=config.model.BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=config.model.BATCH_SIZE, shuffle=True)

    def _train_one_epoch(self, optimizer, activation_function, loss_fn) -> float:
        """
        Train one epoch

        :param optimizer: optimizer
        :param activation_function: activation function for model hidden layers
        :param loss_fn: loss function

        :return: average train loss for current epoch
        """
        running_cum_loss = 0
        for images, masks in tqdm(self.train_loader, desc="Train one Epoch"):
            device = config.model.DEVICE
            images, masks = images.float().to(device), masks.float().to(device)
            optimizer.zero_grad()  # reset the gradients for new batch
            outputs = self.model(images)  # forward
            loss = loss_fn(outputs, masks)  # compute loss
            loss.backward()  # backward
            optimizer.step()  # step of input optimizer

            # mul on batch size because loss is avg loss for batch, so loss=loss/batch_size
            running_cum_loss += loss.item() * images.shape[0]
        return running_cum_loss / len(self.train_dataset)

    def _train_loop(self, optimizer, activation_function, loss_fn) -> None:
        for epoch in tqdm(range(config.model.EPOCH_COUNT), desc="Train model"):
            self.model.train()  # set model in training mode.
            epoch_loss = self._train_one_epoch(
                optimizer=optimizer,
                activation_function=activation_function,
                loss_fn=loss_fn
            )
            self.model.eval()  # set model in evaluation mode.

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
