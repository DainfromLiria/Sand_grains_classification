import torch
from torchvision.models.segmentation import deeplabv3_resnet101


class MicroTextureDetector:

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        self.model = deeplabv3_resnet101(pretrained=True)
        self.model.to(device)
        self.data_loader = None  # TODO

    def _train_one_epoch(self, model, loss_fn, optimizer, activation_function, data_loader):
        running_cum_loss = 0

        for data in data_loader:
            inputs, labels = data

            optimizer.zero_grad()  # Sets the gradients of all optimized torch Tensors to zero.
            outputs = model(inputs, activation_function)  # forward
            loss = loss_fn(outputs, labels)  # compute loss
            loss.backward()  # backward prop
            optimizer.step()  # step of input optimizer

            # find train loss after one epoch and return it
            last_mean_loss = loss.item()  # get loss on this batch
            running_cum_loss += last_mean_loss * inputs.shape[
                0]  # sum of multiplys loss*size_of_batch(last batch not always 32) for compute average train loss
        return running_cum_loss / len(Xtrain)  # return average train loss after one epoch


    def trainNN(self, model, loss_fn, optimizer, epoch_count, printInfo=False,
                acc_arr=None, activation_function=F.relu, onEarlyStop=False, showGraph=False,
                train_data_loader=train_loader, val_data_loader=val_loader):

        best_vloss = 1000000.  # for early stopping
        max_worse = 0
        # for graph
        train_loss = []
        validation_loss = []

        for epoch in range(epoch_count):

            # Train on (gradient tracking is on)
            model.train(True)

            # train one epoch
            avg_loss = train_epoch(model, loss_fn, optimizer, activation_function, train_data_loader)

            # Train off to do reporting
            model.train(False)

            # Compute validation loss and accuraccy on validation data
            running_cum_vloss = 0.0
            vcorrect = 0
            for vdata in val_data_loader:
                vinputs, vlabels = vdata
                # get loss on validation data(same computation like in train_epoch for train data)
                with torch.no_grad():
                    voutputs = model(vinputs)
                    vloss = loss_fn(voutputs, vlabels)
                running_cum_vloss += vloss * vinputs.shape[0]
                # get count the correctly classified samples on val data
                vcorrect += (voutputs.argmax(1) == vlabels).float().sum()
            # get average loss
            avg_vloss = running_cum_vloss / len(Xval)
            # get accuraccy
            vacc = vcorrect / len(Xval)

            # append accuraccy in list for tuning
            if (acc_arr != None):
                acc_arr.append(torch.round(vacc, decimals=3))

            # early stopping
            if (onEarlyStop == True):
                # If 9 times in a row loss is not higher then existent best loss
                if (max_worse >= 9):
                    break

                # If current loss is lower than best loss, save model
                if avg_vloss < best_vloss:
                    best_vloss = avg_vloss
                    model_path = "saves/best_model.pt"
                    torch.save(model.state_dict(), model_path)
                    max_worse = 0
                    print("Best model saved")
                else:
                    max_worse += 1

                    # append values of loss for train and val data
            validation_loss.append(avg_vloss)
            train_loss.append(avg_loss)

            # print info by one epoch
            if (printInfo == True):
                print(f"EPOCH {epoch + 1}: ")
                print(f"TRAIN loss: {avg_loss:.3f}, VALIDATION loss: {avg_vloss:.3f}, accuraccy: {vacc:.3f}")

        # show graph of train and validation loss by epochs
        if (showGraph == True):
            plt.plot(train_loss, 'or-')
            plt.plot(validation_loss, 'ob-')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(['train loss', 'validation loss'])
            plt.show()
