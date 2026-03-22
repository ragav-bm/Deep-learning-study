import torch as t
from dask.dataframe.io.tests.test_parquet import test_local
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

        # self._scheduler = t.optim.lr_scheduler.StepLR(self._optim, step_size=5, gamma=0.5)

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})

    def train_step(self, x, y):

        self._optim.zero_grad()
        output = self._model(x)
        # y = y.long()
        # print(output.shape)
        # print(y.shape)
        loss = self._crit(output, y)
        loss.backward()
        self._optim.step()
        return loss.item()
        #TODO



    def val_test_step(self, x, y):

        # predic
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions

        #TODO
        output = self._model(x)
        loss = self._crit(output, y)
        return loss.item(), output

    def train_epoch(self):
        # set training mode
        self._model.train()
        # iterate through the training set

        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        #TODO
        t_loss  =0
        # print(self._train_dl.shape)
        for x, y in self._train_dl:
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            loss = self.train_step(x, y)
            t_loss += loss
        return t_loss / len(self._train_dl)


    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        #TODO
        self._model.eval()

        t_loss =0
        prediction = []
        labels = []
        with t.no_grad():
            for x, y in self._val_test_dl:
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()
                loss, output = self.val_test_step(x, y)
                # print(output.shape)
                t_loss = t_loss + loss
                output = t.sigmoid(output)
                output = (output>0.3).float()
                prediction.extend(output.cpu().numpy())
                labels.extend(y.cpu().numpy())
        score = f1_score(labels, prediction, average='macro')

        print(f'Validation Loss: {t_loss / len(self._val_test_dl)}, F1 Score: {score}')
        return t_loss/len(self._val_test_dl)


    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0

        epoch =0
        train_loss = []
        val_loss = []
        count =0
        high_loss = float('inf')


        while epochs != epoch:
            epoch += 1
            t_loss = self.train_epoch()
            v_loss = self.val_test()
            train_loss.append(t_loss)
            val_loss.append(v_loss)

            # self._scheduler.step()

            if v_loss < high_loss:
                high_loss = v_loss
                self.save_checkpoint(epoch)
                count = 0
            else:
                count += 1

            if count >= self._early_stopping_patience and self._early_stopping_patience >0:
                print(f'Early stopping patience: {self._early_stopping_patience}')
                break


        return train_loss, val_loss



            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
        #TODO



