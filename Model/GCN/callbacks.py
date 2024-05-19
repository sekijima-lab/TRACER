import torch
import numpy as np


class EarlyStopping:

    def __init__(self, patience=5, verbose=False, path='checkpoint_model.pth'):

        self.patience = patience    # stop cpunter
        self.verbose = verbose 
        self.counter = 0            # current counter
        self.best_score = None      # best score
        self.early_stop = False     # stop flag
        self.val_loss_min = np.Inf   # to memorize previous best score
        self.path = path             # path to save the best model

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:  #1Epoch
            self.best_score = score
            self.checkpoint(val_loss, model)  # save model and show score
        elif score < self.best_score:  # if it can not update best score
            self.counter += 1   # stop counter +1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:  # if it update best score
            self.best_score = score
            self.checkpoint(val_loss, model) # save model and show score
            self.counter = 0  # stop counter is reset

    def checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss