'''
This script allows to obtain the regression of the features from the EEG data.

MINIMA RMSE OTTENUTA: 20.1 (con EEGNet, Adam, 200epoche, batch_size=32, lr=0.001, features=resnet18)
'''


# Define options
import argparse
parser = argparse.ArgumentParser(description="Template")

# Data path
parser.add_argument('-ed', '--data_path', default="../data/THINGS-EEG2/", help="Data folder path")

# Subject selection
parser.add_argument('-subj','--subject', default=1, type=int, help="choose a subject from 1 to 6, default is 1")

# Training options
parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch size")
parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float, help="learning rate")
parser.add_argument('-e', '--epochs', default=100, type=int, help="training epochs")

# Model type and optimizer
parser.add_argument('-mt','--model_type', default='EEGNet', help="Select model EEGNet, LSTM, GRU")
parser.add_argument('-ot','--optimizer_type', default='Adam', help="Select optimizer Adam, SGD")

# Feature selection
parser.add_argument('-fm','--features_model', default='resnet18', help="Select model resnet18, efficientnet")

# Parse arguments
opt = parser.parse_args()
print(opt)


# Import libraries
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from Regression_models import EEGNet, EEGNet_orig
from ignite.metrics import Loss, RootMeanSquaredError
import utils

np.seterr(divide='ignore', invalid='ignore') # ignore the warning of division by zero for F1 score
pd.set_option("display.precision", 2) # set the precision of the pandas dataframe


def return_info(data_path, n_subject):
    # Extract EEG data
    eeg_train_path = os.path.join(data_path, 'preprocessed_data', 'sub-' + str(n_subject).zfill(2), 'preprocessed_eeg_training.npy')
    eeg_train = np.load(eeg_train_path, allow_pickle=True).item()
    n_channels, n_times = eeg_train['preprocessed_eeg_data'].shape[2:]

    # Extract length features
    features_path = os.path.join(data_path, 'image_set_features', opt.features_model, 'training_images_70.npy')
    features = np.load(features_path, allow_pickle=True)
    length_features = features.shape[-1]

    return n_channels, n_times, length_features

class Dataset(Dataset):
  # Class to create the dataset
  def __init__(self, data_path, n_subject):
    # EEG data
    eeg_path = os.path.join(data_path, 'preprocessed_data', 'sub-' + str(n_subject).zfill(2), 'preprocessed_eeg_training.npy')
    eeg = np.load(eeg_path, allow_pickle=True).item()
    n_images, n_repetitions, n_channels, n_times = eeg['preprocessed_eeg_data'].shape
    print(eeg['preprocessed_eeg_data'].shape)
    eeg_reshaped = eeg['preprocessed_eeg_data'].reshape((-1, n_channels, n_times))
    # eeg_reshaped = np.mean(eeg['preprocessed_eeg_data'], axis=1) # mean
    print(eeg_reshaped.shape)

    # Features
    features_path = os.path.join(data_path, 'image_set_features', opt.features_model, 'training_images_70.npy')
    features = np.load(features_path, allow_pickle=True)
    features = features.reshape((-1, features.shape[-1])) # reshape to (n_images*n_repetitions, n_features)

    features_rep = np.repeat(features, n_repetitions, axis=0) # select and repeat the concepts for every trial
    # features_rep = np.log(features_rep+0.01) # log transform (to normalize the distribution)

    
    # features_rep = np.ones((features_rep.shape[0], 512))/20 # MOCK FEATURES (uniform 0.5 values)

    self.X = torch.from_numpy(eeg_reshaped).type(torch.float32) # convert float64 to float32
    self.y = torch.from_numpy(features_rep).type(torch.float32) # convert float64 to float32
    
    self.len = features_rep.shape[0]

  def __len__(self):
    return self.len
  
  def __getitem__(self, index):
    return self.X[index], self.y[index]

class Splitter:
    # class to split the dataset
    def __init__(self, dataset, split_name):
        # Set EEG dataset
        self.dataset = dataset

        # Load split index
        train_idx = np.load(os.path.join(opt.data_path, 'image_metadata/split_rep_train.npy'), allow_pickle=True)
        val_idx = np.load(os.path.join(opt.data_path, 'image_metadata/split_rep_val.npy'), allow_pickle=True)

        # Set split index
        if split_name == 'train':
            self.split_idx = train_idx
        elif split_name == 'val':
            self.split_idx = val_idx

        # print(len(self.split_idx))
        # size = 1 # len(self.split_idx)//100
        # opt.batch_size = size
        # self.split_idx = np.random.choice(self.split_idx, size=size, replace=False) # 
        # print(len(self.split_idx))

        # Compute len
        self.len = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.len

    # Get item
    def __getitem__(self, index):
        return self.dataset[self.split_idx[index]]

from matplotlib import pyplot as plt

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    prova_pred = []
    prova_y = []
    for batch in dataloader:
        X = batch[0].to(device)
        y = batch[1].to(device)

        # Compute prediction error
        pred = model(X) # shape: (batch_size, n_features:512)
        loss = loss_fn(pred, y)
        # loss = torch.median(loss)

        # Backpropagation
        optimizer.zero_grad()
        # model.zero_grad()
        loss.backward()
        optimizer.step()

        prova_pred.append(pred.cpu().detach().numpy())
        prova_y.append(y.cpu().detach().numpy())

    print('Pred (min, max, mean):\t', np.min(prova_pred), np.max(prova_pred), np.mean(prova_pred))
    print('Y (min, max, mean):\t', np.min(prova_y), np.max(prova_y), np.mean(prova_y))

    # plot distribution
    prova_pred_np = np.array(prova_pred).flatten()
    prova_y_np = np.array(prova_y).flatten()
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    # axs[0].hist(np.log(np.random.choice(prova_y_np, 10000)), bins=50)
    # axs[1].hist(np.log(np.random.choice(prova_pred_np, 10000)), bins=50)
    axs[0].hist(np.random.choice(prova_y_np, 10000), bins=50)
    axs[1].hist(np.random.choice(prova_pred_np, 10000), bins=50)

    axs[0].set_xlabel('Distribuzione reale y\'=log(y + 0.01)\n(10000 valori campionati casualmente da 16000immagini x 512features)')
    axs[1].set_xlabel('Distribuzione predetta\n(10000 valori campionati casualmente dopo 20 epoche di addestramento)')
    
    plt.show()

def valid(dataloader, model, loss_fn):
    model.eval()
    loss = Loss(loss_fn)
    rmse = RootMeanSquaredError()

    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)

            pred = model(X)

            loss.update((pred, y))
            rmse.update((pred, y))

    print("Loss: ", round(loss.compute(), 4),
          "\tRMSE: ", round(rmse.compute(), 3))
    
    loss.reset()
    rmse.reset()


# verify GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Info
n_channels, n_times, length_features =  return_info(opt.data_path, opt.subject)

# Dataset
dataset = Dataset(opt.data_path, opt.subject)

# Splitter
dataset_train = Splitter(dataset, 'train')
dataset_val = Splitter(dataset, 'val')

# DataLoader
dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, drop_last=True)
dataloader_val = DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=False, drop_last=True)

# Model
models_dict = {'EEGNet': EEGNet(chunk_size=n_times, num_electrodes=n_channels, num_classes=length_features),
            #    'LSTM': LSTM(num_electrodes= n_channels, num_classes = length_features),
            #    'GRU': GRU(num_electrodes= n_channels, num_classes = length_features)
               }
model = models_dict[opt.model_type].to(device)

# model = EEGNet_orig(chunk_size=n_times, num_electrodes=n_channels, num_classes=length_features).to(device) # oringinal EEGNet

# Print model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')

# Load model weights
# PATH = "C:\\Users\\gianl\\Desktop\\Uni Bicocca\\GitHub\\thesis\\trained_models\\Classification_EEG_Concepts\\model_acc0.9_EEGNet.pt"
# model.load_state_dict(torch.load(PATH), strict=False)

# Optimizer
optimizers_dict = {'Adam' : torch.optim.Adam(model.parameters(), lr=opt.learning_rate),
                   'SGD' : torch.optim.SGD(model.parameters(), lr=opt.learning_rate)}
optimizer = optimizers_dict[opt.optimizer_type]
# optimizer = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# LOSS FUNCTION
class CLIPLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = 0.07 # da cambiare
    
    def forward(self, pred, actual):
        def cross_entropy(preds, targets, reduction='none'):
            log_softmax = torch.nn.LogSoftmax(dim=-1)
            loss = (-targets * log_softmax(preds)).sum(1)
            if reduction == "none":
                return loss
            elif reduction == "mean":
                return loss.mean()
            
        logits = (pred @ torch.transpose(actual, 0, 1)) / self.temperature
        pred_similarity = pred @ torch.transpose(pred, 0, 1)
        actual_similarity = actual @ torch.transpose(actual, 0, 1)
        targets = torch.softmax(
            (pred_similarity + actual_similarity) / 2 * self.temperature, dim=-1
        )
        pred_loss = cross_entropy(logits, targets, reduction='none')
        actual_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (pred_loss + actual_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    

    
    # def forward(self, eeg, img):
    #     eeg = torch.nn.functional.normalize(eeg, p=2, dim=1)
    #     img = torch.nn.functional.normalize(img, p=2, dim=1)

    #     logits = torch.dot(eeg, img.T) * np.exp(self.temperature)

    #     torch.nn.CrossEntropyLoss()
    #     loss_i = cross_entropy_loss(logits, img, axis=0)
    #     loss_t = cross_entropy_loss(logits, img, axis=1) 
    #     loss = (loss_i + loss_t)/2

class RMSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
    
# loss_fn = CLIPLoss() # CLIP loss
# loss_fn = RMSLELoss() # RMSLE loss
loss_fn = torch.nn.MSELoss() # mean squared error loss
# loss_fn = torch.nn.HuberLoss() # Huber loss
# loss_fn = torch.nn.L1Loss() # mean absolute error loss

# Training loop
for t in range(opt.epochs):
    print(f"------------ Epoch {t+1} ---------------")
    train(dataloader_train, model, loss_fn, optimizer)
    valid(dataloader_val, model, loss_fn)
print("Done!") 

# Save model
utils.save_model(model, metric=0, name=os.path.basename(__file__)[:-2])