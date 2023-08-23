# Define options
import argparse
parser = argparse.ArgumentParser(description="Template")

# Data path
parser.add_argument('-ed', '--data_path', default="../data/THINGS-EEG2/", help="Data folder path")

# Subject selecting
parser.add_argument('-subj','--subject', default=1, type=int, help="choose a subject from 1 to 6, default is 1")

# Subsect selecting
parser.add_argument('-subs','--subsect', default='test', type=str, help="choose a subset between training (1654 classes) or test (200 classes)")

# Training options
parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch size")
parser.add_argument('-lr', '--learning-rate', default=0.01, type=float, help="learning rate")
parser.add_argument('-e', '--epochs', default=200, type=int, help="training epochs")

# Model type/options
parser.add_argument('-mt','--model_type', default='EEGNet_sigmoid', help="Select model EEGNet, LSTM, GRU")
parser.add_argument('-ot','--optimizer_type', default='Adam', help="Select optimizer Adam, SGD")

# Parse arguments
opt = parser.parse_args()
print(opt)


# Import libraries
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
# from torcheeg import models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from Classification_models import EEGNet, LSTM, GRU, EEGNet_sigmoid
from ignite.metrics import TopKCategoricalAccuracy, Loss


def return_info(data_path, subset, n_subject):
    # Extract EEG data
    eeg_path = os.path.join(data_path, 'preprocessed_data', 'sub-' + str(n_subject).zfill(2), 'preprocessed_eeg_' + subset + '.npy')
    eeg = np.load(eeg_path, allow_pickle=True).item()
    n_channels, n_times = eeg['preprocessed_eeg_data'].shape[2:]

    # Labels
    image_metadata_path = os.path.join(data_path, 'image_metadata/image_metadata.npy')
    image_metadata = np.load(image_metadata_path, allow_pickle=True).item()
    n_classes = len(set(image_metadata[f'{subset[:5]}_img_concepts']))

    return n_channels, n_times, n_classes
   
class EEGDataset(Dataset):
  # Class to create the dataset
  def __init__(self, data_path, subset, n_subject):
    # EEG data
    eeg_path = os.path.join(data_path, 'preprocessed_data', 'sub-' + str(n_subject).zfill(2), 'preprocessed_eeg_' + subset + '.npy')
    eeg = np.load(eeg_path, allow_pickle=True).item()
    n_images, n_repetitions, n_channels, n_times = eeg['preprocessed_eeg_data'].shape
    eeg_reshaped = eeg['preprocessed_eeg_data'].reshape((-1, n_channels, n_times))

    # Labels
    image_metadata_path = os.path.join(data_path, 'image_metadata/image_metadata.npy')
    image_metadata = np.load(image_metadata_path, allow_pickle=True).item()
    n_classes = len(set(image_metadata[f'{subset[:5]}_img_concepts']))

    le = LabelEncoder()
    le.fit(image_metadata[f'{subset[:5]}_img_concepts'])
    idx = le.transform(image_metadata[f'{subset[:5]}_img_concepts'])
    concepts_int = np.repeat(idx, n_repetitions)

    self.X = torch.from_numpy(eeg_reshaped.astype(np.float32)) # convert float64 to float32
    self.y = torch.from_numpy(concepts_int).type(torch.LongTensor) # convert float64 to Long
    self.len = self.X.shape[0]

  def __len__(self):
    return self.len
  
  def __getitem__(self, index):
    return self.X[index], self.y[index]
  
class Splitter:
    # class to split the dataset
    def __init__(self, dataset, split_name, seed=42):
        # Set EEG dataset
        self.dataset = dataset

        # Create split index
        stratify = [i[1] for i in iter(self.dataset)]
        train_idx, val_idx = train_test_split(range(len(stratify)), test_size=0.25, stratify=stratify, random_state=seed) # stratified split
        
        if split_name == 'train':
            self.split_idx = train_idx
        elif split_name == 'val':
            self.split_idx = val_idx

        # Compute len
        self.len = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.len

    # Get item
    def __getitem__(self, index):
        return self.dataset[self.split_idx[index]]

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch in dataloader:
        X = batch[0].to(device)
        y = batch[1].to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def valid(dataloader, model, loss_fn):
    model.eval()
    top1 = TopKCategoricalAccuracy(k=1)
    top3 = TopKCategoricalAccuracy(k=3)
    top5 = TopKCategoricalAccuracy(k=5)
    loss = Loss(loss_fn)

    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)
            pred = model(X)

            top1.update((pred, y))
            top3.update((pred, y))
            top5.update((pred, y))
            loss.update((pred, y))

    print("Top1: ", round(top1.compute()*100, 1),
          "\tTop3: ", round(top3.compute()*100, 1),
          "\tTop5: ", round(top5.compute()*100, 1),
          "\tloss: ", round(loss.compute(), 3))
    
    top1.reset()
    top3.reset()
    top5.reset()
    loss.reset()


# verify GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Info
n_channels, n_times, n_classes =  return_info(opt.data_path, opt.subsect, opt.subject)

# Dataset
dataset = EEGDataset(opt.data_path, opt.subsect, opt.subject)

# Splitter
seed = 42 # common seed to avoid overlapping splits
dataset_train = Splitter(dataset, 'train', seed=seed)
dataset_val = Splitter(dataset, 'val', seed=seed)

# DataLoader
dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, drop_last=True)
dataloader_val = DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=False, drop_last=True)

# Model
models_dict = {'EEGNet': EEGNet(chunk_size=n_times, num_electrodes=n_channels, num_classes=n_classes),
               'EEGNet_sigmoid': EEGNet_sigmoid(chunk_size=n_times, num_electrodes=n_channels, num_classes=n_classes),
               'LSTM': LSTM(num_electrodes= n_channels, num_classes = n_classes),
               'GRU': GRU(num_electrodes= n_channels, num_classes = n_classes)}

model = models_dict[opt.model_type].to(device)

# Training
optimizers_dict = {'Adam' : torch.optim.Adam(model.parameters(), lr=opt.learning_rate),
                   'SGD' : torch.optim.SGD(model.parameters(), lr=opt.learning_rate)}
optimizer = optimizers_dict[opt.optimizer_type]
loss_fn = torch.nn.CrossEntropyLoss()

for t in range(opt.epochs):
    print(f"------------ Epoch {t+1} ---------------")
    train(dataloader_train, model, loss_fn, optimizer)
    valid(dataloader_val, model, loss_fn)
print("Done!") 