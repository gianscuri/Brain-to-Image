'''
This script allows to classify the EEG data into the concepts of the THINGS-EEG2 dataset.

NB. As the concepts in the training set (1654) and the test set (200) are different,
it is necessary to choose one of the two partitions on which to carry out the classification.
The test set has fewer classes and more repetitions of the same visual signal and therefore simplifies the classification
'''


# Define options
import argparse
parser = argparse.ArgumentParser(description="Template")

# Data path
parser.add_argument('-ed', '--data_path', default="../data/THINGS-EEG2/", help="Data folder path")

# Training options
parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch size")
parser.add_argument('-lr', '--learning-rate', default=0.0005, type=float, help="learning rate")
parser.add_argument('-e', '--epochs', default=100, type=int, help="training epochs")

# Model type/options
parser.add_argument('-ot','--optimizer_type', default='Adam', help="Select optimizer Adam, SGD")

# Feature selection
parser.add_argument('-fm','--features_model', default='resnet18', help="Select model resnet18, efficientnet")
parser.add_argument('-nc','--num_comp', default=100, help="Select [512, 200, 100, 50]")

# Parse arguments
opt = parser.parse_args()
print(opt)


# Import libraries
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from Classification_FEAT_models import FC
from ignite.metrics import TopKCategoricalAccuracy, Loss


def return_info(data_path):

    # Labels
    image_metadata_path = os.path.join(data_path, 'image_metadata/image_metadata.npy')
    image_metadata = np.load(image_metadata_path, allow_pickle=True).item()
    n_classes = len(set(image_metadata['train_img_concepts']))

    return n_classes
   
class Dataset(Dataset):
  # Class to create the dataset
  def __init__(self, data_path):
    # Features
    file = f'training_images_{opt.num_comp}_mean.npy' if opt.num_comp != 512 else 'training_images.npy'
    features_path = os.path.join(data_path, 'image_set_features', opt.features_model, file)
    features = np.load(features_path, allow_pickle=True)
    features = features.reshape((-1, features.shape[-1])) # reshape to (n_images*n_repetitions, n_features)

    # Labels
    image_metadata_path = os.path.join(data_path, 'image_metadata/image_metadata.npy')
    image_metadata = np.load(image_metadata_path, allow_pickle=True).item()

    le = LabelEncoder()
    le.fit(image_metadata['train_img_concepts'])
    idx = le.transform(image_metadata['train_img_concepts'])
    concepts_int = idx
    print('Number of classes:', len(set(idx)),
          '\nRandom chance Top1:', round(1/len(set(idx))*100, 2), '%' )

    self.X = torch.from_numpy(features).type(torch.float32) # convert float64 to float32
    self.y = torch.from_numpy(concepts_int).type(torch.LongTensor) # convert float64 to Long
    self.len = self.X.shape[0]

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
        train_idx = np.load(os.path.join(opt.data_path, 'image_metadata/split_train.npy'), allow_pickle=True)
        val_idx = np.load(os.path.join(opt.data_path, 'image_metadata/split_val.npy'), allow_pickle=True)
        
        # Set split index
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

def valid(dataloader, model):
    model.eval()

    top1.reset()
    top3.reset()
    top5.reset()
    loss.reset()

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
    
    return top1.compute()*100, top5.compute()*100 # return top1 accuracy
    


# verify GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Info
n_classes =  return_info(opt.data_path)

# Dataset
dataset = Dataset(opt.data_path)

# Splitter
dataset_train = Splitter(dataset, 'train')
dataset_val = Splitter(dataset, 'val')

# DataLoader
dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, drop_last=True)
dataloader_val = DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=False, drop_last=True)

# Model
model = FC(input_dim=opt.num_comp, output_dim=n_classes).to(device)
model.eval()


# Training
optimizers_dict = {'Adam' : torch.optim.Adam(model.parameters(), lr=opt.learning_rate),
                   'SGD' : torch.optim.SGD(model.parameters(), lr=opt.learning_rate)}
optimizer = optimizers_dict[opt.optimizer_type]
loss_fn = torch.nn.CrossEntropyLoss()

# Metrics
top1 = TopKCategoricalAccuracy(k=1)
top3 = TopKCategoricalAccuracy(k=3)
top5 = TopKCategoricalAccuracy(k=5)
loss = Loss(loss_fn)

import copy
acc_best = 0
for t in range(opt.epochs):
    print(f"------------ Epoch {t+1} ---------------")
    train(dataloader_train, model, loss_fn, optimizer)
    top1acc, top5acc = valid(dataloader_val, model)

    if top1acc > acc_best:
        acc_best = copy.copy(top1acc)
        acc5_best = copy.copy(top5acc)


print("Done!") 
print("Best Top1: ", round(acc_best, 2), "\tBest Top5: ", round(acc5_best, 2))