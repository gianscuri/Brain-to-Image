'''
This script allows to classify the EEG data into the 27 high level categories of the THINGS-EEG2 dataset.

ROTTO
'''


# Define options
import argparse
parser = argparse.ArgumentParser(description="Template")

# Data path
parser.add_argument('-ed', '--data_path', default="../data/THINGS-EEG2/", help="Data folder path")

# Subsect selecting
parser.add_argument('-subs','--subsect', default='training', type=str, help="choose a subset between training (1654 classes) or test (200 classes)")

# Training options
parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch size")
parser.add_argument('-lr', '--learning-rate', default=0.0005, type=float, help="learning rate")
parser.add_argument('-e', '--epochs', default=30, type=int, help="training epochs")

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
from Classification_FEAT_models import FC
from ignite.metrics import Loss, Precision, Recall

np.seterr(divide='ignore', invalid='ignore') # ignore the warning of division by zero for F1 score
pd.set_option("display.precision", 2) # set the precision of the pandas dataframe


def return_info(data_path, subset):
    # Compute weigths for BCEWithLogitsLoss
    category27_mat_path = os.path.join(data_path, 'image_metadata/category_mat_manual.tsv')
    category27_mat = pd.read_csv(category27_mat_path, sep='\t')
    pos_weight = list(np.sum(category27_mat == 0) / np.sum(category27_mat))
    categories_sum = category27_mat.sum()

    # Labels
    n_classes = 27

    return n_classes, pos_weight, categories_sum

class Dataset(Dataset):
  # Class to create the dataset
  def __init__(self, data_path):
    # Features
    features_path = os.path.join(data_path, 'image_set_features', opt.features_model, 'training_images.npy')
    features = np.load(features_path, allow_pickle=True)
    features = features.reshape((-1, features.shape[-1])) # reshape to (n_concepts(1654)*n_images(10), n_features)

    # Labels
    image_metadata_path = os.path.join(data_path, 'image_metadata/image_metadata.npy')
    image_metadata = np.load(image_metadata_path, allow_pickle=True).item()
    category27_mat_path = os.path.join(data_path, 'image_metadata/category_mat_manual.tsv')
    category27_mat = pd.read_csv(category27_mat_path, sep='\t')

    concepts = sorted([i[6:] for i in set(image_metadata['train_img_concepts'] + image_metadata['test_img_concepts'])])
    concept_selector_list = [i[6:] for i in image_metadata['train_img_concepts']] # concepts in the partitions
    concepts_selector_idx = [i in concept_selector_list for i in concepts] # boolean list to select the concepts in the partition
    # labels = category27_mat[concepts_selector_idx].to_numpy() # ???
    rep = len(features) / sum(concepts_selector_idx)
    labels = np.repeat(category27_mat[concepts_selector_idx].to_numpy(), rep, axis=0) # select and repeat the concepts for every trial

    self.X = torch.from_numpy(features.astype(np.float32)) # convert float64 to float32
    self.y = torch.from_numpy(labels).type(torch.float32) # convert float64 to float32
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
        test_idx = np.load(os.path.join(opt.data_path, 'image_metadata/split_test.npy'), allow_pickle=True)

        # Set split index
        if split_name == 'train':
            self.split_idx = train_idx
        elif split_name == 'val':
            self.split_idx = np.concatenate((val_idx, test_idx))

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

def valid(dataloader, model, loss_fn, full_stats):
    model.eval()
    loss = Loss(loss_fn)
    prec = Precision(is_multilabel=True)
    rec = Recall(is_multilabel=True)

    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)

            pred = model(X)
            pred_bin = torch.where(pred > 0.5, 1, 0) # 0 or 1 with a threshold of 0.5

            loss.update((pred, y))
            prec.update((pred_bin, y))
            rec.update((pred_bin, y))

    p = prec.compute().numpy()*100 # precision for each category
    r = rec.compute().numpy()*100 # recall for each category
    f = 2*p*r/(p+r) # F1 score for each category

    stats = pd.DataFrame([[np.nanmean(f), np.mean(p), np.mean(r), loss.compute()]], index=['MEAN'], columns=['F1', 'Precision', 'Recall', 'Loss'])
    
    if full_stats:
        stats = pd.concat([stats, pd.DataFrame(zip(f, p, r, categories_sum, f/categories_sum*100), index=categories_sum.index, columns=['F1', 'Precision', 'Recall', 'N concepts', 'F1/N conc'])]) # add the stats for each category

    print(stats)
    
    loss.reset()
    prec.reset()
    rec.reset()


# verify GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Info
n_classes, pos_weight, categories_sum =  return_info(opt.data_path, opt.subsect)

# Dataset
dataset = Dataset(opt.data_path)

# Splitter
dataset_train = Splitter(dataset, 'train')
dataset_val = Splitter(dataset, 'val')

# DataLoader
dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, drop_last=True)
dataloader_val = DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=False, drop_last=True)

# Model
model = FC(input_dim=512, output_dim=n_classes).to(device)
model.eval()

# Optimizer
optimizers_dict = {'Adam' : torch.optim.Adam(model.parameters(), lr=opt.learning_rate),
                   'SGD' : torch.optim.SGD(model.parameters(), lr=opt.learning_rate)}
optimizer = optimizers_dict[opt.optimizer_type]

# Loss function
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device)) # pos_weight > 1 increases the recall, pos_weight < 1 increases the precision, len(pos_weigth) = n_classes

# Training loop
for t in range(opt.epochs):
    full_stats = True if t % 5 == 0 else False # print the stats for each category every 5 epochs

    print(f"------------ Epoch {t+1} ---------------")
    train(dataloader_train, model, loss_fn, optimizer)
    valid(dataloader_val, model, loss_fn, full_stats)
print("Done!") 