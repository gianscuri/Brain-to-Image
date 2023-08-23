# Define options
import argparse
parser = argparse.ArgumentParser(description="Template")

# Data path
parser.add_argument('-ed', '--data_path', default="../data/THINGS-EEG2/", help="Data folder path")

# Subject selecting
parser.add_argument('-subj','--subject', default=1, type=int, help="choose a subject from 1 to 6, default is 1")

# # Number of high-level categories
# parser.add_argument('-n_cat','--categories', default=27, type=int, help="choose the number of high-level categories between 27 and , default is 27")

# Training options
parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch size")
parser.add_argument('-lr', '--learning-rate', default=0.01, type=float, help="learning rate")
parser.add_argument('-e', '--epochs', default=200, type=int, help="training epochs")

# Model type/options
parser.add_argument('-mt','--model_type', default='EEGNet', help="Select model EEGNet, LSTM, GRU")
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
from Classification_models import EEGNet, LSTM, GRU
from ignite.metrics import Loss, Precision, Recall, Fbeta, MultiLabelConfusionMatrix


def return_info(data_path, n_subject):
    # Extract EEG data
    eeg_train_path = os.path.join(data_path, 'preprocessed_data', 'sub-' + str(n_subject).zfill(2), 'preprocessed_eeg_training.npy')
    eeg_train = np.load(eeg_train_path, allow_pickle=True).item()
    n_channels, n_times = eeg_train['preprocessed_eeg_data'].shape[2:]
    n_classes = 27

    # Compute weigths for BCEWithLogitsLoss
    category27_mat_path = os.path.join(data_path, 'image_metadata/category_mat_manual.tsv')
    category27_mat = pd.read_csv(category27_mat_path, sep='\t')
    pos_weight = list(len(category27_mat)/np.sum(category27_mat))

    return n_channels, n_times, n_classes, pos_weight


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
    category27_mat_path = os.path.join(data_path, 'image_metadata/category_mat_manual.tsv')
    category27_mat = pd.read_csv(category27_mat_path, sep='\t')

    concepts = sorted([i[6:] for i in set(image_metadata['train_img_concepts'] + image_metadata['test_img_concepts'])])
    concept_selector_list = [i[6:] for i in image_metadata[f'{subset[:5]}_img_concepts']] # concepts in the partitions
    concepts_selector_idx = [i in concept_selector_list for i in concepts] # boolean list to select the concepts in the partition
    rep = n_images * n_repetitions / sum(concepts_selector_idx)
    labels = np.repeat(category27_mat[concepts_selector_idx].to_numpy(), rep, axis=0) # select and repeat the concepts for every trial

    self.X = torch.from_numpy(eeg_reshaped.astype(np.float32)) # convert float64 to float32
    self.y = torch.from_numpy(labels).type(torch.float32) # convert float64 to float32
    self.len = self.X.shape[0]

  def __len__(self):
    return self.len
  
  def __getitem__(self, index):
    return self.X[index], self.y[index]


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
    conf = MultiLabelConfusionMatrix(num_classes=27)

    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)
            pred = model(X)

            loss.update((pred, y))

            pred_bin = torch.where(pred > 0.5, 1, 0) # 0 or 1 with a trheshold of 0.5
            conf.update((pred_bin.type(torch.int32), y.type(torch.int32)))
            prec.update((pred_bin.type(torch.int32), y.type(torch.int32)))
            rec.update((pred_bin.type(torch.int32), y.type(torch.int32)))

    p = torch.mean(prec.compute()).item()*100
    r = torch.mean(rec.compute()).item()*100
    f = 2*p*r/(p+r)
    print("Loss:\t\t", round(loss.compute(), 3),
          "\nMean F1:\t", round(f, 3),
          "\nMean Precision:\t", round(p, 3),
          "\nMean Recall:\t", round(r, 3))
    
    if full_stats:
        print("Confusion matrix:\n", conf.compute().numpy())
    
    loss.reset()
    conf.reset()
    prec.reset()
    rec.reset()


# verify GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Info
n_channels, n_times, n_classes, pos_weight =  return_info(opt.data_path, opt.subject)

# Dataset
dataset_train = EEGDataset(opt.data_path, 'training', opt.subject)
dataset_val = EEGDataset(opt.data_path, 'test', opt.subject)

# DataLoader
dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, drop_last=True)
dataloader_val = DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=False, drop_last=True)

# Model
models_dict = {'EEGNet': EEGNet(chunk_size=n_times, num_electrodes=n_channels, num_classes=n_classes),
               'LSTM': LSTM(num_electrodes= n_channels, num_classes = n_classes),
               'GRU': GRU(num_electrodes= n_channels, num_classes = n_classes)}

model = models_dict[opt.model_type].to(device)

# Optimizer
optimizers_dict = {'Adam' : torch.optim.Adam(model.parameters(), lr=opt.learning_rate),
                   'SGD' : torch.optim.SGD(model.parameters(), lr=opt.learning_rate)}
optimizer = optimizers_dict[opt.optimizer_type]

# Loss function
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device)) # pos_weight > 1 increases the recall, pos_weight < 1 increases the precision, len(pos_weigth) = n_classes

# Training loop
for t in range(opt.epochs):
    full_stats = True if t % 5 == 0 else False # print the confusion matrix every 5 epochs

    print(f"------------ Epoch {t+1} ---------------")
    train(dataloader_train, model, loss_fn, optimizer)
    valid(dataloader_val, model, loss_fn, full_stats)
print("Done!") 