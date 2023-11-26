'''
This script allows to classify the EEG data into the 27 high level categories of the THINGS-EEG2 dataset.
'''


# Define options
import argparse
parser = argparse.ArgumentParser(description="Template")

# Data path
parser.add_argument('-ed', '--data_path', default="../data/THINGS-EEG2/", help="Data folder path")

# Subject selection
parser.add_argument('-subj','--subject', default=1, type=int, help="choose a subject from 1 to 6, default is 1")

# Training options
parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float, help="learning rate")
parser.add_argument('-e', '--epochs', default=30, type=int, help="training epochs")

# Model type and optimizer
parser.add_argument('-mt','--model_type', default='EEGNet', help="Select model [EEGNet, LSTM, TSconv, Spectr]")
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
from Classification_EEG_models import EEGNet, easy, spectr_model, TSconv
from ignite.metrics import Loss, Precision, Recall, Accuracy, Fbeta
import torchaudio
# from copy import copy, deepcopy

np.seterr(divide='ignore', invalid='ignore') # ignore the warning of division by zero for F1 score
pd.set_option("display.precision", 3) # set the precision of the pandas dataframe


def return_info(data_path, n_subject):
    # Extract EEG data
    eeg_train_path = os.path.join(data_path, 'preprocessed_data', 'sub-' + str(n_subject).zfill(2), 'preprocessed_eeg_training.npy')
    eeg_train = np.load(eeg_train_path, allow_pickle=True).item()
    n_channels, n_times = eeg_train['preprocessed_eeg_data'].shape[2:]
    n_classes = 27

    # Compute weigths for BCEWithLogitsLoss
    category27_mat_path = os.path.join(data_path, 'image_metadata/category_mat_manual.tsv')
    category27_mat = pd.read_csv(category27_mat_path, sep='\t')
    # pos_weight = list(np.sum(category27_mat == 0) / np.sum(category27_mat))
    pos_weight = list(1854/np.sum(category27_mat))
    categories_sum = category27_mat.sum()

    return n_channels, n_times, n_classes, pos_weight, categories_sum

class Dataset(Dataset):
  # Class to create the dataset
  def __init__(self, data_path, n_subject, transform):
    self.transform = transform

    # EEG data
    eeg_path = os.path.join(data_path, 'preprocessed_data', 'sub-' + str(n_subject).zfill(2), 'preprocessed_eeg_training.npy')
    eeg = np.load(eeg_path, allow_pickle=True).item()
    n_images, n_repetitions, n_channels, n_times = eeg['preprocessed_eeg_data'].shape
    eeg_reshaped = eeg['preprocessed_eeg_data'].reshape((-1, n_channels, n_times))
    # Labels
    image_metadata_path = os.path.join(data_path, 'image_metadata/image_metadata.npy')
    image_metadata = np.load(image_metadata_path, allow_pickle=True).item()
    category27_mat_path = os.path.join(data_path, 'image_metadata/category_mat_manual.tsv')
    category27_mat = pd.read_csv(category27_mat_path, sep='\t')

    concepts = sorted([i[6:] for i in set(image_metadata['train_img_concepts'] + image_metadata['test_img_concepts'])])
    concept_selector_list = [i[6:] for i in image_metadata['train_img_concepts']] # concepts in the partitions
    concepts_selector_idx = [i in concept_selector_list for i in concepts] # boolean list to select the concepts in the partition
    rep = n_images * n_repetitions / sum(concepts_selector_idx)
    labels = np.repeat(category27_mat[concepts_selector_idx].to_numpy(), rep, axis=0) # select and repeat the concepts for every trial

    self.X = torch.from_numpy(eeg_reshaped.astype(np.float32)) # convert float64 to float32
    self.y = torch.from_numpy(labels).type(torch.float32) # convert float64 to float32
    self.len = self.X.shape[0]

    if self.transform:
        self.X = self.transform(self.X)

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
        test_idx = np.load(os.path.join(opt.data_path, 'image_metadata/split_rep_test.npy'), allow_pickle=True)

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
    rec = Recall(is_multilabel=True) # , average='weighted'
    acc = Accuracy(is_multilabel=True)

    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)

            pred = model(X)
            pred_bin = torch.where(pred > 0.95, 1, 0) # 0 or 1 with a threshold of 0.5

            loss.update((pred, y))
            prec.update((pred_bin, y))
            rec.update((pred_bin, y))
            acc.update((pred_bin, y))

    p = prec.compute().numpy()*100 # precision for each category
    r = rec.compute().numpy()*100 # recall for each category
    a = acc.compute()*100 # accuracy for each category
    f = 2*p*r/(p+r) # F1 score for each category

    print(np.sum(pred_bin.cpu().numpy(), axis=0))
    print(np.sum(y.cpu().numpy(), axis=0))

    print(len(p))
    print(len(pos_weight))
    stats = pd.DataFrame([[np.nanmean(f), np.average(p, weights=pos_weight), np.average(r, weights=pos_weight), np.mean(a), loss.compute()]], index=['MEAN'], columns=['F1', 'Precision', 'Recall', 'Accuracy', 'Loss'])
    print(stats)
    p_mean = np.mean(p)
    r_mean = np.mean(r)
    f_mean = 2*p_mean*r_mean/(p_mean+r_mean)
    stats = pd.DataFrame([[f_mean, p_mean, r_mean, np.mean(a), loss.compute()]], index=['MEAN'], columns=['F1', 'Precision', 'Recall', 'Accuracy', 'Loss'])
    stats_categ = pd.DataFrame(zip(f, p, r, categories_sum, f/categories_sum*100), index=categories_sum.index, columns=['F1', 'Precision', 'Recall', 'N concepts', 'F1/N conc'])

    # if full_stats:
    #     print(pd.concat([stats, stats_categ])) # add the stats for each category
    # else:
    #     print(stats)

    print(stats)
    scheduler.step(loss.compute())
    
    loss.reset()
    prec.reset()
    rec.reset()
    acc.reset()
    return stats, stats_categ


# verify GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Info
n_channels, n_times, n_classes, pos_weight, categories_sum =  return_info(opt.data_path, opt.subject)

# Dataset
if opt.model_type == 'Spectr':
    transform = torchaudio.transforms.Spectrogram(n_fft=22)
    print('Compute spectrogram')
else:
    transform = None
dataset = Dataset(opt.data_path, opt.subject, transform=transform) # spectrogram or None

# Splitter
dataset_train = Splitter(dataset, 'train')
dataset_val = Splitter(dataset, 'val')

# DataLoader
dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, drop_last=True)
dataloader_val = DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=True, drop_last=True)

# Model
models_dict = {'EEGNet': EEGNet(chunk_size=n_times, num_electrodes=n_channels, num_classes=n_classes),
               'LSTM': easy(chunk_size=n_times, num_electrodes=n_channels, num_classes=n_classes),
               'TSconv': TSconv(num_electrodes=n_channels, num_classes=n_classes),
               'Spectr': spectr_model(num_classes=n_classes)
               }
model = models_dict[opt.model_type].to(device)

# Print model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')

# Optimizer
optimizers_dict = {'Adam' : torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.0002, 0.5)),
                   'SGD' : torch.optim.SGD(model.parameters(), lr=opt.learning_rate)}
optimizer = optimizers_dict[opt.optimizer_type]
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True)

# Loss function
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=(torch.tensor(pos_weight)).to(device)) # pos_weight > 1 increases the recall, pos_weight < 1 increases the precision, len(pos_weigth) = n_classes

# Training loop
F1_best = 0
for t in range(opt.epochs):
    full_stats = True if t % 5 == 0 else False # print the stats for each category every 5 epochs

    print(f"------------ Epoch {t+1} ---------------")
    train(dataloader_train, model, loss_fn, optimizer)
    stats, stats_categ = valid(dataloader_val, model, loss_fn, full_stats)
    # acc_best = save_model(model=model, file_name=os.path.basename(__file__)[:-2], new_metric=acc, best_metric=acc_best) # save model if accuracy is better than the previous best
    
    if stats['F1'][0] > F1_best:
        stats_categ_best = stats_categ.copy()
        stats_best = stats.copy()
        F1_best = stats_best['F1'][0]
        # best_model = deepcopy(model)

print("Done!")
print("Best:\n", stats_best[['F1', 'Precision', 'Recall', 'Accuracy']])

# save dataframe
path = os.path.join(opt.data_path, 'others', 'Classification_EEG_Categories')
os.makedirs(path, exist_ok=True)
stats_categ_best.to_csv(os.path.join(path, f'stats_categ_{opt.model_type}_withPW_F{round(F1_best, 2)}.csv'))
