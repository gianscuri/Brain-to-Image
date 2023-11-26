'''
This script allows to classify the EEG data into the concepts of the THINGS-EEG2 dataset.
'''


# Define options
import argparse
parser = argparse.ArgumentParser(description="Template")

# Data path
parser.add_argument('-ed', '--data_path', default="../data/THINGS-EEG2/", help="Data folder path")

# Subject selecting
parser.add_argument('-subj','--subject', default=1, type=int, help="choose a subject from 1 to 6, default is 1")
parser.add_argument('-n_conc','--n_concepts', default=100, type=int, help="number of concepts to classify, default is 100")

# Training options
parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float, help="learning rate")
parser.add_argument('-e', '--epochs', default=100, type=int, help="training epochs")

# Model type/options
parser.add_argument('-mt','--model_type', default='TSconv', help="Select model [EEGNet, LSTM, TSconv, Spectr]")
parser.add_argument('-ot','--optimizer_type', default='Adam', help="Select optimizer [Adam, SGD]")

# Parse arguments
opt = parser.parse_args()
print(opt)


# Import libraries
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from Classification_EEG_models import EEGNet, easy, spectr_model, TSconv
from ignite.metrics import TopKCategoricalAccuracy, Loss
# from ignite.handlers import EarlyStopping
# from utils import save_model, rename_saved_model
import torchaudio
import copy


def return_info(data_path, n_subject):
    # Extract EEG data
    eeg_path = os.path.join(data_path, 'preprocessed_data', 'sub-' + str(n_subject).zfill(2), 'preprocessed_eeg_training.npy')
    eeg = np.load(eeg_path, allow_pickle=True).item()
    n_channels, n_times = eeg['preprocessed_eeg_data'].shape[2:]

    # # Labels
    # image_metadata_path = os.path.join(data_path, 'image_metadata/image_metadata.npy')
    # image_metadata = np.load(image_metadata_path, allow_pickle=True).item()
    # n_classes = len(set(image_metadata['train_img_concepts']))

    return n_channels, n_times#, n_classes
   
class Dataset(Dataset):
  # Class to create the dataset
  def __init__(self, data_path, n_subject, transform=None):
    self.transform = transform

    # EEG data
    eeg_path = os.path.join(data_path, 'preprocessed_data', 'sub-' + str(n_subject).zfill(2), 'preprocessed_eeg_training.npy')
    eeg = np.load(eeg_path, allow_pickle=True).item()
    _, n_repetitions, n_channels, n_times = eeg['preprocessed_eeg_data'].shape
    eeg_reshaped = eeg['preprocessed_eeg_data'].reshape((-1, n_channels, n_times))

    # Labels
    image_metadata_path = os.path.join(data_path, 'image_metadata/image_metadata.npy')
    image_metadata = np.load(image_metadata_path, allow_pickle=True).item()

    le = LabelEncoder()
    le.fit(image_metadata['train_img_concepts'])
    idx = le.transform(image_metadata['train_img_concepts'])
    concepts_int = np.repeat(idx, n_repetitions)

    self.X = torch.from_numpy(eeg_reshaped.astype(np.float32)) # convert float64 to float32
    self.y = torch.from_numpy(concepts_int).type(torch.LongTensor) # convert float64 to Long
    self.len = self.X.shape[0]

    if self.transform:
        self.X = self.transform(self.X)
        print(self.X.shape)

  def __len__(self):
    return self.len
  
  def __getitem__(self, index):
    return self.X[index], self.y[index]
  
class Splitter:
    # class to split the dataset
    def __init__(self, dataset, n_concepts, split_name):
        # Set EEG dataset
        self.dataset = dataset

        # Load split index
        train_idx = np.load(os.path.join(opt.data_path, 'image_metadata/split_rep_train.npy'), allow_pickle=True)
        val_idx = np.load(os.path.join(opt.data_path, 'image_metadata/split_rep_val.npy'), allow_pickle=True)

        # a = 1 # da 0 a 1254
        # train_idx = train_idx[(a*4*8):(n_concepts*4*8+a*4*8)] # 8 images per concept and 4 repetitions
        # val_idx = val_idx[(a*4):(n_concepts*4+a*4)]
        train_idx = train_idx[:n_concepts*4*8] # 8 images per concept and 4 repetitions
        val_idx = val_idx[:n_concepts*4]

        print('random chance accuracy: ', round(100/n_concepts, 2))

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

    print("Top1: ", round(top1.compute()*100, 2),
          "\tTop3: ", round(top3.compute()*100, 2),
          "\tTop5: ", round(top5.compute()*100, 2),
          "\tloss: ", round(loss.compute(), 3))
    
    scheduler.step(loss.compute())

    return top1.compute()*100, top5.compute()*100 # return top1 accuracy


# verify GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Info
n_channels, n_times =  return_info(opt.data_path, opt.subject)
n_classes = opt.n_concepts


# Dataset
if opt.model_type == 'Spectr':
    transform = torchaudio.transforms.Spectrogram(n_fft=22)
    print('Compute spectrogram')
else:
    transform = None
dataset = Dataset(opt.data_path, opt.subject, transform=transform) # spectrogram or None

# Splitter
dataset_train = Splitter(dataset, n_classes, 'train')
dataset_val = Splitter(dataset, n_classes, 'val')

# DataLoader
dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, drop_last=True)
dataloader_val = DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=False, drop_last=True)

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

# Training
optimizers_dict = {'Adam' : torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=(0.0002, 0.5)),
                   'SGD' : torch.optim.SGD(model.parameters(), lr=opt.learning_rate)}
optimizer = optimizers_dict[opt.optimizer_type]
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6, verbose=True)
loss_fn = torch.nn.CrossEntropyLoss()

# Metrics
top1 = TopKCategoricalAccuracy(k=1)
top3 = TopKCategoricalAccuracy(k=3)
top5 = TopKCategoricalAccuracy(k=5)
loss = Loss(loss_fn)


# Training loop
acc_best = 0
for t in range(opt.epochs):
    
    print(f"------------ Epoch {t+1} ---------------")
    train(dataloader_train, model, loss_fn, optimizer)
    top1acc, top5acc = valid(dataloader_val, model)
    # acc_best = save_model(model=model, file_name=os.path.basename(__file__)[:-2], new_metric=acc, best_metric=acc_best) # save model if accuracy is better than the previous best
    
    if top1acc > acc_best:
        acc_best = copy.copy(top1acc)
        acc5_best = copy.copy(top5acc)
        best_model = copy.deepcopy(model)
print("Done!")
print("Best top1: ", round(acc_best, 2), '\tBest top5: ', round(acc5_best, 2))

# Save best model
os.makedirs(os.path.join('../trained_models', os.path.basename(__file__)[:-2]), exist_ok=True) # create folder
path = os.path.join('../trained_models', os.path.basename(__file__)[:-2], f'model_{opt.n_concepts}_acc{round(acc_best, 2)}_{opt.model_type}.pt') # :04.1f
# torch.save(best_model, path) # save best model 

# Rename model file with accuracy
# rename_saved_model(file_name=os.path.basename(__file__)[:-2], metric=acc_best)