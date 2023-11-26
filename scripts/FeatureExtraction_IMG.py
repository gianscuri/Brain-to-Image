"""
Extract and save the feature with ResNet18 (512 features)

Provides 6 files:
    - For THINGS-EEG2: 'training_images', 'test_images'
    - For THINGS: 'training_images', 'test_images', 'training_info.csv', 'test_info.csv'
"""

# Define options
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--THINGS_dir', default='../data/THINGS/', type=str)
parser.add_argument('--THINGSEEG_dir', default='../data/THINGS-EEG2/', type=str)
parser.add_argument('-mt','--model_type', default='resnet18', help="Select model resnet18, efficientnet")

opt = parser.parse_args()
print(opt)

# Import libraries
import numpy as np
import pandas as pd
import os
from PIL import Image
from zipfile import ZipFile
from tqdm import tqdm
import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights, resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names


def find_train_test(path, partition):
	with ZipFile(os.path.join(path, partition), 'r') as zipObj: # open the zip file

		folder_list = [f.filename for f in zipObj.infolist() if f.is_dir()] # list of the folders in the zip file (concepts)
		folder_list = folder_list[1:] # remove the first folder (it is the folder of the partition)
		folder_list = ['_'.join(i.split('_')[2:]) for i in folder_list] # remove the folder and code from the string
		folder_list.sort()
		
		
		image_list = [i for i in zipObj.namelist() if i.endswith(('.jpg', '.JPEG'))] # list of the images in the zip file
		image_list = [i.split('/')[2] for i in image_list] # remove the folder from the string
		image_list.sort()
		
		return folder_list, image_list
	
def extract_features(path):
    zip_list = os.listdir(path)
    zip_list.sort()
    # zip_list = [zip_list[2]] # DA TOGLIERE

    feat_train = []
    feat_test = []
    feat_train_all = []
    feat_test_all = []

    idx_train = []
    idx_test = []
    name_train = []
    name_test = []
    
    for part in tqdm(zip_list, desc='Partitions'): # for each partition

        part_dir = os.path.join(path, part)
        with ZipFile(part_dir, 'r') as zipObj: # open the zip file
            zipObj.setpassword(b'things4all')

            folder_list = [f.filename for f in zipObj.infolist() if f.is_dir()]
            folder_list.sort()
            # print(folder_list)
            # folder_list = folder_list[:10] # DA TOGLEIRE

            for folder in tqdm(folder_list, leave=False, desc='Concepts'): # for each concept

                image_list = [i for i in zipObj.namelist() if i.startswith(folder) and i.endswith(('.jpg', '.JPEG'))]
                image_list.sort()

                img_train = []
                img_test = []

                for image in image_list: # for each image
                    file = zipObj.open(image)
                    img = Image.open(file).convert('RGB')
                    input_img = preprocess(img).unsqueeze(0).to(device) # preprocess and convert to tensor
                    
                    x = model(input_img) # extract features
                    x = x['last_layer'].data.cpu().numpy()[0]

                    if folder in concepts_train: # train set
                        feat_train_all.append(x)
                        idx_train.append(True if image.split('/')[1] in images_train else False) # save the index of the images used in THINGS-EEG2
                        name_train.append(image.split('/')[1])

                        if image.split('/')[1] in images_train: # append the images used in THINGS-EEG2
                            img_train.append(x)
                        
                    else: # test set
                        feat_test_all.append(x)
                        idx_test.append(True if image.split('/')[1] in images_test else False) # append true if the images used in THINGS-EEG2
                        name_test.append(image.split('/')[1])

                        if image.split('/')[1] in images_test: # append the images used in THINGS-EEG2
                            img_test.append(x)

                if folder in concepts_train: # train
                    feat_train.append(img_train)
                else: # test
                    feat_test.append(img_test)

    info_train = pd.DataFrame([name_train, idx_train], index=['name', 'idx']).T
    info_test = pd.DataFrame([name_test, idx_test], index=['name', 'idx']).T

    feat_train = np.array(feat_train)
    feat_test = np.array(feat_test)
    feat_train_all = np.array(feat_train_all)
    feat_test_all = np.array(feat_test_all)

    return feat_train, feat_test, feat_train_all, feat_test_all, info_train, info_test

def scale_features(feat_train, feat_test, feat_train_all, feat_test_all): # range [-1,1]
    max = np.max(feat_train_all)
    min = np.min(feat_train_all)

    print('Min, max, mean: ', min, max, np.mean(feat_train_all))

    feat_train = (feat_train - min) / (max - min)
    feat_test = (feat_test - min) / (max - min)
    feat_train_all = (feat_train_all - min) / (max - min)
    feat_test_all = (feat_test_all - min) / (max - min)

    print('Min, max, mean: ', np.min(feat_train), np.max(feat_train), np.mean(feat_train))
    print('Min, max, mean: ', np.min(feat_test), np.max(feat_test), np.mean(feat_test))
    print('Min, max, mean: ', np.min(feat_train_all), np.max(feat_train_all), np.mean(feat_train_all))
    print('Min, max, mean: ', np.min(feat_test_all), np.max(feat_test_all), np.mean(feat_test_all))

    return feat_train, feat_test, feat_train_all, feat_test_all
    
def save_features(feat_train, feat_test, feat_train_all, feat_test_all, info_train, info_test):
    save_dir = os.path.join(opt.THINGSEEG_dir, 'image_set_features', opt.model_type)
    save_dir_all = os.path.join(opt.THINGS_dir, 'Images_features', opt.model_type)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_all, exist_ok=True)

    np.save(os.path.join(save_dir, 'training_images'), feat_train)
    np.save(os.path.join(save_dir, 'test_images'), feat_test)
    np.save(os.path.join(save_dir_all, 'training_images'), feat_train_all)
    np.save(os.path.join(save_dir_all, 'test_images'), feat_test_all)

    info_train.to_csv(os.path.join(save_dir_all, 'training_info.csv'))
    info_test.to_csv(os.path.join(save_dir_all, 'test_info.csv'))

# verify GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Model selection
models_dict = {'resnet18': (resnet18(weights = ResNet18_Weights.DEFAULT), ResNet18_Weights.DEFAULT),
               'efficientnet': (efficientnet_b1(weights = EfficientNet_B1_Weights.DEFAULT), EfficientNet_B1_Weights.DEFAULT)}
w = models_dict[opt.model_type][1]
m = models_dict[opt.model_type][0]

# Initialize the inference transforms
preprocess = w.transforms()

# print(get_graph_node_names(m), '\n') # obtain the names of the nodes of the model
model = create_feature_extractor(m, return_nodes={'flatten': 'last_layer'}).to(device) # choose the last layer as output (es. 'avgpool' or 'flatten')
model.eval()

# Train and test concepts and images
path = os.path.join(opt.THINGSEEG_dir, 'image_set')

concepts_train, images_train = find_train_test(path, 'training_images.zip')
concepts_test, images_test = find_train_test(path, 'test_images.zip')

print('Train concepts:\t', len(concepts_train),
	  '\nTrain images:\t', len(images_train),
      '\nTest concepts:\t', len(concepts_test),
      '\nTest images:\t', len(images_test))


# Extract and scale features
path = os.path.join(opt.THINGS_dir, 'Images')
feat_train, feat_test, feat_train_all, feat_test_all, info_train, info_test = extract_features(path)
# feat_train, feat_test, feat_train_all, feat_test_all = scale_features(feat_train, feat_test, feat_train_all, feat_test_all)

# Save features
save_features(feat_train, feat_test, feat_train_all, feat_test_all, info_train, info_test)