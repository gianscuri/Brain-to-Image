"""
Extract and save the feature with ResNet18 (512 features) or EfficientNet (2048 features) of the training and test images.

	# test efficientnet:		1.8197047 -0.25439042 -0.03469166
	# train efficientnet: 		3.1342416 -0.27140582 -0.029406885 
	# test resnet:				0.0, 13.193601, 0.057900384
	# train resnet:				0.0, 19.421381, 0.06920943
"""

# Define options
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--project_dir', default='../data/THINGS-EEG2/', type=str)
parser.add_argument('-mt','--model_type', default='resnet18', help="Select model resnet18, efficientnet")

opt = parser.parse_args()
print(opt)


# Import libraries
import numpy as np
import os
from PIL import Image
from zipfile import ZipFile
from tqdm import tqdm
import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights, resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names


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

# Image directories
img_set_dir = os.path.join(opt.project_dir, 'image_set')
img_partitions = os.listdir(img_set_dir)
img_partitions = ['training_images.zip', 'test_images.zip']
# img_partitions = ['test_images.zip']

# Extract features
features = []
# min, max, mean = [], [], []
for part in tqdm(img_partitions, desc='Partitions'): # for each partition

	part_dir = os.path.join(img_set_dir, part)
	with ZipFile(part_dir, 'r') as zipObj: # open the zip file

		folder_list = [f.filename for f in zipObj.infolist() if f.is_dir()][1:]
		folder_list.sort()
		folders = []

		for folder in tqdm(folder_list, leave=False, desc='Concepts'): # for each concept
			image_list = [i for i in zipObj.namelist() if i.startswith(folder) and i.endswith(('.jpg', '.JPEG'))]
			image_list.sort()

			images = []
			for image in image_list: # for each image
				file = zipObj.open(image)
				img = Image.open(file).convert('RGB')
				input_img = preprocess(img).unsqueeze(0).to(device) # preprocess and convert to tensor

				x = model(input_img)
				# print(x['last_layer'].shape)
				
				# max.append(np.max(x['flatten'].data.cpu().numpy().flatten()))
				# min.append(np.min(x['flatten'].data.cpu().numpy().flatten()))
				# mean.append(np.mean(x['flatten'].data.cpu().numpy().flatten()))
				# print(x['flatten'].cpu().detach().numpy().shape)

				images.append(x['last_layer'].data.cpu().numpy())
			folders.append(images)
		features = np.array(folders)
		print(np.mean(features), np.std(features))

	features = np.log(features + 0.01) # peggiora le performance

	if part == 'training_images.zip':
		mean = np.mean(features)
		std = np.std(features)
		max = np.max(features)
		min = np.min(features)

	# mean = 0.008
	# std = 0.12
	# print(features[0][0][0])

	# Normalize features
	# features = (features - mean) / std
	features = (features - min) / (max - min) # normalize between 0 and 1

	# print(features[0][0][0])

	print('Min, max, mean: ', np.min(features), np.max(features), np.mean(features))

	# Save features
	save_dir = os.path.join(opt.project_dir, 'image_set_features', opt.model_type)
	os.makedirs(save_dir, exist_ok=True)
	np.save(os.path.join(save_dir, part.split('.')[0]), features)