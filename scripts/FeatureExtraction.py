"""Extract and save the ResNet-50 feature maps of the training and test images,
and of the ILSVRC-2012 validation and test images.

Parameters
----------
pretrained : bool
	If True use a pretrained network, if False a randomly initialized one.
project_dir : str
	Directory of the project folder.

"""

import argparse
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
import os
from PIL import Image


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--project_dir', default='../data/THINGS-EEG2/', type=str)
args = parser.parse_args()

print('>>> Extract feature maps ResNet-50 <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.use_deterministic_algorithms(True)


# =============================================================================
# Import the model
# =============================================================================
def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
		padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
		bias=False)

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		if self.downsample is not None:
			identity = self.downsample(x)
		out += identity
		out = self.relu(out)
		return out

class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = conv1x1(inplanes, planes)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = conv3x3(planes, planes, stride)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = conv1x1(planes, planes * self.expansion)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv3(out)
		out = self.bn3(out)
		if self.downsample is not None:
			identity = self.downsample(x)
		out += identity
		out = self.relu(out)
		return out

class ResNet(nn.Module):

	def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
		super(ResNet, self).__init__()
		self.inplanes = 64
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
			bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.feat_list = ['block1', 'block2', 'block3', 'block4', 'fc']
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512 * block.expansion, num_classes)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out',
					nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual
		# block behaves like an identity. This improves the model by 0.2~0.3%
		# according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion),
			)
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))
		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x1= self.layer1(x)
		x2 = self.layer2(x1)
		x3 = self.layer3(x2)
		x4 = self.layer4(x3)
		x = self.avgpool(x4)
		x = x.view(x.size(0), -1)
		x5 = self.fc(x)
		return x1, x2, x3, x4, x5

def resnet50(pretrained=args.pretrained, **kwargs):
	"""Constructs a ResNet-50 model. """
	model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
	if pretrained == True:
		model_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
		model.load_state_dict(model_zoo.load_url(model_url))
	return model

model = resnet50()
if torch.cuda.is_available():
	model.cuda()
model.eval()


# =============================================================================
# Define the image preprocessing
# =============================================================================
centre_crop = trn.Compose([
	trn.Resize((224,224)),
	trn.ToTensor(),
	trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =============================================================================
# Load the images and extract the corresponding feature maps
# =============================================================================
# Extract the feature maps of (1) training images, (2) test images,
# (3) ILSVRC-2012 validation images, (4) ILSVRC-2012 test images.
from zipfile import ZipFile
from tqdm import tqdm

# Image directories
img_set_dir = os.path.join(args.project_dir, 'image_set')
img_partitions = os.listdir(img_set_dir)

img_partitions = ['test_images.zip'] # da togliere

for p in img_partitions:
	part_dir = os.path.join(img_set_dir, p)
	with ZipFile(part_dir, 'r') as zipObj:
		image_list = [i for i in zipObj.namelist() if i.endswith('.jpg') or i.endswith('.JPEG')]
	image_list.sort()

	# Create the saving directory if not existing
	save_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
		'full_feature_maps', 'resnet50', 'pretrained-'+str(args.pretrained))
	if os.path.isdir(save_dir) == False:
		os.makedirs(save_dir)



	# Extract and save the feature maps
	image_features = []
	with ZipFile(part_dir, 'r') as zipObj:
		for i, image in enumerate(tqdm(image_list)):
			print(image)
			ifile = zipObj.open(image)
			img = Image.open(ifile).convert('RGB')
			input_img = V(centre_crop(img).unsqueeze(0))
			if torch.cuda.is_available():
				input_img=input_img.cuda()
			x = model.forward(input_img)
			feats = {}
			for f, feat in enumerate(x):
				feats[model.feat_list[f]] = feat.data.cpu().numpy()
			# file_name = p + '_' + format(i+1, '07')
			image_features.append(feats['fc'])
	# print(feats['fc'].shape)
	# np.save(os.path.join(save_dir, p.split('.')[0]), image_features)


# from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

# model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V2)
# model.eval()