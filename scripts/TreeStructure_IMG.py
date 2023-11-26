'''
KDTree creation and saving
'''

import os
from sklearn.neighbors import KDTree
import joblib
import numpy as np

# Paths
path_feat = '../data/THINGS/Images_features/ResNet18/'
path_tree = '../trained_models/Retrieval_IMG/'

# Load features
# feat = np.load(os.path.join(path_feat, 'test_images_100.npy'), allow_pickle=True)
feat = np.load(os.path.join(path_feat, 'training_images_100_mean.npy'), allow_pickle=True)[:1535]
print(feat.shape)
print('Number of images:', feat.shape[0])

print(KDTree.valid_metrics)
# print(BallTree.valid_metrics)
# Create the tree
tree = KDTree(feat, leaf_size=40, metric='euclidean')
print(tree.data.shape)

# Save the tree
os.makedirs(path_tree, exist_ok=True) # create the directory
joblib.dump(tree, os.path.join(path_tree, 'kdtree.joblib')) # save the tree
print('Tree saved!')