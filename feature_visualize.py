import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import cupy as cp
from tqdm import tqdm
from initial import params
from Graph import ng

if params['gpu_nums'] >0:
    xp = cp
else:
    xp = np


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"

data = xp.load('/home/miil/Datasets/ContiMeta/pretrain_train.npz')
feature_load = data['feature']
label_load = data['label']
image_load = data['image']


idx = xp.random.randint(0,feature_load.shape[0], size= params['feature_nums'])


feature_set = feature_load[idx][:]
label_set = label_load[idx]
image_set = image_load[idx][:]


BG = ng(feature_set, params['gpu_nums'], params['vertex_nums'],params['alpha'],params['eta'],params['max_iter'])
BG.pretrain(image_set, label_set)

winner_index = BG.vertices[0].f_idx
idx_ = winner_index[0]
np.array(feature_set[idx_])
#for i in range(1,len(winner_index)):
#    idx_ = winner_index[i]







