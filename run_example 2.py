
#import neural gas model
from Graph import ng

#preparing data ...
from initial import params
import numpy as np
import cupy as cp
import pickle
import os

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


#fitting
BG = ng(feature_set, params['gpu_nums'], params['vertex_nums'],params['alpha'],params['eta'],params['max_iter'])
BG.pretrain(image_set, label_set)
acc = BG.compute_accuracy_for_train(label_set)
print(acc)

"""
#testing acc
test_data = xp.load('/home/miil/Datasets/ContiMeta/pretrain_test.npz')
t_feature_load = data['feature']
t_label_load = data['label']
t_image_load = data['image']

t_idx = xp.random.randint(0,t_feature_load.shape[0], size= params['test_feature_nums'])

test_feature_set = feature_load[t_idx][:]
test_label_set = label_load[t_idx]
test_image_set = image_load[t_idx][:]

acc = []
for i in range(10):
    t_idx = xp.random.randint(0, t_feature_load.shape[0], size=params['test_feature_nums'])

    test_feature_set = feature_load[t_idx][:]
    test_label_set = label_load[t_idx]
    test_image_set = image_load[t_idx][:]

    acc.append(BG.compute_accuracy(test_feature_set,test_label_set))


print(acc)
"""
"""
feature_set = xp.random.rand(params['feature_nums'], params['feature_dim'])
image_set = xp.random.rand(params['feature_nums'])
label_set = xp.random.randint(0,params['class'], size=(params['feature_nums']))


#fitting neural network ...
BG = ng(feature_set, params['gpu_nums'], params['vertex_nums'],params['alpha'],params['eta'],params['max_iter'])
BG.pretrain(image_set,label_set)

#testing accuracy ...
test_feature_set = xp.random.rand(params['test_feature_nums'],params['feature_dim'])
test_label_set = xp.random.randint(0,params['class'], size=(params['feature_nums']))

BG.compute_accuracy(test_feature_set, test_label_set)

#incremental phase
novel_feature_num = params['n_way']*params['k_shot']
new_feature_set = xp.random.rand(novel_feature_num, params['feature_dim'])
new_image_set = xp.random.rand(params['feature_nums'])
new_label_set = xp.random.randint(params['class'],params['class']+params['n_way'], size=novel_feature_num)

BG.update_graph.remote(new_feature_set,params['n_way'],new_image_set, new_label_set)
"""
