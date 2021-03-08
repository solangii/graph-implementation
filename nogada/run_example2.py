from Graph_eq3 import ng
from init2 import params
import numpy as np
import cupy as cp
import os


if params['gpu_nums'] >0:
    xp = cp
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    xp = np


#======================================================
data = xp.load('/home/miil/Datasets/ContiMeta/pretrain_train.npz')
feature_load = data['feature']
label_load = data['label']
image_load = data['image']

idx = xp.random.randint(0,feature_load.shape[0], size= params['feature_nums'])

feature_set = feature_load[idx][:]
label_set = label_load[idx]
image_set = image_load[idx][:]


#fitting
#    def __init__(self, feature_set, gpu, vertex_nums=1200, lambda_i = 10, lambda_f = 0.01, epsilon_i = 0.1, epsilon_f = 0.005 , max_iter = 1000):

BG = ng(feature_set, params['gpu_nums'], params['vertex_nums'], params['lambda_i'],params['lambda_f'])
BG.pretrain(image_set, label_set)

acc_train = BG.compute_accuracy_for_train(label_set)
print("train avg :", acc_train)
print("======================================")


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
    acc.append()

acc = np.array(acc)
avg = np.mean(acc)
print("list : ",acc)
print("======================================")
print("test avg :", avg)
