#import neural gas model
from Graph import ng

#preparing data ...
from initial import params
import numpy as np
import cupy as cp

if params['gpu_nums'] >0:
    xp = cp
else:
    xp = np

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

BG.update_graph(new_feature_set,params['n_way'],new_image_set, new_label_set)




