#import neural gas model
from NeuralGas import ng

#preparing data ...
from initial import params
import numpy as np

feature_set = np.random.rand(params['feature_nums'], params['feature_dim'])
image_set = np.random.rand(params['feature_nums'])
label_set = np.random.rand(params['feature_nums'])

#fitting neural network ...
BG = ng(feature_set, params['gpu_nums'], params['vertex_nums'],params['alpha'],params['eta'],params['max_iter'])
BG.pretrain(image_set,label_set)





