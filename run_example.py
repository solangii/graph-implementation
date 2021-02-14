
from NeuralGas import ng

#preparing data ...
from initial import feature_set
from initial import params
from initial import image_set
from initial import label_set



#fitting neural network ...
BG = ng(feature_set, params['node_nums'],params['alpha'],params['eta'],params['max_iter'])


BG.pretrain(image_set,label_set)





