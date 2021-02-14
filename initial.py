import numpy as np

Nv= 1200
Nf = 30000
feature_dim = 512
feature_set = np.random.rand(Nf, feature_dim)

image_set = np.random.rand(Nf)
label_set = np.random.randint(Nf)

params = {'node_nums' : 1200,
          'alpha':10,
          'eta':0.1,
          'max_iter':100
}


print(feature_set.dtype)