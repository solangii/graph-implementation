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


tsne = TSNE(random_state=0, perplexity=15, early_exaggeration=7)

winner_index_1 = BG.vertices[0].f_idx
feat1 = []
label1 = []
for i in range(1,len(winner_index_1)):
    idx_ = winner_index_1[i]
    feat1.append(feature_set[idx_])
    label1.append(label_set[idx_])

winner_index_2 = BG.vertices[1].f_idx
feat2 = []
label2 = []
for i in range(1, len(winner_index_2)):
    idx_ = winner_index_2[i]
    feat2.append(feature_set[idx_])
    label2.append(label_set[idx_])

feat = np.array(feat1)
label = np.array(label1)
np.concatenate((feat, np.array(feat2)))
np.concatenate((label, np.array(label2)))

colors = []
for i in range(2):
    colors.append(list(np.random.choice(range(256), size=3)))
colors = np.array(colors) / 256
print('Shape of colors: {}'.format(colors.shape))
print('min = {}, max = {}'.format(np.min(colors), np.max(colors)))

digits_tsne = tsne.fit_transform(feat)

for i in range(feat.shape[0]):
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(label[i]),   # x, y , 그룹
             fontdict={'weight': 'bold', 'size':9})


plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max())          # 최소, 최대
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max())          # 최소, 최대
plt.xlabel('t-SNE feature 0')
plt.ylabel('t-SNE feature 1')

plt.show()
plt.savefig('feature_tSNE.png')







