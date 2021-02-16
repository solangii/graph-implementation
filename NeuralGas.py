import numpy as np
from tqdm import tqdm
import os
from scipy.spatial import distance
from numpy.linalg import norm
from cupy.linalg import norm as norm_gpu

#Todo gpu연산으로 바꾸고
#Todo ACCURACY 측정하는거 추가하고
#Todo 중간 파라미터 저장할 수 있는

#Todo 후에 Argument는 parser

class Vertex():
    def __init__(self, m):
        self.c = None
        self.z = None
        self.l = None
        self.m = m
        self.f_idx = []

class ng:
    def __init__(self, feature_set, gpu, vertex_nums=1200, alpha = 10, eta = 0.1, max_iter = 100):
        self.xp = None
        self.feature_set = feature_set
        self.feature_nums = self.feature_set.shape[0]
        self.vertex_nums = vertex_nums
        self.alpha = alpha
        self.eta = eta
        self.max_iter = max_iter
        self.anchor_set = self.init_anchor()
        self.vertices = []
        self.gpu = gpu

        if self.gpu <0:
            self.xp = np
        else:
            import cupy as cp
            self.xp = cp

            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % self.gpu

    def init_anchor(self):
        """
        :return: initialize anchor that randomly chosen at feature set
        """
        rand_idx = np.random.randint(self.feature_set.shape[0], size= self.vertex_nums)
        anchor = self.feature_set[rand_idx, :]
        print("initialize anchor done!")
        return anchor

    def compute_rank_order(self, feature):
        """
        compute distance with feature and m, and ordering it
        :param feature: each feature
        :return: rank order index for feature
        """
        Df = []

        for i in range(self.vertex_nums):
            dis = distance.euclidean(feature, self.anchor_set[i])
            Df.append(dis)
        Rf = np.argsort(Df)

        return Rf

    def update_anchor(self, feature, anchor, i, anchor_index):
        """
        update anchor(m) by equation3 in FSCIL.
        :param feature: each feature
        :param anchor: centroid
        :param i: feature index in feature set
        :param anchor_index: anchor index in anchor set
        """
        rate = self.eta * np.exp(-i/self.alpha)
        updated_anchor = anchor + rate *(feature - anchor)
        self.anchor_set[anchor_index] = updated_anchor

    def update_anchor_set(self):
        """
        update all anchor
        """
        for i in tqdm(range(self.feature_nums), desc="update anchor"):
            f = self.feature_set[i]
            Rf = self.compute_rank_order(f) #해당 feature와 가까운 순서대로 m의 index

            for j in range(self.max_iter):
                idx = Rf[j]
                m = self.anchor_set[idx]
                self.update_anchor(f,m,j,idx)

    def make_vertex(self):
        """
        make nodes(objects) with updated anchor(m)
        """
        for i in range(self.vertex_nums):
            self.vertices.append(Vertex(self.anchor_set[i]))

    def update_vertex(self, image_set, label_set):
        """
        :param image_set:
        :param label_set:
        :return: vertex update
        """
        for i in range(self.feature_nums):
            nearest_m = self.compute_rank_order(self.feature_set[i])[0]
            self.vertices[nearest_m].f_idx.append(i)

        for i in tqdm(range(self.vertex_nums), desc="making vertex : "):
            f_set = []
            for j in range(len(self.vertices[i].f_idx)):
                f_set.append(self.feature_set[self.vertices[i].f_idx[j]])

            f_set = self.xp.array(f_set)

            ##예외처리, F_set이 비어있
            if len(f_set)==0:
                self.vertices[i].l = -1
                self.vertices[i].z = -1
                self.vertices[i].c = -1
            else:
                self.vertices[i].l = self.compute_lambda(f_set)
                self.vertices[i].z, self.vertices[i].c = self.compute_nearest(i, f_set, image_set, label_set)

    def compute_lambda(self, f_set):
        """
        :param f_set:
        :return: compute variance for winner vectors
        """
        cov = self.xp.cov(self.xp.transpose(f_set))
        var = self.xp.diag(cov)
        lam = self.xp.diag(var)
        return lam

    def compute_nearest(self, m_idx, f_set, image_set, label_set):
        """
        :param m_idx: vertex index
        :param f_set: nearest feature set
        :param image_set: image set
        :param label_set: label set
        :return: 모든 Vertex에 대해서 가장 가까운 pseudo image와 label을 찾음
        """
        #dist = distance.euclidean(self.vertices[m_idx].m, f_set[0].T)
        dist = norm(self.vertices[m_idx].m - f_set[0])
        idx = 0
        for i in range(1,len(f_set)):
            if dist > norm(self.vertices[m_idx].m - f_set[i]):
                dist = norm(self.vertices[m_idx].m - f_set[i])
                idx = i

        #real_idx = self.vertices[m_idx].f_set[idx]
        real_idx = self.vertices[m_idx].f_idx[idx]

        z = image_set[real_idx]
        c = label_set[real_idx]
        return z, c

    def pretrain(self, x,y):
        """
        :param x: image_set for pretrain data set
        :param y: label-set for pretrain data set
        :return:
        """
        self.update_anchor_set()
        print("anchor update done!")
        print("============================================================")

        self.make_vertex()
        print("make vertex done!")

        self.update_vertex(x,y)
        print("update vertex done!")
        print("============================================================")