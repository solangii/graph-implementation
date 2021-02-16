import os

import numpy as np
from cupy.linalg import norm as norm_gpu
from numpy.linalg import norm
#from scipy.spatial import distance
from tqdm import tqdm


#Todo ACCURACY 측정하는거 추가하고
#Todo 중간 파라미터 저장할 수 있는
#Todo 후에 Argument는 parser

class Vertex():
    def __init__(self, m, xp):
        self.c = None
        self.z = None
        self.l = None
        self.m = m
        #self.f_idx = xp.empty((0))
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
        self.gpu = gpu
        self.vertices = []

        if self.gpu <0:
            self.xp = np
            self.dist = norm
        else:
            import cupy as cp
            self.xp = cp
            self.dist = norm_gpu

            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % self.gpu

        #self.vertices = self.xp.empty(0)


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
            dis = self.dist(feature - self.anchor_set[i])
            Df.append(dis)
        Df = self.xp.array(Df)
        Rf = self.xp.argsort(Df).tolist()

        return Rf

    def update_anchor(self, feature, anchor, i, anchor_index):
        """
        update anchor(m) by equation3 in FSCIL.
        :param feature: each feature
        :param anchor: centroid
        :param i: feature index in feature set
        :param anchor_index: anchor index in anchor set
        """
        rate = self.eta * self.xp.exp(-i/self.alpha)
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
        for i in tqdm(range(self.vertex_nums), desc="make vertex"):
            self.vertices.append(Vertex(self.anchor_set[i], self.xp))


    def update_vertex(self, image_set, label_set):
        """
        :param image_set:
        :param label_set:
        :return: vertex update
        """

        for i in tqdm(range(self.feature_nums), desc="winner vector compute"):
            nearest_m = self.compute_rank_order(self.feature_set[i])[0]
            self.vertices[nearest_m].f_idx.append(i)

        for i in tqdm(range(self.vertex_nums), desc="update vertex"):
            f_set = []
            for j in range(len(self.vertices[i].f_idx)):
                f_set.append(self.feature_set[self.vertices[i].f_idx[j]])

            f_set = self.xp.array(f_set)

            ##exception
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

        if self.xp is np:
            comp = norm
        else:
            comp = norm_gpu

        dist = comp(self.vertices[m_idx].m - f_set[0])
        idx = 0
        for i in range(1,len(f_set)):
            if dist > comp(self.vertices[m_idx].m - f_set[i]):
                dist = comp(self.vertices[m_idx].m - f_set[i])
                idx = i

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


    def compute_accuracy(self, f_set, label_set):
        """
        compute accuracy for hyper parameter.
        해당 Feature와 winner vertex의 label이 일치한지 계산

        :param f_set: feature set in th same class configuration with pretrained dataset. However, that were not used for pretrain
        :param label_set: feature set's label
        :return: success rate
        """
        print("compute accuracy for hyper-parameter!!")

        success = 0

        if self.xp is np:
            comp = norm
        else:
            comp = norm_gpu

        for i in tqdm(range(len(f_set)),desc="feature"):
            dist = comp(self.vertices[0].m - f_set[i])
            idx = 0
            for m_idx in range(1,self.vertex_nums):
                if dist > comp(self.vertices[m_idx].m - f_set[i]):
                    dist = comp(self.vertices[m_idx].m - f_set[i])
                    idx = i

            if idx == label_set[i]:
                success += 1

        success_rate = float(success / len(f_set) )* 100

        print("the accuracy is %d percent" %success_rate)


