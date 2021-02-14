import numpy as np
import random
from scipy.spatial import distance

#Todo gpu연산으로 바꾸고
#Todo ACCURACY 측정하는거 추가하고
#Todo 중간 파라미터 저장할 수 있는
#Todo vertex 클래스 만들

class Vertex():
    def __init__(self, m):
        self.c = None
        self.z = None
        self.l = None
        self.m = m
        self.f_idx = []


class ng:
    def __init__(self, feature_set, vertex_nums=1200, alpha = 10, eta = 0.1, max_iter = 100):
        self.feature_set = feature_set
        self.feature_nums = self.feature_set.shape[0]
        self.vertex_nums = vertex_nums
        self.alpha = alpha
        self.eta = eta
        self.max_iter = max_iter
        self.anchor_set = self.init_anchor()
        self.vertices = []

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
        :return: rank index for feature
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
        for i in range(self.feature_nums):
            f = self.feature_set[i]
            Rf = self.compute_rank_order(f) #해당 feature와 가까운 순서대로 m의 index

            for j in range(self.max_iter):
                idx = Rf[j]
                m = self.anchor_set[idx]
                self.update_anchor(f,m,j,idx)

            print("update m about", i,"th feature")

    def make_vertex(self):
        """
        make nodes(objects) with updated anchor(m)
        """
        for i in range(self.vertex_nums):
            self.vertices.append(Vertex(self.anchor_set[i]))

    def update_vertex(self, image_set, label_set):
        #모든 feature set 에 대해서 Rf 계산한거 저장해둬야겟다.
        for i in range(self.feature_nums):
            nearest_m = self.compute_rank_order(self.feature_set[i])[0]
            self.vertices[nearest_m].f_idx.append(i)

        for i in range(self.vertex_nums):
            f_set = []
            for j in range(len(self.vertices[i].f_idx)):
                f_set.append(self.feature_set[self.vertices[i].f_idx[j]])

            self.vertices[i].l = self.compute_lambda(f_set)
            self.vertices[i].z, self.vertices[i].c = self.compute_nearest(i, f_set, image_set, label_set)


    def compute_lambda(self, f_set):
        var = np.diag(np.cov(np.transpose(f_set),f_set))
        lam = np.diag(var)
        return lam

    def compute_nearest(self, m_idx, f_set, image_set, label_set):
        dist = distance.euclidean(self.vertices[m_idx].m,f_set[0])
        idx = 0
        for i in range(1,len(f_set)):
            if dist > distance.euclidean(self.vertices[m_idx].m,f_set[i]):
                dist = distance.euclidean(self.vertices[m_idx].m,f_set[i])
                idx = i

        real_idx = self.vertices[m_idx].f_set[idx]

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
        print("m update done!")
        print("============================================================")

        self.make_vertex()
        print("make node done!")
        print("============================================================")

        self.update_vertex(x,y)
        print("update node done!")
        print("============================================================")










