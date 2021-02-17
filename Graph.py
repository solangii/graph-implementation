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
        self.anchor_set = self.init_anchor(self.feature_set, self.vertex_nums)
        self.gpu = gpu
        self.vertices = []

        if self.gpu <0:
            self.xp = np
            self.dist = norm
        else:
            #ToDo GPU번호지정
            import cupy as cp
            self.xp = cp
            self.dist = norm_gpu

            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % self.gpu

    def init_anchor(self, f_set, v_num):
        """
        :return: initialize anchor that randomly chosen at feature set
        """
        rand_idx = np.random.randint(f_set.shape[0], size= v_num)
        anchor = f_set[rand_idx, :]
        #print("initialize anchor done!")
        return anchor #ndarray

    def compute_rank_order(self, feature, anchor_set):
        """
        compute distance with feature and m, and ordering it
        :param feature: each feature
        :param anchor_set:
        :return: rank order index for feature
        """
        Df = []

        for i in range(anchor_set.shape[0]):
            dis = self.dist(feature - anchor_set[i])
            Df.append(dis)
        Df = self.xp.array(Df)
        Rf = self.xp.argsort(Df).tolist()

        return Rf #list

    def update_anchor(self, feature, anchor, i):
        """
        update anchor(m) by equation3 in FSCIL.
        :param feature: each feature
        :param anchor: centroid
        :param i: iterate number
        :return updated anchor position
        """
        rate = self.eta * self.xp.exp(-i/self.alpha)
        updated_anchor = anchor + rate *(feature - anchor)
        return updated_anchor #ndarray

    def update_anchor_set(self, f_set, anchor_set):

        for f_idx in tqdm(range(f_set.shape[0]), desc="update anchor"):
            f = f_set[f_idx]
            Rf = self.compute_rank_order(f, anchor_set)

            for i in range(self.max_iter):
                #Todo 이거 max iter수가 100인데 inc에서는 m이 그만큼 안많아서 예외처리
                if len(Rf) < self.max_iter:
                    break
                m_idx = Rf[i]
                m = self.anchor_set[m_idx]
                anchor_set[m_idx] = self.update_anchor(f,m,i)

        return anchor_set #ndarray

    def make_vertex(self, anchor_set, v_num):
        vertices = []
        for i in tqdm(range(v_num), desc="make vertex"):
            #self.vertices.append(Vertex(self.anchor_set[i], self.xp))
            vertices.append(Vertex(anchor_set[i], self.xp))
        return vertices #list

    def update_vertex(self, image_set, label_set, vertices, f_set, anchor_set):
        for i in tqdm(range(f_set.shape[0]), desc="winner vector compute"):
            nearest_m = self.compute_rank_order(f_set[i],anchor_set)[0]
            vertices[nearest_m].f_idx.append(i)

        for i in tqdm(range(len(vertices)), desc="update vertex"):
            winner_set = []
            for j in range(len(vertices[i].f_idx)):
                winner_set.append(f_set[vertices[i].f_idx[j]])

            winner_set = self.xp.array(winner_set)

            #ToDo exception
            if len(winner_set)==0:
                vertices[i].l = -1
                vertices[i].z = -1
                vertices[i].c = -1
            else:
                vertices[i].l = self.compute_lambda(winner_set)
                vertices[i].z, vertices[i].c = self.compute_nearest(i, winner_set, image_set, label_set, vertices)

        return vertices

    def compute_lambda(self, f_set):
        """
        :param f_set:
        :return: compute variance for winner vectors
        """
        cov = self.xp.cov(self.xp.transpose(f_set))
        var = self.xp.diag(cov)
        lam = self.xp.diag(var)
        return lam #ndarray

    def compute_nearest(self, m_idx, f_set, image_set, label_set, vertices):
        """
        :param vertices:
        :param m_idx: vertex index
        :param f_set: nearest feature set
        :param image_set: image set
        :param label_set: label set
        :return: 모든 Vertex에 대해서 가장 가까운 pseudo image와 label을 찾음
        """
        if self.xp is np:
            comp = norm
        else:
            comp = norm_gpu

        dist = comp(vertices[m_idx].m - f_set[0])
        idx = 0
        for i in range(1,len(f_set)):
            if dist > comp(vertices[m_idx].m - f_set[i]):
                dist = comp(vertices[m_idx].m - f_set[i])
                idx = i

        real_idx = vertices[m_idx].f_idx[idx]

        z = image_set[real_idx]
        c = label_set[real_idx]
        return z, c

    def pretrain(self, x,y):
        """
        :param x: image_set for pretrain data set
        :param y: label-set for pretrain data set
        :return:
        """
        self.anchor_set = self.update_anchor_set(self.feature_set, self.anchor_set)
        print("anchor update done!")
        print("============================================================")

        self.vertices = self.make_vertex(self.anchor_set, self.vertex_nums)
        print("make vertex done!")

        self.vertices = self.update_vertex(x, y, self.vertices, self.feature_set, self.anchor_set)
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

    def update_graph(self, f_set, class_num, image_set, label_set):
        """
        :param label_set:
        :param image_set:
        :param f_set: novel feature set in Sub-Epi (Query key)
        :param class_num: class number = new vertex number
        :return:
        """
        #novel anchor initialize
        new_anchor_set = self.init_anchor(f_set, class_num)

        #novel anchor update
        new_anchor_set = self.update_anchor_set(f_set, new_anchor_set)

        #make new vertex
        vertices = self.make_vertex(new_anchor_set, class_num)

        #update new vertex
        vertices = self.update_vertex(image_set,label_set,vertices,f_set,new_anchor_set)

        #update for graph

        self.xp.concatenate((self.anchor_set, new_anchor_set), axis=0)
        self.vertices.append(vertices)

        print("update done (incremental sub-episode phase)")