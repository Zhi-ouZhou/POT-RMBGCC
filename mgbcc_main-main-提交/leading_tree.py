import numpy as np
from scipy.spatial.distance import cdist


class GB_LeadingTree:
    """
    针对粒球(Granular Ball)优化的引领树
    """

    def __init__(self, centers, radii, counts, lt_num, dc=None):
        """
        :param centers: 粒球中心矩阵 (K, d)
        :param radii: 粒球半径数组 (K,)
        :param counts: 粒球包含的样本数量 (K,) - 重要！用于加权密度
        :param lt_num: 需要构建的树（子树/根节点）的数量，通常设为聚类数 num_classes
        :param dc: 截断距离
        """
        self.centers = centers
        self.radii = radii
        self.counts = counts
        self.lt_num = lt_num
        self.K = len(centers)

        # 粒球间的真实距离 = 中心距离 - 两个球的半径（如果重叠则距离视为0）
        raw_dist = cdist(centers, centers, metric='euclidean')
        radii_matrix = radii[:, None] + radii[None, :]
        self.D = np.maximum(raw_dist - radii_matrix, 0.0)

        # 如果未提供 dc，动态计算（取所有距离的前 2%~10% 作为一个阈值）
        if dc is None:
            self.dc = np.percentile(self.D[self.D > 0], 5) if np.any(self.D > 0) else 0.1
        else:
            self.dc = dc

        self.density = np.zeros(self.K)
        self.Pa = np.zeros(self.K, dtype=int) - 1
        self.layer = np.zeros(self.K, dtype=int)
        self.roots = []

    def fit(self):
        # 1. 计算加权局部密度 (将球的样本容量 counts 加入密度考量)
        tempMat = np.exp(-(self.D / self.dc) ** 2)
        # 密度 = 自身包含的样本数 + 邻居球的样本数按距离衰减
        self.density = self.counts + np.sum(tempMat * self.counts[None, :], axis=1) - self.counts
        Q = np.argsort(self.density)[::-1]  # 密度从大到小排序的索引

        # 2. 计算 delta 和 父节点 Pa
        delta = np.zeros(self.K)
        for i in range(self.K):
            if i == 0:
                delta[Q[i]] = np.max(self.D[Q[i]])
                self.Pa[Q[i]] = -1
            else:
                greaterInds = Q[0:i]  # 密度比当前节点大的所有节点
                D_A = self.D[Q[i], greaterInds]
                delta[Q[i]] = np.min(D_A)
                self.Pa[Q[i]] = greaterInds[np.argmin(D_A)]  # 指向最近的高密度节点

        # 3. 寻找根节点 (gamma = 密度 * delta)
        gamma = self.density * delta
        gamma_D = np.argsort(gamma)[::-1]

        # 断开前 lt_num 个最高 gamma 值的节点，使其成为根节点
        for i in range(min(self.lt_num, self.K)):
            self.Pa[gamma_D[i]] = -1
            self.roots.append(gamma_D[i])

        # 4. 计算层级结构 (Layer)
        for nodei in range(self.K):
            curInd = Q[nodei]
            if self.Pa[curInd] != -1:
                self.layer[curInd] = self.layer[self.Pa[curInd]] + 1

    def get_tree_mask(self):
        """
        生成用于 PyTorch 对比学习的父子邻接矩阵 (单向: 子->父)
        """
        mask = np.zeros((self.K, self.K), dtype=np.float32)
        for i in range(self.K):
            p = self.Pa[i]
            if p != -1:
                mask[i, p] = 1.0  # i 向其父节点 p 靠拢
        return mask


class LeadingTree:
    """
    Leading Tree
    """

    def __init__(self, X_train, dc, lt_num, D):
        self.X_train = X_train
        self.dc = dc
        self.lt_num = lt_num
        self.D = D  # Calculate the distance matrix D
        # print(f'The data type of the distance matrix D is {self.D.dtype}')
        self.density = None
        self.Pa = None
        self.delta = None
        self.gamma = None
        self.gamma_D = None
        self.Q = None
        self.AL = [np.zeros((0, 1), dtype=int) for i in range(lt_num)]  # AL[i] store all indexes of a subtree
        self.layer = np.zeros(len(X_train), dtype=int)

    def ComputeLocalDensity(self, D, dc):
        """
        Calculate the local density of samples
        :param D: The Euclidean distance of all samples
        :param dc:Bandwidth parameters
        :return:
        self.density: local density of all samples
        self.Q: Sort the density index in descending order
        """
        tempMat1 = np.exp(-(D ** 2))
        tempMat = np.power(tempMat1, dc ** (-2))
        self.density = np.sum(tempMat, 1, dtype='float32') - 1
        self.Q = np.argsort(self.density)[::-1]

        # print(f'The data type of density is {self.density.dtype}\n'  #       f'The data type of Q is {self.Q.dtype}')

    def ComputeParentNode(self, D, Q):
        """
        Calculate the distance to the nearest data point of higher density (delta) and the parent node (Pa)
        :param D: The Euclidean distance of all samples
        :param Q:Sort by index in descending order of sample local density
        :return:
        self.delta: the distance of the sample to the closest data point with a higher density
        self.Pa: the index of the parent node of the sample
        """

        self.delta = np.zeros(len(Q), dtype='float32')
        self.Pa = np.zeros(len(Q), dtype=int)
        for i in range(len(Q)):
            if i == 0:
                self.delta[Q[i]] = max(D[Q[i]])
                self.Pa[Q[i]] = -1
            else:
                greaterInds = Q[0:i]
                D_A = D[Q[i], greaterInds]
                self.delta[Q[i]] = min(D_A)
                self.Pa[Q[i]] = greaterInds[np.argmin(D_A)]

        # print(f'The data type of delta is {self.delta.dtype}')

    def ProCenter(self, density, delta, Pa):
        """
        Calculate the probability of being chosen as the center node and Disconnect the Leading Tree
        :param density: local density of all samples
        :param delta: the distance of the sample to the closest data point with a higher density
        :param Pa: the index of the parent node of the sample
        :return:
        self.gamma: the probability of the sample being chosen as a center node
        self.gamma_D: Sort the gamma index in descending order
        """
        self.gamma = density * delta
        self.gamma_D = np.argsort(self.gamma)[::-1]
        # print(f'The data type of gamma is {self.gamma.dtype}')
        # Disconnect the Leading Tree
        for i in range(self.lt_num):
            Pa[self.gamma_D[i]] = -1

    def GetSubtreeR(self, gamma_D, lt_num, Q, pa):
        """
         Subtree
        :param gamma_D:
        :param lt_num: the number of subtrees
        :return:
        self.AL: AL[i] store indexes of a subtrees, i = {0, 1, ..., lt_num-1}
        """
        for i in range(lt_num):
            self.AL[i] = np.append(self.AL[i], gamma_D[i])

        N = len(gamma_D)
        treeID = np.zeros((N, 1), dtype=int) - 1
        for i in range(lt_num):
            treeID[gamma_D[i]] = i

        for nodei in range(N):  ### casscade label assignment
            curInd = Q[nodei]
            if treeID[curInd] > -1:
                continue

            else:
                paID = pa[curInd]
                self.layer[curInd] = self.layer[paID] + 1
                curTreeID = treeID[paID]
                treeID[curInd] = curTreeID
                self.AL[curTreeID[0]] = np.append(self.AL[curTreeID[0]], curInd)

    def Edges(self, Pa):  # store edges of subtrees
        """

        :param Pa:  the index of the parent node of the sample
        :return:
        self. edges: pairs of child node and parent node
        """
        edgesO = np.array(list(zip(range(len(Pa)), Pa)))
        ind = edgesO[:, 1] > -1
        self.edges = edgesO[ind,]

    def fit(self):
        self.ComputeLocalDensity(self.D, self.dc)
        self.ComputeParentNode(self.D, self.Q)
        self.ProCenter(self.density, self.delta, self.Pa)
        self.GetSubtreeR(self.gamma_D, self.lt_num, self.Q, self.Pa)
        self.Edges(self.Pa)
        self.layer = self.layer + 1
