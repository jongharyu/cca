import numpy as np
from numpy.linalg import svd
from sklearn.neighbors import NearestNeighbors

from utils import construct_gram_matrix


class NonparametricCCA:
    def __init__(self):
        """
        Generic nonparamteric CCA that solves multidimensional HGR maximal correlation problem
        """
        pass

    def fit(self, X, Y):
        """
        Parameters
        ----------
        X: np.array; (dx, N)
        Y: np.array; (dy, N)
        """
        self.X = X - X.mean(axis=1, keepdims=True)
        self.Y = Y - Y.mean(axis=1, keepdims=True)

        # 1) Estimate PMI matrix
        self.pointwise_dependence = self.estimate_pointwise_dependence(self.X, self.Y)  # p(x,y)/(p(x)p(y))

        # 2) Compute SVD
        U, sigmas, VT = svd(self.pointwise_dependence, full_matrices=False)  # U.shape = (N, N), VT.shape = (N, N)

        # 3) Scale to find projections of training data
        N = X.shape[1]
        self.canonical_correlations = N * sigmas
        self.projections_x = np.sqrt(N) * U.T  # (N, N)
        self.projections_y = np.sqrt(N) * VT  # (N, N)
        # projections are in the form of (dimensions, samples); that is,
        # l-th row = new samples at the l-th canonical dimension
        # n-th col = a new representation of the n-th sample

        return self

    def estimate_pointwise_dependence(self, X, Y):
        raise NotImplementedError

    def project_x(self, x):
        # return predicted optimal projections based on the Nystrom method
        raise NotImplementedError

    def project_y(self, y):
        # return predicted optimal projections based on the Nystrom method
        raise NotImplementedError


class GaussianKdeNCCA(NonparametricCCA):
    def __init__(self, sigma_x=1.0, sigma_y=1.0, self_tuning_k=None):
        """
        NCCA with Gaussian KDE
        """
        super().__init__()
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.self_tuning = False if self_tuning_k is None else True
        self.self_tuning_k = self_tuning_k

    def estimate_pointwise_dependence(self, X, Y):
        N = X.shape[1]
        allone = np.ones((N, 1))

        # 1) Construct Gram matrices
        self.Gx = construct_gram_matrix(X)
        self.Gy = construct_gram_matrix(Y)

        # TODO: implement heuristic truncation with k-NN
        # 1') Construct weight matrices using Gaussian (i.e., RBF) kernel
        if self.self_tuning:
            self.sigma_x = self.self_tuned_bandwidth(X)
            self.sigma_y = self.self_tuned_bandwidth(Y)
            Wx = np.exp(- self.Gx / (2 * self.sigma_x @ self.sigma_x.T))
            Wx /= np.sqrt(self.sigma_x @ self.sigma_x.T) ** X.shape[0]
            Wy = np.exp(- self.Gy / (2 * self.sigma_y @ self.sigma_y.T))
            Wy /= np.sqrt(self.sigma_y @ self.sigma_y.T) ** Y.shape[0]
        else:
            Wx = np.exp(- self.Gx / (2 * self.sigma_x ** 2))
            Wy = np.exp(- self.Gy / (2 * self.sigma_y ** 2))

        # 2) Normalize to make Wx and Wy to be (right) stochastic
        np.fill_diagonal(Wx, 0)
        np.fill_diagonal(Wy, 0)
        self.Wx = Wx / (Wx @ allone)
        self.Wy = Wy / (Wy @ allone)

        # 3) Compute the sample pointwise dependence matrix
        pointwise_dependence = self.Wx @ self.Wy.T / N  # matrix multiplication

        return pointwise_dependence

    def project_x(self, x):
        # return predicted optimal projections based on the Nystrom method
        # TODO: verify the implementation
        # TODO: modify it for adaptive bandwidth selection
        Wz = np.exp(- (x - self.X) @ np.ones(self.X.shape[0]) / (2 * self.sigma_x ** 2))
        Wz = Wz / np.sum(Wz)
        S_new = np.vstack([self.Wx, Wz]) @ self.Wy

        return ((S_new @ self.projections_y.T) / self.canonical_correlations)

    def self_tuned_bandwidth(self, X):
        # assume that X is of shape (feature_dim, sample_size)
        knn_distances, _ = NearestNeighbors(n_neighbors=self.self_tuning_k + 1).fit(X.T).kneighbors(X.T)
        sigma = knn_distances[:, -1:]  # (num_samples, 1)
        return sigma


class KnnNCCA(NonparametricCCA):
    def __init__(self, kx=3, ky=3, kxy=3):
        """
        NCCA with Gaussian KDE
        """
        super().__init__()
        self.kx = kx
        self.ky = ky
        self.kxy = kxy

    def estimate_pointwise_dependence(self, X, Y):
        N = X.shape[1]

        # assume that X is of shape (feature_dim, sample_size)
        knn_distances_x, _ = NearestNeighbors(n_neighbors=self.kx + 1).fit(X.T).kneighbors(X.T)
        knn_distances_x = knn_distances_x[:, -1:]  # (num_samples, 1)

        knn_distances_y, _ = NearestNeighbors(n_neighbors=self.ky + 1).fit(Y.T).kneighbors(Y.T)
        knn_distances_y = knn_distances_y[:, -1:]  # (num_samples, 1)

        XY = np.vstack([X, Y])  # (dim_xy, N)

        # TODO: more efficient implementation?
        XxY = np.zeros((X.shape[0] + Y.shape[0], N, N))  # (dim_xy, N, N)
        for i in range(N):
            for j in range(N):
                XxY[:, i, j] = np.hstack([X[:, i], Y[:, j]])
        XxY = XxY.reshape((-1, N**2))
        knn_distances_xy, _ = NearestNeighbors(n_neighbors=self.kxy + 1).fit(XY.T).kneighbors(XxY.T)
        knn_distances_xy = knn_distances_xy[:, -1:]  # (N**2, 1)
        knn_distances_xy = knn_distances_xy.reshape((N, N))

        pointwise_dependence = ((knn_distances_x ** X.shape[0]) @ (knn_distances_y ** Y.shape[0]).T) / (knn_distances_xy ** (X.shape[0] + Y.shape[0])) * \
                               self.kxy / (self.kx * self.ky) / N

        return pointwise_dependence

"""
To implement:
- Partially linear NCCA
- Kernel CCA
- Deep CCA
"""
