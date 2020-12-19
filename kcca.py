import numpy as np
from numpy.linalg import inv

from utils import construct_gram_matrix


class KernelCCA:
    def __init__(self, rx=0.1, ry=0.1):
        """
        Kernel CCA
        """
        self.rx = rx
        self.ry = ry

    def construct_kernel_matrix(self, X, Y):
        raise NotImplementedError

    def fit(self, X, Y):
        """
        Parameters
        ----------
        X: np.array; (dx, N)
        Y: np.array; (dy, N)
        """
        self.X = X
        self.Y = Y

        N = X.shape[1]
        Kx, Ky = self.construct_kernel_matrix(X, Y)
        self.Kx, self.Ky = Kx, Ky

        Tx = inv(Kx @ Kx + self.rx * np.eye(N))
        Ty = inv(Ky @ Ky + self.ry * np.eye(N))
        Kxy = Kx @ Ky
        R = Tx @ Kxy
        S = Ty @ Kxy.T
        Mx = R @ S
        My = S @ R

        Dx, U = self.eig(Mx)
        Dy, V = self.eig(My)

        self.Dx = Dx
        self.U = U
        self.Dy = Dy
        self.V = V

        # sort eigenvectors in the descending order of eigenvalues
        # U = U[:, ::-1]
        # V = V[: ,::-1]
        self.projections_x = Kx @ U / np.sqrt(np.sum((Kx @ U) ** 2, 0))
        self.projections_y = Ky @ V / np.sqrt(np.sum((Ky @ V) ** 2, 0))

        return self

    def estimate_pointwise_dependence(self, X, Y):
        raise NotImplementedError

    def project_x(self, x):
        # return predicted optimal projections based on the Nystrom method
        raise NotImplementedError

    def project_y(self, y):
        # return predicted optimal projections based on the Nystrom method
        raise NotImplementedError

    @staticmethod
    def eig(L):
        lambdas, V = np.linalg.eig(L)  # eigenvalues are not necessarily ordered
        idx = np.real(lambdas).argsort()[::-1]
        lambdas = np.real(lambdas)[idx]
        V = np.real(V)[:, idx]
        return lambdas, V


class GaussianKernelCCA(KernelCCA):
    def __init__(self, rx, ry, sigma_x, sigma_y):
        super().__init__(rx, ry)
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def construct_kernel_matrix(self, X, Y):
        self.Gx = construct_gram_matrix(X)  # (n, n)
        self.Gy = construct_gram_matrix(Y)  # (n, n)
        Wx = np.exp(- self.Gx / (2 * self.sigma_x ** 2))
        Wy = np.exp(- self.Gy / (2 * self.sigma_y ** 2))

        return Wx, Wy
    