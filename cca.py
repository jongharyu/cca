import numpy as np
from numpy.linalg import inv, svd
from scipy.linalg import sqrtm


class CCA:
    def __init__(self):
        """Sample CCA
        """
        pass

    @property
    def mean_x(self):
        return self.X.mean(axis=1, keepdims=1)

    @property
    def mean_y(self):
        return self.Y.mean(axis=1, keepdims=1)

    def fit(self, X, Y):
        """
        Parameters
        ----------
        X: np.array; (dx, n)
        Y: np.array; (dy, n)
        """
        self.X = X
        self.Y = Y

        # 1) Compute covariance matrices O(n * max(dx,dy)^2)
        dx = X.shape[0]
        cov = np.cov(X, Y, bias=True)
        self.cov_xx = cov[:dx, :dx]
        self.cov_yy = cov[dx:, dx:]
        self.cov_xy = cov[:dx, dx:]

        # 2) Compute cross-correlation matrix: T = (Cov(X))^{-1/2} Cov(X,Y) (Cov(Y))^{-1/2}
        sqrt_inv_cov_xx = inv(sqrtm(self.cov_xx))
        sqrt_inv_cov_yy = inv(sqrtm(self.cov_yy))
        self.T = sqrt_inv_cov_xx @ self.cov_xy @ sqrt_inv_cov_yy  # (dx, dy)

        # 3) SVD of T
        U, self.canonical_correlations, VT = svd(self.T, full_matrices=False)  # U: (dx, N), V: (dy, N)

        # 4) Compute optimal projection matrices; each column is a canonical direction
        self.canonical_directions_x = sqrt_inv_cov_xx @ U       # (dx, N)
        self.canonical_directions_y = sqrt_inv_cov_yy @ (VT.T)  # (dy, N)

        return self

    def project(self, x, y):
        # return (predicted) optimal projections; each column is a new representation
        return self.project_x(x), self.project_y(y)

    def project_x(self, x):
        # return (predicted) optimal projections
        projections_x = (self.canonical_directions_x.T @ (x - self.mean_x))  # (N, N)
        return projections_x

    def project_y(self, y):
        # return (predicted) optimal projections
        projections_y = (self.canonical_directions_y.T @ (y - self.mean_y))  # (N, N)
        return projections_y
