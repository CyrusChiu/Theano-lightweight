import numpy as np

def mean_subtraction(X):
    return np.mean(X, axis=0)


def zca_whitening(X):
    cov = np.dot(X.T, X) / X.shape[0]
    U, S, V = np.linalg.svd(cov)
    Xrot = np.dot(X, U)
    Xpca_whtie = Xrot / np.sqrt(S + 1e-5)
    Xzca_white = np.dot(Xpca_whtie, U.T)
    return Xzca_white
