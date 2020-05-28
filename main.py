from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
import numpy as np
from numba import jit


@jit(nopython=True)
def fstar(x):
    return np.sin(2 * np.pi * x)


@jit(nopython=True)
def generateData(size, sigma):
    X = np.random.uniform(0, 1, size)
    f = fstar(X)
    y = f + np.random.normal(0, sigma, size=size)
    return X, y


@jit(nopython=True)
def bootWithReturnOnce(data):
    P = data.shape[1]
    return np.sum(data[:, np.random.choice(np.arange(P), P)], 1) / P


@jit(nopython=True)
def boot(data, fbar, iter):
    length, P = data.shape
    bootData = np.zeros((length, iter))
    dists = np.zeros((iter, ))
    for i in range(iter):
        fhatBoot = bootWithReturnOnce(data)
        dists[i] = np.linalg.norm(fbar - fhatBoot)
        bootData[:, i] = fhatBoot
    quantile = np.quantile(dists, 0.95)
    return quantile, bootData[:, dists < quantile]


def fastKRR(X, y, P, sigma, grid):
    n = X.shape[0]
    nu = 5/2
    s = nu + 1/2
    rho = n ** (-(2*s)/(2*s + 1))
    const = sigma ** 2 / (n * rho)
    kernel = ConstantKernel(constant_value=const, constant_value_bounds="fixed") * Matern(length_scale=1.0,
                                                                                    length_scale_bounds="fixed", nu=s)
    gp = GPR(kernel=kernel)

    xPartitioned = np.split(X, P)
    yPartitioned = np.split(y, P)

    fhatp = np.zeros((grid.shape[0], P))

    for (xPartition, yPartition, p) in zip(xPartitioned, yPartitioned, range(P)):
        gp.fit(xPartition.reshape(-1, 1), yPartition)
        fhatp[:, p] = gp.predict(grid.reshape(-1, 1))

    fbar = np.mean(fhatp, 1)

    maxDist, fromTheBall = boot(fhatp, fbar, 1000)

    return fbar,  np.min(fromTheBall, 1), np.max(fromTheBall, 1), maxDist, fromTheBall

