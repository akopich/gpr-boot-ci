# import matplotlib.pyplot as plt
from main import *
from joblib import Parallel, delayed
import time


def checkInterval(N, P):
    sigma = 1
    X, y = generateData(N, sigma)
    gridN = 1000
    grid = np.linspace(0, 1, num=gridN)

    fbar, lower, upper, maxDist, fromTheBall = fastKRR(X, y, P, sigma, grid)
    fstarGrid = fstar(grid)
    isCovered = float(np.linalg.norm(fstarGrid - fbar) < maxDist)
    rmse = np.linalg.norm(fbar - fstarGrid) / np.sqrt(grid.shape[0])
    return np.array([isCovered, rmse, maxDist / np.sqrt(gridN)])


def coverageProbability(N, P):
    def job(i):
        np.random.seed(13 * i)
        return checkInterval(N, P)

    t0 = time.time()
    n_jobs = 48
    n = n_jobs * 20
    probAndRMSEAndDist = sum(Parallel(n_jobs=n_jobs)(delayed(job)(i) for i in range(n))) / n
    print(f"N={N} P={P} prob={probAndRMSEAndDist[0]} rmse={probAndRMSEAndDist[1]} distance={probAndRMSEAndDist[2]} time={time.time() - t0} iter={n}", flush=True)


def main():
    N = 2 ** 17
    coverageProbability(N, 2 ** 6)
    coverageProbability(N, 2 ** 7)
    coverageProbability(N, 2 ** 8)
    coverageProbability(N, 2 ** 9)
    coverageProbability(N, 2 ** 10)
    coverageProbability(N, 2 ** 11)
    coverageProbability(N, 2 ** 12)
    coverageProbability(N, 2 ** 13)


main()

