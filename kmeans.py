import sys
import numpy as np
from scipy.cluster.vq import kmeans2

RAND_SEED = 42

if len(sys.argv) < 4:
    print("Usage: python kmeans.py data.txt clusters.txt centroids.txt init_centroids.txt, K")
    exit(1)

K = int(sys.argv[5])

data = np.loadtxt(sys.argv[1])
N, D = data.shape

np.random.seed(RAND_SEED)
init_centroids = np.random.rand(K, D)
np.savetxt(sys.argv[4], init_centroids, fmt='%.8f')
centroids, labels = kmeans2(data, init_centroids, minit='matrix', iter = 100)

np.savetxt(sys.argv[2], labels, fmt='%d')
np.savetxt(sys.argv[3], centroids, fmt='%.6f')
