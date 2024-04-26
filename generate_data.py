import sys
import numpy as np

if len(sys.argv) < 4:
    print("Usage: python generate_data.py data.txt N D")
    exit(1)

N, D = int(sys.argv[2]), int(sys.argv[3])
data_file = sys.argv[1]

datapoints = np.random.rand(N, D)
np.savetxt(data_file, datapoints, fmt='%.6f')