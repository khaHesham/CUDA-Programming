import torch
import numpy as np
from kmeans_pytorch import kmeans
import sys

def main():
    # Read input arguments
    data_file = sys.argv[1]
    clusters_file = sys.argv[2]
    centroids_file = sys.argv[3]
    num_clusters = int(sys.argv[4])
    
    # Load data
    with open(data_file, 'r') as f:
        lines = f.readlines()
        data = torch.tensor([list(map(float, line.split())) for line in lines]).to(torch.device('cuda:0'))
    
    # kmeans
    cluster_ids_x, cluster_centers = kmeans(
        X=data, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0'),iter_limit = 100
    )
    
    # Save final cluster indices
    torch.save(cluster_ids_x.cpu(), clusters_file)
    
    # Save final centroids
    torch.save(cluster_centers.cpu(), centroids_file)
    
    # Save cluster labels for each data point
    with open(clusters_file, 'w') as f:
        for cluster_idx in cluster_ids_x:
            f.write(f"{cluster_idx.item()}\n")
    

if __name__ == "__main__":
    main()
