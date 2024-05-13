#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 128
#define CONVERGENCE_THRESHOLD 0
#define MAX_ITER 100

typedef unsigned int uint;

typedef struct Params
{
    uint N;
    uint D;
    uint K;
} Params;


// Kernel to compute distances between a datapoint and all centroids
__device__ float computeDistance(const float *datapoint, const float *centroid, int D)
{
    float distance = 0.0f;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Compute distance in parallel
    for (int j = tid; j < D; j += blockSize)
    {
        float diff = datapoint[j] - centroid[j];
        distance += diff * diff;
    }

    // Reduce within the block using shared memory
    __shared__ float shared_distance[BLOCK_SIZE];
    shared_distance[tid] = distance;
    __syncthreads();

    for (int stride = blockSize / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared_distance[tid] += shared_distance[tid + stride];
        }
        __syncthreads();
    }

    // Store the final result in distance
    if (tid == 0)
    {
        distance = shared_distance[0];
    }

    return distance;
}

// Child kernel to find the nearest centroid for each datapoint
__global__ void findNearestCentroids(const float *datapoints, const float *centroids, uint *assignments, int N, int K, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        float minDist = FLT_MAX;
        uint nearestCentroid = 0;
        for (int centroidId = 0; centroidId < K; ++centroidId)
        {
            float dist = computeDistance(&datapoints[idx * D], &centroids[centroidId * D], D);
            if (dist < minDist)
            {
                minDist = dist;
                nearestCentroid = centroidId;
            }
        }
        if (nearestCentroid != assignments[idx])
        {
            assignments[idx] = nearestCentroid;
        }
    }
}

__global__ void assign_points(float *datapoints, float *centroids, uint *assignments, Params params) {
     extern __shared__ char shared_mem[];

    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Since only one shared memory array is allowed, we divide it using pointer arithmetic
    float *centroids_s = (float *)shared_mem;
    float *datapoint_s = (float *)&centroids_s[params.K * params.D];

    // Cache the centroids in shared memory
    for (int cluster = threadIdx.x; cluster < params.K; cluster += blockDim.x)
    {
        for (int j = 0; j < params.D; j++)
            centroids_s[cluster * params.D + j] = centroids[cluster * params.D + j];
    }

    // Cache the datapoints since each will be read K times by a thread
    // Although no data is shared between threads, we still want to get the benefit of caching
    if (idx < params.N)
    {
        for (int j = 0; j < params.D; j++)
            datapoint_s[threadIdx.x * params.D + j] = datapoints[idx * params.D + j];
    }

    __syncthreads();

    // check if we have a multiple dimensions then run computeDistance kernell else run the simple distance calculation
    if (params.D > 10)
    {
        float min_dist = FLT_MAX;
        uint min_cluster = 0;

        for (int cluster = 0; cluster < params.K; cluster++)
        {
            float dist = computeDistance(&datapoint_s[threadIdx.x * params.D], &centroids_s[cluster * params.D], params.D);
            if (dist < min_dist)
            {
                min_dist = dist;
                min_cluster = cluster;
            }
        }

        if (min_cluster != assignments[idx])
            assignments[idx] = min_cluster;
    }
    else
    {
        float min_dist = FLT_MAX;
        uint min_cluster = 0;

        for (int cluster = 0; cluster < params.K; cluster++)
        {
            float dist = (datapoint_s[threadIdx.x] - centroids_s[cluster]) * (datapoint_s[threadIdx.x] - centroids_s[cluster]);
            if (dist < min_dist)
            {
                min_dist = dist;
                min_cluster = cluster;
            }
        }

        if (min_cluster != assignments[idx])
            assignments[idx] = min_cluster;
    }

}

__global__ void update_centroids(float *datapoints, uint *assignments, float *centroids, uint *clusters_count, Params params)
{
    // Define private centroids in shared memory (privatization + shared memory optimization)
    extern __shared__ char shared_mem[];

    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Since only one shared memory array is allowed, we divide it using pointer arithmetic
    float *centroids_s = (float *)shared_mem;
    uint *clusters_count_s = (uint *)&centroids_s[params.K * params.D];

    // Initialize both centroids and centroid counts to zeros
    for (int cluster = threadIdx.x; cluster < params.K; cluster += blockDim.x)
    {
        clusters_count_s[cluster] = 0;

        for (int j = 0; j < params.D; j++)
            centroids_s[cluster * params.D + j] = 0.0f;
    }

    __syncthreads();

    if (idx < params.N)
    {
        // Accumulate in the private shared arrays atomically
        uint cluster_id = assignments[idx];

        atomicAdd(&clusters_count_s[cluster_id], (uint)1);

        for (int j = 0; j < params.D; j++)
            atomicAdd(&centroids_s[cluster_id * params.D + j], datapoints[idx * params.D + j]);
    }

    __syncthreads();

    // Commit results in the global array atomically
    for (int cluster = threadIdx.x; cluster < params.K; cluster += blockDim.x)
    {
        atomicAdd(&clusters_count[cluster], clusters_count_s[cluster]);

        for (int j = 0; j < params.D; j++)
            atomicAdd(&centroids[cluster * params.D + j], centroids_s[cluster * params.D + j]);
    }
}

// Should use templates instead (later)
__global__ void set_to_zero(float *data, uint n)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = 0.0f;
}
__global__ void set_to_zero(uint *data, uint n)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = 0;
}

__global__ void divide(float *dividend, uint *divisor, uint n, uint D)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        uint div = divisor[idx];
        if (div != 0)
        {
            for (int j = 0; j < D; j++)
                dividend[idx * D + j] = dividend[idx * D + j] / div;
        }
    }
}

void initialize_centroids(float *centroids, float *datapoints, Params params)
{
    for (int i = 0; i < params.K; i++)
    {
        uint point_idx = rand() % params.N; // generate a random number between [0, N)
        for (int j = 0; j < params.D; j++)
            centroids[i * params.D + j] = datapoints[point_idx * params.D + j];
    }
    // for (int i = 0; i < params.K * params.D; i++)
    //     centroids[i] = (float)rand() / RAND_MAX;

    // FILE* init_centroids_file = fopen("init.txt", "r");
    // if (init_centroids_file == NULL) {
    //     printf("Error opening inital centroids file.\n");
    //     exit(1);
    // }

    // for (int i = 0; i < params.K * params.D; i++)
    //     fscanf(init_centroids_file, "%f", &centroids[i]);

    // fclose(init_centroids_file);

    // write initial centroids to a file
    FILE *init_centroids_file = fopen("init.txt", "w");
    if (init_centroids_file == NULL)
    {
        printf("Error opening inital centroids file.\n");
        exit(1);
    }

    for (int i = 0; i < params.K; i++)
    {
        for (int j = 0; j < params.D; j++)
            fprintf(init_centroids_file, "%f ", centroids[i * params.D + j]);
        fprintf(init_centroids_file, "\n");
    }
    fclose(init_centroids_file);
}

void write_results(float *centroids, uint *assignments, const char *clusters_path, const char *centroids_path, Params params)
{
    FILE *clusters_file = fopen(clusters_path, "w");
    if (clusters_file == NULL)
    {
        printf("Error opening clusters file.\n");
        exit(1);
    }

    for (int i = 0; i < params.N; i++)
        fprintf(clusters_file, "%d\n", assignments[i]);

    fclose(clusters_file);

    FILE *centroids_file = fopen(centroids_path, "w");
    if (centroids_file == NULL)
    {
        printf("Error opening centroids file.\n");
        exit(1);
    }

    for (int i = 0; i < params.K; i++)
    {
        for (int j = 0; j < params.D; j++)
            fprintf(centroids_file, "%f ", centroids[i * params.D + j]);
        fprintf(centroids_file, "\n");
    }

    fclose(centroids_file);
}

void kmeans(float *datapoints, float *centroids, uint *assignments, Params params)
{
    uint N = params.N, D = params.D, K = params.K;

    // Allocating device memory
    float *d_datapoints, *d_centroids;
    uint *d_assignments, *d_clusters_count;

    cudaMalloc((void **)&d_datapoints, N * D * sizeof(float));
    cudaMalloc((void **)&d_centroids, K * D * sizeof(float));
    cudaMalloc((void **)&d_assignments, N * sizeof(uint));
    cudaMalloc((void **)&d_clusters_count, K * sizeof(int));

    initialize_centroids(centroids, datapoints, params);

    // Copy data to device
    cudaMemcpy(d_datapoints, datapoints, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, K * D * sizeof(float), cudaMemcpyHostToDevice);

    // Grid configuration (Should be reconsidered later)
    dim3 gridDim((N - 1) / BLOCK_SIZE + 1, 1);
    dim3 blockDim(BLOCK_SIZE, 1);

    size_t update_shared_mem = K * D * sizeof(float) + K * sizeof(uint);
    size_t assign_shared_mem = K * D * sizeof(float) + blockDim.x * D * sizeof(float);

    assign_points<<<gridDim, blockDim, assign_shared_mem>>>(d_datapoints, d_centroids, d_assignments, params);
    cudaDeviceSynchronize();

    int iter = 0;
    while (iter < MAX_ITER)
    {
        set_to_zero<<<(K * D - 1) / BLOCK_SIZE + 1, blockDim>>>(d_centroids, K * D);
        set_to_zero<<<(K - 1) / BLOCK_SIZE + 1, blockDim>>>(d_clusters_count, K);
        update_centroids<<<gridDim, blockDim, update_shared_mem>>>(d_datapoints, d_assignments, d_centroids, d_clusters_count, params);
        divide<<<(K - 1) / BLOCK_SIZE + 1, blockDim>>>(d_centroids, d_clusters_count, K, D);

        assign_points<<<gridDim, blockDim, assign_shared_mem>>>(d_datapoints, d_centroids, d_assignments, params);

        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(1);
        }

        iter++;
    }

    cudaMemcpy(centroids, d_centroids, K * D * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(assignments, d_assignments, N * sizeof(uint), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_datapoints);
    cudaFree(d_centroids);
    cudaFree(d_assignments);
    cudaFree(d_clusters_count);
}

int main(int argc, char *argv[])
{
    if (argc < 7)
    {
        printf("Usage: ./kmeans data.txt clusters.txt centroids.txt N D K");
        return 1;
    }

    // set the seed for random number generation
    srand(0);

    uint N = atoi(argv[4]);
    uint D = atoi(argv[5]);
    uint K = atoi(argv[6]);
    Params params = {N, D, K};

    // Allocating host memory
    float *datapoints, *centroids;
    uint *assignments;

    datapoints = (float *)malloc(N * D * sizeof(float));
    centroids = (float *)malloc(K * D * sizeof(float));
    assignments = (uint *)malloc(N * sizeof(uint));

    // Read data points input file
    FILE *data_file = fopen(argv[1], "r");
    if (data_file == NULL)
    {
        printf("Error opening data file.\n");
        return 1;
    }

    for (int i = 0; i < N * D; i++)
        fscanf(data_file, "%f", &datapoints[i]);

    fclose(data_file);

    kmeans(datapoints, centroids, assignments, params);

    write_results(centroids, assignments, argv[2], argv[3], params);

    // Free host memory
    free(datapoints);
    free(centroids);
    free(assignments);

    return 0;
}
