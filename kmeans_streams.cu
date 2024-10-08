#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 128
#define CONVERGENCE_THRESHOLD 0
#define MAX_ITER 100
#define NUM_STREAMS 16


typedef unsigned int uint;

typedef struct Params
{
    uint N;
    uint D;
    uint K;
} Params;

__global__ void assign_points(float *datapoints, float *centroids, uint *assignments, Params params)
{
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

    if (idx < params.N)
    {
        uint nearest_centroid = 0;
        float min_dist = FLT_MAX;

        for (int centroid_id = 0; centroid_id < params.K; centroid_id++)
        {
            float dist = 0;

            for (int j = 0; j < params.D; j++)
            {
                float diff = datapoint_s[threadIdx.x * params.D + j] - centroids_s[centroid_id * params.D + j];
                dist += diff * diff;
            }

            if (dist < min_dist)
            {
                min_dist = dist;
                nearest_centroid = centroid_id;
            }
        }

        if (nearest_centroid != assignments[idx])
        {
            assignments[idx] = nearest_centroid;
            // atomicAdd(changed, 1); // Increment the number of points whose assigned cluster changed
        }
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
    FILE* init_centroids_file = fopen("init.txt", "w");
    if (init_centroids_file == NULL) {
        printf("Error opening inital centroids file.\n");
        exit(1);
    }

    for (int i = 0; i < params.K; i++) {
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

void streamed_assign(float* d_datapoints, uint* d_assignments, float* d_centroids, float* datapoints, uint* assignments, Params params, uint dir=0)
{
    uint N = params.N, D = params.D, K = params.K;

    // Use streams to overlap copying datapoints with the initial assignment kernel
    cudaStream_t streams[NUM_STREAMS];

    for(unsigned int s = 0; s < NUM_STREAMS; s++)
        cudaStreamCreate(&streams[s]);

    unsigned int chunk_size = (N - 1) / NUM_STREAMS + 1;
    size_t assign_shared_mem = K * D * sizeof(float) + BLOCK_SIZE * D * sizeof(float);

    for(unsigned int s = 0; s < NUM_STREAMS; s++){

        unsigned int start = s * chunk_size;
        unsigned int end = min(start + chunk_size, N);
        unsigned int Nchunk = end-start;
        Params chunk_params = {Nchunk, D, K};

        if(dir == 0){
            cudaMemcpyAsync(&d_datapoints[start * D], &datapoints[start * D], Nchunk * D * sizeof(float), cudaMemcpyHostToDevice, streams[s]);
            assign_points<<<(Nchunk - 1) / BLOCK_SIZE + 1, BLOCK_SIZE, assign_shared_mem, streams[s]>>>(&d_datapoints[start * D], d_centroids, &d_assignments[start], chunk_params);            
        }
        else{
            assign_points<<<(Nchunk - 1) / BLOCK_SIZE + 1, BLOCK_SIZE, assign_shared_mem, streams[s]>>>(&d_datapoints[start * D], d_centroids, &d_assignments[start], chunk_params);            
            cudaMemcpyAsync(&assignments[start], &d_assignments[start], Nchunk * sizeof(uint), cudaMemcpyDeviceToHost, streams[s]);
        }
    };
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
    cudaMemcpy(d_centroids, centroids, K * D * sizeof(float), cudaMemcpyHostToDevice);

    streamed_assign(d_datapoints, d_assignments, d_centroids, datapoints, assignments, params, 0);
    cudaDeviceSynchronize();

    // Grid configuration (Should be reconsidered later)
    dim3 gridDim((N - 1) / BLOCK_SIZE + 1, 1);
    dim3 blockDim(BLOCK_SIZE, 1);

    size_t update_shared_mem = K * D * sizeof(float) + K * sizeof(uint);
    size_t assign_shared_mem = K * D * sizeof(float) + blockDim.x * D * sizeof(float);

    int iter = 0;
    while (iter < MAX_ITER)
    {
        set_to_zero<<<(K * D - 1) / BLOCK_SIZE + 1, blockDim>>>(d_centroids, K * D);
        set_to_zero<<<(K - 1) / BLOCK_SIZE + 1, blockDim>>>(d_clusters_count, K);
        update_centroids<<<gridDim, blockDim, update_shared_mem>>>(d_datapoints, d_assignments, d_centroids, d_clusters_count, params);
        divide<<<(K - 1) / BLOCK_SIZE + 1, blockDim>>>(d_centroids, d_clusters_count, K, D);

        if(iter != MAX_ITER-1)
            assign_points<<<gridDim, blockDim, assign_shared_mem>>>(d_datapoints, d_centroids, d_assignments, params);
        else
            streamed_assign(d_datapoints, d_assignments, d_centroids, datapoints, assignments, params, 1);

        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(1);
        }

        iter++;
    }

    cudaMemcpy(centroids, d_centroids, K * D * sizeof(float), cudaMemcpyDeviceToHost);

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

    // Allocate datapoints in pinned memory for streams
    cudaMallocHost((void**)&datapoints, N * D * sizeof(float));
    cudaMallocHost ((void**)&assignments, N * sizeof(uint));

    centroids = (float *)malloc(K * D * sizeof(float));

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
    cudaFreeHost(datapoints);
    cudaFreeHost(assignments);
    free(centroids);

    return 0;
}
