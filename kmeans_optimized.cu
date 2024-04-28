#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define CONVERGENCE_THRESHOLD 0

typedef unsigned int uint;

typedef struct Params {
    uint N;
    uint D;
    uint K;
} Params;


// Naiive version
__global__ void assign_points(float* datapoints, float* centroids, uint* assignments, uint* changed, Params params)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < params.N){
        uint nearest_centroid = 0;
        float min_dist = FLT_MAX;
    
        for(int centroid_id = 0; centroid_id < params.K; centroid_id++){
            float dist = 0;
    
            for(int j = 0; j < params.D; j++){
                float diff = datapoints[idx*params.D + j] - centroids[centroid_id*params.D + j];
                dist += diff * diff;
            }
    
            if(dist < min_dist){
                min_dist = dist;
                nearest_centroid = centroid_id;
            }
        }
    
        if (nearest_centroid != assignments[idx]){
            assignments[idx] = nearest_centroid;
            atomicAdd(changed, 1); // Increment the number of points whose assigned cluster changed
        }
    }
}

__global__ void update_centroids(float* datapoints, uint* assignments, float* centroids, uint* clusters_count, Params params)
{
    // Define private centroids in shared memory (privatization + shared memory optimization)
    extern __shared__ char shared_mem[];

    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Since only one shared memory array is allowed, we divide it using pointer arithmetic
    float* centroids_s = (float*) shared_mem;
    uint* clusters_count_s = (uint*) &centroids_s[params.K * params.D];

    // Initialize both centroids and centroid counts to zeros
    for(int cluster = threadIdx.x; cluster < params.K; cluster += blockDim.x){
        clusters_count_s[cluster] = 0;

        for(int j = 0; j < params.K; j++)
            centroids_s[cluster * params.D + j] = 0.0f;
    }

    __syncthreads();

    if(idx < params.N){
        // Accumulate in the private shared arrays atomically
        uint cluster_id = assignments[idx];
  
        atomicAdd(&clusters_count_s[cluster_id], (uint)1);
  
        for(int j = 0; j < params.D; j++)
            atomicAdd(&centroids_s[cluster_id * params.D + j], datapoints[idx * params.D + j]);
    }

    __syncthreads();

    // Commit results in the global array atomically
    for(int cluster = threadIdx.x; cluster < params.K; cluster += blockDim.x){
        atomicAdd(&clusters_count[cluster], clusters_count_s[cluster]);

        for(int j = 0; j < params.D; j++)
            atomicAdd(&centroids[cluster * params.D + j], centroids_s[cluster * params.D + j]);
    }

}

// Should use templates instead (later)
__global__ void set_to_zero(float* data, uint n)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] = 0.0f;
}
__global__ void set_to_zero(uint* data, uint n)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] = 0;
}

__global__ void divide(float* dividend, uint* divisor, uint n, uint D)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n){
        for(int j = 0; j < D; j++)
            dividend[idx*D + j] = dividend[idx*D + j] / divisor[idx];
    }
}

void initialize_centroids(float* centroids, float* datapoints, Params params) {
    // srand(time(NULL));
    // for (int i = 0; i < params.K; i++) {
    //     uint point_idx = rand() % params.N; // generate a random number between [0, N)
    //     for (int j = 0; j < params.D; j++)
    //         centroids[i * params.D + j] = datapoints[point_idx * params.D + j];
    // }
    // for (int i = 0; i < params.K * params.D; i++)
    //     centroids[i] = (float)rand() / RAND_MAX;

    FILE* init_centroids_file = fopen("init.txt", "r");
    if (init_centroids_file == NULL) {
        printf("Error opening data file.\n");
        exit(1);
    }

    for (int i = 0; i < params.K * params.D; i++)
        fscanf(init_centroids_file, "%f", &centroids[i]);

    fclose(init_centroids_file);
}

void write_results(float* centroids, uint* assignments, const char* clusters_path, const char* centroids_path, Params params){
    FILE* clusters_file = fopen(clusters_path, "w");
    if (clusters_file == NULL) {
        printf("Error opening clusters file.\n");
        exit(1);
    }

    for (int i = 0; i < params.N; i++)
        fprintf(clusters_file, "%d\n", assignments[i]);

    fclose(clusters_file);

    FILE* centroids_file = fopen(centroids_path, "w");
    if (centroids_file == NULL) {
        printf("Error opening centroids file.\n");
        exit(1);
    }
    
    for (int i = 0; i < params.K; i++) {
        for (int j = 0; j < params.D; j++)
            fprintf(centroids_file, "%f ", centroids[i * params.D + j]);
        fprintf(centroids_file, "\n");
    }

    fclose(centroids_file);
}

void kmeans(float* datapoints, float* centroids, uint* assignments, Params params){
    uint N = params.N, D = params.D, K = params.K;

    // Allocating device memory
    float* d_datapoints, * d_centroids;
    uint* d_assignments, * d_clusters_count;

    cudaMalloc((void**)&d_datapoints, N * D * sizeof(float));
    cudaMalloc((void**)&d_centroids, K * D * sizeof(float));
    cudaMalloc((void**)&d_assignments, N * sizeof(uint));
    cudaMalloc((void**)&d_clusters_count, K * sizeof(int));

    initialize_centroids(centroids, datapoints, params);

    // Copy data to device
    cudaMemcpy(d_datapoints, datapoints, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, K * D * sizeof(float), cudaMemcpyHostToDevice);

    // Grid configuration (Should be reconsidered later)
    dim3 gridDim((N-1)/BLOCK_SIZE + 1, 1);
    dim3 blockDim(BLOCK_SIZE, 1);

    bool converged = false;

    uint changed, *d_changed;
    cudaMalloc((void**)&d_changed, sizeof(uint));

    size_t shared_mem_size = K * D * sizeof(float) + K * sizeof(uint);

    assign_points << <gridDim, blockDim >> > (d_datapoints, d_centroids, d_assignments, d_changed, params);
    cudaDeviceSynchronize();

    while(!converged) {
        set_to_zero << <(K*D-1)/BLOCK_SIZE + 1, blockDim >> > (d_centroids, K * D);
        set_to_zero << <(K-1)/BLOCK_SIZE + 1, blockDim >> > (d_clusters_count, K);
        update_centroids << <gridDim, blockDim, shared_mem_size >> > (d_datapoints, d_assignments, d_centroids, d_clusters_count, params);
        divide << <(K-1)/BLOCK_SIZE + 1, blockDim >> > (d_centroids, d_clusters_count, K, D);

        cudaMemset(d_changed, 0, sizeof(uint));
        assign_points << <gridDim, blockDim >> > (d_datapoints, d_centroids, d_assignments, d_changed, params);
        cudaMemcpy(&changed, d_changed, sizeof(uint), cudaMemcpyDeviceToHost);

        converged = (changed <= CONVERGENCE_THRESHOLD);
    }

    cudaMemcpy(centroids, d_centroids, K * D * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(assignments, d_assignments, N * sizeof(uint), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_datapoints);
    cudaFree(d_centroids);
    cudaFree(d_assignments);
    cudaFree(d_clusters_count);
}

int main(int argc, char* argv[])
{
    if (argc < 7) {
        printf("Usage: ./kmeans data.txt clusters.txt centroids.txt N D K");
        return 1;
    }

    uint N = atoi(argv[4]);
    uint D = atoi(argv[5]);
    uint K = atoi(argv[6]);
    Params params = { .N = N, .D = D, .K = K };

    // Allocating host memory
    float* datapoints, * centroids;
    uint* assignments;

    datapoints = (float*)malloc(N * D * sizeof(float));
    centroids = (float*)malloc(K * D * sizeof(float));
    assignments = (uint*)malloc(N * sizeof(uint));

    // Read data points input file
    FILE* data_file = fopen(argv[1], "r");
    if (data_file == NULL) {
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
