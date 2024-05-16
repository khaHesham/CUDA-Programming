// kmeans algorithm on CPU

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <omp.h>


#define MAX_ITER 100

// Function to read data from file
int read_data(float *data, int n_points, int n_dims, char *filename)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("Error: File not found\n");
        return 0;
    }
    for (int i = 0; i < n_points; i++)
    {
        for (int j = 0; j < n_dims; j++)
        {
            fscanf(file, "%f", &data[i * n_dims + j]);
        }
    }
    fclose(file);
    return 1;
}

// Function to write data to file
void write_data(float *data, int n_points, int n_dims, char *filename)
{
    FILE *file = fopen(filename, "w");
    for (int i = 0; i < n_points; i++)
    {
        for (int j = 0; j < n_dims; j++)
        {
            fprintf(file, "%f ", data[i * n_dims + j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

// Function to initialize centroids
void init_centroids(float *data, float *centroids, int n_points, int n_dims, int n_clusters)
{
    for (int i = 0; i < n_clusters; i++)
    {
        int index = rand() % n_points;
        for (int j = 0; j < n_dims; j++)
        {
            centroids[i * n_dims + j] = data[index * n_dims + j];
        }
    }
}

// Function to calculate distance between two points
float distance(float *point, float *centroid, int n_dims)
{
    float sum = 0;
    for (int i = 0; i < n_dims; i++)
    {
        sum += (point[i] - centroid[i]) * (point[i] - centroid[i]);
    }
    return sqrt(sum);
}

// Function to assign points to clusters
void assign_clusters(float *data, float *centroids, int *clusters, int n_points, int n_dims, int n_clusters)
{
    for (int i = 0; i < n_points; i++)
    {
        float min_dist = distance(&data[i * n_dims], &centroids[0], n_dims);
        clusters[i] = 0;
        for (int j = 1; j < n_clusters; j++)
        {
            float dist = distance(&data[i * n_dims], &centroids[j * n_dims], n_dims);
            if (dist < min_dist)
            {
                min_dist = dist;
                clusters[i] = j;
            }
        }
    }
}

// Function to update centroids
void update_centroids(float *data, float *centroids, int *clusters, int n_points, int n_dims, int n_clusters)
{
    int *counts = (int *)malloc(n_clusters * sizeof(int));
    for (int i = 0; i < n_clusters; i++)
    {
        counts[i] = 0;
        for (int j = 0; j < n_dims; j++)
        {
            centroids[i * n_dims + j] = 0;
        }
    }
    for (int i = 0; i < n_points; i++)
    {
        int cluster = clusters[i];
        counts[cluster]++;
        for (int j = 0; j < n_dims; j++)
        {
            centroids[cluster * n_dims + j] += data[i * n_dims + j];
        }
    }
    for (int i = 0; i < n_clusters; i++)
    {
        if (counts[i] > 0)
        {
            for (int j = 0; j < n_dims; j++)
            {
                centroids[i * n_dims + j] /= counts[i];
            }
        }
    }
    free(counts);
}

// Function to run kmeans algorithm
void kmeans(float *data, float *centroids, int *clusters, int n_points, int n_dims, int n_clusters)
{
    int iter = 0;
    while (iter < MAX_ITER)
    {
        assign_clusters(data, centroids, clusters, n_points, n_dims, n_clusters);
        update_centroids(data, centroids, clusters, n_points, n_dims, n_clusters);
        iter++;
    }
}

int main(int argc, char **argv)
{
    if (argc != 6)
    {
        printf("Usage: ./KmeansCPU <data_file> <clusters_file> <n_points> <n_dims> <n_clusters>\n");
        return 0;
    }

    // Read data
    int n_points, n_dims, n_clusters;

    n_points = atoi(argv[3]);
    n_dims = atoi(argv[4]);
    n_clusters = atoi(argv[5]);

    float *data = (float *)malloc(n_points * n_dims * sizeof(float));
    if (!read_data(data, n_points, n_dims, argv[1]))
    {
        return 0;
    }

    printf("Number of points: %d\n", n_points);

    // Initialize centroids
    float *centroids = (float *)malloc(n_clusters * n_dims * sizeof(float));
    int *clusters = (int *)malloc(n_points * sizeof(int));


    // set random seed
    srand(0);

    init_centroids(data, centroids, n_points, n_dims, n_clusters);

    printf("Number of clusters: %d\n", n_clusters);

    // Run kmeans algorithm
    kmeans(data, centroids, clusters, n_points, n_dims, n_clusters);

    printf("Kmeans algorithm completed\n");

    // write clusters to file
    FILE *file = fopen(argv[2], "w");
    for (int i = 0; i < n_points; i++)
    {
        fprintf(file, "%d\n", clusters[i]);
    }
    fclose(file);


    return 0;
}