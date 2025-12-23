#include "core/kmeans_gpu.h"

#include <cuda_runtime.h>

#include <stdexcept>

#include "metrics/timers.h"

namespace kmeans::core {

namespace {

void check_cuda(cudaError_t status, const char *message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

__global__ void assign_kernel(const float *data, const float *centroids, int *labels, int n_samples, int n_features,
                              int n_clusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) {
        return;
    }

    const float *point = data + static_cast<size_t>(idx) * n_features;
    float best_distance = 1e30f;
    int best_cluster = 0;

    for (int k = 0; k < n_clusters; ++k) {
        const float *centroid = centroids + static_cast<size_t>(k) * n_features;
        float distance = 0.0f;
        for (int d = 0; d < n_features; ++d) {
            float diff = point[d] - centroid[d];
            distance += diff * diff;
        }
        if (distance < best_distance) {
            best_distance = distance;
            best_cluster = k;
        }
    }

    labels[idx] = best_cluster;
}

__global__ void reset_kernel(float *centroid_sums, int *counts, int n_clusters, int n_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_clusters * n_features;
    if (idx < total) {
        centroid_sums[idx] = 0.0f;
    }
    if (idx < n_clusters) {
        counts[idx] = 0;
    }
}

__global__ void accumulate_kernel(const float *data, const int *labels, float *centroid_sums, int *counts, int n_samples,
                                  int n_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) {
        return;
    }

    int label = labels[idx];
    atomicAdd(&counts[label], 1);
    const float *point = data + static_cast<size_t>(idx) * n_features;
    float *sum = centroid_sums + static_cast<size_t>(label) * n_features;
    for (int d = 0; d < n_features; ++d) {
        atomicAdd(&sum[d], point[d]);
    }
}

__global__ void finalize_kernel(float *centroids, const float *centroid_sums, const int *counts, int n_clusters,
                                int n_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_clusters * n_features;
    if (idx >= total) {
        return;
    }
    int cluster = idx / n_features;
    int count = counts[cluster];
    if (count == 0) {
        return;
    }
    centroids[idx] = centroid_sums[idx] / static_cast<float>(count);
}

}  // namespace

KMeansGPU::KMeansGPU(int n_clusters, int max_iters, float tol, kmeans::utils::Logger logger, int block_size)
    : KMeansBase(n_clusters, max_iters, tol, std::move(logger)), block_size_(block_size) {}

std::string KMeansGPU::name() const {
    return "gpu_cuda";
}

std::vector<int> KMeansGPU::assign_clusters(const data::Dataset &dataset, const std::vector<float> &centroids,
                                            double &elapsed_ms) {
    metrics::Timer timer;
    timer.start();

    float *d_data = nullptr;
    float *d_centroids = nullptr;
    int *d_labels = nullptr;

    size_t data_bytes = dataset.values.size() * sizeof(float);
    size_t centroid_bytes = centroids.size() * sizeof(float);
    size_t label_bytes = static_cast<size_t>(dataset.n_samples) * sizeof(int);

    check_cuda(cudaMalloc(&d_data, data_bytes), "cudaMalloc data");
    check_cuda(cudaMalloc(&d_centroids, centroid_bytes), "cudaMalloc centroids");
    check_cuda(cudaMalloc(&d_labels, label_bytes), "cudaMalloc labels");

    check_cuda(cudaMemcpy(d_data, dataset.values.data(), data_bytes, cudaMemcpyHostToDevice), "cudaMemcpy data");
    check_cuda(cudaMemcpy(d_centroids, centroids.data(), centroid_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy centroids");

    int blocks = (dataset.n_samples + block_size_ - 1) / block_size_;
    assign_kernel<<<blocks, block_size_>>>(d_data, d_centroids, d_labels, dataset.n_samples, dataset.n_features,
                                           n_clusters_);
    check_cuda(cudaGetLastError(), "assign_kernel");

    std::vector<int> labels(dataset.n_samples);
    check_cuda(cudaMemcpy(labels.data(), d_labels, label_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy labels");

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_labels);

    elapsed_ms = timer.stop();
    return labels;
}

std::vector<float> KMeansGPU::update_centroids(const data::Dataset &dataset, const std::vector<int> &labels,
                                               double &elapsed_ms) {
    metrics::Timer timer;
    timer.start();

    float *d_data = nullptr;
    int *d_labels = nullptr;
    float *d_sums = nullptr;
    int *d_counts = nullptr;
    float *d_centroids = nullptr;

    size_t data_bytes = dataset.values.size() * sizeof(float);
    size_t label_bytes = labels.size() * sizeof(int);
    size_t centroid_bytes = static_cast<size_t>(n_clusters_) * dataset.n_features * sizeof(float);

    check_cuda(cudaMalloc(&d_data, data_bytes), "cudaMalloc data");
    check_cuda(cudaMalloc(&d_labels, label_bytes), "cudaMalloc labels");
    check_cuda(cudaMalloc(&d_sums, centroid_bytes), "cudaMalloc sums");
    check_cuda(cudaMalloc(&d_counts, static_cast<size_t>(n_clusters_) * sizeof(int)), "cudaMalloc counts");
    check_cuda(cudaMalloc(&d_centroids, centroid_bytes), "cudaMalloc centroids");

    check_cuda(cudaMemcpy(d_data, dataset.values.data(), data_bytes, cudaMemcpyHostToDevice), "cudaMemcpy data");
    check_cuda(cudaMemcpy(d_labels, labels.data(), label_bytes, cudaMemcpyHostToDevice), "cudaMemcpy labels");

    int total_centroid_entries = n_clusters_ * dataset.n_features;
    int reset_blocks = (total_centroid_entries + block_size_ - 1) / block_size_;
    reset_kernel<<<reset_blocks, block_size_>>>(d_sums, d_counts, n_clusters_, dataset.n_features);
    check_cuda(cudaGetLastError(), "reset_kernel");

    int data_blocks = (dataset.n_samples + block_size_ - 1) / block_size_;
    accumulate_kernel<<<data_blocks, block_size_>>>(d_data, d_labels, d_sums, d_counts, dataset.n_samples,
                                                    dataset.n_features);
    check_cuda(cudaGetLastError(), "accumulate_kernel");

    finalize_kernel<<<reset_blocks, block_size_>>>(d_centroids, d_sums, d_counts, n_clusters_, dataset.n_features);
    check_cuda(cudaGetLastError(), "finalize_kernel");

    std::vector<float> centroids(static_cast<size_t>(n_clusters_) * dataset.n_features);
    check_cuda(cudaMemcpy(centroids.data(), d_centroids, centroid_bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy centroids");

    cudaFree(d_data);
    cudaFree(d_labels);
    cudaFree(d_sums);
    cudaFree(d_counts);
    cudaFree(d_centroids);

    elapsed_ms = timer.stop();
    return centroids;
}

}  // namespace kmeans::core
