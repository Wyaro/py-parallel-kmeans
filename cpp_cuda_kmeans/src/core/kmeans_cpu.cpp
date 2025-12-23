#include "core/kmeans_cpu.h"

#include <limits>

namespace kmeans::core {

KMeansCPU::KMeansCPU(int n_clusters, int max_iters, float tol, kmeans::utils::Logger logger)
    : KMeansBase(n_clusters, max_iters, tol, std::move(logger)) {}

std::string KMeansCPU::name() const {
    return "cpu_single";
}

std::vector<int> KMeansCPU::assign_clusters(const data::Dataset &dataset, const std::vector<float> &centroids,
                                            double &elapsed_ms) {
    metrics::Timer timer;
    timer.start();

    std::vector<int> labels(dataset.n_samples);
    for (int i = 0; i < dataset.n_samples; ++i) {
        float best_distance = std::numeric_limits<float>::max();
        int best_cluster = 0;
        const size_t offset = static_cast<size_t>(i) * dataset.n_features;

        for (int k = 0; k < n_clusters_; ++k) {
            float distance = 0.0f;
            const size_t centroid_offset = static_cast<size_t>(k) * dataset.n_features;
            for (int d = 0; d < dataset.n_features; ++d) {
                float diff = dataset.values[offset + d] - centroids[centroid_offset + d];
                distance += diff * diff;
            }
            if (distance < best_distance) {
                best_distance = distance;
                best_cluster = k;
            }
        }
        labels[i] = best_cluster;
    }

    elapsed_ms = timer.stop();
    return labels;
}

std::vector<float> KMeansCPU::update_centroids(const data::Dataset &dataset, const std::vector<int> &labels,
                                               double &elapsed_ms) {
    metrics::Timer timer;
    timer.start();

    std::vector<float> centroids(static_cast<size_t>(n_clusters_) * dataset.n_features, 0.0f);
    std::vector<int> counts(n_clusters_, 0);

    for (int i = 0; i < dataset.n_samples; ++i) {
        int label = labels[i];
        ++counts[label];
        const size_t offset = static_cast<size_t>(i) * dataset.n_features;
        const size_t centroid_offset = static_cast<size_t>(label) * dataset.n_features;
        for (int d = 0; d < dataset.n_features; ++d) {
            centroids[centroid_offset + d] += dataset.values[offset + d];
        }
    }

    for (int k = 0; k < n_clusters_; ++k) {
        if (counts[k] == 0) {
            continue;
        }
        const size_t centroid_offset = static_cast<size_t>(k) * dataset.n_features;
        for (int d = 0; d < dataset.n_features; ++d) {
            centroids[centroid_offset + d] /= static_cast<float>(counts[k]);
        }
    }

    elapsed_ms = timer.stop();
    return centroids;
}

}  // namespace kmeans::core
