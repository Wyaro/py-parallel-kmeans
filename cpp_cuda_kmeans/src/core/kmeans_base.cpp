#include "core/kmeans_base.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace kmeans::core {

KMeansBase::KMeansBase(int n_clusters, int max_iters, float tol, kmeans::utils::Logger logger)
    : n_clusters_(n_clusters), max_iters_(max_iters), tol_(tol), logger_(std::move(logger)) {}

FitResult KMeansBase::fit(const data::Dataset &dataset) {
    FitResult result;
    result.centroids = initialize_centroids(dataset);
    result.labels.assign(dataset.n_samples, 0);

    metrics::Timer total_timer;
    total_timer.start();

    for (int iter = 0; iter < max_iters_; ++iter) {
        double assign_ms = 0.0;
        double update_ms = 0.0;

        auto labels = assign_clusters(dataset, result.centroids, assign_ms);
        auto new_centroids = update_centroids(dataset, labels, update_ms);

        const float shift = compute_shift(result.centroids, new_centroids);
        result.centroids = std::move(new_centroids);
        result.labels = std::move(labels);
        result.iterations = iter + 1;

        result.timings.assign_ms += assign_ms;
        result.timings.update_ms += update_ms;

        if (shift <= tol_) {
            logger_.info("Converged after " + std::to_string(result.iterations) + " iterations");
            break;
        }
    }

    result.timings.total_ms = total_timer.stop();
    return result;
}

std::vector<float> KMeansBase::initialize_centroids(const data::Dataset &dataset) const {
    std::vector<float> centroids(static_cast<size_t>(n_clusters_) * dataset.n_features);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, dataset.n_samples - 1);

    for (int k = 0; k < n_clusters_; ++k) {
        int idx = dist(rng);
        const auto offset = static_cast<size_t>(idx) * dataset.n_features;
        for (int d = 0; d < dataset.n_features; ++d) {
            centroids[static_cast<size_t>(k) * dataset.n_features + d] = dataset.values[offset + d];
        }
    }

    return centroids;
}

float KMeansBase::compute_shift(const std::vector<float> &prev, const std::vector<float> &next) const {
    float max_shift = 0.0f;
    for (size_t i = 0; i < prev.size(); ++i) {
        float diff = prev[i] - next[i];
        max_shift = std::max(max_shift, diff * diff);
    }
    return std::sqrt(max_shift);
}

}  // namespace kmeans::core
