#pragma once

#include <string>
#include <vector>

#include "data/dataset.h"
#include "metrics/timers.h"
#include "utils/logging.h"

namespace kmeans::core {

struct FitTimings {
    double total_ms = 0.0;
    double assign_ms = 0.0;
    double update_ms = 0.0;
};

struct FitResult {
    std::vector<float> centroids;
    std::vector<int> labels;
    int iterations = 0;
    FitTimings timings;
};

class KMeansBase {
public:
    KMeansBase(int n_clusters, int max_iters, float tol, kmeans::utils::Logger logger);
    virtual ~KMeansBase() = default;

    FitResult fit(const data::Dataset &dataset);

protected:
    virtual std::string name() const = 0;
    virtual std::vector<int> assign_clusters(const data::Dataset &dataset, const std::vector<float> &centroids,
                                             double &elapsed_ms) = 0;
    virtual std::vector<float> update_centroids(const data::Dataset &dataset, const std::vector<int> &labels,
                                                double &elapsed_ms) = 0;

    int n_clusters_;
    int max_iters_;
    float tol_;
    kmeans::utils::Logger logger_;

private:
    std::vector<float> initialize_centroids(const data::Dataset &dataset) const;
    float compute_shift(const std::vector<float> &prev, const std::vector<float> &next) const;
};

}  // namespace kmeans::core
