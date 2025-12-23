#pragma once

#include "core/kmeans_base.h"

namespace kmeans::core {

class KMeansCPUOpenMP : public KMeansBase {
public:
    KMeansCPUOpenMP(int n_clusters, int max_iters, float tol, kmeans::utils::Logger logger, int threads);

protected:
    std::string name() const override;
    std::vector<int> assign_clusters(const data::Dataset &dataset, const std::vector<float> &centroids,
                                     double &elapsed_ms) override;
    std::vector<float> update_centroids(const data::Dataset &dataset, const std::vector<int> &labels,
                                        double &elapsed_ms) override;

private:
    int threads_;
};

}  // namespace kmeans::core
