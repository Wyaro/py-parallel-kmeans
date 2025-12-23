#pragma once

#include "core/kmeans_base.h"

namespace kmeans::core {

class KMeansGPU : public KMeansBase {
public:
    KMeansGPU(int n_clusters, int max_iters, float tol, kmeans::utils::Logger logger, int block_size = 256);

protected:
    std::string name() const override;
    std::vector<int> assign_clusters(const data::Dataset &dataset, const std::vector<float> &centroids,
                                     double &elapsed_ms) override;
    std::vector<float> update_centroids(const data::Dataset &dataset, const std::vector<int> &labels,
                                        double &elapsed_ms) override;

private:
    int block_size_;
};

}  // namespace kmeans::core
