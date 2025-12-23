#include "data/dataset.h"

namespace kmeans::data {

Dataset Dataset::random_gaussian(const std::string &name, int n_samples, int n_features, int n_clusters,
                                 unsigned int seed) {
    Dataset dataset;
    dataset.name = name;
    dataset.n_samples = n_samples;
    dataset.n_features = n_features;
    dataset.n_clusters = n_clusters;
    dataset.values.resize(static_cast<size_t>(n_samples) * n_features);

    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (auto &value : dataset.values) {
        value = dist(rng);
    }

    return dataset;
}

}  // namespace kmeans::data
