#pragma once

#include <random>
#include <string>
#include <vector>

namespace kmeans::data {

struct Dataset {
    std::string name;
    int n_samples;
    int n_features;
    int n_clusters;
    std::vector<float> values;

    static Dataset random_gaussian(const std::string &name, int n_samples, int n_features, int n_clusters,
                                   unsigned int seed);
};

}  // namespace kmeans::data
