#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace kmeans::experiments {

enum class ExperimentId {
    kExp2ScalingN,
    kExp3ScalingD,
    kExp4ScalingK,
    kAll
};

struct ExperimentConfig {
    ExperimentId id;
    std::string description;
    std::vector<int> values;
    int repeats;
};

ExperimentId parse_experiment_id(const std::string &value);
std::string to_string(ExperimentId id);
const std::unordered_map<ExperimentId, ExperimentConfig> &experiment_configs();

}  // namespace kmeans::experiments
