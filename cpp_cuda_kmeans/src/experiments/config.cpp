#include "experiments/config.h"

#include <stdexcept>

namespace kmeans::experiments {

ExperimentId parse_experiment_id(const std::string &value) {
    if (value == "exp2_scaling_n") {
        return ExperimentId::kExp2ScalingN;
    }
    if (value == "exp3_scaling_d") {
        return ExperimentId::kExp3ScalingD;
    }
    if (value == "exp4_scaling_k") {
        return ExperimentId::kExp4ScalingK;
    }
    if (value == "all") {
        return ExperimentId::kAll;
    }
    throw std::runtime_error("Unknown experiment id: " + value);
}

std::string to_string(ExperimentId id) {
    switch (id) {
        case ExperimentId::kExp2ScalingN:
            return "exp2_scaling_n";
        case ExperimentId::kExp3ScalingD:
            return "exp3_scaling_d";
        case ExperimentId::kExp4ScalingK:
            return "exp4_scaling_k";
        case ExperimentId::kAll:
            return "all";
    }
    return "unknown";
}

const std::unordered_map<ExperimentId, ExperimentConfig> &experiment_configs() {
    static const std::unordered_map<ExperimentId, ExperimentConfig> configs = {
        {ExperimentId::kExp2ScalingN, {ExperimentId::kExp2ScalingN, "Scaling by N", {20000, 50000, 100000}, 5}},
        {ExperimentId::kExp3ScalingD, {ExperimentId::kExp3ScalingD, "Scaling by D", {8, 16, 32, 64}, 5}},
        {ExperimentId::kExp4ScalingK, {ExperimentId::kExp4ScalingK, "Scaling by K", {4, 8, 16, 32}, 5}},
    };
    return configs;
}

}  // namespace kmeans::experiments
