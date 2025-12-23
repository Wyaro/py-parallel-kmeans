#pragma once

#include <string>
#include <vector>

#include "core/kmeans_base.h"
#include "data/dataset.h"

namespace kmeans::experiments {

struct TimingSummary {
    double total_ms = 0.0;
    double assign_ms = 0.0;
    double update_ms = 0.0;
};

struct ImplementationResult {
    std::string implementation;
    TimingSummary timing;
    int iterations = 0;
};

ImplementationResult run_single(core::KMeansBase &model, const data::Dataset &dataset, int repeats);

}  // namespace kmeans::experiments
