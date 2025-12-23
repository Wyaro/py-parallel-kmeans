#pragma once

#include <string>

#include "experiments/config.h"
#include "utils/logging.h"

namespace kmeans::experiments {

class ExperimentSuite {
public:
    explicit ExperimentSuite(kmeans::utils::Logger logger);

    void run(ExperimentId id, const std::string &output_path);

private:
    void run_scaling_n(const std::string &output_path);
    void run_scaling_d(const std::string &output_path);
    void run_scaling_k(const std::string &output_path);

    kmeans::utils::Logger logger_;
};

}  // namespace kmeans::experiments
