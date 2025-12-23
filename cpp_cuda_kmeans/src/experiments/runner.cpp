#include "experiments/runner.h"

namespace kmeans::experiments {

ImplementationResult run_single(core::KMeansBase &model, const data::Dataset &dataset, int repeats) {
    TimingSummary summary;
    int total_iterations = 0;

    for (int i = 0; i < repeats; ++i) {
        auto result = model.fit(dataset);
        summary.total_ms += result.timings.total_ms;
        summary.assign_ms += result.timings.assign_ms;
        summary.update_ms += result.timings.update_ms;
        total_iterations += result.iterations;
    }

    summary.total_ms /= repeats;
    summary.assign_ms /= repeats;
    summary.update_ms /= repeats;

    ImplementationResult output;
    output.implementation = "";
    output.timing = summary;
    output.iterations = repeats > 0 ? total_iterations / repeats : 0;
    return output;
}

}  // namespace kmeans::experiments
