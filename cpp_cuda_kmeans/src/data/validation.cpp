#include "data/validation.h"

namespace kmeans::data {

bool validate_dataset(const Dataset &dataset) {
    if (dataset.n_samples <= 0 || dataset.n_features <= 0 || dataset.n_clusters <= 0) {
        return false;
    }
    return dataset.values.size() == static_cast<size_t>(dataset.n_samples) * dataset.n_features;
}

}  // namespace kmeans::data
