#include "data/registry.h"

#include <stdexcept>

namespace kmeans::data {

void DatasetRegistry::add(const Dataset &dataset) {
    datasets_[dataset.name] = dataset;
}

const Dataset &DatasetRegistry::get(const std::string &name) const {
    auto it = datasets_.find(name);
    if (it == datasets_.end()) {
        throw std::runtime_error("Dataset not found: " + name);
    }
    return it->second;
}

bool DatasetRegistry::contains(const std::string &name) const {
    return datasets_.find(name) != datasets_.end();
}

}  // namespace kmeans::data
