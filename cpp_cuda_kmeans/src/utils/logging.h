#pragma once

#include <string>

namespace kmeans::utils {

enum class LogLevel {
    kInfo,
    kWarning,
    kError
};

class Logger {
public:
    explicit Logger(std::string name);

    void log(LogLevel level, const std::string &message) const;
    void info(const std::string &message) const;
    void warn(const std::string &message) const;
    void error(const std::string &message) const;

private:
    std::string name_;
};

}  // namespace kmeans::utils
