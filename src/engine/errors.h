#pragma once

#include <stdexcept>
#include <string>

namespace bsolver {

// Invalid job specification (HTTP 400, CLI exit 1).
class JobError : public std::runtime_error {
 public:
  explicit JobError(const std::string& msg) : std::runtime_error(msg) {}
};

// Operation conflicts with the manager state (HTTP 409).
class ConflictError : public std::runtime_error {
 public:
  explicit ConflictError(const std::string& msg) : std::runtime_error(msg) {}
};

}  // namespace bsolver
