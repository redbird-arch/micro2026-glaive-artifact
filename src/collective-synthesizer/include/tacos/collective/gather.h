/*
# File name  :    gather.h
# Author     :    Galois
# Time       :    2025/11/20 20:22:43
*/

#pragma once

#include <memory>
#include <tacos/collective/collective.h>
#include <tacos/topology/topology.h>

namespace tacos {

/// @brief Gather collective communication pattern
class Gather final : public Collective {
  public:
    /// @brief Constructor for Gather collective
    /// @param npusCount number of NPUs in the topology
    /// @param collectivesCount number of initial chunks per each NPU
    explicit Gather(int npusCount, int collectivesCount = 1, int root = 0) noexcept;
};
}  // namespace tacos