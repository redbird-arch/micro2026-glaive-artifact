/*
# File name  :    all_to_all.h
# Author     :    Galois
# Time       :    2025/11/20 19:42:00
*/

#pragma once

#include <memory>
#include <tacos/collective/collective.h>
#include <tacos/topology/topology.h>

namespace tacos {

/// @brief All-to-All collective communication pattern
class AlltoAll final : public Collective {
  public:
    /// @brief Constructor for AllGather collective
    /// @param npusCount number of NPUs in the topology
    /// @param collectivesCount number of initial chunks per each NPU
    explicit AlltoAll(int npusCount, int collectivesCount = 1) noexcept;
};
}  // namespace tacos