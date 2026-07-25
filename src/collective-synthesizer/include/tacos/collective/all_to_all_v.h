/*
# File name  :    all_to_all_v.h
# Author     :    Galois
# Time       :    2025/12/25 20:28:57
*/

#pragma once

#include <memory>
#include <vector>
#include <tacos/collective/collective.h>
#include <tacos/topology/topology.h>

namespace tacos {

/// @brief All-to-All-V (variable) collective communication pattern
/// Supports variable data sizes for each source-destination pair
class AlltoAllV final : public Collective {
  public:
    /// @brief Constructor for AlltoAllV collective
    /// @param npusCount number of NPUs in the topology
    /// @param dataMatrix matrix where dataMatrix[i][j] = data size from node i to node j (in bytes)
    /// @param chunkSize chunk size determined by GCD of all data sizes
    explicit AlltoAllV(int npusCount, 
                       const std::vector<std::vector<long long>>& dataMatrix,
                       long long chunkSize) noexcept;
};

}  // namespace tacos

