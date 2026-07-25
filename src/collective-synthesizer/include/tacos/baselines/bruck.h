/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#pragma once

#include <tacos/baselines/baseline_solver.h>

namespace tacos {

/// @brief Bruck algorithm for all-to-all communication
/// Implements the logarithmic-step Bruck algorithm with log2(N) steps
class Bruck : public BaselineSolver {
  public:
    /// @brief Constructor
    /// @param topology network topology
    /// @param shape shape vector for mesh/torus (for XY routing)
    explicit Bruck(std::shared_ptr<Topology> topology, 
                  const std::vector<int>& shape = {}) noexcept;

    /// @brief Run Bruck algorithm
    /// @param dataMatrix data size matrix (dataMatrix[i][j] = bytes from node i to node j)
    /// @return result containing total time and step information
    Result solve(const std::vector<std::vector<long long>>& dataMatrix) override;

  private:
    /// @brief Get data blocks to transfer in step k for node i
    /// @param nodeID node ID
    /// @param step step number (0 to log2(N)-1)
    /// @param dataMatrix data size matrix
    /// @return vector of (destination, dataSize) pairs
    std::vector<std::pair<NpuID, long long>> getDataBlocksForStep(
        NpuID nodeID, int step, const std::vector<std::vector<long long>>& dataMatrix) const noexcept;
};

}  // namespace tacos

