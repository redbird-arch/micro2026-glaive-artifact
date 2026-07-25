/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#pragma once

#include <tacos/baselines/baseline_solver.h>

namespace tacos {

/// @brief Spreadout algorithm for all-to-all communication
/// Implements the linear-step Spreadout algorithm with N-1 steps
class Spreadout : public BaselineSolver {
  public:
    /// @brief Constructor
    /// @param topology network topology
    /// @param shape shape vector for mesh/torus (for XY routing)
    explicit Spreadout(std::shared_ptr<Topology> topology, 
                      const std::vector<int>& shape = {}) noexcept;

    /// @brief Run Spreadout algorithm
    /// @param dataMatrix data size matrix (dataMatrix[i][j] = bytes from node i to node j)
    /// @return result containing total time and step information
    Result solve(const std::vector<std::vector<long long>>& dataMatrix) override;
};

}  // namespace tacos

