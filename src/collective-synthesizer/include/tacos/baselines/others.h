/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#pragma once

#include <tacos/baselines/baseline_solver.h>

namespace tacos {

/// @brief HalfRing+DimRotation algorithm for all-to-all on Torus networks
/// Event-driven half-ring routing with dimension rotation across torus dimensions
class HalfRingDimRotation : public BaselineSolver {
  public:
    /// @brief Constructor
    /// @param topology network topology (must be Torus)
    /// @param shape shape vector for torus
    explicit HalfRingDimRotation(std::shared_ptr<Topology> topology, 
                                 const std::vector<int>& shape = {}) noexcept;

    /// @brief Run HalfRing+DimRotation algorithm
    /// @param dataMatrix data size matrix (dataMatrix[i][j] = bytes from node i to node j)
    /// @return result containing total time and step information
    Result solve(const std::vector<std::vector<long long>>& dataMatrix) override;
};

/// @brief FoldedRing+DimRotation algorithm for all-to-all on Mesh networks
/// Formula-based calculation without event-driven simulation
class FoldedRingDimRotation : public BaselineSolver {
  public:
    /// @brief Constructor
    /// @param topology network topology (must be Mesh)
    /// @param shape shape vector for mesh
    explicit FoldedRingDimRotation(std::shared_ptr<Topology> topology, 
                                  const std::vector<int>& shape = {}) noexcept;

    /// @brief Run FoldedRing+DimRotation algorithm
    /// @param dataMatrix data size matrix (dataMatrix[i][j] = bytes from node i to node j)
    /// @return result containing total time and step information
    Result solve(const std::vector<std::vector<long long>>& dataMatrix) override;
};

}  // namespace tacos
