/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#pragma once

#include <tacos/baselines/baseline_solver.h>

namespace tacos {

/// @brief Bidirectional ring baseline for all-to-all communication.
/// Messages are assigned to clockwise or counter-clockwise directions on a
/// logical ring and are forwarded one logical hop per round.
class BiRing : public BaselineSolver {
  public:
    explicit BiRing(std::shared_ptr<Topology> topology,
                    const std::vector<int>& shape = {}) noexcept;

    Result solve(const std::vector<std::vector<long long>>& dataMatrix) override;
};

}  // namespace tacos
