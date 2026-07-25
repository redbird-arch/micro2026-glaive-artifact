/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#pragma once

#include <memory>
#include <tacos/collective/collective.h>
#include <tacos/topology/topology.h>

namespace tacos {

/// @brief Profiler to compute theoretical lower bounds for collective communication
class Profiler {
  public:
    using Time = Topology::Time;
    using NpuID = Topology::NpuID;
    using Bandwidth = Topology::Bandwidth;
    using ChunkSize = Collective::ChunkSize;

    /// @brief Compute theoretical lower bound for collective communication time
    /// @param topology Target network topology
    /// @param collective Target collective pattern
    /// @param chunkSize Size of each chunk (in bytes)
    /// @return theoretical lower bound time (in microseconds)
    [[nodiscard]] static Time computeLowerBound(
        const std::shared_ptr<Topology>& topology,
        const std::shared_ptr<Collective>& collective,
        ChunkSize chunkSize) noexcept;

  private:
    /// @brief Compute lower bound based on node degrees and communication requirements
    /// @param topology Target network topology
    /// @param collective Target collective pattern
    /// @param chunkSize Size of each chunk (in bytes)
    /// @return lower bound time based on node degree constraints
    [[nodiscard]] static Time computeDegreeBasedLowerBound(
        const std::shared_ptr<Topology>& topology,
        const std::shared_ptr<Collective>& collective,
        ChunkSize chunkSize) noexcept;

    /// @brief Compute lower bound based on bisection bandwidth
    /// @param topology Target network topology
    /// @param collective Target collective pattern
    /// @param chunkSize Size of each chunk (in bytes)
    /// @return lower bound time based on bisection bandwidth
    [[nodiscard]] static Time computeBisectionBasedLowerBound(
        const std::shared_ptr<Topology>& topology,
        const std::shared_ptr<Collective>& collective,
        ChunkSize chunkSize) noexcept;

    /// @brief Find a bisection cut starting from a specific node using BFS
    /// @param topology Target network topology
    /// @param startNode Starting node for BFS
    /// @return pair of (set of nodes in partition A, set of links crossing the cut)
    [[nodiscard]] static std::pair<std::vector<bool>, std::vector<std::pair<NpuID, NpuID>>>
    findBisectionCutFromNode(const std::shared_ptr<Topology>& topology, int startNode) noexcept;

    /// @brief Compute total data volume that needs to cross a bisection cut
    /// @param collective Target collective pattern
    /// @param partitionA Boolean vector indicating which nodes are in partition A
    /// @param chunkSize Size of each chunk (in bytes)
    /// @return total data volume (in bytes) that must cross the cut
    [[nodiscard]] static double computeCrossCutDataVolume(
        const std::shared_ptr<Collective>& collective,
        const std::vector<bool>& partitionA,
        ChunkSize chunkSize) noexcept;
};

}  // namespace tacos

