/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#pragma once

#include <memory>
#include <vector>
#include <tacos/topology/topology.h>
#include <tacos/event_queue/event_queue.h>
#include <tacos/synthesizer/time_expanded_network.h>

namespace tacos {

/// @brief Base class for baseline algorithms (Bruck, Spreadout)
class BaselineSolver {
  public:
    using Time = Topology::Time;
    using NpuID = Topology::NpuID;
    using Bandwidth = Topology::Bandwidth;
    using Latency = Topology::Latency;
    using ChunkSize = long long;

    /// @brief Result structure for baseline algorithm execution
    struct Result {
        Time totalTime;  // Total makespan in microseconds
        std::vector<std::vector<std::pair<NpuID, NpuID>>> stepTransfers;  // Transfers per step
        std::vector<Time> stepTimes;  // Time for each step
        double averageUtilization = 0.0;  // Average link utilization over all links (0..1)
    };

    /// @brief Constructor
    /// @param topology network topology
    /// @param shape shape vector for mesh/torus (for XY routing)
    explicit BaselineSolver(std::shared_ptr<Topology> topology, 
                           const std::vector<int>& shape = {}) noexcept;

    /// @brief Run the baseline algorithm
    /// @param dataMatrix data size matrix (dataMatrix[i][j] = bytes from node i to node j)
    /// @return result containing total time and step information
    virtual Result solve(const std::vector<std::vector<long long>>& dataMatrix) = 0;

  protected:
    /// @brief Get XY routing path between two nodes (or graph shortest path for switch topologies)
    /// @param src source node ID
    /// @param dest destination node ID
    /// @return vector of node IDs representing the path (including src and dest)
    std::vector<NpuID> getXYPath(NpuID src, NpuID dest) const noexcept;

    /// @brief Get shortest path in topology graph (BFS); used for switch topologies
    std::vector<NpuID> getGraphPath(NpuID src, NpuID dest) const noexcept;

    /// @brief Convert NpuID to coordinate
    /// @param npuID node ID
    /// @return coordinate vector
    std::vector<int> npuIDToCoordinate(NpuID npuID) const noexcept;

    /// @brief Convert coordinate to NpuID
    /// @param coord coordinate vector
    /// @return node ID
    NpuID coordinateToNpuID(const std::vector<int>& coord) const noexcept;

    /// @brief Calculate transfer time for a data size over a link (single hop: latency + data/bw)
    Time calculateTransferTime(NpuID src, NpuID dest, long long dataSize) const noexcept;

    /// @brief Cut-through path timing: total = sum(hop latencies) + dataSize/minBw (bandwidth counted once)
    struct PathTiming {
        std::vector<Time> hopLatenciesUs;
        Time dataTimeUs;
        Time totalTimeUs;
    };
    PathTiming getPathTimingCutThrough(const std::vector<NpuID>& path, long long dataSize) const noexcept;

    std::shared_ptr<Topology> topology_;
    std::vector<int> shape_;
    int npusCount_;
};

}  // namespace tacos

