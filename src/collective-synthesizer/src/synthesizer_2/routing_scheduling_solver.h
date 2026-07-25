/*
# File name  :    routing_scheduling_solver.h
# Author     :    Galois
# Time       :    2025/01/05
# Description:    Two-phase solver for non-uniform AllToAll on Mesh/Torus
#                 Phase 1: Routing (shortest path / dimension-order routing)
#                 Phase 2: Scheduling (edge-based weighted scheduling)
*/

#pragma once

#include <vector>
#include <queue>
#include <unordered_map>
#include <memory>
#include <limits>
#include <algorithm>
#include <iostream>
#include <tacos/topology/topology.h>

namespace tacos {

// Forward declarations
class DemandMatrix;
class RoutingResult;
class SchedulingResult;

//=============================================================================
// DemandMatrix: Represents the non-uniform AllToAll demand D[i][j]
//=============================================================================
class DemandMatrix {
public:
    using DataSize = long long;

    DemandMatrix(int npusCount) : npusCount_(npusCount) {
        demand_.resize(npusCount, std::vector<DataSize>(npusCount, 0));
    }

    // Set demand from node i to node j
    void setDemand(int src, int dst, DataSize size) {
        demand_[src][dst] = size;
    }

    // Get demand from node i to node j
    DataSize getDemand(int src, int dst) const {
        return demand_[src][dst];
    }

    // Load from CSV-style matrix
    void loadFromMatrix(const std::vector<std::vector<DataSize>>& matrix) {
        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                demand_[i][j] = matrix[i][j];
            }
        }
    }

    int npusCount() const { return npusCount_; }

    // Get total demand (sum of all D[i][j])
    DataSize totalDemand() const {
        DataSize total = 0;
        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                total += demand_[i][j];
            }
        }
        return total;
    }

    // Get max row sum (max outgoing demand from any node)
    DataSize maxRowSum() const {
        DataSize maxSum = 0;
        for (int i = 0; i < npusCount_; ++i) {
            DataSize sum = 0;
            for (int j = 0; j < npusCount_; ++j) {
                sum += demand_[i][j];
            }
            maxSum = std::max(maxSum, sum);
        }
        return maxSum;
    }

    // Get max column sum (max incoming demand to any node)
    DataSize maxColSum() const {
        DataSize maxSum = 0;
        for (int j = 0; j < npusCount_; ++j) {
            DataSize sum = 0;
            for (int i = 0; i < npusCount_; ++i) {
                sum += demand_[i][j];
            }
            maxSum = std::max(maxSum, sum);
        }
        return maxSum;
    }

    void print() const {
        std::cout << "Demand Matrix (" << npusCount_ << " x " << npusCount_ << "):\n";
        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                std::cout << demand_[i][j] << "\t";
            }
            std::cout << "\n";
        }
    }

private:
    int npusCount_;
    std::vector<std::vector<DataSize>> demand_;
};

//=============================================================================
// EdgeLoad: Tracks load on each edge after routing
//=============================================================================
struct EdgeLoad {
    int src;
    int dst;
    long long load;  // Total bytes that need to traverse this edge

    // List of (commodity_src, commodity_dst, bytes) that use this edge
    std::vector<std::tuple<int, int, long long>> commodities;
};

//=============================================================================
// RoutingResult: Output of the routing phase
//=============================================================================
class RoutingResult {
public:
    // For each commodity (src, dst), store the path as a sequence of nodes
    // path[src][dst] = {src, hop1, hop2, ..., dst}
    std::vector<std::vector<std::vector<int>>> paths;

    // Edge loads after routing
    // edgeLoads[src][dst] = total load on edge (src, dst)
    std::vector<std::vector<long long>> edgeLoads;

    // Max edge load (bottleneck)
    long long maxEdgeLoad = 0;

    void init(int npusCount) {
        paths.resize(npusCount, std::vector<std::vector<int>>(npusCount));
        edgeLoads.resize(npusCount, std::vector<long long>(npusCount, 0));
    }

    void print(int npusCount) const {
        std::cout << "Edge Loads:\n";
        for (int i = 0; i < npusCount; ++i) {
            for (int j = 0; j < npusCount; ++j) {
                if (edgeLoads[i][j] > 0) {
                    std::cout << "  Edge (" << i << " -> " << j << "): "
                              << edgeLoads[i][j] << " bytes\n";
                }
            }
        }
        std::cout << "Max Edge Load: " << maxEdgeLoad << " bytes\n";
    }
};

//=============================================================================
// SchedulingResult: Output of the scheduling phase
//=============================================================================
class SchedulingResult {
public:
    // Total time to complete all transfers (in microseconds)
    double makespan = 0.0;

    // Schedule: list of (time, src, dst, commodity_src, commodity_dst, bytes)
    struct Transfer {
        double startTime;
        double endTime;
        int edgeSrc;
        int edgeDst;
        int commoditySrc;
        int commodityDst;
        long long bytes;
    };
    std::vector<Transfer> schedule;
};

//=============================================================================
// MeshTorusRouter: Routing algorithms for Mesh and Torus topologies
//=============================================================================
class MeshTorusRouter {
public:
    MeshTorusRouter(const std::vector<int>& shape, bool isTorus)
        : shape_(shape), isTorus_(isTorus) {

        dimensions_ = shape.size();
        npusCount_ = 1;
        for (int d : shape) {
            npusCount_ *= d;
        }

        // Compute strides for coordinate conversion
        strides_.resize(dimensions_);
        int stride = 1;
        for (int d = dimensions_ - 1; d >= 0; --d) {
            strides_[d] = stride;
            stride *= shape_[d];
        }
    }

    // Convert NpuID to coordinates
    std::vector<int> idToCoord(int id) const {
        std::vector<int> coord(dimensions_);
        for (int d = 0; d < dimensions_; ++d) {
            coord[d] = (id / strides_[d]) % shape_[d];
        }
        return coord;
    }

    // Convert coordinates to NpuID
    int coordToId(const std::vector<int>& coord) const {
        int id = 0;
        for (int d = 0; d < dimensions_; ++d) {
            id += coord[d] * strides_[d];
        }
        return id;
    }

    // Dimension-Order Routing (XY routing for 2D, generalized for nD)
    // Returns the path from src to dst as a sequence of node IDs
    std::vector<int> dimensionOrderRoute(int src, int dst) const {
        std::vector<int> path;
        path.push_back(src);

        if (src == dst) return path;

        std::vector<int> srcCoord = idToCoord(src);
        std::vector<int> dstCoord = idToCoord(dst);
        std::vector<int> currentCoord = srcCoord;

        // Route dimension by dimension
        for (int d = 0; d < dimensions_; ++d) {
            while (currentCoord[d] != dstCoord[d]) {
                int diff = dstCoord[d] - currentCoord[d];

                if (isTorus_) {
                    // For torus, choose shorter direction
                    int posDist = (dstCoord[d] - currentCoord[d] + shape_[d]) % shape_[d];
                    int negDist = (currentCoord[d] - dstCoord[d] + shape_[d]) % shape_[d];

                    if (posDist <= negDist) {
                        currentCoord[d] = (currentCoord[d] + 1) % shape_[d];
                    } else {
                        currentCoord[d] = (currentCoord[d] - 1 + shape_[d]) % shape_[d];
                    }
                } else {
                    // For mesh, just move towards destination
                    if (diff > 0) {
                        currentCoord[d]++;
                    } else {
                        currentCoord[d]--;
                    }
                }

                path.push_back(coordToId(currentCoord));
            }
        }

        return path;
    }

    // Compute shortest path distance (hop count)
    int shortestPathDistance(int src, int dst) const {
        std::vector<int> srcCoord = idToCoord(src);
        std::vector<int> dstCoord = idToCoord(dst);

        int totalDist = 0;
        for (int d = 0; d < dimensions_; ++d) {
            int diff = std::abs(dstCoord[d] - srcCoord[d]);
            if (isTorus_) {
                diff = std::min(diff, shape_[d] - diff);
            }
            totalDist += diff;
        }
        return totalDist;
    }

    int npusCount() const { return npusCount_; }
    int dimensions() const { return dimensions_; }
    const std::vector<int>& shape() const { return shape_; }
    bool isTorus() const { return isTorus_; }

private:
    std::vector<int> shape_;
    std::vector<int> strides_;
    int dimensions_;
    int npusCount_;
    bool isTorus_;
};

//=============================================================================
// RoutingSchedulingSolver: Main solver class
//=============================================================================
class RoutingSchedulingSolver {
public:
    using Bandwidth = double;  // GiB/sec
    using Latency = double;    // microseconds

    RoutingSchedulingSolver(std::shared_ptr<Topology> topology,
                            const std::vector<int>& shape,
                            bool isTorus,
                            Bandwidth bandwidth,
                            Latency latency)
        : topology_(topology),
          router_(shape, isTorus),
          bandwidth_(bandwidth),
          latency_(latency) {
        npusCount_ = topology->npusCount();
    }

    //=========================================================================
    // Phase 1: Routing
    // Given demand matrix D, compute paths and edge loads
    //=========================================================================
    RoutingResult computeRouting(const DemandMatrix& demand) {
        RoutingResult result;
        result.init(npusCount_);

        // For each commodity (src, dst), compute path and accumulate edge loads
        for (int src = 0; src < npusCount_; ++src) {
            for (int dst = 0; dst < npusCount_; ++dst) {
                if (src == dst) continue;

                long long demandSize = demand.getDemand(src, dst);
                if (demandSize == 0) continue;

                // Compute path using dimension-order routing
                std::vector<int> path = router_.dimensionOrderRoute(src, dst);
                result.paths[src][dst] = path;

                // Accumulate edge loads
                for (size_t i = 0; i + 1 < path.size(); ++i) {
                    int edgeSrc = path[i];
                    int edgeDst = path[i + 1];
                    result.edgeLoads[edgeSrc][edgeDst] += demandSize;
                }
            }
        }

        // Find max edge load
        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                result.maxEdgeLoad = std::max(result.maxEdgeLoad, result.edgeLoads[i][j]);
            }
        }

        return result;
    }

    //=========================================================================
    // Phase 2: Scheduling
    // Given routing result, compute the schedule and makespan
    //=========================================================================
    SchedulingResult computeScheduling(const DemandMatrix& demand,
                                       const RoutingResult& routing) {
        SchedulingResult result;

        // Simple model: each edge processes its load sequentially
        // Makespan = max over all edges of (edge_load / bandwidth + latency * num_hops)

        // More sophisticated: consider store-and-forward delays
        // For now, use a simple lower bound based on bottleneck edge

        // Use alpha-beta model matching original synthesizer:
        // - latency_ is in ns, convert to us by /1000
        // - bandwidth_ is in GB/s, convert to bytes/us: * (1 << 30) / 1e6
        double bandwidthBytesPerUs = bandwidth_ * (1 << 30) / 1e6;  // GB/s to bytes/us
        double latencyUs = latency_ / 1000.0;  // ns to us

        // Lower bound 1: Bottleneck edge
        // Time = max_edge(load / bandwidth)
        double bottleneckTime = 0.0;
        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                if (routing.edgeLoads[i][j] > 0) {
                    double edgeTime = routing.edgeLoads[i][j] / bandwidthBytesPerUs;
                    bottleneckTime = std::max(bottleneckTime, edgeTime);
                }
            }
        }

        // Lower bound 2: Total demand / total bandwidth
        // For mesh/torus, each node has limited number of links

        // Lower bound 3: Max row/column sum
        double maxRowColTime = std::max(demand.maxRowSum(), demand.maxColSum()) / bandwidthBytesPerUs;

        // The actual makespan is at least the maximum of these bounds
        // For store-and-forward, we need to add latency considerations

        // Simple store-and-forward model:
        // Each hop adds latency, and data must wait at intermediate nodes
        result.makespan = std::max(bottleneckTime, maxRowColTime);

        // Add latency for the longest path
        int maxHops = 0;
        for (int src = 0; src < npusCount_; ++src) {
            for (int dst = 0; dst < npusCount_; ++dst) {
                if (!routing.paths[src][dst].empty()) {
                    int hops = routing.paths[src][dst].size() - 1;
                    maxHops = std::max(maxHops, hops);
                }
            }
        }
        result.makespan += maxHops * latencyUs;

        return result;
    }

    //=========================================================================
    // Full solve: Routing + Scheduling
    //=========================================================================
    double solve(const DemandMatrix& demand, bool verbose = false) {
        if (verbose) {
            std::cout << "\n========== Routing-Scheduling Solver ==========\n";
            demand.print();
        }

        // Phase 1: Routing
        RoutingResult routing = computeRouting(demand);
        if (verbose) {
            std::cout << "\n--- Phase 1: Routing ---\n";
            routing.print(npusCount_);
        }

        // Phase 2: Scheduling
        SchedulingResult scheduling = computeScheduling(demand, routing);
        if (verbose) {
            std::cout << "\n--- Phase 2: Scheduling ---\n";
            std::cout << "Estimated Makespan: " << scheduling.makespan << " us\n";
        }

        return scheduling.makespan;
    }

    //=========================================================================
    // Analysis: Compute theoretical lower bounds
    //=========================================================================
    void analyzeLowerBounds(const DemandMatrix& demand) {
        std::cout << "\n========== Lower Bound Analysis ==========\n";

        // Use alpha-beta model matching original synthesizer:
        // - latency_ is in ns, convert to us by /1000
        // - bandwidth_ is in GB/s, convert to bytes/us: * (1 << 30) / 1e6
        double bandwidthBytesPerUs = bandwidth_ * (1 << 30) / 1e6;  // GB/s to bytes/us

        // LB1: Bottleneck edge after routing
        RoutingResult routing = computeRouting(demand);
        double lb_bottleneck = routing.maxEdgeLoad / bandwidthBytesPerUs;
        std::cout << "LB1 (Bottleneck Edge): " << lb_bottleneck << " us\n";
        std::cout << "     Max Edge Load: " << routing.maxEdgeLoad << " bytes\n";

        // LB2: Max row sum (node send constraint)
        double lb_send = demand.maxRowSum() / bandwidthBytesPerUs;
        std::cout << "LB2 (Max Send): " << lb_send << " us\n";
        std::cout << "     Max Row Sum: " << demand.maxRowSum() << " bytes\n";

        // LB3: Max column sum (node receive constraint)
        double lb_recv = demand.maxColSum() / bandwidthBytesPerUs;
        std::cout << "LB3 (Max Recv): " << lb_recv << " us\n";
        std::cout << "     Max Col Sum: " << demand.maxColSum() << " bytes\n";

        // Overall lower bound
        double lb = std::max({lb_bottleneck, lb_send, lb_recv});
        std::cout << "Overall Lower Bound: " << lb << " us\n";
    }

private:
    std::shared_ptr<Topology> topology_;
    MeshTorusRouter router_;
    int npusCount_;
    Bandwidth bandwidth_;
    Latency latency_;
};

} // namespace tacos
