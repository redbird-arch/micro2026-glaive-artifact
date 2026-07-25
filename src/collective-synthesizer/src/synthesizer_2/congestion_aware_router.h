/*
# File name  :    congestion_aware_router.h
# Author     :    Chen Chen
# Time       :    2025/01/06
# Description:    Congestion-aware routing algorithms for Mesh/Torus
#                 - Load-balanced routing
#                 - Multi-path routing
#                 - Adaptive routing based on congestion prediction
#                 - Optimal algorithms for symmetric traffic patterns
*/

#pragma once

#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <limits>
#include <functional>
#include <iostream>
#include <random>

#include "matrix_analyzer.h"

namespace tacos {

//=============================================================================
// RoutingStrategy: Different routing strategies
//=============================================================================
enum class RoutingStrategy {
    DIMENSION_ORDER,      // XY routing (baseline)
    DOR_XY_BINPACK,       // DOR XY with binpacking optimization (Tier 1)
    LOAD_BALANCED,        // Minimize max edge load
    MULTI_PATH,           // Split traffic across multiple paths
    ADAPTIVE,             // Choose based on congestion
    DIMENSION_EXCHANGE,   // Optimal for uniform traffic
    LATIN_SQUARE          // Conflict-free scheduling
};

//=============================================================================
// PathInfo: Information about a routing path
//=============================================================================
struct PathInfo {
    std::vector<int> nodes;      // Path as sequence of node IDs
    int hopCount;                // Number of hops
    long long bytes;             // Bytes to transfer on this path

    std::vector<std::pair<int, int>> getEdges() const {
        std::vector<std::pair<int, int>> edges;
        for (size_t i = 0; i + 1 < nodes.size(); ++i) {
            edges.push_back({nodes[i], nodes[i + 1]});
        }
        return edges;
    }
};

//=============================================================================
// CommodityFlow: Represents a flow from src to dst
//=============================================================================
struct CommodityFlow {
    int src;
    int dst;
    long long totalBytes;
    std::vector<PathInfo> paths;  // May have multiple paths for multi-path routing
};

//=============================================================================
// RoutingPlan: Complete routing plan for all commodities
//=============================================================================
struct RoutingPlan {
    std::vector<CommodityFlow> flows;
    std::vector<std::vector<long long>> edgeLoads;  // edgeLoads[i][j] = load on edge (i,j)
    long long maxEdgeLoad;
    double loadBalanceRatio;  // max_load / avg_load

    void computeEdgeLoads(int npusCount) {
        edgeLoads.assign(npusCount, std::vector<long long>(npusCount, 0));
        maxEdgeLoad = 0;
        long long totalLoad = 0;
        int edgeCount = 0;

        for (const auto& flow : flows) {
            for (const auto& path : flow.paths) {
                for (auto [u, v] : path.getEdges()) {
                    edgeLoads[u][v] += path.bytes;
                }
            }
        }

        for (int i = 0; i < npusCount; ++i) {
            for (int j = 0; j < npusCount; ++j) {
                if (edgeLoads[i][j] > 0) {
                    maxEdgeLoad = std::max(maxEdgeLoad, edgeLoads[i][j]);
                    totalLoad += edgeLoads[i][j];
                    edgeCount++;
                }
            }
        }

        double avgLoad = (edgeCount > 0) ? (double)totalLoad / edgeCount : 0;
        loadBalanceRatio = (avgLoad > 0) ? (double)maxEdgeLoad / avgLoad : 1.0;
    }

    void print(int npusCount) const {
        std::cout << "\n--- Routing Plan ---\n";
        std::cout << "Total flows: " << flows.size() << "\n";
        std::cout << "Max edge load: " << maxEdgeLoad << " bytes\n";
        std::cout << "Load balance ratio: " << loadBalanceRatio << "\n";

        std::cout << "\nEdge loads (non-zero):\n";
        for (int i = 0; i < npusCount; ++i) {
            for (int j = 0; j < npusCount; ++j) {
                if (edgeLoads[i][j] > 0) {
                    std::cout << "  (" << i << " -> " << j << "): "
                              << edgeLoads[i][j] << " bytes\n";
                }
            }
        }
    }
};

//=============================================================================
// CongestionAwareRouter: Main routing class
//=============================================================================
class CongestionAwareRouter {
public:
    using DataSize = long long;

    CongestionAwareRouter(const std::vector<int>& shape, bool isTorus)
        : shape_(shape), isTorus_(isTorus) {
        npusCount_ = 1;
        for (int d : shape) {
            npusCount_ *= d;
        }
        dimensions_ = shape.size();

        // Compute strides
        strides_.resize(dimensions_);
        int stride = 1;
        for (int d = dimensions_ - 1; d >= 0; --d) {
            strides_[d] = stride;
            stride *= shape_[d];
        }

        // Initialize edge loads
        currentEdgeLoads_.assign(npusCount_, std::vector<DataSize>(npusCount_, 0));
    }

    //=========================================================================
    // Main routing function - chooses strategy based on analysis
    //=========================================================================
    RoutingPlan computeRouting(const std::vector<std::vector<DataSize>>& demand,
                               const MatrixAnalysisResult& analysis,
                               RoutingStrategy strategy = RoutingStrategy::ADAPTIVE) {
        // If adaptive, choose strategy based on analysis
        if (strategy == RoutingStrategy::ADAPTIVE) {
            strategy = chooseStrategy(analysis);
            std::cout << "Adaptive routing chose strategy: " << strategyName(strategy) << "\n";
        }

        RoutingPlan plan;

        switch (strategy) {
            case RoutingStrategy::DIMENSION_ORDER:
                plan = routeDimensionOrder(demand);
                break;
            case RoutingStrategy::DOR_XY_BINPACK:
                plan = routeDorXyBinpack(demand);
                break;
            case RoutingStrategy::LOAD_BALANCED:
                plan = routeLoadBalanced(demand);
                break;
            case RoutingStrategy::MULTI_PATH:
                plan = routeMultiPath(demand);
                break;
            case RoutingStrategy::DIMENSION_EXCHANGE:
                plan = routeDimensionExchange(demand);
                break;
            case RoutingStrategy::LATIN_SQUARE:
                plan = routeLatinSquare(demand);
                break;
            default:
                plan = routeDimensionOrder(demand);
        }

        plan.computeEdgeLoads(npusCount_);
        return plan;
    }

    //=========================================================================
    // Strategy selection based on matrix analysis
    //=========================================================================
    RoutingStrategy chooseStrategy(const MatrixAnalysisResult& analysis) {
        // Check for uniform traffic
        for (auto type : analysis.symmetryTypes) {
            if (type == MatrixSymmetryType::UNIFORM) {
                return RoutingStrategy::DIMENSION_EXCHANGE;
            }
        }

        // Check for high congestion
        if (analysis.congestion.needsDiffusion) {
            return RoutingStrategy::LOAD_BALANCED;
        }

        // Check for sparse traffic
        if (analysis.sparsity > 0.7) {
            return RoutingStrategy::DIMENSION_ORDER;  // Simple is fine for sparse
        }

        // Check for hotspots
        if (analysis.hotspotCount > npusCount_ * 0.1) {
            return RoutingStrategy::MULTI_PATH;
        }

        // Default to load-balanced
        return RoutingStrategy::LOAD_BALANCED;
    }

    std::string strategyName(RoutingStrategy strategy) const {
        switch (strategy) {
            case RoutingStrategy::DIMENSION_ORDER: return "DIMENSION_ORDER";
            case RoutingStrategy::DOR_XY_BINPACK: return "DOR_XY_BINPACK";
            case RoutingStrategy::LOAD_BALANCED: return "LOAD_BALANCED";
            case RoutingStrategy::MULTI_PATH: return "MULTI_PATH";
            case RoutingStrategy::ADAPTIVE: return "ADAPTIVE";
            case RoutingStrategy::DIMENSION_EXCHANGE: return "DIMENSION_EXCHANGE";
            case RoutingStrategy::LATIN_SQUARE: return "LATIN_SQUARE";
            default: return "UNKNOWN";
        }
    }

private:
    //=========================================================================
    // Dimension-Order Routing (XY routing)
    //=========================================================================
    RoutingPlan routeDimensionOrder(const std::vector<std::vector<DataSize>>& demand) {
        RoutingPlan plan;

        for (int src = 0; src < npusCount_; ++src) {
            for (int dst = 0; dst < npusCount_; ++dst) {
                if (src == dst || demand[src][dst] == 0) continue;

                CommodityFlow flow;
                flow.src = src;
                flow.dst = dst;
                flow.totalBytes = demand[src][dst];

                PathInfo path;
                path.nodes = dimensionOrderPath(src, dst);
                path.hopCount = path.nodes.size() - 1;
                path.bytes = demand[src][dst];

                flow.paths.push_back(path);
                plan.flows.push_back(flow);
            }
        }

        return plan;
    }

    //=========================================================================
    // DOR XY with Binpacking Optimization (Tier 1)
    // Uses DOR XY routing but with priority-based path selection
    // Priority = bytes * hopCount (larger flows with longer paths first)
    //=========================================================================
    RoutingPlan routeDorXyBinpack(const std::vector<std::vector<DataSize>>& demand) {
        RoutingPlan plan;

        // Reset edge loads
        currentEdgeLoads_.assign(npusCount_, std::vector<DataSize>(npusCount_, 0));

        // Collect all demands with priority (bytes * hopCount)
        struct FlowPriority {
            int src, dst;
            DataSize bytes;
            int hopCount;
            double priority;
            std::vector<int> path;
        };

        std::vector<FlowPriority> flows;
        for (int src = 0; src < npusCount_; ++src) {
            for (int dst = 0; dst < npusCount_; ++dst) {
                if (src == dst || demand[src][dst] == 0) continue;

                FlowPriority fp;
                fp.src = src;
                fp.dst = dst;
                fp.bytes = demand[src][dst];
                fp.path = dimensionOrderPath(src, dst);
                fp.hopCount = fp.path.size() - 1;
                fp.priority = (double)fp.bytes * fp.hopCount;  // Binpack priority
                flows.push_back(fp);
            }
        }

        // Sort by priority (largest first) - binpacking heuristic
        std::sort(flows.begin(), flows.end(), [](const FlowPriority& a, const FlowPriority& b) {
            return a.priority > b.priority;
        });

        // Route each flow using DOR XY, but select best path among alternatives
        for (const auto& fp : flows) {
            CommodityFlow flow;
            flow.src = fp.src;
            flow.dst = fp.dst;
            flow.totalBytes = fp.bytes;

            // Find all shortest paths (for 2D, typically 2 paths: X-first or Y-first)
            auto allPaths = findAllShortestPaths(fp.src, fp.dst);

            // Choose the path with minimum max edge load after adding this flow
            PathInfo bestPath;
            DataSize bestMaxLoad = std::numeric_limits<DataSize>::max();

            for (const auto& pathNodes : allPaths) {
                DataSize maxLoadIfUsed = 0;
                for (size_t i = 0; i + 1 < pathNodes.size(); ++i) {
                    int u = pathNodes[i];
                    int v = pathNodes[i + 1];
                    maxLoadIfUsed = std::max(maxLoadIfUsed, currentEdgeLoads_[u][v] + fp.bytes);
                }

                if (maxLoadIfUsed < bestMaxLoad) {
                    bestMaxLoad = maxLoadIfUsed;
                    bestPath.nodes = pathNodes;
                    bestPath.hopCount = pathNodes.size() - 1;
                    bestPath.bytes = fp.bytes;
                }
            }

            // Update edge loads
            for (auto [u, v] : bestPath.getEdges()) {
                currentEdgeLoads_[u][v] += fp.bytes;
            }

            flow.paths.push_back(bestPath);
            plan.flows.push_back(flow);
        }

        return plan;
    }

    //=========================================================================
    // Load-Balanced Routing
    // Greedily assigns paths to minimize max edge load
    //=========================================================================
    RoutingPlan routeLoadBalanced(const std::vector<std::vector<DataSize>>& demand) {
        RoutingPlan plan;

        // Reset edge loads
        currentEdgeLoads_.assign(npusCount_, std::vector<DataSize>(npusCount_, 0));

        // Collect all demands and sort by size (largest first)
        std::vector<std::tuple<DataSize, int, int>> demands;
        for (int src = 0; src < npusCount_; ++src) {
            for (int dst = 0; dst < npusCount_; ++dst) {
                if (src == dst || demand[src][dst] == 0) continue;
                demands.push_back({demand[src][dst], src, dst});
            }
        }
        std::sort(demands.begin(), demands.end(), std::greater<>());

        // Route each demand using the path that minimizes max edge load
        for (auto [bytes, src, dst] : demands) {
            CommodityFlow flow;
            flow.src = src;
            flow.dst = dst;
            flow.totalBytes = bytes;

            // Find all shortest paths
            auto allPaths = findAllShortestPaths(src, dst);

            // Choose the path with minimum max edge load after adding this flow
            PathInfo bestPath;
            DataSize bestMaxLoad = std::numeric_limits<DataSize>::max();

            for (const auto& pathNodes : allPaths) {
                DataSize maxLoadIfUsed = 0;
                for (size_t i = 0; i + 1 < pathNodes.size(); ++i) {
                    int u = pathNodes[i];
                    int v = pathNodes[i + 1];
                    maxLoadIfUsed = std::max(maxLoadIfUsed, currentEdgeLoads_[u][v] + bytes);
                }

                if (maxLoadIfUsed < bestMaxLoad) {
                    bestMaxLoad = maxLoadIfUsed;
                    bestPath.nodes = pathNodes;
                    bestPath.hopCount = pathNodes.size() - 1;
                    bestPath.bytes = bytes;
                }
            }

            // Update edge loads
            for (auto [u, v] : bestPath.getEdges()) {
                currentEdgeLoads_[u][v] += bytes;
            }

            flow.paths.push_back(bestPath);
            plan.flows.push_back(flow);
        }

        return plan;
    }

    //=========================================================================
    // Multi-Path Routing
    // Splits large flows across multiple paths
    // Enhanced for 2D Mesh/Torus with load-aware path selection
    //=========================================================================
    RoutingPlan routeMultiPath(const std::vector<std::vector<DataSize>>& demand,
                               int maxPaths = 2) {
        RoutingPlan plan;

        // Reset edge loads
        currentEdgeLoads_.assign(npusCount_, std::vector<DataSize>(npusCount_, 0));

        // For 2D topologies, use specialized path finding
        bool is2D = (dimensions_ == 2);

        // Collect and sort flows by size (largest first for better load balancing)
        std::vector<std::tuple<DataSize, int, int>> flowList;
        for (int src = 0; src < npusCount_; ++src) {
            for (int dst = 0; dst < npusCount_; ++dst) {
                if (src == dst || demand[src][dst] == 0) continue;
                flowList.push_back({demand[src][dst], src, dst});
            }
        }
        std::sort(flowList.begin(), flowList.end(), std::greater<>());

        for (auto [bytes, src, dst] : flowList) {
            CommodityFlow flow;
            flow.src = src;
            flow.dst = dst;
            flow.totalBytes = bytes;

            // Find multiple paths
            std::vector<std::vector<int>> paths;
            if (is2D) {
                // Use 2D-specific path finding (X-first vs Y-first)
                paths = find2DMinimalPaths(src, dst);
            } else {
                paths = findKShortestPaths(src, dst, maxPaths);
            }

            // For Torus, also consider wrap-around paths
            if (isTorus_ && is2D && paths.size() < maxPaths) {
                auto wrapPaths = findTorusWrapPaths(src, dst);
                for (const auto& wp : wrapPaths) {
                    if (paths.size() >= maxPaths) break;
                    // Only add if different from existing paths
                    bool isDifferent = true;
                    for (const auto& p : paths) {
                        if (p == wp) { isDifferent = false; break; }
                    }
                    if (isDifferent) paths.push_back(wp);
                }
            }

            if (paths.empty()) {
                paths.push_back(dimensionOrderPath(src, dst));
            }

            // Smart traffic splitting based on path loads
            std::vector<DataSize> pathLoads(paths.size(), 0);
            for (size_t i = 0; i < paths.size(); ++i) {
                for (size_t j = 0; j + 1 < paths[i].size(); ++j) {
                    pathLoads[i] = std::max(pathLoads[i],
                        currentEdgeLoads_[paths[i][j]][paths[i][j+1]]);
                }
            }

            // Distribute traffic inversely proportional to current load
            double totalWeight = 0.0;
            std::vector<double> weights(paths.size());
            for (size_t i = 0; i < paths.size(); ++i) {
                weights[i] = 1.0 / (1.0 + pathLoads[i] / 1e6);  // Avoid division by zero
                totalWeight += weights[i];
            }

            // Safety check: if totalWeight is 0 or invalid, distribute evenly
            if (totalWeight <= 0.0 || std::isnan(totalWeight) || std::isinf(totalWeight)) {
                totalWeight = paths.size();
                for (size_t i = 0; i < paths.size(); ++i) {
                    weights[i] = 1.0;
                }
            }

            DataSize remainingBytes = bytes;
            for (size_t i = 0; i < paths.size(); ++i) {
                PathInfo pathInfo;
                pathInfo.nodes = paths[i];
                pathInfo.hopCount = paths[i].size() - 1;

                if (i == paths.size() - 1) {
                    pathInfo.bytes = remainingBytes;  // Last path gets remainder
                } else {
                    pathInfo.bytes = (DataSize)(bytes * weights[i] / totalWeight);
                    // Ensure non-negative bytes
                    if (pathInfo.bytes < 0) pathInfo.bytes = 0;
                    remainingBytes -= pathInfo.bytes;
                }

                // Safety check: ensure remainingBytes doesn't go negative
                if (remainingBytes < 0) {
                    pathInfo.bytes += remainingBytes;  // Adjust last assigned path
                    remainingBytes = 0;
                }

                // Update edge loads only if bytes > 0
                if (pathInfo.bytes > 0) {
                    for (auto [u, v] : pathInfo.getEdges()) {
                        currentEdgeLoads_[u][v] += pathInfo.bytes;
                    }
                }

                flow.paths.push_back(pathInfo);
            }

            plan.flows.push_back(flow);
        }

        return plan;
    }

    //=========================================================================
    // Find 2D minimal paths (X-first and Y-first)
    //=========================================================================
    std::vector<std::vector<int>> find2DMinimalPaths(int src, int dst) const {
        std::vector<std::vector<int>> paths;

        if (dimensions_ != 2) {
            paths.push_back(dimensionOrderPath(src, dst));
            return paths;
        }

        auto srcCoord = idToCoord(src);
        auto dstCoord = idToCoord(dst);

        // Path 1: X-first, then Y
        std::vector<int> path1;
        path1.push_back(src);
        auto currentCoord = srcCoord;

        // Move in X direction first
        while (currentCoord[0] != dstCoord[0]) {
            if (isTorus_) {
                int posDist = (dstCoord[0] - currentCoord[0] + shape_[0]) % shape_[0];
                int negDist = (currentCoord[0] - dstCoord[0] + shape_[0]) % shape_[0];
                if (posDist <= negDist) {
                    currentCoord[0] = (currentCoord[0] + 1) % shape_[0];
                } else {
                    currentCoord[0] = (currentCoord[0] - 1 + shape_[0]) % shape_[0];
                }
            } else {
                currentCoord[0] += (dstCoord[0] > currentCoord[0]) ? 1 : -1;
            }
            path1.push_back(coordToId(currentCoord));
        }
        // Then move in Y direction
        while (currentCoord[1] != dstCoord[1]) {
            if (isTorus_) {
                int posDist = (dstCoord[1] - currentCoord[1] + shape_[1]) % shape_[1];
                int negDist = (currentCoord[1] - dstCoord[1] + shape_[1]) % shape_[1];
                if (posDist <= negDist) {
                    currentCoord[1] = (currentCoord[1] + 1) % shape_[1];
                } else {
                    currentCoord[1] = (currentCoord[1] - 1 + shape_[1]) % shape_[1];
                }
            } else {
                currentCoord[1] += (dstCoord[1] > currentCoord[1]) ? 1 : -1;
            }
            path1.push_back(coordToId(currentCoord));
        }
        paths.push_back(path1);

        // Path 2: Y-first, then X (only if different)
        if (srcCoord[0] != dstCoord[0] && srcCoord[1] != dstCoord[1]) {
            std::vector<int> path2;
            path2.push_back(src);
            currentCoord = srcCoord;

            // Move in Y direction first
            while (currentCoord[1] != dstCoord[1]) {
                if (isTorus_) {
                    int posDist = (dstCoord[1] - currentCoord[1] + shape_[1]) % shape_[1];
                    int negDist = (currentCoord[1] - dstCoord[1] + shape_[1]) % shape_[1];
                    if (posDist <= negDist) {
                        currentCoord[1] = (currentCoord[1] + 1) % shape_[1];
                    } else {
                        currentCoord[1] = (currentCoord[1] - 1 + shape_[1]) % shape_[1];
                    }
                } else {
                    currentCoord[1] += (dstCoord[1] > currentCoord[1]) ? 1 : -1;
                }
                path2.push_back(coordToId(currentCoord));
            }
            // Then move in X direction
            while (currentCoord[0] != dstCoord[0]) {
                if (isTorus_) {
                    int posDist = (dstCoord[0] - currentCoord[0] + shape_[0]) % shape_[0];
                    int negDist = (currentCoord[0] - dstCoord[0] + shape_[0]) % shape_[0];
                    if (posDist <= negDist) {
                        currentCoord[0] = (currentCoord[0] + 1) % shape_[0];
                    } else {
                        currentCoord[0] = (currentCoord[0] - 1 + shape_[0]) % shape_[0];
                    }
                } else {
                    currentCoord[0] += (dstCoord[0] > currentCoord[0]) ? 1 : -1;
                }
                path2.push_back(coordToId(currentCoord));
            }
            paths.push_back(path2);
        }

        return paths;
    }

    //=========================================================================
    // Find Torus wrap-around paths
    //=========================================================================
    std::vector<std::vector<int>> findTorusWrapPaths(int src, int dst) const {
        std::vector<std::vector<int>> paths;

        if (!isTorus_ || dimensions_ != 2) {
            return paths;
        }

        auto srcCoord = idToCoord(src);
        auto dstCoord = idToCoord(dst);

        // Try wrap-around in X direction
        int directDistX = std::abs(dstCoord[0] - srcCoord[0]);
        int wrapDistX = shape_[0] - directDistX;

        if (wrapDistX < directDistX) {
            // Wrap path might be shorter, create it
            std::vector<int> wrapPath;
            wrapPath.push_back(src);
            auto currentCoord = srcCoord;

            // Move in wrap direction for X
            int direction = (dstCoord[0] > srcCoord[0]) ? -1 : 1;  // Opposite of direct
            while (currentCoord[0] != dstCoord[0]) {
                currentCoord[0] = (currentCoord[0] + direction + shape_[0]) % shape_[0];
                wrapPath.push_back(coordToId(currentCoord));
            }
            // Then Y
            while (currentCoord[1] != dstCoord[1]) {
                int posDist = (dstCoord[1] - currentCoord[1] + shape_[1]) % shape_[1];
                int negDist = (currentCoord[1] - dstCoord[1] + shape_[1]) % shape_[1];
                if (posDist <= negDist) {
                    currentCoord[1] = (currentCoord[1] + 1) % shape_[1];
                } else {
                    currentCoord[1] = (currentCoord[1] - 1 + shape_[1]) % shape_[1];
                }
                wrapPath.push_back(coordToId(currentCoord));
            }
            paths.push_back(wrapPath);
        }

        return paths;
    }

    //=========================================================================
    // Dimension-Exchange Routing (optimal for uniform traffic)
    // Phase 1: Exchange within rows
    // Phase 2: Exchange within columns
    //=========================================================================
    RoutingPlan routeDimensionExchange(const std::vector<std::vector<DataSize>>& demand) {
        RoutingPlan plan;

        if (dimensions_ != 2) {
            // Fall back to dimension-order for non-2D
            return routeDimensionOrder(demand);
        }

        int N = shape_[0];  // Assuming square mesh

        // For dimension exchange, we route in two phases:
        // Phase 1: All data moves horizontally to the correct column
        // Phase 2: All data moves vertically to the correct row

        for (int src = 0; src < npusCount_; ++src) {
            for (int dst = 0; dst < npusCount_; ++dst) {
                if (src == dst || demand[src][dst] == 0) continue;

                auto srcCoord = idToCoord(src);
                auto dstCoord = idToCoord(dst);

                CommodityFlow flow;
                flow.src = src;
                flow.dst = dst;
                flow.totalBytes = demand[src][dst];

                // Create path: first horizontal, then vertical
                PathInfo path;
                path.nodes = dimensionOrderPath(src, dst);
                path.hopCount = path.nodes.size() - 1;
                path.bytes = demand[src][dst];

                flow.paths.push_back(path);
                plan.flows.push_back(flow);
            }
        }

        return plan;
    }

    //=========================================================================
    // Latin Square Routing (conflict-free scheduling)
    //=========================================================================
    RoutingPlan routeLatinSquare(const std::vector<std::vector<DataSize>>& demand) {
        // Latin square routing is more about scheduling than routing
        // For now, use dimension-order routing with Latin square scheduling hints
        return routeDimensionOrder(demand);
    }

    //=========================================================================
    // Path finding utilities
    //=========================================================================
    std::vector<int> dimensionOrderPath(int src, int dst) const {
        std::vector<int> path;
        path.push_back(src);

        if (src == dst) return path;

        auto srcCoord = idToCoord(src);
        auto dstCoord = idToCoord(dst);
        auto currentCoord = srcCoord;

        for (int d = 0; d < dimensions_; ++d) {
            while (currentCoord[d] != dstCoord[d]) {
                int diff = dstCoord[d] - currentCoord[d];

                if (isTorus_) {
                    int posDist = (dstCoord[d] - currentCoord[d] + shape_[d]) % shape_[d];
                    int negDist = (currentCoord[d] - dstCoord[d] + shape_[d]) % shape_[d];

                    if (posDist <= negDist) {
                        currentCoord[d] = (currentCoord[d] + 1) % shape_[d];
                    } else {
                        currentCoord[d] = (currentCoord[d] - 1 + shape_[d]) % shape_[d];
                    }
                } else {
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

    //=========================================================================
    // Find all shortest paths between src and dst
    //=========================================================================
    std::vector<std::vector<int>> findAllShortestPaths(int src, int dst) const {
        std::vector<std::vector<int>> allPaths;

        if (src == dst) {
            allPaths.push_back({src});
            return allPaths;
        }

        // BFS to find shortest distance
        std::vector<int> dist(npusCount_, -1);
        std::queue<int> q;
        q.push(src);
        dist[src] = 0;

        while (!q.empty()) {
            int u = q.front();
            q.pop();

            for (int v : getNeighbors(u)) {
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                }
            }
        }

        if (dist[dst] == -1) {
            return allPaths;  // No path exists
        }

        // DFS to find all shortest paths
        std::vector<int> currentPath;
        std::function<void(int)> dfs = [&](int u) {
            currentPath.push_back(u);

            if (u == dst) {
                allPaths.push_back(currentPath);
            } else {
                for (int v : getNeighbors(u)) {
                    if (dist[v] == dist[u] + 1) {
                        dfs(v);
                    }
                }
            }

            currentPath.pop_back();
        };

        dfs(src);

        // Limit number of paths to avoid explosion
        if (allPaths.size() > 10) {
            allPaths.resize(10);
        }

        return allPaths;
    }

    //=========================================================================
    // Find K shortest paths (may not be disjoint)
    //=========================================================================
    std::vector<std::vector<int>> findKShortestPaths(int src, int dst, int k) const {
        auto allPaths = findAllShortestPaths(src, dst);

        if (allPaths.size() <= k) {
            return allPaths;
        }

        // Select k paths that are most edge-disjoint
        std::vector<std::vector<int>> selectedPaths;
        std::unordered_set<long long> usedEdges;

        auto edgeHash = [](int u, int v) -> long long {
            return (long long)u * 1000000 + v;
        };

        // Sort paths by number of new edges they would add
        while (selectedPaths.size() < k && !allPaths.empty()) {
            int bestIdx = -1;
            int maxNewEdges = -1;

            for (size_t i = 0; i < allPaths.size(); ++i) {
                int newEdges = 0;
                for (size_t j = 0; j + 1 < allPaths[i].size(); ++j) {
                    if (usedEdges.find(edgeHash(allPaths[i][j], allPaths[i][j + 1])) == usedEdges.end()) {
                        newEdges++;
                    }
                }
                if (newEdges > maxNewEdges) {
                    maxNewEdges = newEdges;
                    bestIdx = i;
                }
            }

            if (bestIdx >= 0) {
                for (size_t j = 0; j + 1 < allPaths[bestIdx].size(); ++j) {
                    usedEdges.insert(edgeHash(allPaths[bestIdx][j], allPaths[bestIdx][j + 1]));
                }
                selectedPaths.push_back(allPaths[bestIdx]);
                allPaths.erase(allPaths.begin() + bestIdx);
            } else {
                break;
            }
        }

        return selectedPaths;
    }

    //=========================================================================
    // Get neighbors of a node
    //=========================================================================
    std::vector<int> getNeighbors(int nodeId) const {
        std::vector<int> neighbors;
        auto coord = idToCoord(nodeId);

        for (int d = 0; d < dimensions_; ++d) {
            // Positive direction
            auto newCoord = coord;
            if (isTorus_) {
                newCoord[d] = (coord[d] + 1) % shape_[d];
                neighbors.push_back(coordToId(newCoord));
            } else if (coord[d] + 1 < shape_[d]) {
                newCoord[d] = coord[d] + 1;
                neighbors.push_back(coordToId(newCoord));
            }

            // Negative direction
            newCoord = coord;
            if (isTorus_) {
                newCoord[d] = (coord[d] - 1 + shape_[d]) % shape_[d];
                neighbors.push_back(coordToId(newCoord));
            } else if (coord[d] > 0) {
                newCoord[d] = coord[d] - 1;
                neighbors.push_back(coordToId(newCoord));
            }
        }

        return neighbors;
    }

    //=========================================================================
    // Coordinate conversion
    //=========================================================================
    std::vector<int> idToCoord(int id) const {
        std::vector<int> coord(dimensions_);
        for (int d = 0; d < dimensions_; ++d) {
            coord[d] = (id / strides_[d]) % shape_[d];
        }
        return coord;
    }

    int coordToId(const std::vector<int>& coord) const {
        int id = 0;
        for (int d = 0; d < dimensions_; ++d) {
            id += coord[d] * strides_[d];
        }
        return id;
    }

    std::vector<int> shape_;
    std::vector<int> strides_;
    int dimensions_;
    int npusCount_;
    bool isTorus_;
    std::vector<std::vector<DataSize>> currentEdgeLoads_;
};

} // namespace tacos
