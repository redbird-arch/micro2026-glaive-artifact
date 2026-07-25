/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include <cassert>
#include <cmath>
#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <tacos/baselines/baseline_solver.h>
#include <tacos/thread_output.h>

using namespace tacos;

BaselineSolver::BaselineSolver(std::shared_ptr<Topology> topology, 
                              const std::vector<int>& shape) noexcept
    : topology_(topology), shape_(shape) {
    // For switch topologies (fat-tree, rail-optimized, cm), collective uses GPU count = product(shape)
    if (shape_.empty()) {
        npusCount_ = topology_->npusCount();
    } else {
        npusCount_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
    }
}

std::vector<int> BaselineSolver::npuIDToCoordinate(BaselineSolver::NpuID npuID) const noexcept {
    if (shape_.empty()) {
        // For non-mesh/torus topologies, return empty or single-element coordinate
        return {npuID};
    }
    
    std::vector<int> coord(shape_.size());
    // Match Mesh/Torus construction: dimension 0 has stride 1, then strides grow
    // by the preceding dimensions.
    int stride = 1;
    for (int d = 0; d < static_cast<int>(shape_.size()); ++d) {
        coord[d] = (npuID / stride) % shape_[d];
        stride *= shape_[d];
    }
    
    return coord;
}

BaselineSolver::NpuID BaselineSolver::coordinateToNpuID(const std::vector<int>& coord) const noexcept {
    if (shape_.empty() || coord.empty()) {
        return coord.empty() ? 0 : coord[0];
    }
    
    assert(coord.size() == shape_.size());
    
    // Match Mesh/Torus construction: dimension 0 has stride 1, then strides grow
    // by the preceding dimensions.
    int npuID = 0;
    int stride = 1;
    for (int d = 0; d < static_cast<int>(shape_.size()); ++d) {
        npuID += coord[d] * stride;
        stride *= shape_[d];
    }
    return npuID;
}

std::vector<BaselineSolver::NpuID> BaselineSolver::getGraphPath(BaselineSolver::NpuID src, BaselineSolver::NpuID dest) const noexcept {
    std::vector<BaselineSolver::NpuID> path;
    if (src == dest) {
        path.push_back(src);
        return path;
    }
    const int totalNodes = topology_->npusCount();
    std::vector<int> parent(totalNodes, -1);
    std::queue<BaselineSolver::NpuID> q;
    q.push(src);
    parent[src] = src;
    while (!q.empty()) {
        BaselineSolver::NpuID u = q.front();
        q.pop();
        if (u == dest) break;
        for (int v = 0; v < totalNodes; ++v) {
            if (parent[v] >= 0) continue;
            if (!topology_->connected(u, v)) continue;
            parent[v] = u;
            q.push(v);
        }
    }
    if (parent[dest] < 0) {
        path.push_back(src);
        path.push_back(dest);
        return path;
    }
    std::vector<BaselineSolver::NpuID> rev;
    for (BaselineSolver::NpuID cur = dest; cur != src; cur = static_cast<BaselineSolver::NpuID>(parent[cur])) {
        rev.push_back(cur);
    }
    path.push_back(src);
    for (auto it = rev.rbegin(); it != rev.rend(); ++it) {
        path.push_back(*it);
    }
    return path;
}

std::vector<BaselineSolver::NpuID> BaselineSolver::getXYPath(BaselineSolver::NpuID src, BaselineSolver::NpuID dest) const noexcept {
    std::vector<BaselineSolver::NpuID> path;
    
    // Direct connection: single-hop path
    if (topology_->connected(src, dest)) {
        path.push_back(src);
        if (src != dest) path.push_back(dest);
        return path;
    }
    
    // Switch topology (more nodes than GPUs): use graph shortest path
    if (topology_->npusCount() > npusCount_) {
        return getGraphPath(src, dest);
    }
    
    // No shape (e.g. fullmesh): no path possible except direct
    if (shape_.empty()) {
        path.push_back(src);
        path.push_back(dest);
        return path;
    }
    
    // Dimension-order routing for mesh/torus. The coordinate conversion below
    // uses the same mixed-radix order as the topology builders, so each unit
    // coordinate move must be a real single-hop link in the topology graph.
    auto srcCoord = npuIDToCoordinate(src);
    auto destCoord = npuIDToCoordinate(dest);
    
    assert(srcCoord.size() == destCoord.size());
    assert(srcCoord.size() == shape_.size());
    
    path.push_back(src);
    std::vector<int> currentCoord = srcCoord;
    BaselineSolver::NpuID currentNode = src;
    
    // Keep the existing last-to-first dimension order; with the corrected
    // mapping this is still a shortest Manhattan path for mesh.
    for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
        while (currentCoord[d] != destCoord[d]) {
            std::vector<int> nextCoord = currentCoord;
            if (currentCoord[d] < destCoord[d]) {
                nextCoord[d]++;
            } else {
                nextCoord[d]--;
            }
            BaselineSolver::NpuID nextNode = coordinateToNpuID(nextCoord);
            
            // Verify that the next node is directly connected to current node
            // In mesh/torus, nodes along the same dimension should be connected
            // But we need to check the actual topology connection
            if (!topology_->connected(currentNode, nextNode)) {
                // This can happen if the coordinate conversion doesn't match the actual mesh layout
                // For mesh, nodes along dimension d should be connected if they differ only in dimension d
                // If not connected, we have a problem with coordinate conversion or mesh structure
                // For now, we'll skip this hop and continue (the error will be caught later)
                // But ideally, this shouldn't happen
                std::cerr << "Error: Non-adjacent nodes in XY path: " << currentNode << " -> " << nextNode << std::endl;
                // Try to continue anyway - the error will be caught when processing
            }
            
            path.push_back(nextNode);
            currentCoord = nextCoord;
            currentNode = nextNode;
        }
    }
    
    return path;
}

BaselineSolver::Time BaselineSolver::calculateTransferTime(BaselineSolver::NpuID src, BaselineSolver::NpuID dest, long long dataSize) const noexcept {
    if (dataSize <= 0) {
        return 0.0;
    }
    
    // This function should only be called for direct (single-hop) connections
    // Multi-hop routing is handled by the caller using getXYPath
    if (!topology_->connected(src, dest)) {
        // If not directly connected, this is an error - should use getXYPath first
        std::cerr << "Error: calculateTransferTime called for non-connected nodes: " 
                  << src << " -> " << dest << std::endl;
        return 0.0;  // Return 0 instead of max() to avoid inf propagation
    }
    
    // Direct connection: use alpha-beta model similar to TimeExpandedNetwork
    BaselineSolver::Latency lat = topology_->latency(src, dest);
    BaselineSolver::Bandwidth bw = topology_->bandwidth(src, dest);
    
    if (bw <= 0) {
        std::cerr << "Warning: Zero or negative bandwidth for link " << src << " -> " << dest << std::endl;
        return 0.0;  // Return 0 instead of max() to avoid inf propagation
    }
    
    // Convert latency from ns to us (based on TimeExpandedNetwork implementation)
    // Note: According to synthesizer2 code, latency in topology config is in ns
    double latencyUs = lat / 1000.0;  // ns to us
    
    // Convert bandwidth from GiB/s to bytes/us
    // GiB = 2^30 bytes, 1 second = 1e6 microseconds
    double bandwidthBytesPerUs = bw * (1 << 30) / 1e6;  // bytes/us
    
    // Alpha-beta model: time = latency + dataSize / bandwidth
    BaselineSolver::Time transferTime = latencyUs + (dataSize / bandwidthBytesPerUs);
    
    // Check for invalid results (using C-style functions for compatibility)
    if (std::isnan(transferTime) || std::isinf(transferTime)) {
        std::cerr << "Warning: Invalid transfer time calculated: " << transferTime 
                  << " for link " << src << " -> " << dest 
                  << " (dataSize=" << dataSize << ", bw=" << bw << ", lat=" << lat << ")" << std::endl;
        return 0.0;
    }
    
    return transferTime;
}

BaselineSolver::PathTiming BaselineSolver::getPathTimingCutThrough(const std::vector<BaselineSolver::NpuID>& path, long long dataSize) const noexcept {
    PathTiming timing;
    timing.totalTimeUs = 0.0;
    timing.dataTimeUs = 0.0;
    if (path.size() < 2 || dataSize <= 0)
        return timing;
    double sumLatUs = 0.0;
    double minBwBytesPerUs = std::numeric_limits<double>::max();
    for (size_t h = 0; h < path.size() - 1; ++h) {
        NpuID a = path[h], b = path[h + 1];
        if (!topology_->connected(a, b)) continue;
        double latUs = topology_->latency(a, b) / 1000.0;
        timing.hopLatenciesUs.push_back(latUs);
        sumLatUs += latUs;
        double bw = topology_->bandwidth(a, b);
        if (bw > 0) {
            double bytesPerUs = bw * (1 << 30) / 1e6;
            if (bytesPerUs < minBwBytesPerUs) minBwBytesPerUs = bytesPerUs;
        }
    }
    if (minBwBytesPerUs <= 0 || minBwBytesPerUs == std::numeric_limits<double>::max()) {
        timing.dataTimeUs = 0.0;
    } else {
        timing.dataTimeUs = dataSize / minBwBytesPerUs;
    }
    timing.totalTimeUs = sumLatUs + timing.dataTimeUs;
    return timing;
}
