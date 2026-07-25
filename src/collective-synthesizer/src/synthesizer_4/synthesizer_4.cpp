/*
# File name  :    synthesizer_4.cpp
# Author     :    Galois
# Time       :    2026/01/21
# Description:    Implementation of Synthesizer4 interface
*/

#include "synthesizer_4.h"
#include <algorithm>
#include <numeric>
#include <functional>
#include <unordered_set>
#include <queue>
#include <set>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace tacos {

Synthesizer4::Synthesizer4(const std::vector<int>& shape, bool isTorus,
                           Bandwidth bandwidth, Latency latency)
    : shape_(shape), isTorus_(isTorus),
      bandwidth_(bandwidth), latency_(latency), topology_(nullptr), totalNodeCount_(0) {
    
    npusCount_ = 1;
    for (int d : shape) {
        npusCount_ *= d;
    }
    gpuNodeCount_ = npusCount_;  // For direct-connect topologies, all nodes are GPUs

    // Default configuration
    verbose_ = true;
    printSchedule_ = false;
    cleanMode_ = false;
    
    // Build connection graph
    buildConnectionGraph();
}

Synthesizer4::Synthesizer4(std::shared_ptr<Topology> topology, const std::vector<int>& shape,
                           Bandwidth bandwidth, Latency latency)
    : shape_(shape), isTorus_(false),
      bandwidth_(bandwidth), latency_(latency), topology_(topology), totalNodeCount_(0) {
    
    // For switch-based topologies, npusCount_ is the number of GPU nodes only
    npusCount_ = 1;
    for (int d : shape) {
        npusCount_ *= d;
    }
    gpuNodeCount_ = npusCount_;  // GPU nodes are 0 to gpuNodeCount_ - 1
    // Total nodes in topology (including switches) is topology_->npusCount()

    // Default configuration
    verbose_ = true;
    printSchedule_ = false;
    cleanMode_ = false;
    
    // Build connection graph
    buildConnectionGraph();
}

void Synthesizer4::setVerbose(bool verbose) {
    verbose_ = verbose;
}

void Synthesizer4::setPrintSchedule(bool printSchedule) {
    printSchedule_ = printSchedule;
}

int Synthesizer4::getHopIndex(const TransferEvent4& event) const {
    for (size_t hop = 0; hop + 1 < event.path.size(); ++hop) {
        if (event.path[hop] == event.src && event.path[hop + 1] == event.dst) {
            return static_cast<int>(hop);
        }
    }
    return std::numeric_limits<int>::max();
}

void Synthesizer4::sortEventsForScheduling(std::vector<TransferEvent4>& events) const {
    std::stable_sort(events.begin(), events.end(),
                     [this](const TransferEvent4& a, const TransferEvent4& b) {
                         if (a.chunkId != b.chunkId) return a.chunkId < b.chunkId;

                         int hopA = getHopIndex(a);
                         int hopB = getHopIndex(b);
                         if (hopA != hopB) return hopA < hopB;

                         if (a.flowSrc != b.flowSrc) return a.flowSrc < b.flowSrc;
                         if (a.flowDst != b.flowDst) return a.flowDst < b.flowDst;
                         if (a.src != b.src) return a.src < b.src;
                         return a.dst < b.dst;
                     });
}

void Synthesizer4::validateScheduleOrThrow(
    const std::vector<TransferEvent4>& events,
    const std::string& scheduleName) const {
    if (events.empty()) {
        return;
    }

    constexpr double kTolerance = 1e-9;
    constexpr int kMaxReportedErrors = 8;
    int errorCount = 0;

    auto reportError = [&](const std::string& msg) {
        if (errorCount < kMaxReportedErrors) {
            std::cerr << '[' << scheduleName << "] " << msg << std::endl;
        }
        ++errorCount;
    };

    std::map<std::pair<int, int>, std::vector<const TransferEvent4*>> edgeEvents;
    std::map<int, std::vector<const TransferEvent4*>> chunkEvents;

    for (const auto& event : events) {
        edgeEvents[{event.src, event.dst}].push_back(&event);
        if (event.chunkId >= 0) {
            chunkEvents[event.chunkId].push_back(&event);
        }
    }

    for (auto& [edge, edgeSchedule] : edgeEvents) {
        std::sort(edgeSchedule.begin(), edgeSchedule.end(),
                  [](const TransferEvent4* a, const TransferEvent4* b) {
                      if (a->startTime != b->startTime) return a->startTime < b->startTime;
                      if (a->endTime != b->endTime) return a->endTime < b->endTime;
                      return a->chunkId < b->chunkId;
                  });

        for (size_t i = 1; i < edgeSchedule.size(); ++i) {
            const auto* prev = edgeSchedule[i - 1];
            const auto* curr = edgeSchedule[i];
            if (curr->startTime + kTolerance < prev->endTime) {
                std::ostringstream oss;
                oss << "Link overlap on " << formatNodeName(edge.first)
                    << "->" << formatNodeName(edge.second)
                    << ": chunk " << prev->chunkId << " ends at " << prev->endTime
                    << " us, but chunk " << curr->chunkId << " starts at " << curr->startTime << " us";
                reportError(oss.str());
            }
        }
    }

    for (auto& [chunkId, chunkSchedule] : chunkEvents) {
        std::sort(chunkSchedule.begin(), chunkSchedule.end(),
                  [this](const TransferEvent4* a, const TransferEvent4* b) {
                      int hopA = getHopIndex(*a);
                      int hopB = getHopIndex(*b);
                      if (hopA != hopB) return hopA < hopB;
                      if (a->startTime != b->startTime) return a->startTime < b->startTime;
                      if (a->endTime != b->endTime) return a->endTime < b->endTime;
                      return a->src < b->src;
                  });

        if (chunkSchedule.empty()) {
            continue;
        }

        const auto* first = chunkSchedule.front();
        const auto* last = chunkSchedule.back();
        if (first->src != first->flowSrc) {
            std::ostringstream oss;
            oss << "Chunk " << chunkId << " starts from " << formatNodeName(first->src)
                << " instead of source " << formatNodeName(first->flowSrc);
            reportError(oss.str());
        }
        if (last->dst != last->flowDst) {
            std::ostringstream oss;
            oss << "Chunk " << chunkId << " ends at " << formatNodeName(last->dst)
                << " instead of destination " << formatNodeName(last->flowDst);
            reportError(oss.str());
        }

        for (size_t i = 0; i + 1 < chunkSchedule.size(); ++i) {
            const auto* curr = chunkSchedule[i];
            const auto* next = chunkSchedule[i + 1];
            if (curr->dst != next->src) {
                std::ostringstream oss;
                oss << "Chunk " << chunkId << " has broken path continuity: "
                    << formatNodeName(curr->dst) << " -> " << formatNodeName(next->src);
                reportError(oss.str());
            }
            if (next->startTime + kTolerance < curr->endTime) {
                std::ostringstream oss;
                oss << "Chunk " << chunkId << " violates store-and-forward: hop "
                    << formatNodeName(curr->src) << "->" << formatNodeName(curr->dst)
                    << " ends at " << curr->endTime << " us, but next hop "
                    << formatNodeName(next->src) << "->" << formatNodeName(next->dst)
                    << " starts at " << next->startTime << " us";
                reportError(oss.str());
            }
        }
    }

    if (errorCount > 0) {
        throw std::runtime_error(scheduleName + " validation failed");
    }
}

//=============================================================================
// Helper functions for coordinate conversion
//=============================================================================
std::vector<int> Synthesizer4::nodeIDToCoordinate(int nodeID) const {
    if (shape_.empty()) {
        return {nodeID};
    }
    
    std::vector<int> coord(shape_.size());
    int remaining = nodeID;
    
    // Row-major order: convert from first dimension to last
    for (int d = 0; d < static_cast<int>(shape_.size()); ++d) {
        int stride = 1;
        for (int i = d + 1; i < static_cast<int>(shape_.size()); ++i) {
            stride *= shape_[i];
        }
        coord[d] = remaining / stride;
        remaining %= stride;
    }
    
    return coord;
}

int Synthesizer4::coordinateToNodeID(const std::vector<int>& coord) const {
    if (shape_.empty() || coord.empty()) {
        return coord.empty() ? 0 : coord[0];
    }
    
    if (coord.size() != shape_.size()) {
        return -1;
    }
    
    // Row-major order: calculate from first dimension to last
    int nodeID = 0;
    for (int d = 0; d < static_cast<int>(shape_.size()); ++d) {
        int stride = 1;
        for (int i = d + 1; i < static_cast<int>(shape_.size()); ++i) {
            stride *= shape_[i];
        }
        nodeID += coord[d] * stride;
    }
    return nodeID;
}

//=============================================================================
// Compute node degree
//=============================================================================
int Synthesizer4::computeNodeDegree(int nodeID) const {
    if (topology_ != nullptr) {
        // Switch-based topology: count connections to switches (for GPU nodes)
        // or connections to GPUs and other switches (for switch nodes)
        if (isGPUNode(nodeID)) {
            // GPU node: count connections to switches only
            int degree = 0;
            const auto& neighbors = getNeighborsFromGraph(nodeID);
            for (int neighbor : neighbors) {
                if (isSwitchNode(neighbor)) {
                    degree++;
                }
            }
            return degree;
        } else if (isSwitchNode(nodeID)) {
            // Switch node: count all connections
            return getNeighborsFromGraph(nodeID).size();
        } else {
            return 0;
        }
    } else {
        // Direct-connect topology: use coordinate-based computation
        auto coord = nodeIDToCoordinate(nodeID);
        
        int degree = 0;
        for (size_t d = 0; d < shape_.size(); ++d) {
            if (isTorus_) {
                // Torus: each node has 2 neighbors in each dimension (wrap around)
                degree += 2;
            } else {
                // Mesh: count neighbors based on position in dimension
                if (coord[d] == 0 || coord[d] == shape_[d] - 1) {
                    // Boundary node: 1 neighbor in this dimension
                    degree += 1;
                } else {
                    // Internal node: 2 neighbors in this dimension
                    degree += 2;
                }
            }
        }
        return degree;
    }
}

int Synthesizer4::computeMinDegree() const {
    int minDegree = std::numeric_limits<int>::max();
    for (int i = 0; i < npusCount_; ++i) {
        int degree = computeNodeDegree(i);
        if (degree < minDegree) {
            minDegree = degree;
        }
    }
    return minDegree;
}

//=============================================================================
// Build connection graph (pre-compute adjacency list)
//=============================================================================
void Synthesizer4::buildConnectionGraph() {
    if (topology_ != nullptr) {
        // Switch-based topology: build full graph including switches
        int totalNodes = topology_->npusCount();
        connectionGraph_.clear();
        connectionGraph_.resize(totalNodes);

        // Build adjacency list from topology's connected() method
        for (int src = 0; src < totalNodes; ++src) {
            for (int dst = 0; dst < totalNodes; ++dst) {
                if (topology_->connected(src, dst)) {
                    connectionGraph_[src].push_back(dst);
                }
            }
        }
    } else {
        // Direct-connect topology: build graph only for GPU nodes
        connectionGraph_.clear();
        connectionGraph_.resize(npusCount_);

        // Build adjacency list based on coordinate-based neighbors
        for (int node = 0; node < npusCount_; ++node) {
            auto coord = nodeIDToCoordinate(node);

            for (size_t d = 0; d < shape_.size(); ++d) {
                // Forward neighbor
                std::vector<int> nextCoord = coord;
                if (coord[d] < shape_[d] - 1) {
                    nextCoord[d]++;
                } else if (isTorus_) {
                    nextCoord[d] = 0;  // Wrap around
                } else {
                    continue;  // No wrap around for mesh
                }
                int next = coordinateToNodeID(nextCoord);
                if (next >= 0 && next < npusCount_) {
                    connectionGraph_[node].push_back(next);
                }

                // Backward neighbor
                nextCoord = coord;
                if (coord[d] > 0) {
                    nextCoord[d]--;
                } else if (isTorus_) {
                    nextCoord[d] = shape_[d] - 1;  // Wrap around
                } else {
                    continue;  // No wrap around for mesh
                }
                next = coordinateToNodeID(nextCoord);
                if (next >= 0 && next < npusCount_) {
                    connectionGraph_[node].push_back(next);
                }
            }
        }
    }

    bfsPathCache_.clear();
    shortestPathCache_.clear();
    timeShortestPathCache_.clear();
    initializeStaticCaches();
}

void Synthesizer4::initializeStaticCaches() {
    totalNodeCount_ = static_cast<int>(connectionGraph_.size());
    const double defaultLatencyUs = latency_ / 1000.0;
    const double defaultBandwidthBytesPerUs = bandwidth_ * (1 << 30) / 1e6;

    distanceMatrix_.assign(totalNodeCount_ * totalNodeCount_, std::numeric_limits<int>::max());
    linkLatencyUs_.assign(totalNodeCount_ * totalNodeCount_, defaultLatencyUs);
    linkBandwidthBytesPerUs_.assign(totalNodeCount_ * totalNodeCount_, defaultBandwidthBytesPerUs);

    for (int src = 0; src < totalNodeCount_; ++src) {
        distanceMatrix_[edgeIndex(src, src)] = 0;

        std::vector<int> dist(totalNodeCount_, -1);
        std::queue<int> q;
        q.push(src);
        dist[src] = 0;

        while (!q.empty()) {
            int u = q.front();
            q.pop();

            for (int v : connectionGraph_[u]) {
                if (v >= 0 && v < totalNodeCount_ && dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                }
            }
        }

        for (int dst = 0; dst < totalNodeCount_; ++dst) {
            if (dist[dst] >= 0) {
                distanceMatrix_[edgeIndex(src, dst)] = dist[dst];
            }
        }

        for (int dst : connectionGraph_[src]) {
            const int idx = edgeIndex(src, dst);
            if (topology_ != nullptr) {
                linkLatencyUs_[idx] = topology_->latency(src, dst) / 1000.0;
                linkBandwidthBytesPerUs_[idx] = topology_->bandwidth(src, dst) * (1 << 30) / 1e6;
            }
        }
    }
}

int Synthesizer4::edgeIndex(int src, int dst) const {
    return src * totalNodeCount_ + dst;
}

std::uint64_t Synthesizer4::pairCacheKey(int src, int dst) const {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(src)) << 32) |
           static_cast<std::uint32_t>(dst);
}

//=============================================================================
// Check if node is a GPU node
//=============================================================================
bool Synthesizer4::isGPUNode(int nodeID) const {
    if (topology_ == nullptr) {
        // Direct-connect topology: all nodes are GPUs
        return nodeID >= 0 && nodeID < npusCount_;
    } else {
        // Switch-based topology: GPU nodes are 0 to gpuNodeCount_ - 1
        return nodeID >= 0 && nodeID < gpuNodeCount_;
    }
}

//=============================================================================
// Check if node is a switch node
//=============================================================================
bool Synthesizer4::isSwitchNode(int nodeID) const {
    if (topology_ == nullptr) {
        // Direct-connect topology: no switches
        return false;
    } else {
        // Switch-based topology: switches are gpuNodeCount_ to topology_->npusCount() - 1
        return nodeID >= gpuNodeCount_ && nodeID < topology_->npusCount();
    }
}

//=============================================================================
// Get neighbors from pre-built connection graph
//=============================================================================
const std::vector<int>& Synthesizer4::getNeighborsFromGraph(int node) const {
    if (node < 0 || node >= static_cast<int>(connectionGraph_.size())) {
        static const std::vector<int> empty;
        return empty;
    }
    return connectionGraph_[node];
}

//=============================================================================
// Format node name for display
//=============================================================================
std::string Synthesizer4::formatNodeName(int nodeID) const {
    if (topology_ == nullptr) {
        // Direct-connect topology: all nodes are GPUs
        return "node " + std::to_string(nodeID);
    } else {
        // Switch-based topology
        if (isGPUNode(nodeID)) {
            return "node " + std::to_string(nodeID);
        } else if (isSwitchNode(nodeID)) {
            // For switch nodes, we need to determine the switch level and index
            // Switch ID range: gpuNodeCount_ to topology_->npusCount() - 1
            int switchID = nodeID - gpuNodeCount_;
            
            // Try to determine switch level based on topology structure
            // For fat-tree and rail-optimized, we can infer from switch ID ranges
            // This is a simplified version - we can enhance it later if needed
            // For now, just show switch index
            return "switch " + std::to_string(switchID);
        } else {
            return "unknown " + std::to_string(nodeID);
        }
    }
}

//=============================================================================
// Compute column sums for the demand matrix
//=============================================================================
std::vector<DataSize> Synthesizer4::computeColumnSums(
    const std::vector<std::vector<DataSize>>& demand) const {
    
    std::vector<DataSize> columnSums(npusCount_, 0);
    
    for (int i = 0; i < npusCount_; ++i) {
        for (int j = 0; j < npusCount_; ++j) {
            columnSums[j] += demand[i][j];
        }
    }
    
    return columnSums;
}

//=============================================================================
// Find hotspot columns based on column sums
//=============================================================================
std::set<int> Synthesizer4::findHotspotColumns(
    const std::vector<DataSize>& columnSums) const {
    
    std::set<int> hotspotColumns;
    
    if (columnSums.empty()) {
        return hotspotColumns;
    }
    
    // Create pairs of (column_index, column_sum) for sorting
    std::vector<std::pair<int, DataSize>> indexedSums;
    for (size_t i = 0; i < columnSums.size(); ++i) {
        indexedSums.push_back({static_cast<int>(i), columnSums[i]});
    }
    
    // Sort by column sum in descending order
    std::sort(indexedSums.begin(), indexedSums.end(),
              [](const std::pair<int, DataSize>& a, const std::pair<int, DataSize>& b) {
                  return a.second > b.second;
              });
    
    // Calculate statistics to identify hotspots
    // Method 1: Use top-K columns (K = sqrt(npusCount) or at least 1)
    int topK = std::max(1, static_cast<int>(std::sqrt(npusCount_)));
    
    // Method 2: Use threshold based on median and standard deviation
    // Calculate median
    std::vector<DataSize> sortedSums = columnSums;
    std::sort(sortedSums.begin(), sortedSums.end());
    DataSize median = sortedSums[sortedSums.size() / 2];
    
    // Calculate mean
    DataSize sumTotal = std::accumulate(columnSums.begin(), columnSums.end(), static_cast<DataSize>(0));
    double mean = static_cast<double>(sumTotal) / columnSums.size();
    
    // Calculate standard deviation
    double variance = 0.0;
    for (DataSize val : columnSums) {
        double diff = static_cast<double>(val) - mean;
        variance += diff * diff;
    }
    double stdDev = std::sqrt(variance / columnSums.size());
    
    // Threshold: mean + 2 * stdDev (for significantly above average)
    // Or use a simpler threshold: 2x the median
    DataSize threshold = std::max(
        static_cast<DataSize>(mean + 2.0 * stdDev),
        median * 2
    );
    
    // Collect hotspot columns using both methods
    // Take top-K that also exceed threshold
    int count = 0;
    for (const auto& [colIdx, colSum] : indexedSums) {
        if (colSum > threshold || count < topK) {
            if (colSum > 0) {  // Only consider non-zero columns
                hotspotColumns.insert(colIdx);
                count++;
            }
        }
        // Early stop if we've found enough and remaining values are below threshold
        if (count >= topK && colSum <= threshold) {
            break;
        }
    }
    
    if (verbose_) {
        std::cout << "\n[Hotspot Detection] Column Sum Analysis\n";
        std::cout << "  Total columns: " << columnSums.size() << "\n";
        std::cout << "  Mean column sum: " << mean << " bytes\n";
        std::cout << "  Median column sum: " << median << " bytes\n";
        std::cout << "  Std deviation: " << stdDev << " bytes\n";
        std::cout << "  Threshold: " << threshold << " bytes\n";
        std::cout << "  Top-K: " << topK << "\n";
        std::cout << "  Identified hotspots: ";
        for (int col : hotspotColumns) {
            std::cout << col << " (" << columnSums[col] << " bytes), ";
        }
        std::cout << "\n";
    }
    
    return hotspotColumns;
}

//=============================================================================
// Profiling phase: Decompose matrix into regular and hotspot matrices
//=============================================================================
ProfilingResult4 Synthesizer4::profileMatrix(
    const std::vector<std::vector<DataSize>>& demand) {
    
    ProfilingResult4 result;
    
    // Compute column sums
    result.columnSums = computeColumnSums(demand);
    
    // Find hotspot columns
    result.hotspotColumns = findHotspotColumns(result.columnSums);
    
    // Initialize matrices
    result.regularMatrix.clear();
    result.regularMatrix.resize(npusCount_);
    for (int i = 0; i < npusCount_; ++i) {
        result.regularMatrix[i].resize(npusCount_, 0);
    }
    
    result.hotspotMatrix.clear();
    result.hotspotMatrix.resize(npusCount_);
    for (int i = 0; i < npusCount_; ++i) {
        result.hotspotMatrix[i].resize(npusCount_, 0);
    }
    
    // Decompose matrix: columns in hotspotColumns go to hotspotMatrix, others to regularMatrix
    for (int i = 0; i < npusCount_; ++i) {
        for (int j = 0; j < npusCount_; ++j) {
            if (i == j || demand[i][j] == 0) continue;
            
            if (result.hotspotColumns.find(j) != result.hotspotColumns.end()) {
                // Destination is a hotspot: put in hotspot matrix
                result.hotspotMatrix[i][j] = demand[i][j];
            } else {
                // Destination is regular: put in regular matrix
                result.regularMatrix[i][j] = demand[i][j];
            }
        }
    }
    
    // Count non-zeros and total bytes
    result.regularMatrixNonZeros = 0;
    result.regularMatrixTotalBytes = 0;
    for (int i = 0; i < npusCount_; ++i) {
        for (int j = 0; j < npusCount_; ++j) {
            if (result.regularMatrix[i][j] > 0) {
                result.regularMatrixNonZeros++;
                result.regularMatrixTotalBytes += result.regularMatrix[i][j];
            }
        }
    }
    
    result.hotspotMatrixNonZeros = 0;
    result.hotspotMatrixTotalBytes = 0;
    for (int i = 0; i < npusCount_; ++i) {
        for (int j = 0; j < npusCount_; ++j) {
            if (result.hotspotMatrix[i][j] > 0) {
                result.hotspotMatrixNonZeros++;
                result.hotspotMatrixTotalBytes += result.hotspotMatrix[i][j];
            }
        }
    }
    
    return result;
}

//=============================================================================
// Compute distance (hop count) between two nodes
//=============================================================================
int Synthesizer4::computeDistance(int src, int dst) const {
    if (src == dst) return 0;

    if (src < 0 || dst < 0 || src >= totalNodeCount_ || dst >= totalNodeCount_ ||
        distanceMatrix_.empty()) {
        return std::numeric_limits<int>::max();
    }

    return distanceMatrix_[edgeIndex(src, dst)];
}

//=============================================================================
// Compute shortest path using BFS
//=============================================================================
std::vector<int> Synthesizer4::bfsPath(int src, int dst) const {
    if (src == dst) {
        return {src};
    }

    const auto key = pairCacheKey(src, dst);
    auto cached = bfsPathCache_.find(key);
    if (cached != bfsPathCache_.end()) {
        return cached->second;
    }

    if (src < 0 || dst < 0 || src >= totalNodeCount_ || dst >= totalNodeCount_) {
        return {src, dst};
    }

    std::vector<int> prev(totalNodeCount_, -1);
    std::queue<int> q;

    q.push(src);
    prev[src] = src;

    while (!q.empty()) {
        int current = q.front();
        q.pop();

        const auto& neighbors = getNeighborsFromGraph(current);
        for (int next : neighbors) {
            if (next >= 0 && next < totalNodeCount_ && prev[next] == -1) {
                prev[next] = current;
                if (next == dst) {
                    std::vector<int> pathOut;
                    int node = dst;
                    while (node != src) {
                        pathOut.push_back(node);
                        node = prev[node];
                    }
                    pathOut.push_back(src);
                    std::reverse(pathOut.begin(), pathOut.end());
                    bfsPathCache_.emplace(key, pathOut);
                    return pathOut;
                }
                q.push(next);
            }
        }
    }

    std::vector<int> fallback = {src, dst};
    bfsPathCache_.emplace(key, fallback);
    return fallback;
}

//=============================================================================
// Get neighbors of a node
//=============================================================================
std::vector<int> Synthesizer4::getNeighbors(int node) const {
    // Use pre-built connection graph
    return getNeighborsFromGraph(node);
}

//=============================================================================
// Find all shortest paths between src and dst
//=============================================================================
std::vector<std::vector<int>> Synthesizer4::findAllShortestPaths(int src, int dst) const {
    const auto key = pairCacheKey(src, dst);
    auto cached = shortestPathCache_.find(key);
    if (cached != shortestPathCache_.end()) {
        return cached->second;
    }

    std::vector<std::vector<int>> allPaths;

    if (src == dst) {
        allPaths.push_back({src});
        shortestPathCache_.emplace(key, allPaths);
        return allPaths;
    }

    if (src < 0 || dst < 0 || src >= totalNodeCount_ || dst >= totalNodeCount_ ||
        distanceMatrix_.empty()) {
        shortestPathCache_.emplace(key, allPaths);
        return allPaths;
    }

    constexpr int kMaxPaths = 10;
    const int shortestDistance = distanceMatrix_[edgeIndex(src, dst)];
    if (shortestDistance == std::numeric_limits<int>::max()) {
        shortestPathCache_.emplace(key, allPaths);
        return allPaths;
    }

    std::vector<int> currentPath;
    std::function<void(int)> dfs = [&](int u) {
        if (allPaths.size() >= kMaxPaths) {
            return;
        }

        currentPath.push_back(u);
        if (u == dst) {
            allPaths.push_back(currentPath);
            currentPath.pop_back();
            return;
        }

        const int currentDistance = distanceMatrix_[edgeIndex(src, u)];
        for (int v : getNeighborsFromGraph(u)) {
            if (v < 0 || v >= totalNodeCount_) {
                continue;
            }

            const int nextDistance = distanceMatrix_[edgeIndex(src, v)];
            const int remainingDistance = distanceMatrix_[edgeIndex(v, dst)];
            if (nextDistance == std::numeric_limits<int>::max() ||
                remainingDistance == std::numeric_limits<int>::max()) {
                continue;
            }

            if (nextDistance == currentDistance + 1 &&
                nextDistance + remainingDistance == shortestDistance) {
                dfs(v);
                if (allPaths.size() >= kMaxPaths) {
                    break;
                }
            }
        }

        currentPath.pop_back();
    };

    dfs(src);
    shortestPathCache_.emplace(key, allPaths);
    return allPaths;
}

//=============================================================================
// Select best path from multiple paths based on link load
//=============================================================================
std::vector<int> Synthesizer4::selectBestPath(
    const std::vector<std::vector<int>>& paths,
    const std::map<std::pair<int, int>, double>& linkBusyUntil,
    double currentTime) const {
    
    if (paths.empty()) {
        return {};
    }
    
    if (paths.size() == 1) {
        return paths[0];
    }
    
    // Select path with earliest available time
    std::vector<int> bestPath = paths[0];
    double bestEarliestTime = currentTime;
    
    for (const auto& path : paths) {
        double pathEarliestTime = currentTime;
        
        // Find the earliest time this path can start
        for (size_t i = 0; i + 1 < path.size(); ++i) {
            std::pair<int, int> edge = {path[i], path[i + 1]};
            if (linkBusyUntil.find(edge) != linkBusyUntil.end()) {
                pathEarliestTime = std::max(pathEarliestTime, linkBusyUntil.at(edge));
            }
        }
        
        // Choose path with earliest available time
        if (pathEarliestTime < bestEarliestTime) {
            bestEarliestTime = pathEarliestTime;
            bestPath = path;
        }
    }
    
    return bestPath;
}

//=============================================================================
// Compute shortest path with load-aware selection
//=============================================================================
std::vector<int> Synthesizer4::computeShortestPath(
    int src, int dst,
    const std::map<std::pair<int, int>, double>& linkBusyUntil) const {
    
    // Find all shortest paths
    auto allPaths = findAllShortestPaths(src, dst);
    
    if (allPaths.empty()) {
        // Fallback to BFS if no paths found
        return bfsPath(src, dst);
    }
    
    // If no link load information, return first path
    if (linkBusyUntil.empty()) {
        return allPaths[0];
    }
    
    // Select best path based on link load
    double currentTime = 0.0;
    if (!linkBusyUntil.empty()) {
        // Find maximum current time from link busy times
        for (const auto& [edge, time] : linkBusyUntil) {
            currentTime = std::max(currentTime, time);
        }
    }
    
    return selectBestPath(allPaths, linkBusyUntil, currentTime);
}

//=============================================================================
// Find K shortest paths (may include slightly longer paths for load balancing)
//=============================================================================
std::vector<std::vector<int>> Synthesizer4::findKShortestPaths(int src, int dst, int k) const {
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
    while (selectedPaths.size() < static_cast<size_t>(k) && !allPaths.empty()) {
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
            selectedPaths.push_back(allPaths[bestIdx]);
            for (size_t j = 0; j + 1 < allPaths[bestIdx].size(); ++j) {
                usedEdges.insert(edgeHash(allPaths[bestIdx][j], allPaths[bestIdx][j + 1]));
            }
            allPaths.erase(allPaths.begin() + bestIdx);
        } else {
            break;
        }
    }
    
    return selectedPaths;
}

//=============================================================================
// Find alternative paths (may be longer than shortest) for detour routing
//=============================================================================
std::vector<std::vector<int>> Synthesizer4::findAlternativePaths(int src, int dst, int maxHops) const {
    std::vector<std::vector<int>> alternativePaths;
    
    if (src == dst) {
        return {{src}};
    }
    
    int shortestDistance = computeDistance(src, dst);
    int maxAllowedHops = shortestDistance + 2;  // Allow up to 2 extra hops
    if (maxHops > 0) {
        maxAllowedHops = std::min(maxAllowedHops, maxHops);
    }
    
    // BFS to find paths up to maxAllowedHops
    struct PathNode {
        int node;
        std::vector<int> path;
        int hops;
    };
    
    std::queue<PathNode> q;
    q.push({src, {src}, 0});
    
    std::set<std::vector<int>> seenPaths;  // Avoid duplicate paths
    
    while (!q.empty() && alternativePaths.size() < 10) {  // Limit to 10 alternative paths
        PathNode current = q.front();
        q.pop();
        
        if (current.node == dst && current.hops >= shortestDistance) {
            // Found a path to destination
            if (seenPaths.find(current.path) == seenPaths.end()) {
                alternativePaths.push_back(current.path);
                seenPaths.insert(current.path);
            }
            continue;
        }
        
        if (current.hops >= maxAllowedHops) {
            continue;
        }
        
        // Explore neighbors
        for (int neighbor : getNeighbors(current.node)) {
            // Avoid cycles
            if (std::find(current.path.begin(), current.path.end(), neighbor) != current.path.end()) {
                continue;
            }
            
            std::vector<int> newPath = current.path;
            newPath.push_back(neighbor);
            q.push({neighbor, newPath, current.hops + 1});
        }
    }
    
    return alternativePaths;
}

//=============================================================================
// Evaluate path cost considering congestion and detour penalty
//=============================================================================
double Synthesizer4::evaluatePathCost(
    const std::vector<int>& path,
    const std::map<std::pair<int, int>, double>& linkBusyUntil,
    const std::map<std::pair<int, int>, DataSize>& edgeLoads,
    DataSize chunkSize,
    double currentTime,
    int shortestPathLength) const {
    
    if (path.size() < 2) {
        return std::numeric_limits<double>::max();
    }
    
    double pathCost = 0.0;
    double maxWaitTime = 0.0;
    DataSize maxEdgeLoad = 0;
    
    // Calculate path metrics
    for (size_t i = 0; i + 1 < path.size(); ++i) {
        std::pair<int, int> edge = {path[i], path[i + 1]};
        
        // Check link busy time (waiting time)
        double waitTime = 0.0;
        if (linkBusyUntil.find(edge) != linkBusyUntil.end()) {
            waitTime = std::max(0.0, linkBusyUntil.at(edge) - currentTime);
        }
        maxWaitTime = std::max(maxWaitTime, waitTime);
        
        // Check edge load
        if (edgeLoads.find(edge) != edgeLoads.end()) {
            maxEdgeLoad = std::max(maxEdgeLoad, edgeLoads.at(edge));
        }
        
        // Transfer time for this hop
        double transferTime = computeTransferTime(chunkSize, 1);
        pathCost += transferTime;
    }
    
    // Total cost = waiting time + transfer time + detour penalty
    int pathLength = path.size() - 1;
    int detourHops = pathLength - shortestPathLength;
    double detourPenalty = detourHops > 0 ? (detourHops * computeTransferTime(chunkSize, 1)) : 0.0;
    
    // Cost = max wait time (congestion) + path transfer time + detour penalty
    // Weight congestion more heavily
    double totalCost = maxWaitTime * 2.0 + pathCost + detourPenalty * 0.5;
    
    return totalCost;
}

//=============================================================================
// Select best path for load balancing (considering edge loads)
//=============================================================================
std::vector<int> Synthesizer4::selectBestPathForLoadBalancing(
    const std::vector<std::vector<int>>& paths,
    const std::map<std::pair<int, int>, double>& linkBusyUntil,
    const std::map<std::pair<int, int>, DataSize>& edgeLoads,
    double currentTime) const {
    
    if (paths.empty()) {
        return {};
    }
    
    if (paths.size() == 1) {
        return paths[0];
    }
    
    // Select path with minimum max edge load and earliest available time
    std::vector<int> bestPath = paths[0];
    double bestScore = std::numeric_limits<double>::max();
    
    for (const auto& path : paths) {
        double pathEarliestTime = currentTime;
        DataSize pathMaxLoad = 0;
        
        // Calculate path metrics
        for (size_t i = 0; i + 1 < path.size(); ++i) {
            std::pair<int, int> edge = {path[i], path[i + 1]};
            
            // Check link busy time
            if (linkBusyUntil.find(edge) != linkBusyUntil.end()) {
                pathEarliestTime = std::max(pathEarliestTime, linkBusyUntil.at(edge));
            }
            
            // Check edge load
            if (edgeLoads.find(edge) != edgeLoads.end()) {
                pathMaxLoad = std::max(pathMaxLoad, edgeLoads.at(edge));
            }
        }
        
        // Score: prioritize paths with lower max load, then earlier available time
        // Lower score is better
        double score = pathMaxLoad * 1e9 + pathEarliestTime;
        
        if (score < bestScore) {
            bestScore = score;
            bestPath = path;
        }
    }
    
    return bestPath;
}

//=============================================================================
// Compute transfer time using alpha-beta model
//=============================================================================
double Synthesizer4::computeTransferTime(DataSize bytes, int hops) const {
    if (bytes <= 0) return 0.0;
    
    // Convert latency from ns to us
    double latencyUs = latency_ / 1000.0;
    
    // Convert bandwidth from GB/s to bytes/us
    double bandwidthBytesPerUs = bandwidth_ * (1 << 30) / 1e6;
    
    // Alpha-beta model: time = latency * hops + bytes / bandwidth
    double transferTime = latencyUs * hops + (bytes / bandwidthBytesPerUs);
    
    return transferTime;
}

//=============================================================================
// Compute transfer time for a single link (supports heterogeneous links)
//=============================================================================
double Synthesizer4::computeLinkTransferTime(int src, int dst, DataSize bytes) const {
    if (bytes <= 0) return 0.0;

    if (src >= 0 && dst >= 0 && src < totalNodeCount_ && dst < totalNodeCount_ &&
        !linkBandwidthBytesPerUs_.empty()) {
        const int idx = edgeIndex(src, dst);
        const double bandwidthBytesPerUs = linkBandwidthBytesPerUs_[idx];
        if (bandwidthBytesPerUs > 0.0) {
            return linkLatencyUs_[idx] + (bytes / bandwidthBytesPerUs);
        }
    }

    const double fallbackLatencyUs = latency_ / 1000.0;
    const double fallbackBandwidthBytesPerUs = bandwidth_ * (1 << 30) / 1e6;
    return fallbackLatencyUs + (bytes / fallbackBandwidthBytesPerUs);
}

bool Synthesizer4::isSwitchTopology() const {
    return (topology_ != nullptr && topology_->npusCount() > gpuNodeCount_);
}

Synthesizer4::PathTiming4 Synthesizer4::computePathTimingCutThrough(const std::vector<int>& path, DataSize bytes) const {
    PathTiming4 timing;
    timing.dataTimeUs = 0.0;
    timing.totalTimeUs = 0.0;
    if (path.size() < 2 || bytes <= 0 || topology_ == nullptr) return timing;
    double sumLatUs = 0.0;
    double minBwBytesPerUs = std::numeric_limits<double>::max();
    for (size_t h = 0; h < path.size() - 1; ++h) {
        int a = path[h], b = path[h + 1];
        if (!topology_->connected(a, b)) continue;
        double latUs = topology_->latency(a, b);
        timing.hopLatenciesUs.push_back(latUs);
        sumLatUs += latUs;
        double bw = topology_->bandwidth(a, b);
        if (bw > 0) {
            double bytesPerUs = bw * (1 << 30) / 1e6;
            if (bytesPerUs < minBwBytesPerUs) minBwBytesPerUs = bytesPerUs;
        }
    }
    if (minBwBytesPerUs <= 0 || minBwBytesPerUs == std::numeric_limits<double>::max())
        timing.dataTimeUs = 0.0;
    else
        timing.dataTimeUs = bytes / minBwBytesPerUs;
    timing.totalTimeUs = sumLatUs + timing.dataTimeUs;
    return timing;
}

//=============================================================================
// Find time-shortest paths (based on transfer time, not just hop count)
// Supports heterogeneous link bandwidths
//=============================================================================
std::vector<std::vector<int>> Synthesizer4::computeTimeShortestPathsUncached(int src, int dst, DataSize bytes) const {
    constexpr double kPathEpsilon = 1e-9;
    constexpr std::size_t kMaxPaths = 10;

    std::vector<std::vector<int>> timeShortestPaths;

    if (src == dst) {
        timeShortestPaths.push_back({src});
        return timeShortestPaths;
    }

    if (src < 0 || dst < 0 || src >= totalNodeCount_ || dst >= totalNodeCount_) {
        return timeShortestPaths;
    }

    std::vector<double> dist(totalNodeCount_, std::numeric_limits<double>::max());
    std::vector<std::vector<int>> prev(totalNodeCount_);
    for (int node = 0; node < totalNodeCount_; ++node) {
        prev[node].reserve(getNeighborsFromGraph(node).size());
    }

    using QueueEntry = std::pair<double, int>;
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, std::greater<QueueEntry>> pq;

    dist[src] = 0.0;
    pq.push({0.0, src});

    while (!pq.empty()) {
        const double currentDist = pq.top().first;
        const int u = pq.top().second;
        pq.pop();

        if (currentDist > dist[u] + kPathEpsilon) {
            continue;
        }

        if (dist[dst] < std::numeric_limits<double>::max() &&
            currentDist > dist[dst] + kPathEpsilon) {
            break;
        }

        if (u == dst && std::abs(currentDist - dist[dst]) <= kPathEpsilon) {
            continue;
        }

        const auto& neighbors = getNeighborsFromGraph(u);
        for (int v : neighbors) {
            if (v < 0 || v >= totalNodeCount_) {
                continue;
            }

            const double linkTime = computeLinkTransferTime(u, v, bytes);
            const double newDist = dist[u] + linkTime;

            if (newDist < dist[v] - kPathEpsilon) {
                dist[v] = newDist;
                prev[v].clear();
                prev[v].push_back(u);
                pq.push({newDist, v});
            } else if (std::abs(newDist - dist[v]) <= kPathEpsilon) {
                prev[v].push_back(u);
            }
        }
    }

    if (dist[dst] == std::numeric_limits<double>::max()) {
        return findAllShortestPaths(src, dst);
    }

    std::vector<int> reversedPath;
    reversedPath.reserve(totalNodeCount_);
    std::function<void(int)> dfs = [&](int node) {
        if (timeShortestPaths.size() >= kMaxPaths) {
            return;
        }

        reversedPath.push_back(node);
        if (node == src) {
            timeShortestPaths.emplace_back(reversedPath.rbegin(), reversedPath.rend());
        } else {
            for (int p : prev[node]) {
                dfs(p);
                if (timeShortestPaths.size() >= kMaxPaths) {
                    break;
                }
            }
        }
        reversedPath.pop_back();
    };

    dfs(dst);
    return timeShortestPaths;
}

//=============================================================================
// Find time-shortest paths (based on transfer time, not just hop count)
// Supports heterogeneous link bandwidths
//=============================================================================
std::vector<std::vector<int>> Synthesizer4::findTimeShortestPaths(int src, int dst, DataSize bytes) const {
    TimeShortestPathCacheKey cacheKey{src, dst, bytes};
    auto cached = timeShortestPathCache_.find(cacheKey);
    if (cached != timeShortestPathCache_.end()) {
        return cached->second;
    }

    auto timeShortestPaths = computeTimeShortestPathsUncached(src, dst, bytes);
    timeShortestPathCache_.emplace(cacheKey, timeShortestPaths);
    return timeShortestPaths;
}

//=============================================================================
// Select best path based on accumulated link load and load balancing
// NEW: Uses accumulatedLinkLoad instead of linkBusyUntil
//=============================================================================
std::vector<int> Synthesizer4::selectBestPathByLoad(
    const std::vector<std::vector<int>>& paths,
    const std::vector<double>& accumulatedLinkLoad,
    double maxAccumulatedLinkLoad,
    DataSize chunkSize) const {
    
    if (paths.empty()) {
        return {};
    }
    
    if (paths.size() == 1) {
        return paths[0];
    }

    const double maxLoad = std::max(maxAccumulatedLinkLoad, 1.0);
    const double fallbackBandwidthBytesPerUs = bandwidth_ * (1 << 30) / 1e6;
    std::vector<double> pathDataTransfer(paths.size(), 0.0);
    double maxPathDataTransfer = 0.0;

    for (size_t p = 0; p < paths.size(); ++p) {
        const auto& path = paths[p];
        for (size_t i = 0; i + 1 < path.size(); ++i) {
            const int idx = edgeIndex(path[i], path[i + 1]);
            const double transferTime = computeLinkTransferTime(path[i], path[i + 1], chunkSize);
            double linkBandwidthBytesPerUs = fallbackBandwidthBytesPerUs;
            if (!linkBandwidthBytesPerUs_.empty()) {
                const double cachedBandwidthBytesPerUs = linkBandwidthBytesPerUs_[idx];
                if (cachedBandwidthBytesPerUs > 0.0) {
                    linkBandwidthBytesPerUs = cachedBandwidthBytesPerUs;
                }
            }
            pathDataTransfer[p] += transferTime * linkBandwidthBytesPerUs;
        }
        maxPathDataTransfer = std::max(maxPathDataTransfer, pathDataTransfer[p]);
    }
    if (maxPathDataTransfer < 1e-9) {
        maxPathDataTransfer = 1.0;
    }

    std::vector<int> bestPath = paths[0];
    double bestScore = std::numeric_limits<double>::max();

    for (size_t p = 0; p < paths.size(); ++p) {
        const auto& path = paths[p];
        double pathSumLoad = 0.0;
        double pathMaxLoad = 0.0;

        for (size_t i = 0; i + 1 < path.size(); ++i) {
            const int idx = edgeIndex(path[i], path[i + 1]);
            const double load = accumulatedLinkLoad[idx];
            pathSumLoad += load;
            pathMaxLoad = std::max(pathMaxLoad, load);
        }

        const double normalizedSumLoad = pathSumLoad / maxLoad;
        const double normalizedMaxLoad = pathMaxLoad / maxLoad;
        const double normalizedPathDataTransfer = pathDataTransfer[p] / maxPathDataTransfer;
        const double score = normalizedSumLoad * 0.3 + normalizedMaxLoad * 0.2 + normalizedPathDataTransfer * 0.5;

        if (score < bestScore) {
            bestScore = score;
            bestPath = path;
        }
    }
    
    return bestPath;
}

//=============================================================================
// Select best path for regular matrix (simplified: time-shortest, then min normalizedSumLoad)
//=============================================================================
std::vector<int> Synthesizer4::selectBestPathForRegular(
    const std::vector<std::vector<int>>& paths,
    const std::vector<double>& accumulatedLinkLoad,
    double maxAccumulatedLinkLoad,
    DataSize bytes) const {
    
    if (paths.empty()) {
        return {};
    }
    
    if (paths.size() == 1) {
        return paths[0];
    }

    const double maxLoad = std::max(maxAccumulatedLinkLoad, 1.0);
    std::vector<double> pathTransferTimes(paths.size(), 0.0);
    double minTime = std::numeric_limits<double>::max();
    for (size_t p = 0; p < paths.size(); ++p) {
        const auto& path = paths[p];
        for (size_t i = 0; i + 1 < path.size(); ++i) {
            pathTransferTimes[p] += computeLinkTransferTime(path[i], path[i + 1], bytes);
        }
        minTime = std::min(minTime, pathTransferTimes[p]);
    }

    std::vector<size_t> shortestPathIndices;
    shortestPathIndices.reserve(paths.size());
    for (size_t p = 0; p < paths.size(); ++p) {
        if (std::abs(pathTransferTimes[p] - minTime) < 1e-9) {
            shortestPathIndices.push_back(p);
        }
    }

    if (shortestPathIndices.size() == 1) {
        return paths[shortestPathIndices[0]];
    }

    std::vector<int> bestPath = paths[shortestPathIndices[0]];
    double bestNormalizedSumLoad = std::numeric_limits<double>::max();

    for (size_t idx : shortestPathIndices) {
        const auto& path = paths[idx];
        double pathSumLoad = 0.0;

        for (size_t i = 0; i + 1 < path.size(); ++i) {
            pathSumLoad += accumulatedLinkLoad[edgeIndex(path[i], path[i + 1])];
        }

        const double normalizedSumLoad = pathSumLoad / maxLoad;
        if (normalizedSumLoad < bestNormalizedSumLoad) {
            bestNormalizedSumLoad = normalizedSumLoad;
            bestPath = path;
        }
    }
    
    return bestPath;
}

//=============================================================================
// Allocate time slots for events (for individual matrix scheduling display)
//=============================================================================
double Synthesizer4::allocateTimeSlotsForEvents(std::vector<TransferEvent4>& events) const {
    if (events.empty()) {
        return 0.0;
    }

    std::vector<std::vector<std::pair<double, double>>> linkTimeSlots(totalNodeCount_ * totalNodeCount_);
    int maxChunkId = -1;
    for (const auto& event : events) {
        maxChunkId = std::max(maxChunkId, event.chunkId);
    }
    std::vector<std::vector<double>> dataArrivalTime;
    if (maxChunkId >= 0) {
        dataArrivalTime.assign(totalNodeCount_, std::vector<double>(maxChunkId + 1, -1.0));
    }

    sortEventsForScheduling(events);

    const auto slotLess = [](const std::pair<double, double>& a,
                             const std::pair<double, double>& b) {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        return a.second < b.second;
    };

    for (auto& event : events) {
        const double transferTime = computeLinkTransferTime(event.src, event.dst, event.bytes);
        double earliestStartTime = 0.0;

        if (event.chunkId >= 0 && event.src != event.flowSrc && !dataArrivalTime.empty()) {
            const double arrivalTime = dataArrivalTime[event.src][event.chunkId];
            if (arrivalTime >= 0.0) {
                earliestStartTime = arrivalTime;
            }
        }

        auto& slots = linkTimeSlots[edgeIndex(event.src, event.dst)];
        if (!slots.empty()) {
            bool foundSlot = false;
            if (earliestStartTime + transferTime <= slots.front().first) {
                foundSlot = true;
            } else {
                for (size_t i = 0; i + 1 < slots.size(); ++i) {
                    const double gapStart = slots[i].second;
                    const double gapEnd = slots[i + 1].first;
                    const double requiredStart = std::max(earliestStartTime, gapStart);
                    if (requiredStart + transferTime <= gapEnd) {
                        earliestStartTime = requiredStart;
                        foundSlot = true;
                        break;
                    }
                }
            }
            if (!foundSlot) {
                earliestStartTime = std::max(earliestStartTime, slots.back().second);
            }
        }

        event.startTime = earliestStartTime;
        event.endTime = earliestStartTime + transferTime;
        const auto insertPos = std::upper_bound(
            slots.begin(), slots.end(), std::make_pair(event.startTime, event.endTime), slotLess);
        slots.insert(insertPos, {event.startTime, event.endTime});
        if (event.chunkId >= 0 && !dataArrivalTime.empty()) {
            dataArrivalTime[event.dst][event.chunkId] = event.endTime;
        }
    }

    double maxTime = 0.0;
    for (const auto& e : events) {
        if (e.endTime > maxTime) {
            maxTime = e.endTime;
        }
    }

    return maxTime;
}

//=============================================================================
// Scheduling phase - Regular matrix
//=============================================================================
std::vector<TransferEvent4> Synthesizer4::scheduleRegularMatrix(
    const std::vector<std::vector<DataSize>>& regularMatrix) {
    
    if (verbose_) {
        std::cout << "\n[Scheduling] Processing Regular Matrix\n";
    }
    
    std::vector<TransferEvent4> events;
    
    // Collect all regular flows
    struct RegularFlow {
        int src;
        int dst;
        DataSize bytes;
        int distance;
    };
    
    std::vector<RegularFlow> flows;
    for (int i = 0; i < npusCount_; ++i) {
        for (int j = 0; j < npusCount_; ++j) {
            if (regularMatrix[i][j] > 0) {
                int distance = computeDistance(i, j);
                flows.push_back({i, j, regularMatrix[i][j], distance});
            }
        }
    }
    
    // Sort by distance (farther first) to prioritize long-distance flows
    std::sort(flows.begin(), flows.end(),
              [](const RegularFlow& a, const RegularFlow& b) {
                  return a.distance > b.distance;
              });
    
    // SIMPLIFIED: For regular matrix, only track accumulated link load (no time slot allocation here)
    // Time slot allocation will be done together with hotspot matrix later
    // Track accumulated link load: accumulatedLinkLoad[edge] = sum of transfer times on this link
    std::vector<double> accumulatedLinkLoad(totalNodeCount_ * totalNodeCount_, 0.0);
    double maxAccumulatedLinkLoad = 0.0;
    
    int flowId = 0;
    for (const auto& flow : flows) {
        // Find time-shortest paths (based on transfer time, supports heterogeneous links)
        auto allPaths = findTimeShortestPaths(flow.src, flow.dst, flow.bytes);
        if (allPaths.empty()) {
            // Fallback to hop-based shortest paths
            allPaths = findAllShortestPaths(flow.src, flow.dst);
        }
        if (allPaths.empty()) continue;
        
        // Select time-shortest path, if multiple, choose one with min normalizedSumLoad
        std::vector<int> path = selectBestPathForRegular(allPaths, accumulatedLinkLoad, maxAccumulatedLinkLoad, flow.bytes);
        if (path.size() < 2) continue;
        
        // Always create one event per hop (store-and-forward, including switches)
        for (size_t hop = 0; hop < path.size() - 1; ++hop) {
            int hopSrc = path[hop];
            int hopDst = path[hop + 1];
            const int edgeIdx = edgeIndex(hopSrc, hopDst);
            double hopTransferTime = computeLinkTransferTime(hopSrc, hopDst, flow.bytes);
            TransferEvent4 event;
            event.src = hopSrc;
            event.dst = hopDst;
            event.bytes = flow.bytes;
            event.startTime = 0.0;
            event.endTime = 0.0;
            event.path = path;
            event.chunkId = flowId;
            event.flowSrc = flow.src;
            event.flowDst = flow.dst;
            event.isRegularMatrix = true;
            events.push_back(event);
            accumulatedLinkLoad[edgeIdx] += hopTransferTime;
                maxAccumulatedLinkLoad = std::max(maxAccumulatedLinkLoad, accumulatedLinkLoad[edgeIdx]);
        }
        
        flowId++;
    }
    
    // Allocate time slots for regular matrix events (for display purposes)
    // This is separate from the unified time slot allocation in fuseSchedules
    double makespan = allocateTimeSlotsForEvents(events);
    validateScheduleOrThrow(events, "Regular Matrix");
    
    if (verbose_) {
        std::cout << "  Scheduled " << events.size() << " transfer events\n";
        std::cout << "  Note: Each transfer event represents a single-hop transmission (from one node to an adjacent node)\n";
        std::cout << "  Regular matrix makespan: " << std::fixed << std::setprecision(2) << makespan << " us\n";
        
        // Calculate and print bandwidth utilization
        double utilization = calculateBandwidthUtilization(events, makespan);
        std::cout << "  Average Bandwidth Utilization: " << std::fixed << std::setprecision(2)
                  << (utilization * 100.0) << "%\n";

        // Print per-link busy intervals (ns) and utilization for regular matrix
        printLinkBusyIntervals(events, makespan, "Regular Matrix");
        
        // Print step-by-step schedule
        printStepByStepSchedule(events, "Regular Matrix");
    }
    
    return events;
}

//=============================================================================
// Compute chunk count for hotspot matrix
//=============================================================================
// Compute minimum chunk size based on bandwidth * latency
//=============================================================================
DataSize Synthesizer4::computeMinChunkSize(int src, int dst) const {
    double latencyUs = latency_ / 1000.0;
    double bandwidthBytesPerUs = bandwidth_ * (1 << 30) / 1e6;

    if (src >= 0 && dst >= 0 && src < totalNodeCount_ && dst < totalNodeCount_ &&
        !linkBandwidthBytesPerUs_.empty()) {
        const int directIdx = edgeIndex(src, dst);
        if (linkBandwidthBytesPerUs_[directIdx] > 0.0) {
            latencyUs = linkLatencyUs_[directIdx];
            bandwidthBytesPerUs = linkBandwidthBytesPerUs_[directIdx];
        }
    }

    if (topology_ != nullptr) {
        auto path = bfsPath(src, dst);
        if (path.size() > 1) {
            double minBandwidthBytesPerUs = std::numeric_limits<double>::max();
            double totalLatencyUs = 0.0;

            for (size_t i = 0; i + 1 < path.size(); ++i) {
                const int idx = edgeIndex(path[i], path[i + 1]);
                const double hopBandwidthBytesPerUs = linkBandwidthBytesPerUs_[idx];
                if (hopBandwidthBytesPerUs <= 0.0) {
                    continue;
                }
                totalLatencyUs += linkLatencyUs_[idx];
                minBandwidthBytesPerUs = std::min(minBandwidthBytesPerUs, hopBandwidthBytesPerUs);
            }

            if (minBandwidthBytesPerUs < std::numeric_limits<double>::max()) {
                latencyUs = totalLatencyUs;
                bandwidthBytesPerUs = minBandwidthBytesPerUs;
            }
        }
    }

    double minChunkSize = bandwidthBytesPerUs * latencyUs;
    return std::max(static_cast<DataSize>(1), static_cast<DataSize>(minChunkSize));
}

//=============================================================================
// Compute chunk count for hotspot matrix
//=============================================================================
int Synthesizer4::computeChunkCount(int src, int dst, DataSize bytes) const {
    int srcDegree = computeNodeDegree(src);
    int dstDegree = computeNodeDegree(dst);
    int maxChunks = std::min(srcDegree, dstDegree);
    
    // Compute minimum chunk size based on bandwidth * latency
    DataSize minChunkSize = computeMinChunkSize(src, dst);
    
    // Ensure each chunk is at least minChunkSize
    // Calculate maximum number of chunks that satisfy this constraint
    int maxChunksBySize = static_cast<int>(bytes / minChunkSize);
    
    // Take the minimum of degree-based and size-based constraints
    int numChunks = std::min(maxChunks, maxChunksBySize);
    
    // Ensure at least 1 chunk
    return std::max(1, numChunks);
}

//=============================================================================
// Scheduling phase - Hotspot matrix
//=============================================================================
std::vector<TransferEvent4> Synthesizer4::scheduleHotspotMatrix(
    const std::vector<std::vector<DataSize>>& hotspotMatrix) {
    
    if (verbose_) {
        std::cout << "\n[Scheduling] Processing Hotspot Matrix\n";
    }
    
    std::vector<TransferEvent4> events;
    
    // Collect all hotspot flows (gather communication to hotspot destinations)
    struct HotspotFlow {
        int src;
        int dst;
        DataSize bytes;
        int distance;
        double priority;  // bytes * distance
    };
    
    std::vector<HotspotFlow> flows;
    for (int i = 0; i < npusCount_; ++i) {
        for (int j = 0; j < npusCount_; ++j) {
            if (hotspotMatrix[i][j] > 0) {
                int distance = computeDistance(i, j);
                double priority = static_cast<double>(hotspotMatrix[i][j]) * distance;
                flows.push_back({i, j, hotspotMatrix[i][j], distance, priority});
            }
        }
    }
    
    // Sort by priority (descending): data * distance
    std::sort(flows.begin(), flows.end(),
              [](const HotspotFlow& a, const HotspotFlow& b) {
                  return a.priority > b.priority;
              });
    
    // NEW: Event-driven scheduling using accumulated link load
    // Time slot allocation will be done together with regular matrix later
    // Track accumulated link load: accumulatedLinkLoad[edge] = sum of transfer times on this link
    std::vector<double> accumulatedLinkLoad(totalNodeCount_ * totalNodeCount_, 0.0);
    double maxAccumulatedLinkLoad = 0.0;
    
    int chunkId = 0;
    for (const auto& flow : flows) {
        // Compute chunk count
        int numChunks = computeChunkCount(flow.src, flow.dst, flow.bytes);
        DataSize chunkSize = flow.bytes / numChunks;
        DataSize remainder = flow.bytes % numChunks;
        
        // Schedule each chunk with load-balanced path selection
        for (int c = 0; c < numChunks; ++c) {
            DataSize currentChunkSize = chunkSize;
            if (c < remainder) {
                currentChunkSize++;  // Distribute remainder
            }
            
            // Find time-shortest paths (supports heterogeneous links)
            auto allPaths = findTimeShortestPaths(flow.src, flow.dst, currentChunkSize);
            if (allPaths.empty()) {
                // Fallback to hop-based shortest paths
                int path_k = 3;
                allPaths = findKShortestPaths(flow.src, flow.dst, path_k);
                if (allPaths.empty()) {
                    allPaths = findAllShortestPaths(flow.src, flow.dst);
                }
            }
            if (allPaths.empty()) continue;
            
            // Select best path based on accumulated link load (hotspot matrix uses full load balancing)
            std::vector<int> path = selectBestPathByLoad(allPaths, accumulatedLinkLoad, maxAccumulatedLinkLoad, currentChunkSize);
            if (path.size() < 2) continue;
            
            // Always create one event per hop (store-and-forward, including switches)
            for (size_t hop = 0; hop < path.size() - 1; ++hop) {
                int hopSrc = path[hop];
                int hopDst = path[hop + 1];
                const int edgeIdx = edgeIndex(hopSrc, hopDst);
                double hopTransferTime = computeLinkTransferTime(hopSrc, hopDst, currentChunkSize);
                TransferEvent4 event;
                event.src = hopSrc;
                event.dst = hopDst;
                event.bytes = currentChunkSize;
                event.startTime = 0.0;
                event.endTime = 0.0;
                event.path = path;
                event.chunkId = chunkId;
                event.flowSrc = flow.src;
                event.flowDst = flow.dst;
                event.isRegularMatrix = false;
                events.push_back(event);
                accumulatedLinkLoad[edgeIdx] += hopTransferTime;
                maxAccumulatedLinkLoad = std::max(maxAccumulatedLinkLoad, accumulatedLinkLoad[edgeIdx]);
            }
            
            chunkId++;
        }
    }
    
    // Allocate time slots for hotspot matrix events (for display purposes)
    // This is separate from the unified time slot allocation in fuseSchedules
    double makespan = allocateTimeSlotsForEvents(events);
    validateScheduleOrThrow(events, "Hotspot Matrix");
    
    if (verbose_) {
        std::cout << "  Scheduled " << events.size() << " transfer events\n";
        std::cout << "  Note: Each transfer event represents a single-hop transmission (from one node to an adjacent node)\n";
        std::cout << "  Total chunks: " << chunkId << "\n";
        std::cout << "  Hotspot matrix makespan: " << std::fixed << std::setprecision(2) << makespan << " us\n";
        
        // Calculate and print bandwidth utilization
        double utilization = calculateBandwidthUtilization(events, makespan);
        std::cout << "  Average Bandwidth Utilization: " << std::fixed << std::setprecision(2)
                  << (utilization * 100.0) << "%\n";

        // Print per-link busy intervals (ns) and utilization for hotspot matrix
        printLinkBusyIntervals(events, makespan, "Hotspot Matrix");
        
        // Print step-by-step schedule
        printStepByStepSchedule(events, "Hotspot Matrix");
    }
    
    return events;
}

//=============================================================================
// Fusion phase: Merge schedules
//=============================================================================
ScheduleResult4 Synthesizer4::fuseSchedules(
    const std::vector<TransferEvent4>& regularEvents,
    const std::vector<TransferEvent4>& hotspotEvents) {
    
    if (verbose_) {
        std::cout << "\n[Fusion] Merging Regular and Hotspot Schedules\n";
    }
    
    ScheduleResult4 result;
    
    double regularMakespan = 0.0;
    for (const auto& e : regularEvents) {
        if (e.endTime > regularMakespan) {
            regularMakespan = e.endTime;
        }
    }
    
    double hotspotMakespan = 0.0;
    for (const auto& e : hotspotEvents) {
        if (e.endTime > hotspotMakespan) {
            hotspotMakespan = e.endTime;
        }
    }
    
    result.regularMatrixMakespan = regularMakespan;
    result.hotspotMatrixMakespan = hotspotMakespan;
    
    std::vector<TransferEvent4> allEvents = regularEvents;
    int nextChunkId = 0;
    for (const auto& event : allEvents) {
        nextChunkId = std::max(nextChunkId, event.chunkId + 1);
    }
    for (auto event : hotspotEvents) {
        event.chunkId += nextChunkId;
        allEvents.push_back(std::move(event));
    }
    
    double fusedMakespan = simulateFusedSchedule(allEvents);
    validateScheduleOrThrow(allEvents, "Fusion");
    
    std::sort(allEvents.begin(), allEvents.end(),
              [](const TransferEvent4& a, const TransferEvent4& b) {
                  if (a.startTime != b.startTime) return a.startTime < b.startTime;
                  if (a.chunkId != b.chunkId) return a.chunkId < b.chunkId;
                  return a.src < b.src;
              });
    
    result.fusedMakespan = fusedMakespan;
    result.makespan = fusedMakespan;
    result.events = allEvents;
    
    if (verbose_) {
        std::cout << "  Regular matrix makespan: " << regularMakespan << " us\n";
        std::cout << "  Hotspot matrix makespan: " << hotspotMakespan << " us\n";
        std::cout << "  Fused makespan: " << fusedMakespan << " us\n";
        
        double fusedUtilization = calculateBandwidthUtilization(allEvents, fusedMakespan);
        std::cout << "  Fused Average Bandwidth Utilization: " << std::fixed << std::setprecision(2)
                  << (fusedUtilization * 100.0) << "%\n";
    }
    
    return result;
}

//=============================================================================
// Event-driven simulation for fusion with unified time slot allocation
//=============================================================================
double Synthesizer4::simulateFusedSchedule(
    std::vector<TransferEvent4>& allEvents) const {

    std::vector<std::vector<std::pair<double, double>>> linkTimeSlots(totalNodeCount_ * totalNodeCount_);
    int maxChunkId = -1;
    for (const auto& event : allEvents) {
        maxChunkId = std::max(maxChunkId, event.chunkId);
    }
    std::vector<std::vector<double>> dataArrivalTime;
    if (maxChunkId >= 0) {
        dataArrivalTime.assign(totalNodeCount_, std::vector<double>(maxChunkId + 1, -1.0));
    }

    sortEventsForScheduling(allEvents);

    const auto slotLess = [](const std::pair<double, double>& a,
                             const std::pair<double, double>& b) {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        return a.second < b.second;
    };

    for (auto& event : allEvents) {
        const double transferTime = computeLinkTransferTime(event.src, event.dst, event.bytes);
        double earliestStartTime = 0.0;

        if (event.chunkId >= 0 && event.src != event.flowSrc && !dataArrivalTime.empty()) {
            const double arrivalTime = dataArrivalTime[event.src][event.chunkId];
            if (arrivalTime >= 0.0) {
                earliestStartTime = arrivalTime;
            }
        }

        auto& slots = linkTimeSlots[edgeIndex(event.src, event.dst)];
        if (!slots.empty()) {
            bool foundSlot = false;
            if (earliestStartTime + transferTime <= slots.front().first) {
                foundSlot = true;
            } else {
                for (size_t i = 0; i + 1 < slots.size(); ++i) {
                    const double gapStart = slots[i].second;
                    const double gapEnd = slots[i + 1].first;
                    const double requiredStart = std::max(earliestStartTime, gapStart);
                    if (requiredStart + transferTime <= gapEnd) {
                        earliestStartTime = requiredStart;
                        foundSlot = true;
                        break;
                    }
                }
            }
            if (!foundSlot) {
                earliestStartTime = std::max(earliestStartTime, slots.back().second);
            }
        }

        event.startTime = earliestStartTime;
        event.endTime = earliestStartTime + transferTime;
        const auto insertPos = std::upper_bound(
            slots.begin(), slots.end(), std::make_pair(event.startTime, event.endTime), slotLess);
        slots.insert(insertPos, {event.startTime, event.endTime});
        if (event.chunkId >= 0 && !dataArrivalTime.empty()) {
            dataArrivalTime[event.dst][event.chunkId] = event.endTime;
        }
    }

    double maxTime = 0.0;
    for (const auto& e : allEvents) {
        if (e.endTime > maxTime) {
            maxTime = e.endTime;
        }
    }

    return maxTime;
}

//=============================================================================
// Calculate bandwidth utilization
//=============================================================================
double Synthesizer4::calculateBandwidthUtilization(
    const std::vector<TransferEvent4>& events,
    double makespan) const {

    if (makespan <= 0.0 || events.empty()) {
        return 0.0;
    }

    std::vector<double> linkBusyTime(totalNodeCount_ * totalNodeCount_, 0.0);
    std::vector<bool> linkUsed(totalNodeCount_ * totalNodeCount_, false);
    int linkCount = 0;

    for (const auto& e : events) {
        const int idx = edgeIndex(e.src, e.dst);
        linkBusyTime[idx] += e.endTime - e.startTime;
        if (!linkUsed[idx]) {
            linkUsed[idx] = true;
            ++linkCount;
        }
    }

    if (linkCount == 0) {
        return 0.0;
    }

    double totalUtilization = 0.0;
    for (int idx = 0; idx < static_cast<int>(linkBusyTime.size()); ++idx) {
        if (linkUsed[idx]) {
            totalUtilization += linkBusyTime[idx] / makespan;
        }
    }

    return totalUtilization / linkCount;
}

//=============================================================================
// Print per-link busy intervals (ns) and per-link utilization
//=============================================================================
void Synthesizer4::printLinkBusyIntervals(
    const std::vector<TransferEvent4>& events,
    double makespan,
    const std::string& scopeName) const {

    if (events.empty() || makespan <= 0.0) {
        return;
    }

    std::vector<std::vector<std::pair<double, double>>> linkIntervals(totalNodeCount_ * totalNodeCount_);
    std::vector<double> linkBusyTime(totalNodeCount_ * totalNodeCount_, 0.0);

    for (const auto& e : events) {
        const int idx = edgeIndex(e.src, e.dst);
        linkIntervals[idx].push_back({e.startTime, e.endTime});
        linkBusyTime[idx] += (e.endTime - e.startTime);
    }

    std::cout << "[" << scopeName << "] Link Busy Intervals (ns) and Utilization:\n";

    for (int src = 0; src < totalNodeCount_; ++src) {
        for (int dst = 0; dst < totalNodeCount_; ++dst) {
            auto& intervals = linkIntervals[edgeIndex(src, dst)];
            if (intervals.empty()) {
                continue;
            }

            std::sort(intervals.begin(), intervals.end(),
                      [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
                          if (a.first != b.first) {
                              return a.first < b.first;
                          }
                          return a.second < b.second;
                      });

            std::cout << "  Link(" << formatNodeName(src)
                      << "->" << formatNodeName(dst) << "): intervals=[";

            for (size_t k = 0; k < intervals.size(); ++k) {
                const auto seg = intervals[k];
                const long long startNs = static_cast<long long>(std::llround(seg.first * 1000.0));
                const long long endNs = static_cast<long long>(std::llround(seg.second * 1000.0));
                std::cout << "[" << startNs << ", " << endNs << "]";
                if (k + 1 < intervals.size()) {
                    std::cout << ", ";
                }
            }

            const double util = linkBusyTime[edgeIndex(src, dst)] / makespan;
            std::cout << "], utilization=" << std::fixed << std::setprecision(2)
                      << (util * 100.0) << "%\n";
        }
    }
}

//=============================================================================
// Print step-by-step schedule
//=============================================================================
void Synthesizer4::printStepByStepSchedule(
    const std::vector<TransferEvent4>& events,
    const std::string& matrixName) const {
    
    if (events.empty()) {
        return;
    }
    
    std::cout << "\n[" << matrixName << " - Step-by-Step Link Schedule]\n";
    std::cout << "Format: [Time] Link(src->dst): ChunkID, Bytes\n";
    
    // Sort events by time
    std::vector<TransferEvent4> sortedEvents = events;
    std::sort(sortedEvents.begin(), sortedEvents.end(),
              [](const TransferEvent4& a, const TransferEvent4& b) {
                  if (a.startTime != b.startTime) return a.startTime < b.startTime;
                  if (a.src != b.src) return a.src < b.src;
                  return a.dst < b.dst;
              });
    
    // Group events by time (with small epsilon for grouping)
    const double TIME_EPSILON = 0.001;  // 1 nanosecond
    double currentTime = -1.0;
    
    for (const auto& e : sortedEvents) {
        // Check if we need a new time group
        if (std::abs(e.startTime - currentTime) > TIME_EPSILON) {
            currentTime = e.startTime;
            std::cout << "\n[Time " << std::fixed << std::setprecision(6) 
                      << currentTime << " us]\n";
        }
        
        // Print link with formatted node names
        std::string srcName = formatNodeName(e.src);
        std::string dstName = formatNodeName(e.dst);
        std::cout << "  Link(" << srcName << "->" << dstName << "): "
                  << "Chunk " << e.chunkId << ", " << e.bytes << " bytes";
        
        // Print full path if available and different from direct link
        if (!e.path.empty() && e.path.size() > 2) {
            std::cout << " [Path: ";
            for (size_t i = 0; i < e.path.size(); ++i) {
                std::cout << formatNodeName(e.path[i]);
                if (i + 1 < e.path.size()) std::cout << "->";
            }
            std::cout << "]";
        }
        std::cout << "\n";
    }
    
    std::cout << "\n";
}

//=============================================================================
// Fused Scheduling: Schedule from scratch with mixed network state
//=============================================================================
std::vector<TransferEvent4> Synthesizer4::fusedScheduling(
    const std::vector<std::vector<DataSize>>& regularMatrix,
    const std::vector<std::vector<DataSize>>& hotspotMatrix) {
    
    if (verbose_) {
        std::cout << "\n[Fused Scheduling - Step 4] Scheduling from scratch with mixed network state\n";
    }
    
    std::vector<TransferEvent4> events;
    
    struct FlowInfo {
        int src;
        int dst;
        DataSize bytes;
        int distance;
        bool isRegularMatrix;
        double priority;
    };

    struct FlowQuery {
        int src;
        int dst;
        DataSize bytes;
        bool isRegularMatrix;
    };
    
    std::vector<FlowInfo> allFlows;
    
    // Collect regular matrix flows (priority 1)
    for (int i = 0; i < npusCount_; ++i) {
        for (int j = 0; j < npusCount_; ++j) {
            if (regularMatrix[i][j] > 0) {
                int distance = computeDistance(i, j);
                allFlows.push_back({i, j, regularMatrix[i][j], distance, true,
                                   static_cast<double>(distance)});
            }
        }
    }
    
    // Collect hotspot matrix flows (priority 2)
    for (int i = 0; i < npusCount_; ++i) {
        for (int j = 0; j < npusCount_; ++j) {
            if (hotspotMatrix[i][j] > 0) {
                int distance = computeDistance(i, j);
                double priority = static_cast<double>(hotspotMatrix[i][j]) * distance;
                allFlows.push_back({i, j, hotspotMatrix[i][j], distance, false, priority});
            }
        }
    }
    
    // Sort flows: hotspot matrix first (by priority descending), then regular matrix (by distance descending)
    std::sort(allFlows.begin(), allFlows.end(),
              [](const FlowInfo& a, const FlowInfo& b) {
                  if (a.isRegularMatrix != b.isRegularMatrix) {
                      return a.isRegularMatrix < b.isRegularMatrix;
                  }
                  return a.priority > b.priority;
              });

    std::vector<FlowQuery> queries;
    queries.reserve(allFlows.size());
    for (const auto& flow : allFlows) {
        if (flow.isRegularMatrix) {
            queries.push_back({flow.src, flow.dst, flow.bytes, true});
            continue;
        }

        const int numChunks = computeChunkCount(flow.src, flow.dst, flow.bytes);
        const DataSize chunkSize = flow.bytes / numChunks;
        const DataSize remainder = flow.bytes % numChunks;
        for (int c = 0; c < numChunks; ++c) {
            DataSize currentChunkSize = chunkSize;
            if (c < remainder) {
                ++currentChunkSize;
            }
            queries.push_back({flow.src, flow.dst, currentChunkSize, false});
        }
    }

    std::vector<std::vector<std::vector<int>>> precomputedPaths(queries.size());
    const unsigned hardwareThreads = std::thread::hardware_concurrency();
    const unsigned workerCount =
        (queries.size() >= 512 && totalNodeCount_ >= 32 && hardwareThreads > 1)
            ? std::min<unsigned>(8, std::min<unsigned>(hardwareThreads, static_cast<unsigned>(queries.size())))
            : 1u;

    const bool useParallelPrecompute = workerCount > 1;
    if (useParallelPrecompute) {
        std::vector<std::thread> workers;
        workers.reserve(workerCount);
        const size_t blockSize = (queries.size() + workerCount - 1) / workerCount;

        for (unsigned worker = 0; worker < workerCount; ++worker) {
            const size_t begin = worker * blockSize;
            const size_t end = std::min(queries.size(), begin + blockSize);
            if (begin >= end) {
                break;
            }

            workers.emplace_back([this, &queries, &precomputedPaths, begin, end]() {
                for (size_t i = begin; i < end; ++i) {
                    const auto& query = queries[i];
                    precomputedPaths[i] = computeTimeShortestPathsUncached(query.src, query.dst, query.bytes);
                }
            });
        }

        for (auto& worker : workers) {
            worker.join();
        }
    }
    
    std::vector<double> accumulatedLinkLoad(totalNodeCount_ * totalNodeCount_, 0.0);
    double maxAccumulatedLinkLoad = 0.0;
    int chunkId = 0;

    for (size_t queryIndex = 0; queryIndex < queries.size(); ++queryIndex) {
        const auto& query = queries[queryIndex];
        auto allPaths = useParallelPrecompute
            ? precomputedPaths[queryIndex]
            : findTimeShortestPaths(query.src, query.dst, query.bytes);

        if (allPaths.empty()) {
            if (query.isRegularMatrix) {
                allPaths = findAllShortestPaths(query.src, query.dst);
            } else {
                int path_k = 3;
                allPaths = findKShortestPaths(query.src, query.dst, path_k);
                if (allPaths.empty()) {
                    allPaths = findAllShortestPaths(query.src, query.dst);
                }
            }
        }
        if (allPaths.empty()) {
            continue;
        }

        std::vector<int> path = query.isRegularMatrix
            ? selectBestPathForRegular(allPaths, accumulatedLinkLoad, maxAccumulatedLinkLoad, query.bytes)
            : selectBestPathByLoad(allPaths, accumulatedLinkLoad, maxAccumulatedLinkLoad, query.bytes);
        if (path.size() < 2) {
            continue;
        }

        for (size_t hop = 0; hop < path.size() - 1; ++hop) {
            const int hopSrc = path[hop];
            const int hopDst = path[hop + 1];
            const int edgeIdx = edgeIndex(hopSrc, hopDst);
            const double hopTransferTime = computeLinkTransferTime(hopSrc, hopDst, query.bytes);
            TransferEvent4 event;
            event.src = hopSrc;
            event.dst = hopDst;
            event.bytes = query.bytes;
            event.startTime = 0.0;
            event.endTime = 0.0;
            event.path = path;
            event.chunkId = chunkId;
            event.flowSrc = query.src;
            event.flowDst = query.dst;
            event.isRegularMatrix = query.isRegularMatrix;
            events.push_back(event);
            accumulatedLinkLoad[edgeIdx] += hopTransferTime;
            maxAccumulatedLinkLoad = std::max(maxAccumulatedLinkLoad, accumulatedLinkLoad[edgeIdx]);
        }

        ++chunkId;
    }
    
    double fusedMakespan = simulateFusedSchedule(events);
    validateScheduleOrThrow(events, "Fused Scheduling (Step 4)");
    
    if (verbose_) {
        std::cout << "  Scheduled " << events.size() << " transfer events from scratch\n";
        std::cout << "  Fused scheduling makespan: " << std::fixed << std::setprecision(2) << fusedMakespan << " us\n";
        
        double utilization = calculateBandwidthUtilization(events, fusedMakespan);
        std::cout << "  Average Bandwidth Utilization: " << std::fixed << std::setprecision(2)
                  << (utilization * 100.0) << "%\n";
    }
    
    printLinkBusyIntervals(events, fusedMakespan, "Fused Scheduling (Step 4)");
    
    return events;
}

//=============================================================================
// Main solve function
//=============================================================================
Synthesizer4::Time Synthesizer4::solve(
    const std::vector<std::vector<DataSize>>& demand) {
    
    if (verbose_) {
        std::cout << "\n";
        std::cout << "========================================================\n";
        std::cout << "         SYNTHESIZER4: FOUR-STEP WORKFLOW               \n";
        std::cout << "========================================================\n";
    }
    
    // Step 1: Profiling
    if (verbose_) {
        std::cout << "\n[Step 1] Profiling: Hotspot-Based Matrix Decomposition\n";
        std::cout << "========================================================\n";
    }
    
    profilingResult_ = profileMatrix(demand);
    
    if (verbose_) {
        profilingResult_.print();
        profilingResult_.printMatrices(npusCount_);
    }
    
    // Step 2 & Step 3: Scheduling and Fusion (skipped in clean mode)
    if (!cleanMode_) {
        if (verbose_) {
            std::cout << "\n[Step 2] Scheduling: Separate Matrix Scheduling\n";
            std::cout << "========================================================\n";
        }
        
        auto regularEvents = scheduleRegularMatrix(profilingResult_.regularMatrix);
        auto hotspotEvents = scheduleHotspotMatrix(profilingResult_.hotspotMatrix);
        
        if (verbose_) {
            std::cout << "\n[Step 3] Fusion: Merging Schedules (Regular First, Then Hotspot)\n";
            std::cout << "========================================================\n";
        }
        
        scheduleResult_ = fuseSchedules(regularEvents, hotspotEvents);
    } else {
        // In clean mode, initialize scheduleResult_ fields that would normally be set by fuseSchedules
        scheduleResult_.regularMatrixMakespan = 0.0;
        scheduleResult_.hotspotMatrixMakespan = 0.0;
        scheduleResult_.fusedMakespan = 0.0;
        scheduleResult_.events.clear();
    }
    
    // Step 4: Fused Scheduling (from scratch with mixed network state)
    if (verbose_) {
        std::cout << "\n[Step 4] Fused Scheduling: Scheduling from scratch with mixed network state\n";
        std::cout << "========================================================\n";
    }
    
    auto fusedSchedulingEvents = fusedScheduling(
        profilingResult_.regularMatrix, 
        profilingResult_.hotspotMatrix);
    
    // Update schedule result with fused scheduling makespan
    double fusedSchedulingMakespan = 0.0;
    if (!fusedSchedulingEvents.empty()) {
        for (const auto& e : fusedSchedulingEvents) {
            if (e.endTime > fusedSchedulingMakespan) {
                fusedSchedulingMakespan = e.endTime;
            }
        }
    }
    scheduleResult_.fusedSchedulingMakespan = fusedSchedulingMakespan;
    
    // Use fused scheduling result as final schedule
    scheduleResult_.events = fusedSchedulingEvents;
    scheduleResult_.makespan = fusedSchedulingMakespan;
    
    // Store formatNodeName function in scheduleResult for later use (e.g., in main.cpp)
    scheduleResult_.formatNodeNameFunc = [this](int nodeID) { return formatNodeName(nodeID); };
    
    if (verbose_ && !cleanMode_) {
        scheduleResult_.print();
    }
    
    if (printSchedule_ && !cleanMode_) {
        scheduleResult_.printEvents();
    }
    
    return scheduleResult_.makespan;
}

} // namespace tacos

