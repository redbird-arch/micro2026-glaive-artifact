/*
# File name  :    synthesizer_3.cpp
# Author     :    Galois
# Time       :    2026/01/14
# Description:    Implementation of Synthesizer3 interface
*/

#include "synthesizer_3.h"
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
#include <cstdlib>

namespace tacos {

namespace {

double readPositiveEnvDouble(const char* name, double fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return fallback;
    }

    char* end = nullptr;
    const double value = std::strtod(raw, &end);
    if (end == raw || value <= 0.0) {
        return fallback;
    }
    return value;
}

int readPositiveEnvInt(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return fallback;
    }

    char* end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || value <= 0) {
        return fallback;
    }
    return static_cast<int>(value);
}

int configuredMaxBandwidthFlows(int npusCount) {
    const int absoluteLimit = readPositiveEnvInt("GLAIVE_MAX_BANDWIDTH_FLOWS", -1);
    if (absoluteLimit > 0) {
        return std::max(1, absoluteLimit);
    }

    const double multiplier = readPositiveEnvDouble("GLAIVE_MAX_BANDWIDTH_FLOW_MULT", 4.0);
    return std::max(1, static_cast<int>(std::ceil(multiplier * npusCount)));
}

}  // namespace

Synthesizer3::Synthesizer3(const std::vector<int>& shape, bool isTorus,
                           Bandwidth bandwidth, Latency latency)
    : Synthesizer3(shape,
                   isTorus ? DirectTopologyKind::Torus : DirectTopologyKind::Mesh,
                   bandwidth,
                   latency) {
}

Synthesizer3::Synthesizer3(const std::vector<int>& shape,
                           DirectTopologyKind directTopologyKind,
                           Bandwidth bandwidth,
                           Latency latency)
    : shape_(shape), isTorus_(directTopologyKind == DirectTopologyKind::Torus),
      directTopologyKind_(directTopologyKind),
      npusCount_(0),
      gpuNodeCount_(0),
      bandwidth_(bandwidth),
      latency_(latency),
      topology_(nullptr),
      totalNodeCount_(0) {
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

Synthesizer3::Synthesizer3(std::shared_ptr<Topology> topology, const std::vector<int>& shape,
                           Bandwidth bandwidth, Latency latency)
    : shape_(shape), isTorus_(false),
      directTopologyKind_(DirectTopologyKind::Mesh),
      npusCount_(0),
      gpuNodeCount_(0),
      bandwidth_(bandwidth),
      latency_(latency),
      topology_(topology),
      totalNodeCount_(0) {
    
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

void Synthesizer3::setVerbose(bool verbose) {
    verbose_ = verbose;
}

void Synthesizer3::setPrintSchedule(bool printSchedule) {
    printSchedule_ = printSchedule;
}

int Synthesizer3::getHopIndex(const TransferEvent3& event) const {
    for (size_t hop = 0; hop + 1 < event.path.size(); ++hop) {
        if (event.path[hop] == event.src && event.path[hop + 1] == event.dst) {
            return static_cast<int>(hop);
        }
    }
    return std::numeric_limits<int>::max();
}

void Synthesizer3::sortEventsForScheduling(std::vector<TransferEvent3>& events) const {
    std::stable_sort(events.begin(), events.end(),
                     [this](const TransferEvent3& a, const TransferEvent3& b) {
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

void Synthesizer3::validateScheduleOrThrow(
    const std::vector<TransferEvent3>& events,
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

    std::map<std::pair<int, int>, std::vector<const TransferEvent3*>> edgeEvents;
    std::map<int, std::vector<const TransferEvent3*>> chunkEvents;

    for (const auto& event : events) {
        edgeEvents[{event.src, event.dst}].push_back(&event);
        if (event.chunkId >= 0) {
            chunkEvents[event.chunkId].push_back(&event);
        }
    }

    for (auto& [edge, edgeSchedule] : edgeEvents) {
        std::sort(edgeSchedule.begin(), edgeSchedule.end(),
                  [](const TransferEvent3* a, const TransferEvent3* b) {
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
                  [this](const TransferEvent3* a, const TransferEvent3* b) {
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
std::vector<int> Synthesizer3::nodeIDToCoordinate(int nodeID) const {
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

int Synthesizer3::coordinateToNodeID(const std::vector<int>& coord) const {
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
int Synthesizer3::computeNodeDegree(int nodeID) const {
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
        if (directTopologyKind_ == DirectTopologyKind::FullMesh) {
            int degree = 0;
            for (int dimSize : shape_) {
                degree += std::max(0, dimSize - 1);
            }
            return degree;
        }

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

int Synthesizer3::computeMinDegree() const {
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
void Synthesizer3::buildConnectionGraph() {
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
        // Direct-connect topology: build graph only for GPU nodes.
        // FullMesh shortest paths are generated formulaically, so we avoid materializing
        // O(N^2) neighbor lists for large scalability runs.
        connectionGraph_.clear();
        connectionGraph_.resize(npusCount_);

        if (directTopologyKind_ == DirectTopologyKind::FullMesh) {
            bfsPathCache_.clear();
            shortestPathCache_.clear();
            timeShortestPathCache_.clear();
            initializeStaticCaches();
            return;
        }

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

void Synthesizer3::initializeStaticCaches() {
    totalNodeCount_ = static_cast<int>(connectionGraph_.size());
    if (topology_ == nullptr) {
        distanceMatrix_.clear();
        linkLatencyUs_.clear();
        linkBandwidthBytesPerUs_.clear();
        return;
    }

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
            linkLatencyUs_[idx] = topology_->latency(src, dst) / 1000.0;
            linkBandwidthBytesPerUs_[idx] = topology_->bandwidth(src, dst) * (1 << 30) / 1e6;
        }
    }
}

int Synthesizer3::edgeIndex(int src, int dst) const {
    return src * totalNodeCount_ + dst;
}

std::uint64_t Synthesizer3::pairCacheKey(int src, int dst) const {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(src)) << 32) |
           static_cast<std::uint32_t>(dst);
}

std::uint64_t Synthesizer3::chunkArrivalKey(int node, int chunkId) const {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(node)) << 32) |
           static_cast<std::uint32_t>(chunkId);
}

bool Synthesizer3::usesFormulaRouting() const {
    return topology_ == nullptr;
}

//=============================================================================
// Check if node is a GPU node
//=============================================================================
bool Synthesizer3::isGPUNode(int nodeID) const {
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
bool Synthesizer3::isSwitchNode(int nodeID) const {
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
const std::vector<int>& Synthesizer3::getNeighborsFromGraph(int node) const {
    if (node < 0 || node >= static_cast<int>(connectionGraph_.size())) {
        static const std::vector<int> empty;
        return empty;
    }
    return connectionGraph_[node];
}

//=============================================================================
// Format node name for display
//=============================================================================
std::string Synthesizer3::formatNodeName(int nodeID) const {
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
// Compute bandwidth-latency threshold
//=============================================================================
double Synthesizer3::computeBwLatThreshold() const {
    // Convert latency from ns to us
    double latencyUs = latency_ / 1000.0;
    
    // Convert bandwidth from GB/s to bytes/us
    // GB = 2^30 bytes, 1 second = 1e6 microseconds
    double bandwidthBytesPerUs = bandwidth_ * (1 << 30) / 1e6;
    
    // Threshold = bandwidth * latency * min_degree
    // This represents the data size where latency and bandwidth contribute equally
    int minDegree = computeMinDegree();
    
    const double thresholdScale = readPositiveEnvDouble("GLAIVE_BWLAT_THRESHOLD_SCALE", 1.0);
    double threshold = bandwidthBytesPerUs * latencyUs * minDegree * thresholdScale;
    
    if (verbose_) {
        std::cout << "\n[Profiling] Computing Bandwidth-Latency Threshold\n";
        std::cout << "  Bandwidth: " << bandwidth_ << " GB/s = " 
                  << bandwidthBytesPerUs << " bytes/us\n";
        std::cout << "  Latency: " << latency_ << " ns = " 
                  << latencyUs << " us\n";
        std::cout << "  Min Degree: " << minDegree << "\n";
        std::cout << "  Threshold Scale: " << thresholdScale << "\n";
        std::cout << "  Threshold: " << threshold << " bytes\n";
    }
    
    return threshold;
}

//=============================================================================
// Profiling phase: Decompose matrix
//=============================================================================
ProfilingResult Synthesizer3::profileMatrix(
    const std::vector<std::vector<DataSize>>& demand) {
    
    ProfilingResult result;
    
    // Compute threshold
    result.bwLatThreshold = computeBwLatThreshold();
    
    // Initialize matrices
    result.latencyMatrix.clear();
    result.latencyMatrix.resize(npusCount_);
    for (int i = 0; i < npusCount_; ++i) {
        result.latencyMatrix[i].resize(npusCount_, 0);
    }
    
    result.bandwidthMatrix.clear();
    result.bandwidthMatrix.resize(npusCount_);
    for (int i = 0; i < npusCount_; ++i) {
        result.bandwidthMatrix[i].resize(npusCount_, 0);
    }
    
    // Collect bandwidth matrix candidates (data > threshold)
    struct FlowCandidate {
        int src;
        int dst;
        DataSize bytes;
    };
    
    std::vector<FlowCandidate> bandwidthCandidates;
    
    for (int i = 0; i < npusCount_; ++i) {
        for (int j = 0; j < npusCount_; ++j) {
            if (i == j || demand[i][j] == 0) continue;
            
            if (demand[i][j] > result.bwLatThreshold) {
                // Bandwidth-dominant flow candidate
                bandwidthCandidates.push_back({i, j, demand[i][j]});
            } else {
                // Latency-dominant flow (directly assign to latency matrix)
                result.latencyMatrix[i][j] = demand[i][j];
                result.latencyFlows.push_back({i, j, demand[i][j]});
            }
        }
    }
    
    // Sort bandwidth candidates by bytes (descending) - select top-N largest data transfers
    std::sort(bandwidthCandidates.begin(), bandwidthCandidates.end(),
              [](const FlowCandidate& a, const FlowCandidate& b) {
                  return a.bytes > b.bytes;  // Sort by bytes size, largest first
              });
    
    // Select the configured number of flows for the bandwidth matrix.
    // int maxBandwidthFlows = npusCount_;
    int maxBandwidthFlows = configuredMaxBandwidthFlows(npusCount_);
    int selectedCount = std::min(static_cast<int>(bandwidthCandidates.size()), maxBandwidthFlows);
    
    // Track which candidates were selected for bandwidth matrix
    std::set<std::pair<int, int>> selectedForBandwidth;
    
    for (int i = 0; i < selectedCount; ++i) {
        const auto& candidate = bandwidthCandidates[i];
        result.bandwidthMatrix[candidate.src][candidate.dst] = candidate.bytes;
        result.bandwidthFlows.push_back({candidate.src, candidate.dst, candidate.bytes});
        selectedForBandwidth.insert({candidate.src, candidate.dst});
    }
    
    // Put unselected bandwidth candidates into latency matrix
    // This ensures: latencyMatrix + bandwidthMatrix = original demand matrix
    for (int i = selectedCount; i < static_cast<int>(bandwidthCandidates.size()); ++i) {
        const auto& candidate = bandwidthCandidates[i];
        result.latencyMatrix[candidate.src][candidate.dst] = candidate.bytes;
        result.latencyFlows.push_back({candidate.src, candidate.dst, candidate.bytes});
    }

    finalizeProfilingStats(result);
    
    return result;
}

ProfilingResult Synthesizer3::profileFlows(const std::vector<DemandEntry3>& demand) {
    ProfilingResult result;
    result.bwLatThreshold = computeBwLatThreshold();

    struct FlowCandidate {
        int src;
        int dst;
        DataSize bytes;
    };

    std::vector<FlowCandidate> bandwidthCandidates;
    for (const auto& flow : demand) {
        if (flow.src == flow.dst || flow.bytes <= 0) {
            continue;
        }

        if (flow.bytes > result.bwLatThreshold) {
            bandwidthCandidates.push_back({flow.src, flow.dst, flow.bytes});
        } else {
            result.latencyFlows.push_back(flow);
        }
    }

    std::sort(bandwidthCandidates.begin(), bandwidthCandidates.end(),
              [](const FlowCandidate& a, const FlowCandidate& b) {
                  return a.bytes > b.bytes;
              });

    const int maxBandwidthFlows = configuredMaxBandwidthFlows(npusCount_);
    const int selectedCount = std::min(static_cast<int>(bandwidthCandidates.size()), maxBandwidthFlows);
    for (int i = 0; i < selectedCount; ++i) {
        const auto& candidate = bandwidthCandidates[i];
        result.bandwidthFlows.push_back({candidate.src, candidate.dst, candidate.bytes});
    }
    for (int i = selectedCount; i < static_cast<int>(bandwidthCandidates.size()); ++i) {
        const auto& candidate = bandwidthCandidates[i];
        result.latencyFlows.push_back({candidate.src, candidate.dst, candidate.bytes});
    }

    finalizeProfilingStats(result);
    return result;
}

std::vector<DemandEntry3> Synthesizer3::collectMatrixFlows(
    const std::vector<std::vector<DataSize>>& matrix) const {
    std::vector<DemandEntry3> flows;
    for (int i = 0; i < npusCount_; ++i) {
        for (int j = 0; j < npusCount_; ++j) {
            if (i == j || matrix[i][j] <= 0) {
                continue;
            }
            flows.push_back({i, j, matrix[i][j]});
        }
    }
    return flows;
}

void Synthesizer3::finalizeProfilingStats(ProfilingResult& result) const {
    result.latencyMatrixNonZeros = static_cast<int>(result.latencyFlows.size());
    result.latencyMatrixTotalBytes = 0;
    for (const auto& flow : result.latencyFlows) {
        result.latencyMatrixTotalBytes += flow.bytes;
    }

    result.bandwidthMatrixNonZeros = static_cast<int>(result.bandwidthFlows.size());
    result.bandwidthMatrixTotalBytes = 0;
    for (const auto& flow : result.bandwidthFlows) {
        result.bandwidthMatrixTotalBytes += flow.bytes;
    }
}

//=============================================================================
// Compute distance (hop count) between two nodes
//=============================================================================
int Synthesizer3::computeDistance(int src, int dst) const {
    if (src == dst) return 0;

    if (usesFormulaRouting()) {
        if (src < 0 || dst < 0 || src >= totalNodeCount_ || dst >= totalNodeCount_) {
            return std::numeric_limits<int>::max();
        }

        auto srcCoord = nodeIDToCoordinate(src);
        auto dstCoord = nodeIDToCoordinate(dst);
        if (directTopologyKind_ == DirectTopologyKind::FullMesh) {
            int differingDims = 0;
            for (size_t d = 0; d < srcCoord.size(); ++d) {
                if (srcCoord[d] != dstCoord[d]) {
                    ++differingDims;
                }
            }
            return differingDims;
        }

        int distance = 0;
        for (size_t d = 0; d < srcCoord.size(); ++d) {
            const int diff = std::abs(dstCoord[d] - srcCoord[d]);
            if (directTopologyKind_ == DirectTopologyKind::Torus) {
                distance += std::min(diff, shape_[d] - diff);
            } else {
                distance += diff;
            }
        }
        return distance;
    }

    if (src < 0 || dst < 0 || src >= totalNodeCount_ || dst >= totalNodeCount_ ||
        distanceMatrix_.empty()) {
        return std::numeric_limits<int>::max();
    }

    return distanceMatrix_[edgeIndex(src, dst)];
}

//=============================================================================
// Compute shortest path using BFS
//=============================================================================
std::vector<int> Synthesizer3::bfsPath(int src, int dst) const {
    if (src == dst) {
        return {src};
    }

    if (usesFormulaRouting()) {
        auto directPaths = enumerateDirectShortestPaths(src, dst);
        if (!directPaths.empty()) {
            return directPaths.front();
        }
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
std::vector<int> Synthesizer3::getNeighbors(int node) const {
    if (usesFormulaRouting() && directTopologyKind_ == DirectTopologyKind::FullMesh) {
        std::vector<int> neighbors;
        if (node < 0 || node >= totalNodeCount_) {
            return neighbors;
        }
        auto coord = nodeIDToCoordinate(node);
        for (size_t d = 0; d < shape_.size(); ++d) {
            for (int value = 0; value < shape_[d]; ++value) {
                if (value == coord[d]) {
                    continue;
                }
                auto nextCoord = coord;
                nextCoord[d] = value;
                neighbors.push_back(coordinateToNodeID(nextCoord));
            }
        }
        return neighbors;
    }

    // Use pre-built connection graph
    return getNeighborsFromGraph(node);
}

//=============================================================================
// Find all shortest paths between src and dst
//=============================================================================
std::vector<std::vector<int>> Synthesizer3::findAllShortestPaths(int src, int dst) const {
    if (usesFormulaRouting()) {
        return enumerateDirectShortestPaths(src, dst);
    }

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

std::vector<std::vector<int>> Synthesizer3::enumerateDirectShortestPaths(int src, int dst) const {
    constexpr std::size_t kMaxPaths = 10;

    std::vector<std::vector<int>> allPaths;
    if (src < 0 || dst < 0 || src >= totalNodeCount_ || dst >= totalNodeCount_) {
        return allPaths;
    }
    if (src == dst) {
        allPaths.push_back({src});
        return allPaths;
    }

    const auto key = pairCacheKey(src, dst);
    auto cached = shortestPathCache_.find(key);
    if (cached != shortestPathCache_.end()) {
        return cached->second;
    }

    auto srcCoord = nodeIDToCoordinate(src);
    auto dstCoord = nodeIDToCoordinate(dst);

    if (directTopologyKind_ == DirectTopologyKind::FullMesh) {
        std::vector<int> differingDims;
        for (size_t d = 0; d < srcCoord.size(); ++d) {
            if (srcCoord[d] != dstCoord[d]) {
                differingDims.push_back(static_cast<int>(d));
            }
        }

        if (differingDims.empty()) {
            allPaths.push_back({src});
        } else {
            std::sort(differingDims.begin(), differingDims.end());
            do {
                auto currentCoord = srcCoord;
                std::vector<int> path{src};
                for (int dim : differingDims) {
                    currentCoord[dim] = dstCoord[dim];
                    path.push_back(coordinateToNodeID(currentCoord));
                }
                allPaths.push_back(std::move(path));
                if (allPaths.size() >= kMaxPaths) {
                    break;
                }
            } while (std::next_permutation(differingDims.begin(), differingDims.end()));
        }

        shortestPathCache_.emplace(key, allPaths);
        return allPaths;
    }

    struct DirectionChoice {
        std::vector<int> remainingSteps;
        std::vector<int> stepDirections;
    };

    std::vector<DirectionChoice> directionChoices(1, DirectionChoice{
        std::vector<int>(shape_.size(), 0),
        std::vector<int>(shape_.size(), 0),
    });

    for (size_t d = 0; d < shape_.size(); ++d) {
        const int srcValue = srcCoord[d];
        const int dstValue = dstCoord[d];
        if (srcValue == dstValue) {
            continue;
        }

        std::vector<DirectionChoice> expandedChoices;
        for (const auto& choice : directionChoices) {
            if (directTopologyKind_ == DirectTopologyKind::Torus) {
                const int forward = (dstValue - srcValue + shape_[d]) % shape_[d];
                const int backward = (srcValue - dstValue + shape_[d]) % shape_[d];
                if (forward == backward) {
                    auto plusChoice = choice;
                    plusChoice.remainingSteps[d] = forward;
                    plusChoice.stepDirections[d] = 1;
                    expandedChoices.push_back(std::move(plusChoice));

                    auto minusChoice = choice;
                    minusChoice.remainingSteps[d] = backward;
                    minusChoice.stepDirections[d] = -1;
                    expandedChoices.push_back(std::move(minusChoice));
                    continue;
                }

                auto updatedChoice = choice;
                if (forward < backward) {
                    updatedChoice.remainingSteps[d] = forward;
                    updatedChoice.stepDirections[d] = 1;
                } else {
                    updatedChoice.remainingSteps[d] = backward;
                    updatedChoice.stepDirections[d] = -1;
                }
                expandedChoices.push_back(std::move(updatedChoice));
                continue;
            }

            auto updatedChoice = choice;
            updatedChoice.remainingSteps[d] = std::abs(dstValue - srcValue);
            updatedChoice.stepDirections[d] = (dstValue > srcValue) ? 1 : -1;
            expandedChoices.push_back(std::move(updatedChoice));
        }
        directionChoices = std::move(expandedChoices);
    }

    for (const auto& choice : directionChoices) {
        std::vector<int> currentPath{src};
        auto remaining = choice.remainingSteps;
        appendDirectShortestPaths(srcCoord,
                                  remaining,
                                  choice.stepDirections,
                                  currentPath,
                                  allPaths,
                                  kMaxPaths);
        if (allPaths.size() >= kMaxPaths) {
            break;
        }
    }

    shortestPathCache_.emplace(key, allPaths);
    return allPaths;
}

void Synthesizer3::appendDirectShortestPaths(
    const std::vector<int>& coord,
    std::vector<int>& remainingSteps,
    const std::vector<int>& stepDirections,
    std::vector<int>& currentPath,
    std::vector<std::vector<int>>& allPaths,
    std::size_t maxPaths) const {
    if (allPaths.size() >= maxPaths) {
        return;
    }

    bool done = true;
    for (int steps : remainingSteps) {
        if (steps > 0) {
            done = false;
            break;
        }
    }
    if (done) {
        allPaths.push_back(currentPath);
        return;
    }

    for (size_t d = 0; d < remainingSteps.size(); ++d) {
        if (remainingSteps[d] <= 0) {
            continue;
        }

        auto nextCoord = coord;
        nextCoord[d] += stepDirections[d];
        if (directTopologyKind_ == DirectTopologyKind::Torus) {
            nextCoord[d] = (nextCoord[d] % shape_[d] + shape_[d]) % shape_[d];
        }
        int nextNode = coordinateToNodeID(nextCoord);
        remainingSteps[d]--;
        currentPath.push_back(nextNode);
        appendDirectShortestPaths(nextCoord,
                                  remainingSteps,
                                  stepDirections,
                                  currentPath,
                                  allPaths,
                                  maxPaths);
        currentPath.pop_back();
        remainingSteps[d]++;

        if (allPaths.size() >= maxPaths) {
            return;
        }
    }
}

//=============================================================================
// Find time-shortest paths (based on transfer time, not just hop count)
// Supports heterogeneous link bandwidths
//=============================================================================
std::vector<std::vector<int>> Synthesizer3::computeTimeShortestPathsUncached(int src, int dst, DataSize bytes) const {
    constexpr double kPathEpsilon = 1e-9;
    constexpr std::size_t kMaxPaths = 10;

    std::vector<std::vector<int>> timeShortestPaths;

    if (usesFormulaRouting()) {
        return findAllShortestPaths(src, dst);
    }

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
std::vector<std::vector<int>> Synthesizer3::findTimeShortestPaths(int src, int dst, DataSize bytes) const {
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
// Select best path from multiple paths based on link load (OLD - using linkBusyUntil)
//=============================================================================
std::vector<int> Synthesizer3::selectBestPath(
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
std::vector<int> Synthesizer3::computeShortestPath(
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
std::vector<std::vector<int>> Synthesizer3::findKShortestPaths(int src, int dst, int k) const {
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
std::vector<std::vector<int>> Synthesizer3::findAlternativePaths(int src, int dst, int maxHops) const {
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
double Synthesizer3::evaluatePathCost(
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
std::vector<int> Synthesizer3::selectBestPathForLoadBalancing(
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
// Select best path based on accumulated link load and load balancing
// NEW: Uses accumulatedLinkLoad instead of linkBusyUntil
//=============================================================================
std::vector<int> Synthesizer3::selectBestPathByLoad(
    const std::vector<std::vector<int>>& paths,
    const std::unordered_map<int, double>& accumulatedLinkLoad,
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
            const auto it = accumulatedLinkLoad.find(idx);
            const double load = (it != accumulatedLinkLoad.end()) ? it->second : 0.0;
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
// Select best path for latency matrix (simplified: time-shortest, then min normalizedSumLoad)
//=============================================================================
std::vector<int> Synthesizer3::selectBestPathForLatency(
    const std::vector<std::vector<int>>& paths,
    const std::unordered_map<int, double>& accumulatedLinkLoad,
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
            const int idx = edgeIndex(path[i], path[i + 1]);
            const auto it = accumulatedLinkLoad.find(idx);
            pathSumLoad += (it != accumulatedLinkLoad.end()) ? it->second : 0.0;
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
// Compute transfer time using alpha-beta model
//=============================================================================
double Synthesizer3::computeTransferTime(DataSize bytes, int hops) const {
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
double Synthesizer3::computeLinkTransferTime(int src, int dst, DataSize bytes) const {
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

//=============================================================================
// Scheduling phase - Latency matrix
//=============================================================================
std::vector<TransferEvent3> Synthesizer3::scheduleLatencyMatrix(
    const std::vector<std::vector<DataSize>>& latencyMatrix) {
    return scheduleLatencyFlows(collectMatrixFlows(latencyMatrix));
}

std::vector<TransferEvent3> Synthesizer3::scheduleLatencyFlows(
    const std::vector<DemandEntry3>& latencyFlows) {
    
    if (verbose_) {
        std::cout << "\n[Scheduling] Processing Latency Matrix\n";
    }
    
    std::vector<TransferEvent3> events;
    
    // Collect all latency-dominant flows
    struct LatencyFlow {
        int src;
        int dst;
        DataSize bytes;
        int distance;
    };
    
    std::vector<LatencyFlow> flows;
    flows.reserve(latencyFlows.size());
    for (const auto& flow : latencyFlows) {
        if (flow.bytes <= 0) {
            continue;
        }
        int distance = computeDistance(flow.src, flow.dst);
        flows.push_back({flow.src, flow.dst, flow.bytes, distance});
    }
    
    // Sort by distance (farther first) to prioritize long-distance flows
    std::sort(flows.begin(), flows.end(),
              [](const LatencyFlow& a, const LatencyFlow& b) {
                  return a.distance > b.distance;
              });
    
    // SIMPLIFIED: For latency matrix, only track accumulated link load (no time slot allocation here)
    // Time slot allocation will be done together with bandwidth matrix later
    // Track accumulated link load: accumulatedLinkLoad[edge] = sum of transfer times on this link
    std::unordered_map<int, double> accumulatedLinkLoad;
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
        
        // SIMPLIFIED: Select time-shortest path, if multiple, choose one with min normalizedSumLoad
        std::vector<int> path = selectBestPathForLatency(allPaths, accumulatedLinkLoad, maxAccumulatedLinkLoad, flow.bytes);
        
        if (path.size() < 2) continue;
        
        // Create events for this path (without time slot allocation)
        // We'll just record the path and update accumulated load
        // Time scheduling will be done later together with bandwidth matrix
        for (size_t hop = 0; hop < path.size() - 1; ++hop) {
            int hopSrc = path[hop];
            int hopDst = path[hop + 1];
            const int edgeIdx = edgeIndex(hopSrc, hopDst);
            
            // Compute transfer time for this hop (using link-specific properties)
            double hopTransferTime = computeLinkTransferTime(hopSrc, hopDst, flow.bytes);
            
            // Create event with placeholder times (will be scheduled later)
            TransferEvent3 event;
            event.src = hopSrc;
            event.dst = hopDst;
            event.bytes = flow.bytes;
            event.startTime = 0.0;  // Placeholder, will be set during time slot allocation
            event.endTime = 0.0;    // Placeholder, will be set during time slot allocation
            event.path = path;
            event.chunkId = flowId;  // Use flowId as chunkId for latency matrix
            event.flowSrc = flow.src;
            event.flowDst = flow.dst;
            event.isLatencyMatrix = true;
            events.push_back(event);
            
            // Update accumulated link load (just sum transfer times)
            double& edgeLoad = accumulatedLinkLoad[edgeIdx];
            edgeLoad += hopTransferTime;
            maxAccumulatedLinkLoad = std::max(maxAccumulatedLinkLoad, edgeLoad);
        }
        
        flowId++;
    }
    
    // Allocate time slots for latency matrix events (for display purposes)
    // This is separate from the unified time slot allocation in fuseSchedules
    double makespan = allocateTimeSlotsForEvents(events);
    validateScheduleOrThrow(events, "Latency Matrix");
    
    if (verbose_) {
        std::cout << "  Scheduled " << events.size() << " transfer events\n";
        std::cout << "  Note: Each transfer event represents a single-hop transmission (from one node to an adjacent node)\n";
        std::cout << "  Latency matrix makespan: " << std::fixed << std::setprecision(2) << makespan << " us\n";
        
        // Calculate and print bandwidth utilization
        double utilization = calculateBandwidthUtilization(events, makespan);
        std::cout << "  Average Bandwidth Utilization: " << std::fixed << std::setprecision(2)
                  << (utilization * 100.0) << "%\n";
        
        // Print step-by-step schedule
        printStepByStepSchedule(events, "Latency Matrix");
    }
    
    return events;
}

//=============================================================================
// Compute minimum chunk size based on bandwidth * latency
//=============================================================================
DataSize Synthesizer3::computeMinChunkSize(int src, int dst) const {
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
// Compute chunk count for bandwidth matrix
//=============================================================================
int Synthesizer3::computeChunkCount(int src, int dst, DataSize bytes) const {
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
// Scheduling phase - Bandwidth matrix
//=============================================================================
std::vector<TransferEvent3> Synthesizer3::scheduleBandwidthMatrix(
    const std::vector<std::vector<DataSize>>& bandwidthMatrix) {
    return scheduleBandwidthFlows(collectMatrixFlows(bandwidthMatrix));
}

std::vector<TransferEvent3> Synthesizer3::scheduleBandwidthFlows(
    const std::vector<DemandEntry3>& bandwidthFlows) {
    
    if (verbose_) {
        std::cout << "\n[Scheduling] Processing Bandwidth Matrix\n";
    }
    
    std::vector<TransferEvent3> events;
    
    // Collect all bandwidth-dominant flows
    struct BandwidthFlow {
        int src;
        int dst;
        DataSize bytes;
        int distance;
        double priority;  // bytes * distance
    };
    
    std::vector<BandwidthFlow> flows;
    flows.reserve(bandwidthFlows.size());
    for (const auto& flow : bandwidthFlows) {
        if (flow.bytes <= 0) {
            continue;
        }
        int distance = computeDistance(flow.src, flow.dst);
        double priority = static_cast<double>(flow.bytes) * distance;
        flows.push_back({flow.src, flow.dst, flow.bytes, distance, priority});
    }
    
    // Sort by priority (descending): data * distance
    std::sort(flows.begin(), flows.end(),
              [](const BandwidthFlow& a, const BandwidthFlow& b) {
                  return a.priority > b.priority;
              });
    
    // NEW: Event-driven scheduling using accumulated link load
    // Time slot allocation will be done together with latency matrix later
    // Track accumulated link load: accumulatedLinkLoad[edge] = sum of transfer times on this link
    std::unordered_map<int, double> accumulatedLinkLoad;
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
            
            // Select best path based on accumulated link load (bandwidth matrix uses full load balancing)
            std::vector<int> path = selectBestPathByLoad(allPaths, accumulatedLinkLoad, maxAccumulatedLinkLoad, currentChunkSize);
            
            if (path.size() < 2) continue;
            
            // Create events for this path (without time slot allocation)
            // Time scheduling will be done later together with latency matrix
            for (size_t hop = 0; hop < path.size() - 1; ++hop) {
                int hopSrc = path[hop];
                int hopDst = path[hop + 1];
                const int edgeIdx = edgeIndex(hopSrc, hopDst);
                
                // Compute transfer time (using link-specific properties)
                double hopTransferTime = computeLinkTransferTime(hopSrc, hopDst, currentChunkSize);
                
                // Create event with placeholder times (will be scheduled later)
                TransferEvent3 event;
                event.src = hopSrc;
                event.dst = hopDst;
                event.bytes = currentChunkSize;
                event.startTime = 0.0;  // Placeholder, will be set during time slot allocation
                event.endTime = 0.0;    // Placeholder, will be set during time slot allocation
                event.path = path;
                event.chunkId = chunkId;
                event.flowSrc = flow.src;
                event.flowDst = flow.dst;
                event.isLatencyMatrix = false;
                events.push_back(event);
                
                // Update accumulated link load (just sum transfer times)
                double& edgeLoad = accumulatedLinkLoad[edgeIdx];
                edgeLoad += hopTransferTime;
                maxAccumulatedLinkLoad = std::max(maxAccumulatedLinkLoad, edgeLoad);
            }
            
            chunkId++;
        }
    }
    
    // Allocate time slots for bandwidth matrix events (for display purposes)
    // This is separate from the unified time slot allocation in fuseSchedules
    double makespan = allocateTimeSlotsForEvents(events);
    validateScheduleOrThrow(events, "Bandwidth Matrix");
    
    if (verbose_) {
        std::cout << "  Scheduled " << events.size() << " transfer events\n";
        std::cout << "  Note: Each transfer event represents a single-hop transmission (from one node to an adjacent node)\n";
        std::cout << "  Total chunks: " << chunkId << "\n";
        std::cout << "  Bandwidth matrix makespan: " << std::fixed << std::setprecision(2) << makespan << " us\n";
        
        // Calculate and print bandwidth utilization
        double utilization = calculateBandwidthUtilization(events, makespan);
        std::cout << "  Average Bandwidth Utilization: " << std::fixed << std::setprecision(2)
                  << (utilization * 100.0) << "%\n";
        
        // Print step-by-step schedule
        printStepByStepSchedule(events, "Bandwidth Matrix");
    }
    
    return events;
}

//=============================================================================
// Fusion phase: Merge schedules
//=============================================================================
ScheduleResult3 Synthesizer3::fuseSchedules(
    const std::vector<TransferEvent3>& latencyEvents,
    const std::vector<TransferEvent3>& bandwidthEvents) {
    
    if (verbose_) {
        std::cout << "\n[Fusion] Merging Latency and Bandwidth Schedules\n";
    }
    
    ScheduleResult3 result;
    
    // Compute individual makespans
    double latencyMakespan = 0.0;
    for (const auto& e : latencyEvents) {
        if (e.endTime > latencyMakespan) {
            latencyMakespan = e.endTime;
        }
    }
    
    double bandwidthMakespan = 0.0;
    for (const auto& e : bandwidthEvents) {
        if (e.endTime > bandwidthMakespan) {
            bandwidthMakespan = e.endTime;
        }
    }
    
    result.latencyMatrixMakespan = latencyMakespan;
    result.bandwidthMatrixMakespan = bandwidthMakespan;
    
    // Merge events and simulate with conflict resolution.
    // Keep latency events first, and offset bandwidth chunk IDs to avoid cross-matrix aliasing.
    std::vector<TransferEvent3> allEvents = latencyEvents;
    int nextChunkId = 0;
    for (const auto& event : allEvents) {
        nextChunkId = std::max(nextChunkId, event.chunkId + 1);
    }
    for (auto event : bandwidthEvents) {
        event.chunkId += nextChunkId;
        allEvents.push_back(std::move(event));
    }
    
    // Re-simulate with conflict resolution (this will update event times)
    double fusedMakespan = simulateFusedSchedule(allEvents);
    validateScheduleOrThrow(allEvents, "Fusion");
    
    // Sort by start time after fusion
    std::sort(allEvents.begin(), allEvents.end(),
              [](const TransferEvent3& a, const TransferEvent3& b) {
                  if (a.startTime != b.startTime) return a.startTime < b.startTime;
                  if (a.chunkId != b.chunkId) return a.chunkId < b.chunkId;
                  return a.src < b.src;
              });
    
    result.fusedMakespan = fusedMakespan;
    result.makespan = fusedMakespan;
    result.events = allEvents;
    
    if (verbose_) {
        std::cout << "  Latency matrix makespan: " << latencyMakespan << " us\n";
        std::cout << "  Bandwidth matrix makespan: " << bandwidthMakespan << " us\n";
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
double Synthesizer3::simulateFusedSchedule(
    std::vector<TransferEvent3>& allEvents) const {

    std::unordered_map<int, std::vector<std::pair<double, double>>> linkTimeSlots;
    std::unordered_map<std::uint64_t, double> dataArrivalTime;

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

        if (event.chunkId >= 0 && event.src != event.flowSrc) {
            const auto arrivalIt = dataArrivalTime.find(chunkArrivalKey(event.src, event.chunkId));
            if (arrivalIt != dataArrivalTime.end()) {
                earliestStartTime = arrivalIt->second;
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
        if (event.chunkId >= 0) {
            dataArrivalTime[chunkArrivalKey(event.dst, event.chunkId)] = event.endTime;
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
// Allocate time slots for events (for individual matrix scheduling display)
//=============================================================================
double Synthesizer3::allocateTimeSlotsForEvents(std::vector<TransferEvent3>& events) const {
    if (events.empty()) {
        return 0.0;
    }

    std::unordered_map<int, std::vector<std::pair<double, double>>> linkTimeSlots;
    std::unordered_map<std::uint64_t, double> dataArrivalTime;

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

        if (event.chunkId >= 0 && event.src != event.flowSrc) {
            const auto arrivalIt = dataArrivalTime.find(chunkArrivalKey(event.src, event.chunkId));
            if (arrivalIt != dataArrivalTime.end()) {
                earliestStartTime = arrivalIt->second;
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
        if (event.chunkId >= 0) {
            dataArrivalTime[chunkArrivalKey(event.dst, event.chunkId)] = event.endTime;
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
// Calculate bandwidth utilization
//=============================================================================
double Synthesizer3::calculateBandwidthUtilization(
    const std::vector<TransferEvent3>& events,
    double makespan) const {

    if (makespan <= 0.0 || events.empty()) {
        return 0.0;
    }

    std::unordered_map<int, double> linkBusyTime;

    for (const auto& e : events) {
        const int idx = edgeIndex(e.src, e.dst);
        linkBusyTime[idx] += e.endTime - e.startTime;
    }

    const int linkCount = static_cast<int>(linkBusyTime.size());
    if (linkCount == 0) {
        return 0.0;
    }

    double totalUtilization = 0.0;
    for (const auto& [idx, busyTime] : linkBusyTime) {
        totalUtilization += busyTime / makespan;
    }

    return totalUtilization / linkCount;
}

//=============================================================================
// Print per-link busy intervals (ns) and per-link utilization
//=============================================================================
void Synthesizer3::printLinkBusyIntervals(
    const std::vector<TransferEvent3>& events,
    double makespan,
    const std::string& scopeName) const {

    if (events.empty() || makespan <= 0.0) {
        return;
    }

    std::unordered_map<int, std::vector<std::pair<double, double>>> linkIntervals;
    std::unordered_map<int, double> linkBusyTime;

    for (const auto& e : events) {
        const int idx = edgeIndex(e.src, e.dst);
        linkIntervals[idx].push_back({e.startTime, e.endTime});
        linkBusyTime[idx] += (e.endTime - e.startTime);
    }

    std::cout << '[' << scopeName << "] Link Busy Intervals (ns) and Utilization:\n";

    std::vector<int> edgeIds;
    edgeIds.reserve(linkIntervals.size());
    for (const auto& [edgeId, _] : linkIntervals) {
        edgeIds.push_back(edgeId);
    }
    std::sort(edgeIds.begin(), edgeIds.end());

    for (int edgeId : edgeIds) {
        int src = edgeId / totalNodeCount_;
        int dst = edgeId % totalNodeCount_;
        auto& intervals = linkIntervals[edgeId];
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
            std::cout << '[' << startNs << ", " << endNs << ']';
            if (k + 1 < intervals.size()) {
                std::cout << ", ";
            }
        }

        const double util = linkBusyTime[edgeId] / makespan;
        std::cout << "], utilization=" << std::fixed << std::setprecision(2)
                  << (util * 100.0) << "%\n";
    }
}

//=============================================================================
// Print step-by-step schedule
//=============================================================================
void Synthesizer3::printStepByStepSchedule(
    const std::vector<TransferEvent3>& events,
    const std::string& matrixName) const {
    
    if (events.empty()) {
        return;
    }
    
    std::cout << "\n[" << matrixName << " - Step-by-Step Link Schedule]\n";
    std::cout << "Format: [Time] Link(src->dst): ChunkID, Bytes\n";
    
    // Sort events by time
    std::vector<TransferEvent3> sortedEvents = events;
    std::sort(sortedEvents.begin(), sortedEvents.end(),
              [](const TransferEvent3& a, const TransferEvent3& b) {
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
std::vector<TransferEvent3> Synthesizer3::fusedScheduling(
    const std::vector<std::vector<DataSize>>& latencyMatrix,
    const std::vector<std::vector<DataSize>>& bandwidthMatrix) {
    return fusedScheduling(collectMatrixFlows(latencyMatrix), collectMatrixFlows(bandwidthMatrix));
}

std::vector<TransferEvent3> Synthesizer3::fusedScheduling(
    const std::vector<DemandEntry3>& latencyFlows,
    const std::vector<DemandEntry3>& bandwidthFlows) {
    
    if (verbose_) {
        std::cout << "\n[Fused Scheduling - Step 4] Scheduling from scratch with mixed network state\n";
    }
    
    std::vector<TransferEvent3> events;
    
    struct FlowInfo {
        int src;
        int dst;
        DataSize bytes;
        int distance;
        bool isLatencyMatrix;
        double priority;
    };

    struct FlowQuery {
        int src;
        int dst;
        DataSize bytes;
        bool isLatencyMatrix;
    };
    
    std::vector<FlowInfo> allFlows;
    
    // Collect latency matrix flows (priority 1)
    allFlows.reserve(latencyFlows.size() + bandwidthFlows.size());
    for (const auto& flow : latencyFlows) {
        if (flow.bytes <= 0) {
            continue;
        }
        int distance = computeDistance(flow.src, flow.dst);
        allFlows.push_back({flow.src, flow.dst, flow.bytes, distance, true,
                           static_cast<double>(distance)});
    }
    
    // Collect bandwidth matrix flows (priority 2)
    for (const auto& flow : bandwidthFlows) {
        if (flow.bytes <= 0) {
            continue;
        }
        int distance = computeDistance(flow.src, flow.dst);
        double priority = static_cast<double>(flow.bytes) * distance;
        allFlows.push_back({flow.src, flow.dst, flow.bytes, distance, false, priority});
    }
    
    // Sort flows: bandwidth matrix first (by priority descending), then latency matrix (by distance descending)
    std::sort(allFlows.begin(), allFlows.end(),
              [](const FlowInfo& a, const FlowInfo& b) {
                  if (a.isLatencyMatrix != b.isLatencyMatrix) {
                      return a.isLatencyMatrix < b.isLatencyMatrix;
                  }
                  return a.priority > b.priority;
              });

    std::vector<FlowQuery> queries;
    queries.reserve(allFlows.size());
    for (const auto& flow : allFlows) {
        if (flow.isLatencyMatrix) {
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

    const bool useParallelPrecompute = workerCount > 1 && !usesFormulaRouting();
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
    
    std::unordered_map<int, double> accumulatedLinkLoad;
    double maxAccumulatedLinkLoad = 0.0;
    int chunkId = 0;

    for (size_t queryIndex = 0; queryIndex < queries.size(); ++queryIndex) {
        const auto& query = queries[queryIndex];
        auto allPaths = useParallelPrecompute
            ? precomputedPaths[queryIndex]
            : findTimeShortestPaths(query.src, query.dst, query.bytes);

        if (allPaths.empty()) {
            if (query.isLatencyMatrix) {
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

        std::vector<int> path = query.isLatencyMatrix
            ? selectBestPathForLatency(allPaths, accumulatedLinkLoad, maxAccumulatedLinkLoad, query.bytes)
            : selectBestPathByLoad(allPaths, accumulatedLinkLoad, maxAccumulatedLinkLoad, query.bytes);

        if (path.size() < 2) {
            continue;
        }

        for (size_t hop = 0; hop < path.size() - 1; ++hop) {
            const int hopSrc = path[hop];
            const int hopDst = path[hop + 1];
            const int edgeIdx = edgeIndex(hopSrc, hopDst);
            const double hopTransferTime = computeLinkTransferTime(hopSrc, hopDst, query.bytes);

            TransferEvent3 event;
            event.src = hopSrc;
            event.dst = hopDst;
            event.bytes = query.bytes;
            event.startTime = 0.0;
            event.endTime = 0.0;
            event.path = path;
            event.chunkId = chunkId;
            event.flowSrc = query.src;
            event.flowDst = query.dst;
            event.isLatencyMatrix = query.isLatencyMatrix;
            events.push_back(event);

            double& edgeLoad = accumulatedLinkLoad[edgeIdx];
            edgeLoad += hopTransferTime;
            maxAccumulatedLinkLoad = std::max(maxAccumulatedLinkLoad, edgeLoad);
        }

        ++chunkId;
    }
    
    double fusedMakespan = simulateFusedSchedule(events);
    validateScheduleOrThrow(events, "Fused Scheduling (Step 4)");
    
    if (cleanMode_ && printSchedule_) {
        printLinkBusyIntervals(events, fusedMakespan, "Fused Scheduling (Step 4)");
    }
    
    if (verbose_) {
        std::cout << "  Scheduled " << events.size() << " transfer events from scratch\n";
        std::cout << "  Fused scheduling makespan: " << fusedMakespan << " us\n";
        
        double utilization = calculateBandwidthUtilization(events, fusedMakespan);
        std::cout << "  Average Bandwidth Utilization: " << std::fixed << std::setprecision(2)
                  << (utilization * 100.0) << "%\n";
    }
    
    return events;
}

//=============================================================================
// Main solve function
//=============================================================================
Synthesizer3::Time Synthesizer3::solve(
    const std::vector<std::vector<DataSize>>& demand) {
    
    if (verbose_) {
        std::cout << "\n";
        std::cout << "========================================================\n";
        std::cout << "         SYNTHESIZER3: FOUR-STEP WORKFLOW               \n";
        std::cout << "========================================================\n";
    }
    
    // Step 1: Profiling
    if (verbose_) {
        std::cout << "\n[Step 1] Profiling: Matrix Decomposition\n";
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
        
        auto latencyEvents = scheduleLatencyFlows(profilingResult_.latencyFlows);
        auto bandwidthEvents = scheduleBandwidthFlows(profilingResult_.bandwidthFlows);
        
        if (verbose_) {
            std::cout << "\n[Step 3] Fusion: Merging Schedules\n";
            std::cout << "========================================================\n";
        }
        
        scheduleResult_ = fuseSchedules(latencyEvents, bandwidthEvents);
    } else {
        scheduleResult_.latencyMatrixMakespan = 0.0;
        scheduleResult_.bandwidthMatrixMakespan = 0.0;
        scheduleResult_.fusedMakespan = 0.0;
        scheduleResult_.events.clear();
    }
    
    // Step 4: Fused Scheduling (from scratch with mixed network state)
    if (verbose_) {
        std::cout << "\n[Step 4] Fused Scheduling: Scheduling from scratch with mixed network state\n";
        std::cout << "========================================================\n";
    }
    
    auto fusedSchedulingEvents = fusedScheduling(
        profilingResult_.latencyFlows,
        profilingResult_.bandwidthFlows);
    
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

Synthesizer3::Time Synthesizer3::solveSparse(
    const std::vector<DemandEntry3>& demand) {

    if (verbose_) {
        std::cout << "\n";
        std::cout << "========================================================\n";
        std::cout << "         SYNTHESIZER3: FOUR-STEP WORKFLOW               \n";
        std::cout << "========================================================\n";
    }

    if (verbose_) {
        std::cout << "\n[Step 1] Profiling: Matrix Decomposition\n";
        std::cout << "========================================================\n";
    }

    profilingResult_ = profileFlows(demand);

    if (verbose_) {
        profilingResult_.print();
    }

    if (!cleanMode_) {
        if (verbose_) {
            std::cout << "\n[Step 2] Scheduling: Separate Matrix Scheduling\n";
            std::cout << "========================================================\n";
        }

        auto latencyEvents = scheduleLatencyFlows(profilingResult_.latencyFlows);
        auto bandwidthEvents = scheduleBandwidthFlows(profilingResult_.bandwidthFlows);

        if (verbose_) {
            std::cout << "\n[Step 3] Fusion: Merging Schedules\n";
            std::cout << "========================================================\n";
        }

        scheduleResult_ = fuseSchedules(latencyEvents, bandwidthEvents);
    } else {
        scheduleResult_.latencyMatrixMakespan = 0.0;
        scheduleResult_.bandwidthMatrixMakespan = 0.0;
        scheduleResult_.fusedMakespan = 0.0;
        scheduleResult_.events.clear();
    }

    if (verbose_) {
        std::cout << "\n[Step 4] Fused Scheduling: Scheduling from scratch with mixed network state\n";
        std::cout << "========================================================\n";
    }

    auto fusedSchedulingEvents = fusedScheduling(
        profilingResult_.latencyFlows,
        profilingResult_.bandwidthFlows);

    double fusedSchedulingMakespan = 0.0;
    if (!fusedSchedulingEvents.empty()) {
        for (const auto& e : fusedSchedulingEvents) {
            if (e.endTime > fusedSchedulingMakespan) {
                fusedSchedulingMakespan = e.endTime;
            }
        }
    }
    scheduleResult_.fusedSchedulingMakespan = fusedSchedulingMakespan;
    scheduleResult_.events = fusedSchedulingEvents;
    scheduleResult_.makespan = fusedSchedulingMakespan;
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
