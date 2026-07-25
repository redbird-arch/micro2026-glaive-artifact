/*
# File name  :    standard_synthesizer.cpp
# Description:    Implementation of the standard-mode fast path.
*/

#include "standard_synthesizer.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>

namespace tacos {

namespace {
constexpr std::size_t kMaxPaths = 10;
constexpr long long kDenseEdgeLimit = 1LL << 20;
constexpr std::size_t kInitialSlotReserve = 16;
constexpr std::size_t kBinarySlotSearchThreshold = 64;
constexpr double kPathEpsilon = 1e-9;

bool shouldUseDenseEdges(int totalNodeCount) {
    const long long edges = static_cast<long long>(totalNodeCount) * totalNodeCount;
    return edges > 0 && edges <= kDenseEdgeLimit;
}
} // namespace

std::size_t StandardSynthesizer::TimeShortestPathCacheKeyHash::operator()(
    const TimeShortestPathCacheKey& key) const {
    std::size_t seed = std::hash<int>{}(key.src);
    seed ^= std::hash<int>{}(key.dst) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<DataSize>{}(key.bytes) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
}

StandardSynthesizer::SlotState::SlotState(int totalNodeCountIn, bool useDense)
    : totalNodeCount(totalNodeCountIn),
      dense(useDense) {
    if (dense) {
        const std::size_t edgeCount = static_cast<std::size_t>(totalNodeCount) * totalNodeCount;
        denseSlotIds.assign(edgeCount, -1);
        denseSlots.reserve(static_cast<std::size_t>(std::max(1, totalNodeCount)) * 4);
        denseSlotHints.reserve(static_cast<std::size_t>(std::max(1, totalNodeCount)) * 4);
    }
}

StandardSynthesizer::LoadState::LoadState(int totalNodeCount, bool denseEnabled)
    : useDense(denseEnabled) {
    if (useDense) {
        dense.assign(static_cast<std::size_t>(totalNodeCount) * totalNodeCount, 0.0);
    }
}

double StandardSynthesizer::LoadState::get(int edgeId) const {
    if (useDense) {
        return dense[edgeId];
    }
    const auto it = sparse.find(edgeId);
    return (it == sparse.end()) ? 0.0 : it->second;
}

void StandardSynthesizer::LoadState::add(int edgeId, double value) {
    if (useDense) {
        dense[edgeId] += value;
    } else {
        sparse[edgeId] += value;
    }
}

StandardSynthesizer::StandardSynthesizer(const std::vector<int>& shape,
                                       Synthesizer::DirectTopologyKind directTopologyKind,
                                       Bandwidth bandwidth,
                                       Latency latency)
    : shape_(shape),
      directTopologyKind_(directTopologyKind),
      isTorus_(directTopologyKind == Synthesizer::DirectTopologyKind::Torus),
      npusCount_(1),
      gpuNodeCount_(0),
      totalNodeCount_(0),
      bandwidth_(bandwidth),
      latency_(latency),
      topology_(nullptr) {
    for (int dim : shape_) {
        npusCount_ *= dim;
    }
    gpuNodeCount_ = npusCount_;
    buildConnectionGraph();
}

StandardSynthesizer::StandardSynthesizer(std::shared_ptr<Topology> topology,
                                       const std::vector<int>& shape,
                                       Bandwidth bandwidth,
                                       Latency latency)
    : shape_(shape),
      directTopologyKind_(Synthesizer::DirectTopologyKind::Mesh),
      isTorus_(false),
      npusCount_(1),
      gpuNodeCount_(0),
      totalNodeCount_(0),
      bandwidth_(bandwidth),
      latency_(latency),
      topology_(std::move(topology)) {
    for (int dim : shape_) {
        npusCount_ *= dim;
    }
    gpuNodeCount_ = npusCount_;
    buildConnectionGraph();
}

void StandardSynthesizer::setPathScoreWeights(double sumLoadWeight,
                                             double maxLoadWeight,
                                             double dataTransferWeight) {
    if (sumLoadWeight < 0.0 || maxLoadWeight < 0.0 || dataTransferWeight < 0.0) {
        throw std::invalid_argument("Standard path score weights must be non-negative");
    }
    if (sumLoadWeight + maxLoadWeight + dataTransferWeight <= 0.0) {
        throw std::invalid_argument("At least one Standard path score weight must be positive");
    }
    pathScoreSumLoadWeight_ = sumLoadWeight;
    pathScoreMaxLoadWeight_ = maxLoadWeight;
    pathScoreDataTransferWeight_ = dataTransferWeight;
}

StandardResult StandardSynthesizer::solve(const std::vector<std::vector<DataSize>>& demand) {
    return solveProfiled(profileMatrix(demand));
}

StandardResult StandardSynthesizer::solveSparse(const std::vector<DemandEntry>& demand) {
    return solveProfiled(profileSparse(demand));
}

StandardResult StandardSynthesizer::solveHotOnly(const std::vector<std::vector<DataSize>>& demand) {
    return solveProfiledSubset(profileMatrix(demand), false, true);
}

StandardResult StandardSynthesizer::solveColdOnly(const std::vector<std::vector<DataSize>>& demand) {
    return solveProfiledSubset(profileMatrix(demand), true, false);
}

StandardResult StandardSynthesizer::solveSparseHotOnly(const std::vector<DemandEntry>& demand) {
    return solveProfiledSubset(profileSparse(demand), false, true);
}

StandardResult StandardSynthesizer::solveSparseColdOnly(const std::vector<DemandEntry>& demand) {
    return solveProfiledSubset(profileSparse(demand), true, false);
}

StandardProfileSummary StandardSynthesizer::profileSummary(
    const std::vector<std::vector<DataSize>>& demand) const {
    return summarizeProfiled(profileMatrix(demand));
}

StandardProfileSummary StandardSynthesizer::profileSparseSummary(
    const std::vector<DemandEntry>& demand) const {
    return summarizeProfiled(profileSparse(demand));
}

StandardCacheStats StandardSynthesizer::cacheStats() const {
    StandardCacheStats stats;
    stats.gpuNodeCount = gpuNodeCount_;
    stats.totalNodeCount = totalNodeCount_;
    stats.allSmallSwitchPairsDominant = allSmallSwitchPairsDominant_;
    if (usesFormulaRouting()) {
        if (directTopologyKind_ == Synthesizer::DirectTopologyKind::FullMesh) {
            stats.routingKind = "formula_fullmesh";
        } else if (directTopologyKind_ == Synthesizer::DirectTopologyKind::Torus) {
            stats.routingKind = "formula_torus";
        } else {
            stats.routingKind = "formula_mesh";
        }
    } else if (switchFastPathKind_ == SwitchFastPathKind::FatTreeSingleGlobal) {
        stats.routingKind = "switch_fat_tree_single_global";
    } else if (switchFastPathKind_ == SwitchFastPathKind::CmSingleRail) {
        stats.routingKind = "switch_cm_single_rail";
    } else if (allSmallSwitchPairsDominant_) {
        stats.routingKind = "switch_small_dominant";
    } else {
        stats.routingKind = "switch_generic";
    }

    stats.approximateBytes += connectionGraph_.capacity() * sizeof(std::vector<int>);
    for (const auto& neighbors : connectionGraph_) {
        stats.connectionEdges += neighbors.size();
        stats.approximateBytes += neighbors.capacity() * sizeof(int);
    }

    stats.approximateBytes += nodeDegreeCache_.capacity() * sizeof(int);
    stats.distanceMatrixEntries = distanceMatrix_.size();
    stats.approximateBytes += distanceMatrix_.capacity() * sizeof(int);
    stats.linkCacheEntries = linkLatencyUs_.size();
    stats.approximateBytes += linkLatencyUs_.capacity() * sizeof(double);
    stats.approximateBytes += linkBandwidthBytesPerUs_.capacity() * sizeof(double);
    stats.approximateBytes += linkInvBandwidthUsPerByte_.capacity() * sizeof(double);

    stats.approximateBytes += smallSwitchCandidatePaths_.capacity() * sizeof(std::vector<FixedPath>);
    for (const auto& candidates : smallSwitchCandidatePaths_) {
        if (!candidates.empty()) {
            ++stats.smallSwitchCandidatePairs;
            stats.smallSwitchCandidatePaths += candidates.size();
        }
        stats.approximateBytes += candidates.capacity() * sizeof(FixedPath);
    }
    for (const auto& path : smallSwitchDominantPaths_) {
        if (!path.empty()) {
            ++stats.smallSwitchDominantPaths;
        }
    }
    stats.approximateBytes += smallSwitchDominantPaths_.capacity() * sizeof(FixedPath);
    stats.gpuPairMinChunkEntries = gpuPairMinChunkSize_.size();
    stats.approximateBytes += gpuPairMinChunkSize_.capacity() * sizeof(DataSize);

    stats.shortestPathCacheEntries = shortestPathCache_.size();
    stats.approximateBytes += shortestPathCache_.bucket_count() * sizeof(void*);
    for (const auto& item : shortestPathCache_) {
        stats.approximateBytes += sizeof(item.first);
        stats.approximateBytes += item.second.capacity() * sizeof(std::vector<int>);
        for (const auto& path : item.second) {
            stats.approximateBytes += path.capacity() * sizeof(int);
        }
    }

    stats.timeShortestPathCacheEntries = timeShortestPathCache_.size();
    stats.approximateBytes += timeShortestPathCache_.bucket_count() * sizeof(void*);
    for (const auto& item : timeShortestPathCache_) {
        stats.approximateBytes += sizeof(item.first);
        stats.approximateBytes += item.second.capacity() * sizeof(std::vector<int>);
        for (const auto& path : item.second) {
            stats.approximateBytes += path.capacity() * sizeof(int);
        }
    }
    return stats;
}

void StandardSynthesizer::buildConnectionGraph() {
    if (topology_ != nullptr) {
        const int totalNodes = topology_->npusCount();
        connectionGraph_.assign(totalNodes, {});
        for (int src = 0; src < totalNodes; ++src) {
            for (int dst = 0; dst < totalNodes; ++dst) {
                if (topology_->connected(src, dst)) {
                    connectionGraph_[src].push_back(dst);
                }
            }
        }
    } else {
        connectionGraph_.assign(npusCount_, {});
        if (directTopologyKind_ != Synthesizer::DirectTopologyKind::FullMesh) {
            for (int node = 0; node < npusCount_; ++node) {
                auto coord = nodeIDToCoordinate(node);
                for (std::size_t dim = 0; dim < shape_.size(); ++dim) {
                    auto nextCoord = coord;
                    if (coord[dim] < shape_[dim] - 1) {
                        ++nextCoord[dim];
                    } else if (isTorus_) {
                        nextCoord[dim] = 0;
                    } else {
                        continue;
                    }
                    int next = coordinateToNodeID(nextCoord);
                    if (next >= 0 && next < npusCount_) {
                        connectionGraph_[node].push_back(next);
                    }

                    nextCoord = coord;
                    if (coord[dim] > 0) {
                        --nextCoord[dim];
                    } else if (isTorus_) {
                        nextCoord[dim] = shape_[dim] - 1;
                    } else {
                        continue;
                    }
                    next = coordinateToNodeID(nextCoord);
                    if (next >= 0 && next < npusCount_) {
                        connectionGraph_[node].push_back(next);
                    }
                }
            }
        }
    }

    shortestPathCache_.clear();
    timeShortestPathCache_.clear();
    initializeStaticCaches();
    initializeSmallSwitchCandidatePaths();
}

void StandardSynthesizer::initializeStaticCaches() {
    totalNodeCount_ = static_cast<int>(connectionGraph_.size());
    switchFastPathKind_ = SwitchFastPathKind::None;
    if (topology_ != nullptr && shape_.size() == 2 && shape_[0] > 0 && shape_[1] > 0) {
        const int gpusPerNode = shape_[0];
        const int numNodes = shape_[1];
        const int expectedGpuCount = gpusPerNode * numNodes;
        const int localSwitchBase = expectedGpuCount;
        const int globalSwitch = expectedGpuCount + numNodes;
        if (gpuNodeCount_ == expectedGpuCount && totalNodeCount_ == expectedGpuCount + numNodes + 1) {
            bool allGpuLocal = true;
            bool allGpuGlobal = true;
            for (int gpu = 0; gpu < gpuNodeCount_; ++gpu) {
                const int node = gpu / gpusPerNode;
                allGpuLocal = allGpuLocal && topology_->connected(gpu, localSwitchBase + node);
                allGpuGlobal = allGpuGlobal && topology_->connected(gpu, globalSwitch);
            }

            bool allLocalGlobal = true;
            for (int node = 0; node < numNodes; ++node) {
                allLocalGlobal = allLocalGlobal &&
                    topology_->connected(localSwitchBase + node, globalSwitch);
            }

            if (allGpuLocal && allGpuGlobal) {
                switchFastPathKind_ = SwitchFastPathKind::FatTreeSingleGlobal;
            } else if (allGpuLocal && !allGpuGlobal && allLocalGlobal) {
                switchFastPathKind_ = SwitchFastPathKind::CmSingleRail;
            }
        }
    }
    nodeDegreeCache_.assign(totalNodeCount_, 0);
    minGpuDegree_ = std::numeric_limits<int>::max();

    for (int node = 0; node < totalNodeCount_; ++node) {
        int degree = 0;
        if (topology_ != nullptr) {
            if (node >= 0 && node < gpuNodeCount_) {
                for (int neighbor : getNeighborsFromGraph(node)) {
                    if (neighbor >= gpuNodeCount_ && neighbor < totalNodeCount_) {
                        ++degree;
                    }
                }
            } else if (node >= gpuNodeCount_ && node < totalNodeCount_) {
                degree = static_cast<int>(getNeighborsFromGraph(node).size());
            }
        } else if (directTopologyKind_ == Synthesizer::DirectTopologyKind::FullMesh) {
            for (int dimSize : shape_) {
                degree += std::max(0, dimSize - 1);
            }
        } else {
            const auto coord = nodeIDToCoordinate(node);
            for (std::size_t dim = 0; dim < shape_.size(); ++dim) {
                if (isTorus_) {
                    degree += 2;
                } else if (coord[dim] == 0 || coord[dim] == shape_[dim] - 1) {
                    degree += 1;
                } else {
                    degree += 2;
                }
            }
        }
        nodeDegreeCache_[node] = degree;
        if (node < gpuNodeCount_) {
            minGpuDegree_ = std::min(minGpuDegree_, degree);
        }
    }
    if (minGpuDegree_ == std::numeric_limits<int>::max()) {
        minGpuDegree_ = 0;
    }

    if (topology_ == nullptr) {
        distanceMatrix_.clear();
        linkLatencyUs_.clear();
        linkBandwidthBytesPerUs_.clear();
        linkInvBandwidthUsPerByte_.clear();
        return;
    }

    const double defaultLatencyUs = latency_ / 1000.0;
    const double defaultBandwidthBytesPerUs = bandwidth_ * (1 << 30) / 1e6;
    distanceMatrix_.assign(static_cast<std::size_t>(totalNodeCount_) * totalNodeCount_,
                           std::numeric_limits<int>::max());
    linkLatencyUs_.assign(static_cast<std::size_t>(totalNodeCount_) * totalNodeCount_,
                          defaultLatencyUs);
    linkBandwidthBytesPerUs_.assign(static_cast<std::size_t>(totalNodeCount_) * totalNodeCount_,
                                    defaultBandwidthBytesPerUs);
    linkInvBandwidthUsPerByte_.assign(static_cast<std::size_t>(totalNodeCount_) * totalNodeCount_,
                                      1.0 / defaultBandwidthBytesPerUs);

    for (int src = 0; src < totalNodeCount_; ++src) {
        distanceMatrix_[edgeIndex(src, src)] = 0;
        std::vector<int> dist(totalNodeCount_, -1);
        std::queue<int> q;
        q.push(src);
        dist[src] = 0;

        while (!q.empty()) {
            const int u = q.front();
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
            linkInvBandwidthUsPerByte_[idx] = 1.0 / linkBandwidthBytesPerUs_[idx];
        }
    }
}

void StandardSynthesizer::initializeSmallSwitchCandidatePaths() {
    smallSwitchCandidatePaths_.clear();
    smallSwitchDominantPaths_.clear();
    gpuPairMinChunkSize_.clear();
    allSmallSwitchPairsDominant_ = false;
    if (topology_ == nullptr || gpuNodeCount_ > 64) {
        return;
    }

    smallSwitchCandidatePaths_.resize(static_cast<std::size_t>(gpuNodeCount_) * gpuNodeCount_);
    smallSwitchDominantPaths_.resize(static_cast<std::size_t>(gpuNodeCount_) * gpuNodeCount_);

    auto addCandidate = [&](int src, int dst, const std::vector<int>& path) {
        if (path.size() < 2 || path.size() > FixedPath{}.nodes.size()) {
            return;
        }
        for (std::size_t hop = 0; hop + 1 < path.size(); ++hop) {
            if (!topology_->connected(path[hop], path[hop + 1])) {
                return;
            }
        }
        FixedPath fixedPath;
        fixedPath.length = static_cast<int>(path.size());
        for (int i = 0; i < fixedPath.length; ++i) {
            fixedPath.nodes[i] = path[i];
        }
        auto& candidates = smallSwitchCandidatePaths_[src * gpuNodeCount_ + dst];
        if (std::find(candidates.begin(), candidates.end(), fixedPath) == candidates.end()) {
            candidates.push_back(fixedPath);
        }
    };

    for (int src = 0; src < gpuNodeCount_; ++src) {
        const auto& srcNeighbors = getNeighborsFromGraph(src);
        for (int dst = 0; dst < gpuNodeCount_; ++dst) {
            if (src == dst) {
                continue;
            }

            if (topology_->connected(src, dst)) {
                addCandidate(src, dst, {src, dst});
            }

            for (int sw : srcNeighbors) {
                if (isSwitchNode(sw) && topology_->connected(sw, dst)) {
                    addCandidate(src, dst, {src, sw, dst});
                }
            }

            for (int sw1 : srcNeighbors) {
                if (!isSwitchNode(sw1)) {
                    continue;
                }
                for (int mid : getNeighborsFromGraph(sw1)) {
                    if (!isSwitchNode(mid) || mid == sw1) {
                        continue;
                    }
                    for (int sw2 : getNeighborsFromGraph(mid)) {
                        if (isSwitchNode(sw2) && sw2 != mid && topology_->connected(sw2, dst)) {
                            addCandidate(src, dst, {src, sw1, mid, sw2, dst});
                        }
                    }
                }
            }
        }
    }

    auto linearCost = [&](const FixedPath& path) {
        double latencyIntercept = 0.0;
        double byteSlope = 0.0;
        for (int hop = 0; hop + 1 < path.length; ++hop) {
            const int idx = edgeIndex(path.nodes[hop], path.nodes[hop + 1]);
            double linkLatencyUs = latency_ / 1000.0;
            double linkBandwidthBytesPerUs = bandwidth_ * (1 << 30) / 1e6;
            if (!linkLatencyUs_.empty()) {
                linkLatencyUs = linkLatencyUs_[idx];
            }
            if (!linkBandwidthBytesPerUs_.empty() && linkBandwidthBytesPerUs_[idx] > 0.0) {
                linkBandwidthBytesPerUs = linkBandwidthBytesPerUs_[idx];
            }
            latencyIntercept += linkLatencyUs;
            if (!linkInvBandwidthUsPerByte_.empty()) {
                byteSlope += linkInvBandwidthUsPerByte_[idx];
            } else {
                byteSlope += 1.0 / linkBandwidthBytesPerUs;
            }
        }
        return std::pair<double, double>{latencyIntercept, byteSlope};
    };

    for (std::size_t idx = 0; idx < smallSwitchCandidatePaths_.size(); ++idx) {
        const auto& candidates = smallSwitchCandidatePaths_[idx];
        if (candidates.empty()) {
            continue;
        }
        if (candidates.size() == 1) {
            smallSwitchDominantPaths_[idx] = candidates.front();
            continue;
        }

        std::vector<std::pair<double, double>> costs;
        costs.reserve(candidates.size());
        for (const auto& candidate : candidates) {
            costs.push_back(linearCost(candidate));
        }

        int dominantIndex = -1;
        bool uniqueDominant = false;
        for (int candidateIndex = 0; candidateIndex < static_cast<int>(candidates.size()); ++candidateIndex) {
            bool dominatesAll = true;
            bool tied = false;
            for (int otherIndex = 0; otherIndex < static_cast<int>(candidates.size()); ++otherIndex) {
                if (candidateIndex == otherIndex) {
                    continue;
                }
                const auto [candidateLatency, candidateSlope] = costs[candidateIndex];
                const auto [otherLatency, otherSlope] = costs[otherIndex];
                const bool sameCost =
                    std::abs(candidateLatency - otherLatency) <= kPathEpsilon &&
                    std::abs(candidateSlope - otherSlope) <= kPathEpsilon;
                if (sameCost) {
                    tied = true;
                    break;
                }
                if (candidateLatency > otherLatency + kPathEpsilon ||
                    candidateSlope > otherSlope + kPathEpsilon) {
                    dominatesAll = false;
                    break;
                }
            }
            if (dominatesAll && !tied) {
                dominantIndex = candidateIndex;
                uniqueDominant = true;
                break;
            }
        }

        if (uniqueDominant) {
            smallSwitchDominantPaths_[idx] = candidates[dominantIndex];
        }
    }

    allSmallSwitchPairsDominant_ = true;
    for (int src = 0; src < gpuNodeCount_ && allSmallSwitchPairsDominant_; ++src) {
        for (int dst = 0; dst < gpuNodeCount_; ++dst) {
            if (src == dst) {
                continue;
            }
            const std::size_t idx = static_cast<std::size_t>(src) * gpuNodeCount_ + dst;
            if (smallSwitchDominantPaths_[idx].empty()) {
                allSmallSwitchPairsDominant_ = false;
                break;
            }
        }
    }

    gpuPairMinChunkSize_.assign(static_cast<std::size_t>(gpuNodeCount_) * gpuNodeCount_, 0);
    for (int src = 0; src < gpuNodeCount_; ++src) {
        for (int dst = 0; dst < gpuNodeCount_; ++dst) {
            if (src == dst) {
                continue;
            }

            double latencyUs = latency_ / 1000.0;
            double bandwidthBytesPerUs = bandwidth_ * (1 << 30) / 1e6;
            const int directIdx = edgeIndex(src, dst);
            if (!linkBandwidthBytesPerUs_.empty() && linkBandwidthBytesPerUs_[directIdx] > 0.0) {
                latencyUs = linkLatencyUs_[directIdx];
                bandwidthBytesPerUs = linkBandwidthBytesPerUs_[directIdx];
            }

            auto path = computeSwitchFastPath(src, dst, 1);
            if (path.empty()) {
                path = bfsPath(src, dst);
            }
            if (path.size() > 1) {
                double minBandwidthBytesPerUs = std::numeric_limits<double>::max();
                double totalLatencyUs = 0.0;
                for (std::size_t hop = 0; hop + 1 < path.size(); ++hop) {
                    const int idx = edgeIndex(path[hop], path[hop + 1]);
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

            gpuPairMinChunkSize_[static_cast<std::size_t>(src) * gpuNodeCount_ + dst] =
                std::max(static_cast<DataSize>(1),
                         static_cast<DataSize>(bandwidthBytesPerUs * latencyUs));
        }
    }
}

bool StandardSynthesizer::isGPUNode(int nodeID) const {
    return nodeID >= 0 && nodeID < gpuNodeCount_;
}

bool StandardSynthesizer::isSwitchNode(int nodeID) const {
    return topology_ != nullptr && nodeID >= gpuNodeCount_ && nodeID < totalNodeCount_;
}

std::uint64_t StandardSynthesizer::pairCacheKey(int src, int dst) const {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(src)) << 32) |
           static_cast<std::uint32_t>(dst);
}

std::vector<int> StandardSynthesizer::nodeIDToCoordinate(int nodeID) const {
    if (shape_.empty()) {
        return {nodeID};
    }
    std::vector<int> coord(shape_.size(), 0);
    int remaining = nodeID;
    for (int dim = 0; dim < static_cast<int>(shape_.size()); ++dim) {
        int stride = 1;
        for (int next = dim + 1; next < static_cast<int>(shape_.size()); ++next) {
            stride *= shape_[next];
        }
        coord[dim] = remaining / stride;
        remaining %= stride;
    }
    return coord;
}

int StandardSynthesizer::coordinateToNodeID(const std::vector<int>& coord) const {
    if (shape_.empty() || coord.empty()) {
        return coord.empty() ? 0 : coord[0];
    }
    if (coord.size() != shape_.size()) {
        return -1;
    }
    int nodeID = 0;
    for (int dim = 0; dim < static_cast<int>(shape_.size()); ++dim) {
        int stride = 1;
        for (int next = dim + 1; next < static_cast<int>(shape_.size()); ++next) {
            stride *= shape_[next];
        }
        nodeID += coord[dim] * stride;
    }
    return nodeID;
}

int StandardSynthesizer::computeDistance(int src, int dst) const {
    if (src == dst) {
        return 0;
    }
    if (usesFormulaRouting()) {
        if (src < 0 || dst < 0 || src >= totalNodeCount_ || dst >= totalNodeCount_) {
            return std::numeric_limits<int>::max();
        }
        auto srcCoord = nodeIDToCoordinate(src);
        auto dstCoord = nodeIDToCoordinate(dst);
        if (directTopologyKind_ == Synthesizer::DirectTopologyKind::FullMesh) {
            int differingDims = 0;
            for (std::size_t dim = 0; dim < srcCoord.size(); ++dim) {
                if (srcCoord[dim] != dstCoord[dim]) {
                    ++differingDims;
                }
            }
            return differingDims;
        }
        int distance = 0;
        for (std::size_t dim = 0; dim < srcCoord.size(); ++dim) {
            const int diff = std::abs(dstCoord[dim] - srcCoord[dim]);
            if (directTopologyKind_ == Synthesizer::DirectTopologyKind::Torus) {
                distance += std::min(diff, shape_[dim] - diff);
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

int StandardSynthesizer::computeNodeDegree(int nodeID) const {
    if (nodeID >= 0 && nodeID < static_cast<int>(nodeDegreeCache_.size())) {
        return nodeDegreeCache_[nodeID];
    }

    if (topology_ != nullptr) {
        if (isGPUNode(nodeID)) {
            int degree = 0;
            for (int neighbor : getNeighborsFromGraph(nodeID)) {
                if (isSwitchNode(neighbor)) {
                    ++degree;
                }
            }
            return degree;
        }
        if (isSwitchNode(nodeID)) {
            return static_cast<int>(getNeighborsFromGraph(nodeID).size());
        }
        return 0;
    }

    if (directTopologyKind_ == Synthesizer::DirectTopologyKind::FullMesh) {
        int degree = 0;
        for (int dimSize : shape_) {
            degree += std::max(0, dimSize - 1);
        }
        return degree;
    }

    auto coord = nodeIDToCoordinate(nodeID);
    int degree = 0;
    for (std::size_t dim = 0; dim < shape_.size(); ++dim) {
        if (isTorus_) {
            degree += 2;
        } else if (coord[dim] == 0 || coord[dim] == shape_[dim] - 1) {
            degree += 1;
        } else {
            degree += 2;
        }
    }
    return degree;
}

int StandardSynthesizer::computeMinDegree() const {
    if (minGpuDegree_ >= 0) {
        return minGpuDegree_;
    }

    int minDegree = std::numeric_limits<int>::max();
    for (int node = 0; node < npusCount_; ++node) {
        minDegree = std::min(minDegree, computeNodeDegree(node));
    }
    return minDegree;
}

double StandardSynthesizer::computeBwLatThreshold() const {
    const double latencyUs = latency_ / 1000.0;
    const double bandwidthBytesPerUs = bandwidth_ * (1 << 30) / 1e6;
    return bandwidthBytesPerUs * latencyUs * computeMinDegree();
}

double StandardSynthesizer::computeLinkTransferTime(int src, int dst, DataSize bytes) const {
    if (bytes <= 0) {
        return 0.0;
    }
    if (src >= 0 && dst >= 0 && src < totalNodeCount_ && dst < totalNodeCount_ &&
        !linkInvBandwidthUsPerByte_.empty()) {
        return computeEdgeTransferTime(edgeIndex(src, dst), bytes);
    }
    const double latencyUs = latency_ / 1000.0;
    const double bandwidthBytesPerUs = bandwidth_ * (1 << 30) / 1e6;
    return latencyUs + (bytes / bandwidthBytesPerUs);
}

double StandardSynthesizer::computeEdgeTransferTime(int edgeId, DataSize bytes) const {
    if (bytes <= 0) {
        return 0.0;
    }
    if (!linkInvBandwidthUsPerByte_.empty()) {
        return linkLatencyUs_[edgeId] + (bytes * linkInvBandwidthUsPerByte_[edgeId]);
    }
    const double latencyUs = latency_ / 1000.0;
    const double bandwidthBytesPerUs = bandwidth_ * (1 << 30) / 1e6;
    return latencyUs + (bytes / bandwidthBytesPerUs);
}

DataSize StandardSynthesizer::computeMinChunkSize(int src, int dst) const {
    if (!gpuPairMinChunkSize_.empty() &&
        src >= 0 && dst >= 0 && src < gpuNodeCount_ && dst < gpuNodeCount_) {
        const DataSize cached =
            gpuPairMinChunkSize_[static_cast<std::size_t>(src) * gpuNodeCount_ + dst];
        if (cached > 0) {
            return cached;
        }
    }

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
            for (std::size_t hop = 0; hop + 1 < path.size(); ++hop) {
                const int idx = edgeIndex(path[hop], path[hop + 1]);
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

    return std::max(static_cast<DataSize>(1),
                    static_cast<DataSize>(bandwidthBytesPerUs * latencyUs));
}

int StandardSynthesizer::computeChunkCount(int src, int dst, DataSize bytes) const {
    const int maxChunks = std::min(computeNodeDegree(src), computeNodeDegree(dst));
    const DataSize minChunkSize = computeMinChunkSize(src, dst);
    const int maxChunksBySize = static_cast<int>(bytes / minChunkSize);
    return std::max(1, std::min(maxChunks, maxChunksBySize));
}

StandardSynthesizer::ProfiledFlows StandardSynthesizer::profileMatrix(
    const std::vector<std::vector<DataSize>>& demand) const {
    const double threshold = computeBwLatThreshold();
    std::vector<Flow> latencyFlows;
    std::vector<Flow> bandwidthCandidates;

    for (int src = 0; src < static_cast<int>(demand.size()); ++src) {
        for (int dst = 0; dst < static_cast<int>(demand[src].size()); ++dst) {
            if (src == dst || demand[src][dst] <= 0) {
                continue;
            }
            if (demand[src][dst] > threshold) {
                bandwidthCandidates.push_back({src, dst, demand[src][dst]});
            } else {
                latencyFlows.push_back({src, dst, demand[src][dst]});
            }
        }
    }

    return profileCandidates(latencyFlows, bandwidthCandidates, threshold);
}

StandardSynthesizer::ProfiledFlows StandardSynthesizer::profileSparse(
    const std::vector<DemandEntry>& demand) const {
    const double threshold = computeBwLatThreshold();
    std::vector<Flow> latencyFlows;
    std::vector<Flow> bandwidthCandidates;
    latencyFlows.reserve(demand.size());

    for (const auto& flow : demand) {
        if (flow.src == flow.dst || flow.bytes <= 0) {
            continue;
        }
        if (flow.bytes > threshold) {
            bandwidthCandidates.push_back({flow.src, flow.dst, flow.bytes});
        } else {
            latencyFlows.push_back({flow.src, flow.dst, flow.bytes});
        }
    }

    return profileCandidates(latencyFlows, bandwidthCandidates, threshold);
}

StandardSynthesizer::ProfiledFlows StandardSynthesizer::profileCandidates(
    std::vector<Flow>& latencyFlows,
    std::vector<Flow>& bandwidthCandidates,
    double threshold) const {
    ProfiledFlows result;
    result.threshold = threshold;

    std::sort(bandwidthCandidates.begin(), bandwidthCandidates.end(),
              [](const Flow& a, const Flow& b) {
                  return a.bytes > b.bytes;
              });

    const int maxBandwidthFlows = maxBandwidthFlowsOverride_ > 0
        ? maxBandwidthFlowsOverride_
        : 4 * npusCount_;
    const int selectedCount = std::min(static_cast<int>(bandwidthCandidates.size()),
                                       maxBandwidthFlows);
    result.latencyFlows = std::move(latencyFlows);
    result.latencyBytes = 0;
    for (const auto& flow : result.latencyFlows) {
        result.latencyBytes += flow.bytes;
    }
    result.bandwidthFlows.reserve(selectedCount);
    for (int i = 0; i < selectedCount; ++i) {
        const auto& candidate = bandwidthCandidates[i];
        result.bandwidthFlows.push_back({candidate.src, candidate.dst, candidate.bytes});
        result.bandwidthBytes += candidate.bytes;
    }
    for (int i = selectedCount; i < static_cast<int>(bandwidthCandidates.size()); ++i) {
        const auto& candidate = bandwidthCandidates[i];
        result.latencyFlows.push_back({candidate.src, candidate.dst, candidate.bytes});
        result.latencyBytes += candidate.bytes;
    }

    return result;
}

StandardProfileSummary StandardSynthesizer::summarizeProfiled(const ProfiledFlows& profiled) const {
    StandardProfileSummary summary;
    summary.threshold = profiled.threshold;
    summary.coldFlowCount = static_cast<int>(profiled.latencyFlows.size());
    summary.hotFlowCount = static_cast<int>(profiled.bandwidthFlows.size());
    summary.coldBytes = profiled.latencyBytes;
    summary.hotBytes = profiled.bandwidthBytes;
    return summary;
}

StandardResult StandardSynthesizer::solveProfiledSubset(ProfiledFlows profiled,
                                                      bool includeLatencyFlows,
                                                      bool includeBandwidthFlows) {
    if (!includeLatencyFlows) {
        profiled.latencyFlows.clear();
        profiled.latencyBytes = 0;
    }
    if (!includeBandwidthFlows) {
        profiled.bandwidthFlows.clear();
        profiled.bandwidthBytes = 0;
    }
    return solveProfiled(profiled);
}

StandardResult StandardSynthesizer::solveProfiled(const ProfiledFlows& profiled) {
    StandardResult result;
    result.latencyFlowCount = static_cast<int>(profiled.latencyFlows.size());
    result.bandwidthFlowCount = static_cast<int>(profiled.bandwidthFlows.size());
    result.latencyBytes = profiled.latencyBytes;
    result.bandwidthBytes = profiled.bandwidthBytes;

    std::vector<RankedFlow> allFlows;
    allFlows.reserve(profiled.latencyFlows.size() + profiled.bandwidthFlows.size());
    for (const auto& flow : profiled.latencyFlows) {
        const int distance = computeDistance(flow.src, flow.dst);
        allFlows.push_back({flow.src, flow.dst, flow.bytes, distance, true,
                            static_cast<double>(distance)});
    }
    for (const auto& flow : profiled.bandwidthFlows) {
        const int distance = computeDistance(flow.src, flow.dst);
        allFlows.push_back({flow.src, flow.dst, flow.bytes, distance, false,
                            static_cast<double>(flow.bytes) * distance});
    }

    std::sort(allFlows.begin(), allFlows.end(),
              [](const RankedFlow& a, const RankedFlow& b) {
                  if (a.isLatencyMatrix != b.isLatencyMatrix) {
                      return a.isLatencyMatrix < b.isLatencyMatrix;
                  }
                  return a.priority > b.priority;
              });

    const bool denseEdges = shouldUseDenseEdges(totalNodeCount_);
    const bool deterministicSwitchPaths =
        allSmallSwitchPairsDominant_ || switchFastPathKind_ != SwitchFastPathKind::None;
    const bool skipLoadTracking = !captureSchedule_ && deterministicSwitchPaths;
    LoadState accumulatedLinkLoad(totalNodeCount_, denseEdges && !skipLoadTracking);
    SlotState slots(totalNodeCount_, denseEdges);
    double maxAccumulatedLinkLoad = 0.0;
    int chunkId = 0;

    if (captureSchedule_) {
        result.events.reserve(allFlows.size() * 4);
    }

    auto scheduleQuery = [&](const Query& query) {
        const FixedPath fixedPath =
            (skipLoadTracking && allSmallSwitchPairsDominant_)
                ? smallSwitchDominantPaths_[static_cast<std::size_t>(query.src) * gpuNodeCount_ + query.dst]
                : computeSwitchFastPathFixed(query.src, query.dst, query.bytes);
        if (!fixedPath.empty()) {
            std::vector<int> eventPath;
            if (captureSchedule_) {
                eventPath = fixedPathToVector(fixedPath);
            }

            double readyTime = 0.0;
            for (int hop = 0; hop + 1 < fixedPath.length; ++hop) {
                const int hopSrc = fixedPath.nodes[hop];
                const int hopDst = fixedPath.nodes[hop + 1];
                const int edgeId = edgeIndex(hopSrc, hopDst);
                const double transferTime = computeEdgeTransferTime(edgeId, query.bytes);
                if (!skipLoadTracking) {
                    accumulatedLinkLoad.add(edgeId, transferTime);
                    maxAccumulatedLinkLoad = std::max(maxAccumulatedLinkLoad,
                                                      accumulatedLinkLoad.get(edgeId));
                }

                const auto times = scheduleEdge(slots, edgeId, readyTime, transferTime);
                readyTime = times.second;
                result.makespan = std::max(result.makespan, times.second);
                ++result.scheduledEvents;

                if (captureSchedule_) {
                    result.events.push_back({hopSrc, hopDst, query.bytes, times.first, times.second,
                                             eventPath, chunkId, query.src, query.dst,
                                             query.isLatencyMatrix});
                }
            }

            ++result.scheduledChunks;
            ++chunkId;
            return;
        }

        std::vector<int> path;
        auto paths = findTimeShortestPaths(query.src, query.dst, query.bytes);
        if (paths.empty()) {
            paths = findAllShortestPaths(query.src, query.dst);
        }
        if (paths.empty()) {
            ++chunkId;
            return;
        }

        path = query.isLatencyMatrix
            ? selectBestPathForLatency(paths, accumulatedLinkLoad, maxAccumulatedLinkLoad, query.bytes)
            : selectBestPathByLoad(paths, accumulatedLinkLoad, maxAccumulatedLinkLoad, query.bytes);

        if (path.size() < 2) {
            ++chunkId;
            return;
        }

        double readyTime = 0.0;
        for (std::size_t hop = 0; hop + 1 < path.size(); ++hop) {
            const int hopSrc = path[hop];
            const int hopDst = path[hop + 1];
            const int edgeId = edgeIndex(hopSrc, hopDst);
            const double transferTime = computeEdgeTransferTime(edgeId, query.bytes);
            accumulatedLinkLoad.add(edgeId, transferTime);
            maxAccumulatedLinkLoad = std::max(maxAccumulatedLinkLoad,
                                              accumulatedLinkLoad.get(edgeId));

            const auto times = scheduleEdge(slots, edgeId, readyTime, transferTime);
            readyTime = times.second;
            result.makespan = std::max(result.makespan, times.second);
            ++result.scheduledEvents;

            if (captureSchedule_) {
                result.events.push_back({hopSrc, hopDst, query.bytes, times.first, times.second,
                                         path, chunkId, query.src, query.dst,
                                         query.isLatencyMatrix});
            }
        }

        ++result.scheduledChunks;
        ++chunkId;
    };

    auto appendFlowQueries = [&](const RankedFlow& flow, std::vector<Query>& queries) {
        if (flow.isLatencyMatrix) {
            queries.emplace_back(flow.src, flow.dst, flow.bytes, true);
            return;
        }

        const int numChunks = computeChunkCount(flow.src, flow.dst, flow.bytes);
        const DataSize chunkSize = flow.bytes / numChunks;
        const DataSize remainder = flow.bytes % numChunks;
        for (int chunk = 0; chunk < numChunks; ++chunk) {
            DataSize currentChunkSize = chunkSize;
            if (chunk < remainder) {
                ++currentChunkSize;
            }
            queries.emplace_back(flow.src, flow.dst, currentChunkSize, false);
        }
    };

    auto scheduleFlow = [&](const RankedFlow& flow) {
        if (flow.isLatencyMatrix) {
            const Query query(flow.src, flow.dst, flow.bytes, true);
            scheduleQuery(query);
            return;
        }

        const int numChunks = computeChunkCount(flow.src, flow.dst, flow.bytes);
        const DataSize chunkSize = flow.bytes / numChunks;
        const DataSize remainder = flow.bytes % numChunks;
        for (int chunk = 0; chunk < numChunks; ++chunk) {
            DataSize currentChunkSize = chunkSize;
            if (chunk < remainder) {
                ++currentChunkSize;
            }
            scheduleQuery(Query(flow.src, flow.dst, currentChunkSize, false));
        }
    };

    const bool useSingleNodeAggregate =
        !captureSchedule_ && topology_ != nullptr && gpuNodeCount_ <= 64 &&
        shape_.size() == 2 && shape_[1] == 1;
    if (useSingleNodeAggregate) {
        std::vector<Query> queries;
        queries.reserve(allFlows.size() + profiled.bandwidthFlows.size());
        for (const auto& flow : allFlows) {
            appendFlowQueries(flow, queries);
        }

        std::vector<double> denseEdgeLoad;
        std::vector<double> denseFirstRelease;
        std::vector<double> denseEdgeNextEnd;
        std::unordered_map<int, double> sparseEdgeLoad;
        std::unordered_map<int, double> sparseFirstRelease;
        std::unordered_map<int, double> sparseEdgeNextEnd;
        if (denseEdges) {
            const std::size_t edgeCount = static_cast<std::size_t>(totalNodeCount_) * totalNodeCount_;
            denseEdgeLoad.assign(edgeCount, 0.0);
            denseFirstRelease.assign(edgeCount, -1.0);
            denseEdgeNextEnd.assign(edgeCount, 0.0);
        }

        bool aggregateSupported = true;
        int chunkIdForAggregate = 0;
        int aggregateChunks = 0;
        int aggregateEvents = 0;
        for (const auto& query : queries) {
            auto path = computeSwitchFastPath(query.src, query.dst, query.bytes);
            if (path.size() < 2) {
                aggregateSupported = false;
                break;
            }

            double readyTime = 0.0;
            for (std::size_t hop = 0; hop + 1 < path.size(); ++hop) {
                const int edgeId = edgeIndex(path[hop], path[hop + 1]);
                const double transferTime = computeEdgeTransferTime(edgeId, query.bytes);
                double startTime = readyTime;
                if (denseEdges) {
                    startTime = std::max(startTime, denseEdgeNextEnd[edgeId]);
                    if (denseFirstRelease[edgeId] < 0.0) {
                        denseFirstRelease[edgeId] = startTime;
                    }
                    denseEdgeLoad[edgeId] += transferTime;
                    denseEdgeNextEnd[edgeId] = startTime + transferTime;
                } else {
                    const auto nextEndIt = sparseEdgeNextEnd.find(edgeId);
                    if (nextEndIt != sparseEdgeNextEnd.end()) {
                        startTime = std::max(startTime, nextEndIt->second);
                    }
                    if (sparseFirstRelease.find(edgeId) == sparseFirstRelease.end()) {
                        sparseFirstRelease[edgeId] = startTime;
                    }
                    sparseEdgeLoad[edgeId] += transferTime;
                    sparseEdgeNextEnd[edgeId] = startTime + transferTime;
                }
                readyTime = startTime + transferTime;
                ++aggregateEvents;
            }
            ++aggregateChunks;
            ++chunkIdForAggregate;
        }

        if (aggregateSupported) {
            if (denseEdges) {
                for (std::size_t edgeId = 0; edgeId < denseEdgeNextEnd.size(); ++edgeId) {
                    if (denseFirstRelease[edgeId] >= 0.0) {
                        result.makespan = std::max(result.makespan, denseEdgeNextEnd[edgeId]);
                    }
                }
            } else {
                for (const auto& [edgeId, nextEnd] : sparseEdgeNextEnd) {
                    result.makespan = std::max(result.makespan, nextEnd);
                }
            }
            result.scheduledChunks = aggregateChunks;
            result.scheduledEvents = aggregateEvents;
            return result;
        }

        for (const auto& query : queries) {
            scheduleQuery(query);
        }
        return result;
    }

    for (const auto& flow : allFlows) {
        scheduleFlow(flow);
    }

    return result;
}

std::vector<std::vector<int>> StandardSynthesizer::findTimeShortestPaths(
    int src, int dst, DataSize bytes) {
    TimeShortestPathCacheKey key{src, dst, bytes};
    const auto cached = timeShortestPathCache_.find(key);
    if (cached != timeShortestPathCache_.end()) {
        return cached->second;
    }
    auto paths = computeTimeShortestPathsUncached(src, dst, bytes);
    timeShortestPathCache_.emplace(key, paths);
    return paths;
}

std::vector<std::vector<int>> StandardSynthesizer::computeTimeShortestPathsUncached(
    int src, int dst, DataSize bytes) const {
    if (usesFormulaRouting()) {
        auto mutableThis = const_cast<StandardSynthesizer*>(this);
        return mutableThis->findAllShortestPaths(src, dst);
    }

    std::vector<std::vector<int>> paths;
    auto fastPath = computeSwitchFastPath(src, dst, bytes);
    if (!fastPath.empty()) {
        paths.push_back(std::move(fastPath));
        return paths;
    }

    if (src == dst) {
        paths.push_back({src});
        return paths;
    }
    if (src < 0 || dst < 0 || src >= totalNodeCount_ || dst >= totalNodeCount_) {
        return paths;
    }

    std::vector<double> dist(totalNodeCount_, std::numeric_limits<double>::max());
    std::vector<std::vector<int>> prev(totalNodeCount_);
    for (int node = 0; node < totalNodeCount_; ++node) {
        prev[node].reserve(getNeighborsFromGraph(node).size());
    }

    using Entry = std::pair<double, int>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;
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
        for (int v : getNeighborsFromGraph(u)) {
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
        auto mutableThis = const_cast<StandardSynthesizer*>(this);
        return mutableThis->findAllShortestPaths(src, dst);
    }

    std::vector<int> reversedPath;
    reversedPath.reserve(totalNodeCount_);
    std::function<void(int)> dfs = [&](int node) {
        if (paths.size() >= kMaxPaths) {
            return;
        }
        reversedPath.push_back(node);
        if (node == src) {
            paths.emplace_back(reversedPath.rbegin(), reversedPath.rend());
        } else {
            for (int parent : prev[node]) {
                dfs(parent);
                if (paths.size() >= kMaxPaths) {
                    break;
                }
            }
        }
        reversedPath.pop_back();
    };
    dfs(dst);
    return paths;
}

StandardSynthesizer::FixedPath StandardSynthesizer::computeSwitchFastPathFixed(
    int src, int dst, DataSize bytes) const {
    auto smallPath = computeSmallSwitchFastPathFixed(src, dst, bytes);
    if (!smallPath.empty()) {
        return smallPath;
    }
    if (topology_ == nullptr || switchFastPathKind_ == SwitchFastPathKind::None ||
        shape_.size() != 2 || src < 0 || dst < 0 || src >= gpuNodeCount_ ||
        dst >= gpuNodeCount_ || src == dst) {
        return {};
    }

    const int gpusPerNode = shape_[0];
    const int numNodes = shape_[1];
    const int srcNode = src / gpusPerNode;
    const int dstNode = dst / gpusPerNode;
    const int localSwitchBase = gpuNodeCount_;
    const int globalSwitch = gpuNodeCount_ + numNodes;

    FixedPath path;
    if (switchFastPathKind_ == SwitchFastPathKind::FatTreeSingleGlobal) {
        path.length = 3;
        path.nodes[0] = src;
        path.nodes[1] = (srcNode == dstNode) ? (localSwitchBase + srcNode) : globalSwitch;
        path.nodes[2] = dst;
        return path;
    }

    if (switchFastPathKind_ == SwitchFastPathKind::CmSingleRail) {
        if (srcNode == dstNode && topology_->connected(src, dst)) {
            path.length = 2;
            path.nodes[0] = src;
            path.nodes[1] = dst;
            return path;
        }
        if (srcNode == dstNode) {
            path.length = 3;
            path.nodes[0] = src;
            path.nodes[1] = localSwitchBase + srcNode;
            path.nodes[2] = dst;
            return path;
        }
        path.length = 5;
        path.nodes[0] = src;
        path.nodes[1] = localSwitchBase + srcNode;
        path.nodes[2] = globalSwitch;
        path.nodes[3] = localSwitchBase + dstNode;
        path.nodes[4] = dst;
        return path;
    }

    return {};
}

std::vector<int> StandardSynthesizer::computeSwitchFastPath(
    int src, int dst, DataSize bytes) const {
    return fixedPathToVector(computeSwitchFastPathFixed(src, dst, bytes));
}

StandardSynthesizer::FixedPath StandardSynthesizer::computeSmallSwitchFastPathFixed(
    int src, int dst, DataSize bytes) const {
    if (topology_ == nullptr || gpuNodeCount_ > 64 ||
        src < 0 || dst < 0 || src >= gpuNodeCount_ || dst >= gpuNodeCount_ ||
        src == dst || smallSwitchCandidatePaths_.empty()) {
        return {};
    }

    const std::size_t pathIndex = static_cast<std::size_t>(src) * gpuNodeCount_ + dst;
    if (!smallSwitchDominantPaths_.empty() && !smallSwitchDominantPaths_[pathIndex].empty()) {
        return smallSwitchDominantPaths_[pathIndex];
    }

    double bestTime = std::numeric_limits<double>::max();
    FixedPath bestPath;
    int bestCount = 0;

    const auto& candidates = smallSwitchCandidatePaths_[pathIndex];
    for (const auto& path : candidates) {
        double time = 0.0;
        for (int hop = 0; hop + 1 < path.length; ++hop) {
            time += computeLinkTransferTime(path.nodes[hop], path.nodes[hop + 1], bytes);
        }
        if (time + kPathEpsilon < bestTime) {
            bestTime = time;
            bestPath = path;
            bestCount = 1;
        } else if (std::abs(time - bestTime) <= kPathEpsilon) {
            ++bestCount;
        }
    }

    return bestCount == 1 ? bestPath : FixedPath{};
}

std::vector<int> StandardSynthesizer::computeSmallSwitchFastPath(
    int src, int dst, DataSize bytes) const {
    return fixedPathToVector(computeSmallSwitchFastPathFixed(src, dst, bytes));
}

std::vector<int> StandardSynthesizer::fixedPathToVector(const FixedPath& path) const {
    if (path.empty()) {
        return {};
    }

    std::vector<int> out;
    out.reserve(path.length);
    for (int i = 0; i < path.length; ++i) {
        out.push_back(path.nodes[i]);
    }
    return out;
}

std::vector<std::vector<int>> StandardSynthesizer::findAllShortestPaths(int src, int dst) {
    if (usesFormulaRouting()) {
        return enumerateDirectShortestPaths(src, dst);
    }

    const auto key = pairCacheKey(src, dst);
    const auto cached = shortestPathCache_.find(key);
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

std::vector<std::vector<int>> StandardSynthesizer::enumerateDirectShortestPaths(int src, int dst) {
    std::vector<std::vector<int>> allPaths;
    if (src < 0 || dst < 0 || src >= totalNodeCount_ || dst >= totalNodeCount_) {
        return allPaths;
    }
    if (src == dst) {
        allPaths.push_back({src});
        return allPaths;
    }

    const auto key = pairCacheKey(src, dst);
    const auto cached = shortestPathCache_.find(key);
    if (cached != shortestPathCache_.end()) {
        return cached->second;
    }

    const auto srcCoord = nodeIDToCoordinate(src);
    const auto dstCoord = nodeIDToCoordinate(dst);

    if (directTopologyKind_ == Synthesizer::DirectTopologyKind::FullMesh) {
        std::vector<int> differingDims;
        for (std::size_t dim = 0; dim < srcCoord.size(); ++dim) {
            if (srcCoord[dim] != dstCoord[dim]) {
                differingDims.push_back(static_cast<int>(dim));
            }
        }
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
        } while (!differingDims.empty() &&
                 std::next_permutation(differingDims.begin(), differingDims.end()));
        shortestPathCache_.emplace(key, allPaths);
        return allPaths;
    }

    struct DirectionChoice {
        std::vector<int> remainingSteps;
        std::vector<int> stepDirections;
    };
    std::vector<DirectionChoice> choices(1, DirectionChoice{
        std::vector<int>(shape_.size(), 0),
        std::vector<int>(shape_.size(), 0),
    });

    for (std::size_t dim = 0; dim < shape_.size(); ++dim) {
        const int srcValue = srcCoord[dim];
        const int dstValue = dstCoord[dim];
        if (srcValue == dstValue) {
            continue;
        }

        std::vector<DirectionChoice> expanded;
        for (const auto& choice : choices) {
            if (directTopologyKind_ == Synthesizer::DirectTopologyKind::Torus) {
                const int forward = (dstValue - srcValue + shape_[dim]) % shape_[dim];
                const int backward = (srcValue - dstValue + shape_[dim]) % shape_[dim];
                if (forward == backward) {
                    auto plusChoice = choice;
                    plusChoice.remainingSteps[dim] = forward;
                    plusChoice.stepDirections[dim] = 1;
                    expanded.push_back(std::move(plusChoice));

                    auto minusChoice = choice;
                    minusChoice.remainingSteps[dim] = backward;
                    minusChoice.stepDirections[dim] = -1;
                    expanded.push_back(std::move(minusChoice));
                    continue;
                }

                auto updated = choice;
                if (forward < backward) {
                    updated.remainingSteps[dim] = forward;
                    updated.stepDirections[dim] = 1;
                } else {
                    updated.remainingSteps[dim] = backward;
                    updated.stepDirections[dim] = -1;
                }
                expanded.push_back(std::move(updated));
                continue;
            }

            auto updated = choice;
            updated.remainingSteps[dim] = std::abs(dstValue - srcValue);
            updated.stepDirections[dim] = (dstValue > srcValue) ? 1 : -1;
            expanded.push_back(std::move(updated));
        }
        choices = std::move(expanded);
    }

    for (const auto& choice : choices) {
        std::vector<int> currentPath{src};
        auto remaining = choice.remainingSteps;
        appendDirectShortestPaths(srcCoord, remaining, choice.stepDirections,
                                  currentPath, allPaths, kMaxPaths);
        if (allPaths.size() >= kMaxPaths) {
            break;
        }
    }

    shortestPathCache_.emplace(key, allPaths);
    return allPaths;
}

void StandardSynthesizer::appendDirectShortestPaths(
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

    for (std::size_t dim = 0; dim < remainingSteps.size(); ++dim) {
        if (remainingSteps[dim] <= 0) {
            continue;
        }
        auto nextCoord = coord;
        nextCoord[dim] += stepDirections[dim];
        if (directTopologyKind_ == Synthesizer::DirectTopologyKind::Torus) {
            nextCoord[dim] = (nextCoord[dim] % shape_[dim] + shape_[dim]) % shape_[dim];
        }
        const int nextNode = coordinateToNodeID(nextCoord);
        --remainingSteps[dim];
        currentPath.push_back(nextNode);
        appendDirectShortestPaths(nextCoord, remainingSteps, stepDirections,
                                  currentPath, allPaths, maxPaths);
        currentPath.pop_back();
        ++remainingSteps[dim];
        if (allPaths.size() >= maxPaths) {
            return;
        }
    }
}

std::vector<int> StandardSynthesizer::bfsPath(int src, int dst) const {
    if (src == dst) {
        return {src};
    }
    if (usesFormulaRouting()) {
        auto mutableThis = const_cast<StandardSynthesizer*>(this);
        const auto paths = mutableThis->findAllShortestPaths(src, dst);
        if (!paths.empty()) {
            return paths.front();
        }
    }
    if (src < 0 || dst < 0 || src >= totalNodeCount_ || dst >= totalNodeCount_) {
        return {src, dst};
    }

    std::vector<int> prev(totalNodeCount_, -1);
    std::queue<int> q;
    q.push(src);
    prev[src] = src;
    while (!q.empty()) {
        const int current = q.front();
        q.pop();
        for (int next : getNeighborsFromGraph(current)) {
            if (next >= 0 && next < totalNodeCount_ && prev[next] == -1) {
                prev[next] = current;
                if (next == dst) {
                    std::vector<int> path;
                    int node = dst;
                    while (node != src) {
                        path.push_back(node);
                        node = prev[node];
                    }
                    path.push_back(src);
                    std::reverse(path.begin(), path.end());
                    return path;
                }
                q.push(next);
            }
        }
    }
    return {src, dst};
}

const std::vector<int>& StandardSynthesizer::getNeighborsFromGraph(int node) const {
    if (node < 0 || node >= static_cast<int>(connectionGraph_.size())) {
        static const std::vector<int> empty;
        return empty;
    }
    return connectionGraph_[node];
}

std::vector<int> StandardSynthesizer::selectBestPathForLatency(
    const std::vector<std::vector<int>>& paths,
    const LoadState& accumulatedLinkLoad,
    double maxAccumulatedLinkLoad,
    DataSize bytes) const {
    if (paths.empty()) {
        return {};
    }
    if (paths.size() == 1) {
        return paths.front();
    }

    std::vector<double> pathTransferTimes(paths.size(), 0.0);
    double minTime = std::numeric_limits<double>::max();
    for (std::size_t pathIndex = 0; pathIndex < paths.size(); ++pathIndex) {
        const auto& path = paths[pathIndex];
        for (std::size_t hop = 0; hop + 1 < path.size(); ++hop) {
            pathTransferTimes[pathIndex] +=
                computeEdgeTransferTime(edgeIndex(path[hop], path[hop + 1]), bytes);
        }
        minTime = std::min(minTime, pathTransferTimes[pathIndex]);
    }

    const double maxLoad = std::max(maxAccumulatedLinkLoad, 1.0);
    std::vector<int> bestPath = paths.front();
    double bestNormalizedSumLoad = std::numeric_limits<double>::max();
    for (std::size_t pathIndex = 0; pathIndex < paths.size(); ++pathIndex) {
        if (std::abs(pathTransferTimes[pathIndex] - minTime) >= kPathEpsilon) {
            continue;
        }
        const auto& path = paths[pathIndex];
        double pathSumLoad = 0.0;
        for (std::size_t hop = 0; hop + 1 < path.size(); ++hop) {
            pathSumLoad += accumulatedLinkLoad.get(edgeIndex(path[hop], path[hop + 1]));
        }
        const double normalizedSumLoad = pathSumLoad / maxLoad;
        if (normalizedSumLoad < bestNormalizedSumLoad) {
            bestNormalizedSumLoad = normalizedSumLoad;
            bestPath = path;
        }
    }
    return bestPath;
}

std::vector<int> StandardSynthesizer::selectBestPathByLoad(
    const std::vector<std::vector<int>>& paths,
    const LoadState& accumulatedLinkLoad,
    double maxAccumulatedLinkLoad,
    DataSize bytes) const {
    if (paths.empty()) {
        return {};
    }
    if (paths.size() == 1) {
        return paths.front();
    }

    const double maxLoad = std::max(maxAccumulatedLinkLoad, 1.0);
    const double fallbackBandwidthBytesPerUs = bandwidth_ * (1 << 30) / 1e6;
    std::vector<double> pathDataTransfer(paths.size(), 0.0);
    double maxPathDataTransfer = 0.0;

    for (std::size_t pathIndex = 0; pathIndex < paths.size(); ++pathIndex) {
        const auto& path = paths[pathIndex];
        for (std::size_t hop = 0; hop + 1 < path.size(); ++hop) {
            const int idx = edgeIndex(path[hop], path[hop + 1]);
            const double transferTime = computeEdgeTransferTime(idx, bytes);
            double linkBandwidthBytesPerUs = fallbackBandwidthBytesPerUs;
            if (!linkBandwidthBytesPerUs_.empty() && linkBandwidthBytesPerUs_[idx] > 0.0) {
                linkBandwidthBytesPerUs = linkBandwidthBytesPerUs_[idx];
            }
            pathDataTransfer[pathIndex] += transferTime * linkBandwidthBytesPerUs;
        }
        maxPathDataTransfer = std::max(maxPathDataTransfer, pathDataTransfer[pathIndex]);
    }
    if (maxPathDataTransfer < kPathEpsilon) {
        maxPathDataTransfer = 1.0;
    }

    std::vector<int> bestPath = paths.front();
    double bestScore = std::numeric_limits<double>::max();
    for (std::size_t pathIndex = 0; pathIndex < paths.size(); ++pathIndex) {
        const auto& path = paths[pathIndex];
        double pathSumLoad = 0.0;
        double pathMaxLoad = 0.0;
        for (std::size_t hop = 0; hop + 1 < path.size(); ++hop) {
            const double load = accumulatedLinkLoad.get(edgeIndex(path[hop], path[hop + 1]));
            pathSumLoad += load;
            pathMaxLoad = std::max(pathMaxLoad, load);
        }
        const double normalizedSumLoad = pathSumLoad / maxLoad;
        const double normalizedMaxLoad = pathMaxLoad / maxLoad;
        const double normalizedPathDataTransfer = pathDataTransfer[pathIndex] / maxPathDataTransfer;
        const double score =
            normalizedSumLoad * pathScoreSumLoadWeight_ +
            normalizedMaxLoad * pathScoreMaxLoadWeight_ +
            normalizedPathDataTransfer * pathScoreDataTransferWeight_;
        if (score < bestScore) {
            bestScore = score;
            bestPath = path;
        }
    }
    return bestPath;
}

std::pair<double, double> StandardSynthesizer::scheduleEdge(
    SlotState& slots,
    int edgeId,
    double earliestStartTime,
    double transferTime) const {
    double startTime = earliestStartTime;

    std::vector<std::pair<double, double>>* edgeSlotsPtr = nullptr;
    std::size_t* denseHint = nullptr;
    if (slots.dense) {
        int slotId = slots.denseSlotIds[edgeId];
        if (slotId < 0) {
            slotId = static_cast<int>(slots.denseSlots.size());
            slots.denseSlotIds[edgeId] = slotId;
            slots.denseSlots.emplace_back();
            slots.denseSlotHints.push_back(0);
        }
        edgeSlotsPtr = &slots.denseSlots[slotId];
        denseHint = &slots.denseSlotHints[slotId];
    } else {
        edgeSlotsPtr = &slots.sparseSlots[edgeId];
    }
    auto& edgeSlots = *edgeSlotsPtr;

    if (edgeSlots.empty()) {
        edgeSlots.reserve(kInitialSlotReserve);
    }

    std::size_t insertIndex = edgeSlots.size();
    if (!edgeSlots.empty()) {
        if (startTime >= edgeSlots.back().second) {
            const double endTime = startTime + transferTime;
            edgeSlots.push_back({startTime, endTime});
            if (denseHint != nullptr) {
                *denseHint = edgeSlots.size() - 1;
            }
            return {startTime, endTime};
        }

        bool foundSlot = false;
        std::size_t scanStart = 0;
        if (denseHint != nullptr && *denseHint < edgeSlots.size() &&
            startTime >= edgeSlots[*denseHint].second) {
            scanStart = *denseHint;
        } else if (edgeSlots.size() >= kBinarySlotSearchThreshold) {
            const auto firstSlotEndingAfterStart =
                std::upper_bound(edgeSlots.begin(), edgeSlots.end(), startTime,
                                 [](double value, const std::pair<double, double>& slot) {
                                     return value < slot.second;
                                 });
            if (firstSlotEndingAfterStart != edgeSlots.begin()) {
                scanStart = static_cast<std::size_t>(
                    std::distance(edgeSlots.begin(), firstSlotEndingAfterStart) - 1);
            }
        }

        if (scanStart == 0 && startTime + transferTime <= edgeSlots.front().first) {
            foundSlot = true;
            insertIndex = 0;
        } else {
            for (std::size_t idx = scanStart; idx + 1 < edgeSlots.size(); ++idx) {
                const double gapStart = edgeSlots[idx].second;
                const double gapEnd = edgeSlots[idx + 1].first;
                const double requiredStart = std::max(startTime, gapStart);
                if (requiredStart + transferTime <= gapEnd) {
                    startTime = requiredStart;
                    foundSlot = true;
                    insertIndex = idx + 1;
                    break;
                }
            }
        }
        if (!foundSlot) {
            startTime = std::max(startTime, edgeSlots.back().second);
            insertIndex = edgeSlots.size();
        }
    }

    const double endTime = startTime + transferTime;
    if (insertIndex == edgeSlots.size()) {
        edgeSlots.push_back({startTime, endTime});
    } else {
        edgeSlots.insert(edgeSlots.begin() + static_cast<std::ptrdiff_t>(insertIndex),
                         {startTime, endTime});
    }
    if (denseHint != nullptr) {
        *denseHint = insertIndex;
    }
    return {startTime, endTime};
}

std::string StandardSynthesizer::formatNodeName(int nodeID) const {
    if (topology_ != nullptr) {
        return topology_->formatNodeName(nodeID);
    }
    return "node " + std::to_string(nodeID);
}

void StandardSynthesizer::printEvents(const std::vector<StandardEvent>& events) const {
    if (events.empty()) {
        return;
    }
    std::cout << "\n[Transfer Events - Standard Schedule]\n";
    for (const auto& event : events) {
        std::cout << "  Current [chunk, dest]: " << event.chunkId
                  << " (node " << event.flowSrc << " -> node " << event.flowDst
                  << ", size: " << event.bytes << " bytes), "
                  << formatNodeName(event.flowDst) << '\n';
        std::cout << "    " << std::fixed << std::setprecision(6)
                  << "[EventTime " << event.startTime << " us] Chunk "
                  << event.chunkId << ": " << formatNodeName(event.src)
                  << " -> " << formatNodeName(event.dst) << '\n';
    }
}

} // namespace tacos
