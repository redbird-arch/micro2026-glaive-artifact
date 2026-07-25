/*
# File name  :    standard_synthesizer.h
# Description:    Standard-mode fast path for Glaive/Synthesizer3 clean runs.
*/

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <tacos/topology/topology.h>
#include "synthesizer_3/synthesizer_3.h"

namespace tacos {

struct StandardEvent {
    int src;
    int dst;
    DataSize bytes;
    double startTime;
    double endTime;
    std::vector<int> path;
    int chunkId;
    int flowSrc;
    int flowDst;
    bool isLatencyMatrix;
};

struct StandardResult {
    double makespan = 0.0;
    int latencyFlowCount = 0;
    int bandwidthFlowCount = 0;
    DataSize latencyBytes = 0;
    DataSize bandwidthBytes = 0;
    int scheduledChunks = 0;
    int scheduledEvents = 0;
    std::vector<StandardEvent> events;
};

struct StandardProfileSummary {
    double threshold = 0.0;
    int coldFlowCount = 0;
    int hotFlowCount = 0;
    DataSize coldBytes = 0;
    DataSize hotBytes = 0;
};

struct StandardCacheStats {
    int gpuNodeCount = 0;
    int totalNodeCount = 0;
    std::string routingKind;
    bool allSmallSwitchPairsDominant = false;
    std::size_t connectionEdges = 0;
    std::size_t distanceMatrixEntries = 0;
    std::size_t linkCacheEntries = 0;
    std::size_t smallSwitchCandidatePairs = 0;
    std::size_t smallSwitchCandidatePaths = 0;
    std::size_t smallSwitchDominantPaths = 0;
    std::size_t gpuPairMinChunkEntries = 0;
    std::size_t shortestPathCacheEntries = 0;
    std::size_t timeShortestPathCacheEntries = 0;
    std::size_t approximateBytes = 0;
};

class StandardSynthesizer {
public:
    using Bandwidth = double;
    using Latency = double;

    StandardSynthesizer(const std::vector<int>& shape,
                       Synthesizer3::DirectTopologyKind directTopologyKind,
                       Bandwidth bandwidth,
                       Latency latency);
    StandardSynthesizer(std::shared_ptr<Topology> topology,
                       const std::vector<int>& shape,
                       Bandwidth bandwidth,
                       Latency latency);

    void setCaptureSchedule(bool captureSchedule) { captureSchedule_ = captureSchedule; }
    void setMaxBandwidthFlowsOverride(int maxBandwidthFlows) { maxBandwidthFlowsOverride_ = maxBandwidthFlows; }
    void setPathScoreWeights(double sumLoadWeight,
                             double maxLoadWeight,
                             double dataTransferWeight);

    StandardResult solve(const std::vector<std::vector<DataSize>>& demand);
    StandardResult solveSparse(const std::vector<DemandEntry3>& demand);
    StandardResult solveHotOnly(const std::vector<std::vector<DataSize>>& demand);
    StandardResult solveColdOnly(const std::vector<std::vector<DataSize>>& demand);
    StandardResult solveSparseHotOnly(const std::vector<DemandEntry3>& demand);
    StandardResult solveSparseColdOnly(const std::vector<DemandEntry3>& demand);
    StandardProfileSummary profileSummary(const std::vector<std::vector<DataSize>>& demand) const;
    StandardProfileSummary profileSparseSummary(const std::vector<DemandEntry3>& demand) const;
    StandardCacheStats cacheStats() const;

    std::string formatNodeName(int nodeID) const;
    void printEvents(const std::vector<StandardEvent>& events) const;

private:
    struct Flow {
        int src;
        int dst;
        DataSize bytes;
    };

    struct RankedFlow {
        int src;
        int dst;
        DataSize bytes;
        int distance;
        bool isLatencyMatrix;
        double priority;
    };

    struct Query {
        Query(int srcIn, int dstIn, DataSize bytesIn, bool isLatencyMatrixIn)
            : src(srcIn),
              dst(dstIn),
              bytes(bytesIn),
              isLatencyMatrix(isLatencyMatrixIn) {}

        int src;
        int dst;
        DataSize bytes;
        bool isLatencyMatrix;
    };

    struct FixedPath {
        std::array<int, 5> nodes{};
        int length = 0;

        bool empty() const { return length < 2; }
        bool operator==(const FixedPath& other) const {
            if (length != other.length) {
                return false;
            }
            for (int i = 0; i < length; ++i) {
                if (nodes[i] != other.nodes[i]) {
                    return false;
                }
            }
            return true;
        }
    };

    enum class SwitchFastPathKind {
        None,
        FatTreeSingleGlobal,
        CmSingleRail,
    };

    struct TimeShortestPathCacheKey {
        int src;
        int dst;
        DataSize bytes;

        bool operator==(const TimeShortestPathCacheKey& other) const {
            return src == other.src && dst == other.dst && bytes == other.bytes;
        }
    };

    struct TimeShortestPathCacheKeyHash {
        std::size_t operator()(const TimeShortestPathCacheKey& key) const;
    };

    struct SlotState {
        std::vector<int> denseSlotIds;
        std::vector<std::vector<std::pair<double, double>>> denseSlots;
        std::vector<std::size_t> denseSlotHints;
        std::unordered_map<int, std::vector<std::pair<double, double>>> sparseSlots;
        int totalNodeCount;
        bool dense;

        SlotState(int totalNodeCount, bool useDense);
    };

    struct ProfiledFlows {
        double threshold = 0.0;
        std::vector<Flow> latencyFlows;
        std::vector<Flow> bandwidthFlows;
        DataSize latencyBytes = 0;
        DataSize bandwidthBytes = 0;
    };

    struct LoadState {
        std::vector<double> dense;
        std::unordered_map<int, double> sparse;
        bool useDense;

        LoadState(int totalNodeCount, bool denseEnabled);
        double get(int edgeId) const;
        void add(int edgeId, double value);
    };

    void buildConnectionGraph();
    void initializeStaticCaches();
    void initializeSmallSwitchCandidatePaths();

    bool usesFormulaRouting() const { return topology_ == nullptr; }
    bool isGPUNode(int nodeID) const;
    bool isSwitchNode(int nodeID) const;
    int edgeIndex(int src, int dst) const { return src * totalNodeCount_ + dst; }
    std::uint64_t pairCacheKey(int src, int dst) const;

    std::vector<int> nodeIDToCoordinate(int nodeID) const;
    int coordinateToNodeID(const std::vector<int>& coord) const;
    int computeDistance(int src, int dst) const;
    int computeNodeDegree(int nodeID) const;
    int computeMinDegree() const;
    double computeBwLatThreshold() const;
    double computeLinkTransferTime(int src, int dst, DataSize bytes) const;
    double computeEdgeTransferTime(int edgeId, DataSize bytes) const;
    DataSize computeMinChunkSize(int src, int dst) const;
    int computeChunkCount(int src, int dst, DataSize bytes) const;

    ProfiledFlows profileMatrix(const std::vector<std::vector<DataSize>>& demand) const;
    ProfiledFlows profileSparse(const std::vector<DemandEntry3>& demand) const;
    ProfiledFlows profileCandidates(std::vector<Flow>& latencyFlows,
                                    std::vector<Flow>& bandwidthCandidates,
                                    double threshold) const;
    StandardProfileSummary summarizeProfiled(const ProfiledFlows& profiled) const;
    StandardResult solveProfiledSubset(ProfiledFlows profiled,
                                      bool includeLatencyFlows,
                                      bool includeBandwidthFlows);
    StandardResult solveProfiled(const ProfiledFlows& profiled);

    std::vector<std::vector<int>> findTimeShortestPaths(int src, int dst, DataSize bytes);
    std::vector<std::vector<int>> computeTimeShortestPathsUncached(int src, int dst, DataSize bytes) const;
    FixedPath computeSwitchFastPathFixed(int src, int dst, DataSize bytes) const;
    std::vector<int> computeSwitchFastPath(int src, int dst, DataSize bytes) const;
    FixedPath computeSmallSwitchFastPathFixed(int src, int dst, DataSize bytes) const;
    std::vector<int> computeSmallSwitchFastPath(int src, int dst, DataSize bytes) const;
    std::vector<int> fixedPathToVector(const FixedPath& path) const;
    std::vector<std::vector<int>> findAllShortestPaths(int src, int dst);
    std::vector<std::vector<int>> enumerateDirectShortestPaths(int src, int dst);
    void appendDirectShortestPaths(const std::vector<int>& coord,
                                   std::vector<int>& remainingSteps,
                                   const std::vector<int>& stepDirections,
                                   std::vector<int>& currentPath,
                                   std::vector<std::vector<int>>& allPaths,
                                   std::size_t maxPaths) const;
    std::vector<int> bfsPath(int src, int dst) const;
    const std::vector<int>& getNeighborsFromGraph(int node) const;

    std::vector<int> selectBestPathForLatency(
        const std::vector<std::vector<int>>& paths,
        const LoadState& accumulatedLinkLoad,
        double maxAccumulatedLinkLoad,
        DataSize bytes) const;
    std::vector<int> selectBestPathByLoad(
        const std::vector<std::vector<int>>& paths,
        const LoadState& accumulatedLinkLoad,
        double maxAccumulatedLinkLoad,
        DataSize bytes) const;

    std::pair<double, double> scheduleEdge(SlotState& slots,
                                           int edgeId,
                                           double earliestStartTime,
                                           double transferTime) const;

    std::vector<int> shape_;
    Synthesizer3::DirectTopologyKind directTopologyKind_;
    bool isTorus_;
    int npusCount_;
    int gpuNodeCount_;
    int totalNodeCount_;
    Bandwidth bandwidth_;
    Latency latency_;
    std::shared_ptr<Topology> topology_;
    double pathScoreSumLoadWeight_ = 0.3;
    double pathScoreMaxLoadWeight_ = 0.2;
    double pathScoreDataTransferWeight_ = 0.5;

    std::vector<std::vector<int>> connectionGraph_;
    std::vector<int> nodeDegreeCache_;
    int minGpuDegree_ = -1;
    std::vector<int> distanceMatrix_;
    std::vector<double> linkLatencyUs_;
    std::vector<double> linkBandwidthBytesPerUs_;
    std::vector<double> linkInvBandwidthUsPerByte_;
    std::unordered_map<std::uint64_t, std::vector<std::vector<int>>> shortestPathCache_;
    std::vector<std::vector<FixedPath>> smallSwitchCandidatePaths_;
    std::vector<FixedPath> smallSwitchDominantPaths_;
    std::vector<DataSize> gpuPairMinChunkSize_;
    std::unordered_map<TimeShortestPathCacheKey,
                       std::vector<std::vector<int>>,
                       TimeShortestPathCacheKeyHash> timeShortestPathCache_;

    bool captureSchedule_ = false;
    int maxBandwidthFlowsOverride_ = 0;
    bool allSmallSwitchPairsDominant_ = false;
    SwitchFastPathKind switchFastPathKind_ = SwitchFastPathKind::None;
};

} // namespace tacos
