/*
# File name  :    synthesizer.h
# Author     :    Galois
# Time       :    2026/01/14
# Description:    Synthesizer - Profiling-Scheduling-Fusion three-step workflow
#                 Designed for non-uniform AllToAll scenarios
#                 - Profiling: Decompose matrix into latency-dominant and bandwidth-dominant matrices
#                 - Scheduling: Fast scheduling for latency matrix, chunk-based scheduling for bandwidth matrix
#                 - Fusion: Merge two schedules to minimize total makespan
*/

#pragma once

#include <vector>
#include <memory>
#include <string>
#include <map>
#include <set>
#include <iostream>
#include <iomanip>
#include <queue>
#include <algorithm>
#include <limits>
#include <functional>
#include <unordered_map>
#include <cstdint>
#include <tacos/topology/topology.h>

namespace tacos {

// Forward declare DataSize type
using DataSize = long long;

struct DemandEntry {
    int src;
    int dst;
    DataSize bytes;
};

//=============================================================================
// TransferEvent: A single transfer event in the schedule
//=============================================================================
struct TransferEvent {
    int src;
    int dst;
    DataSize bytes;
    double startTime;        // Continuous start time (us)
    double endTime;          // Continuous end time (us)
    std::vector<int> path;   // Path from src to dst
    int chunkId;             // Chunk ID (for bandwidth matrix chunks)
    int flowSrc;             // Original source of the flow
    int flowDst;             // Original destination of the flow
    bool isLatencyMatrix;    // Whether this event belongs to latency matrix

    bool operator<(const TransferEvent& other) const {
        if (startTime != other.startTime) return startTime < other.startTime;
        return src < other.src;
    }
};

//=============================================================================
// ProfilingResult: Result from profiling phase
//=============================================================================
struct ProfilingResult {
    double bwLatThreshold;
    std::vector<std::vector<DataSize>> latencyMatrix;
    std::vector<std::vector<DataSize>> bandwidthMatrix;
    std::vector<DemandEntry> latencyFlows;
    std::vector<DemandEntry> bandwidthFlows;
    int latencyMatrixNonZeros;
    int bandwidthMatrixNonZeros;
    DataSize latencyMatrixTotalBytes;
    DataSize bandwidthMatrixTotalBytes;

    void print() const {
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                    PROFILING RESULT                         ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Bandwidth-Latency Threshold: " << std::setw(20)
                  << std::fixed << std::setprecision(2) << bwLatThreshold << " bytes ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Latency Matrix:                                             ║\n";
        std::cout << "║   - Threshold: <= " << std::setw(20)
                  << std::fixed << std::setprecision(2) << bwLatThreshold << " bytes ║\n";

        // Calculate min and max for latency matrix
        DataSize latencyMin = std::numeric_limits<DataSize>::max();
        DataSize latencyMax = 0;
        for (int i = 0; i < static_cast<int>(latencyMatrix.size()); ++i) {
            for (int j = 0; j < static_cast<int>(latencyMatrix[i].size()); ++j) {
                if (latencyMatrix[i][j] > 0) {
                    latencyMin = std::min(latencyMin, latencyMatrix[i][j]);
                    latencyMax = std::max(latencyMax, latencyMatrix[i][j]);
                }
            }
        }
        if (latencyMin == std::numeric_limits<DataSize>::max()) latencyMin = 0;
        std::cout << "║   - Min value: " << std::setw(22) << latencyMin << " bytes              ║\n";
        std::cout << "║   - Max value: " << std::setw(22) << latencyMax << " bytes              ║\n";
        std::cout << "║   - Non-zero entries: " << std::setw(6) << latencyMatrixNonZeros << "                    ║\n";
        std::cout << "║   - Total bytes: " << std::setw(15) << latencyMatrixTotalBytes
                  << " bytes              ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Bandwidth Matrix:                                           ║\n";
        std::cout << "║   - Threshold: > " << std::setw(21)
                  << std::fixed << std::setprecision(2) << bwLatThreshold << " bytes ║\n";

        // Calculate min and max for bandwidth matrix
        DataSize bandwidthMin = std::numeric_limits<DataSize>::max();
        DataSize bandwidthMax = 0;
        for (int i = 0; i < static_cast<int>(bandwidthMatrix.size()); ++i) {
            for (int j = 0; j < static_cast<int>(bandwidthMatrix[i].size()); ++j) {
                if (bandwidthMatrix[i][j] > 0) {
                    bandwidthMin = std::min(bandwidthMin, bandwidthMatrix[i][j]);
                    bandwidthMax = std::max(bandwidthMax, bandwidthMatrix[i][j]);
                }
            }
        }
        if (bandwidthMin == std::numeric_limits<DataSize>::max()) bandwidthMin = 0;
        std::cout << "║   - Min value: " << std::setw(22) << bandwidthMin << " bytes              ║\n";
        std::cout << "║   - Max value: " << std::setw(22) << bandwidthMax << " bytes              ║\n";
        std::cout << "║   - Non-zero entries: " << std::setw(6) << bandwidthMatrixNonZeros << "                    ║\n";
        std::cout << "║   - Total bytes: " << std::setw(15) << bandwidthMatrixTotalBytes
                  << " bytes              ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    }

    void printMatrices(int npusCount) const {
        // Print full matrices in CSV format (values divided by 4096 to show original units)
        const long long unitSize = 4 * 1024;  // 4096 bytes (4KB)

        std::cout << "\n[Latency Matrix (Full)]\n";
        std::cout << "Format: CSV-like matrix (values in original per-token-size units)\n";
        for (int i = 0; i < npusCount; ++i) {
            for (int j = 0; j < npusCount; ++j) {
                long long originalValue = latencyMatrix[i][j] / unitSize;
                std::cout << originalValue;
                if (j < npusCount - 1) {
                    std::cout << ",";
                }
            }
            std::cout << "\n";
        }

        std::cout << "\n[Bandwidth Matrix (Full)]\n";
        std::cout << "Format: CSV-like matrix (values in original per-token-size units)\n";
        for (int i = 0; i < npusCount; ++i) {
            for (int j = 0; j < npusCount; ++j) {
                long long originalValue = bandwidthMatrix[i][j] / unitSize;
                std::cout << originalValue;
                if (j < npusCount - 1) {
                    std::cout << ",";
                }
            }
            std::cout << "\n";
        }
    }
};

//=============================================================================
// ScheduleResult: Complete schedule result
//=============================================================================
struct ScheduleResult {
    std::vector<TransferEvent> events;
    double makespan;
    double latencyMatrixMakespan;
    double bandwidthMatrixMakespan;
    double fusedMakespan;
    double fusedSchedulingMakespan;  // Makespan from fused scheduling (from scratch)

    // Store formatNodeName function for printing (set by Synthesizer)
    std::function<std::string(int)> formatNodeNameFunc;

    void print() const {
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                  SCHEDULE RESULT SUMMARY                     ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Latency Matrix Makespan: " << std::setw(20)
                  << std::fixed << std::setprecision(2) << latencyMatrixMakespan << " us ║\n";
        std::cout << "║ Bandwidth Matrix Makespan: " << std::setw(19)
                  << bandwidthMatrixMakespan << " us ║\n";
        std::cout << "║ Fused Makespan: " << std::setw(28)
                  << fusedMakespan << " us ║\n";
        std::cout << "║ Fused Scheduling Makespan: " << std::setw(20)
                  << fusedSchedulingMakespan << " us ║\n";
        std::cout << "║ Total Events: " << std::setw(31) << events.size() << "        ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    }

    void printEvents(std::function<std::string(int)> formatNodeName = nullptr) const {
        auto formatter = formatNodeName ? formatNodeName : formatNodeNameFunc;
        std::cout << "\n[Transfer Events - Step by Step Schedule]\n";

        // Aggregate per-chunk information
        std::map<int, int> chunkFinalDest;                 // chunkId -> final destination
        std::map<int, std::pair<int, int>> chunkFlow;      // chunkId -> (flowSrc, flowDst)
        std::map<int, DataSize> chunkBytes;                // chunkId -> bytes (per chunk)
        std::map<int, double> chunkFirstTime;              // chunkId -> earliest start time
        std::map<int, std::vector<const TransferEvent*>> chunkEvents; // chunkId -> per-hop events

        for (const auto& e : events) {
            if (e.chunkId < 0) continue;
            int cid = e.chunkId;
            chunkFinalDest[cid] = e.flowDst;
            chunkFlow.emplace(cid, std::make_pair(e.flowSrc, e.flowDst));
            chunkBytes[cid] = e.bytes;
            auto it = chunkFirstTime.find(cid);
            if (it == chunkFirstTime.end() || e.startTime < it->second) {
                chunkFirstTime[cid] = e.startTime;
            }
            chunkEvents[cid].push_back(&e);
        }

        // Assign block index per (flowSrc, flowDst)
        std::map<int, int> chunkBlockIndex;  // chunkId -> which block (1-based) for that flow
        {
            std::map<std::pair<int, int>, std::set<int>> pairToChunkIds;
            for (const auto& kv : chunkFlow) {
                int cid = kv.first;
                auto flow = kv.second;
                pairToChunkIds[flow].insert(cid);
            }
            for (auto& kv : pairToChunkIds) {
                int idx = 1;
                for (int cid : kv.second) {
                    chunkBlockIndex[cid] = idx++;
                }
            }
        }

        // Order chunks by earliest start time, then by the earlier second hop.
        std::vector<int> orderedChunks;
        orderedChunks.reserve(chunkFirstTime.size());
        for (const auto& kv : chunkFirstTime) {
            orderedChunks.push_back(kv.first);
        }
        std::sort(orderedChunks.begin(), orderedChunks.end(),
                  [&](int a, int b) {
                      double ta = chunkFirstTime[a];
                      double tb = chunkFirstTime[b];
                      if (ta != tb) return ta < tb;

                      // tie-break 1: compare second-hop start time (if available)
                      auto getSecondHopTime = [&](int cid) -> double {
                          auto it = chunkEvents.find(cid);
                          if (it == chunkEvents.end()) return std::numeric_limits<double>::infinity();
                          auto evs = it->second;
                          std::sort(evs.begin(), evs.end(),
                                    [](const TransferEvent* x, const TransferEvent* y) {
                                        if (x->startTime != y->startTime) return x->startTime < y->startTime;
                                        if (x->src != y->src) return x->src < y->src;
                                        return x->dst < y->dst;
                                    });
                          if (evs.size() < 2) {
                              return evs.empty() ? std::numeric_limits<double>::infinity()
                                                 : evs.front()->startTime;
                          }
                          return evs[1]->startTime;
                      };

                      double sa = getSecondHopTime(a);
                      double sb = getSecondHopTime(b);
                      if (sa != sb) return sa < sb;

                      // final tie-break: chunkId
                      return a < b;
                  });

        // Print chunks in time order, and within each chunk print hops in time order
        for (int cid : orderedChunks) {
            auto flowIt = chunkFlow.find(cid);
            if (flowIt == chunkFlow.end()) continue;
            int flowSrc = flowIt->second.first;
            int flowDst = flowIt->second.second;
            int finalDest = chunkFinalDest[cid];
            std::string destName = formatter ? formatter(finalDest) : std::to_string(finalDest);
            DataSize bytes = chunkBytes[cid];
            int blockNo = chunkBlockIndex.count(cid) ? chunkBlockIndex.at(cid) : 0;

            std::cout << "  Current [chunk, dest]: " << cid
                      << " (node " << flowSrc << " -> node " << flowDst
                      << ", size: " << bytes << " bytes [block " << blockNo << "]), "
                      << destName << std::endl;

            auto& evs = chunkEvents[cid];
            std::sort(evs.begin(), evs.end(),
                      [](const TransferEvent* a, const TransferEvent* b) {
                          if (a->startTime != b->startTime) return a->startTime < b->startTime;
                          if (a->src != b->src) return a->src < b->src;
                          return a->dst < b->dst;
                      });

            for (const auto* e : evs) {
                std::string srcName = formatter ? formatter(e->src) : std::to_string(e->src);
                std::string dstName = formatter ? formatter(e->dst) : std::to_string(e->dst);
                std::cout << "    " << std::fixed << std::setprecision(6)
                          << "[EventTime " << e->startTime << " us] Chunk "
                          << e->chunkId << ": " << srcName << " -> " << dstName << std::endl;
            }
        }
    }
};

/**
 * Synthesizer: Three-step workflow synthesizer for non-uniform AllToAll
 *
 * Workflow:
 * 1. Profiling: Decompose demand matrix into latency-dominant and bandwidth-dominant matrices
 * 2. Scheduling:
 *    - Latency matrix: Shortest path routing, minimize congestion, fast scheduling
 *    - Bandwidth matrix: Chunk-based scheduling, maximize bandwidth utilization
 * 3. Fusion: Merge schedules and optimize makespan
 */
class Synthesizer {
public:
    using DataSize = long long;
    using Time = double;
    using Bandwidth = double;
    using Latency = double;

    enum class DirectTopologyKind {
        Mesh,
        Torus,
        FullMesh
    };

    /**
     * Constructor
     * @param shape Topology shape (e.g., {8, 8} for 8x8 mesh)
     * @param isTorus Whether the topology is a torus (true) or mesh (false)
     * @param bandwidth Link bandwidth in GB/s
     * @param latency Link latency in nanoseconds
     */
    Synthesizer(const std::vector<int>& shape, bool isTorus,
                 Bandwidth bandwidth, Latency latency);

    Synthesizer(const std::vector<int>& shape,
                 DirectTopologyKind directTopologyKind,
                 Bandwidth bandwidth, Latency latency);

    /**
     * Constructor with Topology object (for switch-based topologies)
     * @param topology Topology object (for path computation)
     * @param shape Topology shape (e.g., {8, 8} for GPU layout)
     * @param bandwidth Link bandwidth in GB/s
     * @param latency Link latency in microseconds
     */
    Synthesizer(std::shared_ptr<Topology> topology, const std::vector<int>& shape,
                 Bandwidth bandwidth, Latency latency);

    /**
     * Configure solver options
     */
    void setVerbose(bool verbose);
    void setPrintSchedule(bool printSchedule);
    void setCleanMode(bool cleanMode) { cleanMode_ = cleanMode; }

    /**
     * Solve the non-uniform AllToAll problem
     * @param demand The demand matrix where demand[i][j] = bytes from node i to node j
     * @return The makespan (total time) in microseconds
     */
    Time solve(const std::vector<std::vector<DataSize>>& demand);

    Time solveSparse(const std::vector<DemandEntry>& demand);

    /**
     * Get profiling result
     */
    const ProfilingResult& getProfilingResult() const { return profilingResult_; }

    /**
     * Get schedule result
     */
    const ScheduleResult& getScheduleResult() const { return scheduleResult_; }

    /**
     * Get NPU count
     */
    int npusCount() const { return npusCount_; }

private:
    struct TimeShortestPathCacheKey {
        int src;
        int dst;
        DataSize bytes;

        bool operator==(const TimeShortestPathCacheKey& other) const {
            return src == other.src && dst == other.dst && bytes == other.bytes;
        }
    };

    struct TimeShortestPathCacheKeyHash {
        std::size_t operator()(const TimeShortestPathCacheKey& key) const {
            std::size_t seed = std::hash<int>{}(key.src);
            seed ^= std::hash<int>{}(key.dst) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= std::hash<DataSize>{}(key.bytes) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };

    //=========================================================================
    // Profiling phase
    //=========================================================================
    ProfilingResult profileMatrix(const std::vector<std::vector<DataSize>>& demand);
    ProfilingResult profileFlows(const std::vector<DemandEntry>& demand);
    std::vector<DemandEntry> collectMatrixFlows(
        const std::vector<std::vector<DataSize>>& matrix) const;
    void finalizeProfilingStats(ProfilingResult& result) const;

    // Compute bandwidth-latency threshold
    double computeBwLatThreshold() const;

    // Compute minimum degree in topology
    int computeMinDegree() const;

    // Compute node degree
    int computeNodeDegree(int nodeID) const;

    // Convert node ID to coordinates
    std::vector<int> nodeIDToCoordinate(int nodeID) const;

    // Convert coordinates to node ID
    int coordinateToNodeID(const std::vector<int>& coord) const;

    //=========================================================================
    // Scheduling phase - Latency matrix
    //=========================================================================
    std::vector<TransferEvent> scheduleLatencyMatrix(
        const std::vector<std::vector<DataSize>>& latencyMatrix);
    std::vector<TransferEvent> scheduleLatencyFlows(
        const std::vector<DemandEntry>& latencyFlows);

    // Calculate bandwidth utilization for a set of events
    double calculateBandwidthUtilization(
        const std::vector<TransferEvent>& events,
        double makespan) const;

    // Print step-by-step schedule for a matrix
    void printStepByStepSchedule(
        const std::vector<TransferEvent>& events,
        const std::string& matrixName) const;

    // Print per-link busy intervals (ns) and per-link utilization for a set of events
    void printLinkBusyIntervals(
        const std::vector<TransferEvent>& events,
        double makespan,
        const std::string& scopeName) const;

    // Compute shortest path between two nodes (with load-aware selection)
    std::vector<int> computeShortestPath(int src, int dst,
                                         const std::map<std::pair<int, int>, double>& linkBusyUntil = {}) const;

    // BFS to find shortest path (fallback)
    std::vector<int> bfsPath(int src, int dst) const;

    // Find all shortest paths between two nodes
    std::vector<std::vector<int>> findAllShortestPaths(int src, int dst) const;
    std::vector<std::vector<int>> enumerateDirectShortestPaths(int src, int dst) const;
    void appendDirectShortestPaths(
        const std::vector<int>& coord,
        std::vector<int>& remainingSteps,
        const std::vector<int>& stepDirections,
        std::vector<int>& currentPath,
        std::vector<std::vector<int>>& allPaths,
        std::size_t maxPaths) const;

    // Get neighbors of a node
    std::vector<int> getNeighbors(int node) const;

    // Select best path from multiple paths based on link load (OLD - using linkBusyUntil)
    std::vector<int> selectBestPath(const std::vector<std::vector<int>>& paths,
                                    const std::map<std::pair<int, int>, double>& linkBusyUntil,
                                    double currentTime) const;

    // Select best path for load balancing (considering edge loads, not just busy times) (OLD)
    std::vector<int> selectBestPathForLoadBalancing(
        const std::vector<std::vector<int>>& paths,
        const std::map<std::pair<int, int>, double>& linkBusyUntil,
        const std::map<std::pair<int, int>, DataSize>& edgeLoads,
        double currentTime) const;

    // NEW: Find time-shortest paths (based on transfer time, not just hop count)
    // Supports heterogeneous link bandwidths
    std::vector<std::vector<int>> findTimeShortestPaths(int src, int dst, DataSize bytes) const;

    // Compute time-shortest paths without touching shared caches.
    std::vector<std::vector<int>> computeTimeShortestPathsUncached(
        int src, int dst, DataSize bytes) const;

    // NEW: Select best path based on accumulated link load and load balancing
    std::vector<int> selectBestPathByLoad(
        const std::vector<std::vector<int>>& paths,
        const std::unordered_map<int, double>& accumulatedLinkLoad,
        double maxAccumulatedLinkLoad,
        DataSize chunkSize) const;

    // NEW: Select best path for latency matrix (simplified: time-shortest, then min normalizedSumLoad)
    std::vector<int> selectBestPathForLatency(
        const std::vector<std::vector<int>>& paths,
        const std::unordered_map<int, double>& accumulatedLinkLoad,
        double maxAccumulatedLinkLoad,
        DataSize bytes) const;

    // NEW: Optimize event scheduling by filling gaps (bubbles) with independent tasks
    void optimizeEventTiming(
        std::vector<TransferEvent>& events,
        const std::map<int, std::set<int>>& chunkDependencies) const;

    // Allocate time slots for events (for individual matrix scheduling display)
    double allocateTimeSlotsForEvents(std::vector<TransferEvent>& events) const;

    // Determine the hop index of an event inside its path
    int getHopIndex(const TransferEvent& event) const;

    // Sort events in chunk/path order before time-slot allocation
    void sortEventsForScheduling(std::vector<TransferEvent>& events) const;

    // Validate that a scheduled event list is link-safe and causally consistent
    void validateScheduleOrThrow(
        const std::vector<TransferEvent>& events,
        const std::string& scheduleName) const;

    // Find K shortest paths (may include slightly longer paths for load balancing)
    std::vector<std::vector<int>> findKShortestPaths(int src, int dst, int k) const;

    // Find alternative paths (may be longer than shortest) for detour routing
    std::vector<std::vector<int>> findAlternativePaths(int src, int dst, int maxHops) const;

    // Evaluate path cost considering congestion and detour penalty
    double evaluatePathCost(const std::vector<int>& path,
                           const std::map<std::pair<int, int>, double>& linkBusyUntil,
                           const std::map<std::pair<int, int>, DataSize>& edgeLoads,
                           DataSize chunkSize,
                           double currentTime,
                           int shortestPathLength) const;

    //=========================================================================
    // Scheduling phase - Bandwidth matrix
    //=========================================================================
    std::vector<TransferEvent> scheduleBandwidthMatrix(
        const std::vector<std::vector<DataSize>>& bandwidthMatrix);
    std::vector<TransferEvent> scheduleBandwidthFlows(
        const std::vector<DemandEntry>& bandwidthFlows);

    // Compute chunk count for a flow
    // Compute minimum chunk size based on bandwidth * latency
    DataSize computeMinChunkSize(int src, int dst) const;

    int computeChunkCount(int src, int dst, DataSize bytes) const;

    // Compute distance between two nodes (hop count)
    int computeDistance(int src, int dst) const;

    //=========================================================================
    // Fusion phase
    //=========================================================================
    ScheduleResult fuseSchedules(
        const std::vector<TransferEvent>& latencyEvents,
        const std::vector<TransferEvent>& bandwidthEvents);

    // Compute transfer time using alpha-beta model
    double computeTransferTime(DataSize bytes, int hops) const;

    // Compute transfer time for a single link (supports heterogeneous links)
    double computeLinkTransferTime(int src, int dst, DataSize bytes) const;

    // Event-driven simulation for fusion
    double simulateFusedSchedule(
        std::vector<TransferEvent>& allEvents) const;

    //=========================================================================
    // Fused Scheduling phase (from scratch with mixed network state)
    //=========================================================================
    std::vector<TransferEvent> fusedScheduling(
        const std::vector<std::vector<DataSize>>& latencyMatrix,
        const std::vector<std::vector<DataSize>>& bandwidthMatrix);
    std::vector<TransferEvent> fusedScheduling(
        const std::vector<DemandEntry>& latencyFlows,
        const std::vector<DemandEntry>& bandwidthFlows);

    //=========================================================================
    // Helper functions
    //=========================================================================
    // Check if link is available at given time
    bool isLinkAvailable(int src, int dst, double time,
                        const std::map<std::pair<int, int>, double>& linkBusyUntil) const;

    // Update link busy time
    void updateLinkBusyTime(int src, int dst, double endTime,
                           std::map<std::pair<int, int>, double>& linkBusyUntil) const;

    // Check if node has data available at given time (store-and-forward)
    bool isDataAvailable(int node, int chunkId, double time,
                        const std::map<int, std::map<int, double>>& dataArrivalTime) const;

    // Update data arrival time
    void updateDataArrivalTime(int node, int chunkId, double arrivalTime,
                              std::map<int, std::map<int, double>>& dataArrivalTime) const;

    //=========================================================================
    // Topology connection pre-computation
    //=========================================================================
    // Build connection graph (adjacency list) for all topologies
    void buildConnectionGraph();

    // Build cached edge metadata and all-pairs hop distances for the static graph
    void initializeStaticCaches();

    // Check if a node is a GPU node (for switch-based topologies)
    bool isGPUNode(int nodeID) const;

    // Check if a node is a switch node (for switch-based topologies)
    bool isSwitchNode(int nodeID) const;

    // Get neighbors from pre-built connection graph
    const std::vector<int>& getNeighborsFromGraph(int node) const;

    // Format node name for display (e.g., "node 0", "switch 1-2")
    std::string formatNodeName(int nodeID) const;

    bool usesFormulaRouting() const;

    // Flatten a directed edge into a dense array index
    int edgeIndex(int src, int dst) const;

    // Build a stable cache key for src/dst path lookups
    std::uint64_t pairCacheKey(int src, int dst) const;
    std::uint64_t chunkArrivalKey(int node, int chunkId) const;

    std::vector<int> shape_;
    bool isTorus_;
    DirectTopologyKind directTopologyKind_;
    int npusCount_;
    int gpuNodeCount_;         // Number of GPU nodes (for switch-based topologies)
    Bandwidth bandwidth_;      // GB/s
    Latency latency_;          // nanoseconds
    std::shared_ptr<Topology> topology_;  // For switch-based topologies

    // Pre-built connection graph: adjacency list for all nodes (including switches)
    // For direct-connect topologies: only GPU nodes, neighbors are direct GPU neighbors
    // For switch-based topologies: includes switches, GPU nodes connect to switches, switches connect to GPUs and other switches
    std::vector<std::vector<int>> connectionGraph_;
    int totalNodeCount_;
    std::vector<int> distanceMatrix_;
    std::vector<double> linkLatencyUs_;
    std::vector<double> linkBandwidthBytesPerUs_;
    mutable std::unordered_map<std::uint64_t, std::vector<int>> bfsPathCache_;
    mutable std::unordered_map<std::uint64_t, std::vector<std::vector<int>>> shortestPathCache_;
    mutable std::unordered_map<TimeShortestPathCacheKey,
                               std::vector<std::vector<int>>,
                               TimeShortestPathCacheKeyHash> timeShortestPathCache_;

    bool verbose_;
    bool printSchedule_;
    bool cleanMode_;

    ProfilingResult profilingResult_;
    ScheduleResult scheduleResult_;
};

} // namespace tacos
