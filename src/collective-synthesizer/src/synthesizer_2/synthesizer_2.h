/*
# File name  :    synthesizer_2.h
# Author     :    Chen Chen
# Time       :    2025/01/06
# Description:    Synthesizer2 - Unified solver interface for non-uniform AllToAll
#                 Wraps the unified solver modules for integration with main.cpp
#                 Similar interface to the original Synthesizer class
*/

#pragma once

#include <vector>
#include <memory>
#include <string>
#include <map>
#include <set>
#include <iostream>
#include <iomanip>

#include "unified_solver.h"
#include <tacos/topology/topology.h>

namespace tacos {

// Forward declare DataSize type
using DataSize = long long;

//=============================================================================
// ChunkInfo: Information about a single chunk
//=============================================================================
struct ChunkInfo {
    int chunkId;
    int srcNpu;      // Original source NPU
    int dstNpu;      // Final destination NPU
    DataSize byteSize;  // Size in bytes
};

//=============================================================================
// ChunkTransferEvent: A transfer event for a chunk
//=============================================================================
struct ChunkTransferEvent {
    int chunkId;
    int fromNpu;
    int toNpu;
    double eventTime;  // in microseconds

    bool operator<(const ChunkTransferEvent& other) const {
        if (eventTime != other.eventTime) return eventTime < other.eventTime;
        return chunkId < other.chunkId;
    }
};

//=============================================================================
// ChunkBasedResult: Result with chunk-level granularity
//=============================================================================
struct ChunkBasedResult {
    // Chunk information
    std::vector<ChunkInfo> chunks;
    int totalChunks;
    DataSize chunkSizeBytes;  // GCD-based chunk size

    // Precondition: chunks initially at each NPU
    // precondition[npu] = list of chunk IDs
    std::vector<std::vector<int>> precondition;

    // Postcondition: chunks finally at each NPU
    // postcondition[npu] = list of chunk IDs
    std::vector<std::vector<int>> postcondition;

    // Transfer events sorted by time
    std::vector<ChunkTransferEvent> transferEvents;

    // Makespan
    double makespan;

    void printPrecondition() const {
        std::cout << "[AlltoAllV] Precondition:\n";
        for (size_t npu = 0; npu < precondition.size(); ++npu) {
            std::cout << "\tNPU " << npu << ": ";
            for (int chunkId : precondition[npu]) {
                std::cout << chunkId << " ";
            }
            std::cout << "\n";
        }
    }

    void printPostcondition() const {
        std::cout << "[AlltoAllV] Postcondition:\n";
        for (size_t npu = 0; npu < postcondition.size(); ++npu) {
            std::cout << "\tNPU " << npu << ": ";
            for (int chunkId : postcondition[npu]) {
                std::cout << chunkId << " ";
            }
            std::cout << "\n";
        }
    }

    void printTransferEvents() const {
        for (const auto& event : transferEvents) {
            std::cout << std::fixed << std::setprecision(6)
                      << "[EventTime " << event.eventTime << " us] Chunk "
                      << event.chunkId << ": " << event.fromNpu << " -> " << event.toNpu << "\n";
        }
    }

    void printPostconditionsList() const {
        std::cout << "Postconditions (final) [chunk, dest]: ";
        bool first = true;
        for (size_t npu = 0; npu < postcondition.size(); ++npu) {
            for (int chunkId : postcondition[npu]) {
                // Only print chunks that need to move (not already at destination)
                if (chunks[chunkId].srcNpu != static_cast<int>(npu)) {
                    if (!first) std::cout << ", ";
                    std::cout << "[" << chunkId << ", " << npu << "]";
                    first = false;
                }
            }
        }
        std::cout << ", \n";
    }
};

/**
 * Synthesizer2: Alternative synthesizer for non-uniform AllToAll
 *
 * This class provides a similar interface to the original Synthesizer,
 * but uses a different algorithm based on:
 * - Matrix analysis for traffic pattern detection
 * - Congestion-aware routing
 * - Optimal scheduling with discrete event simulation
 * - Optional diffusion-based load balancing
 */
class Synthesizer2 {
public:
    using DataSize = long long;
    using Time = double;
    using Bandwidth = double;
    using Latency = double;

    /**
     * Constructor
     * @param shape Topology shape (e.g., {4, 4} for 4x4 mesh)
     * @param isTorus Whether the topology is a torus (true) or mesh (false)
     * @param bandwidth Link bandwidth in GB/s
     * @param latency Link latency in nanoseconds
     */
    Synthesizer2(const std::vector<int>& shape, bool isTorus,
                 Bandwidth bandwidth, Latency latency);

    /**
     * Constructor with Topology object (for switch-based topologies)
     * @param topology Topology object (for path computation)
     * @param shape Topology shape (e.g., {8, 8} for GPU layout)
     * @param bandwidth Link bandwidth in GB/s
     * @param latency Link latency in nanoseconds
     */
    Synthesizer2(std::shared_ptr<Topology> topology, const std::vector<int>& shape,
                 Bandwidth bandwidth, Latency latency);

    /**
     * Configure solver options
     */
    void setVerbose(bool verbose);
    void setPrintSchedule(bool printSchedule);
    void setEnableDiffusion(bool enable);
    void setRoutingStrategy(RoutingStrategy strategy);
    void setSchedulingPolicy(SchedulingPolicy policy);
    void setChunkSize(DataSize chunkSize);

    /**
     * Solve the non-uniform AllToAll problem
     * @param demand The demand matrix where demand[i][j] = bytes from node i to node j
     * @return The makespan (total time) in microseconds
     */
    Time solve(const std::vector<std::vector<DataSize>>& demand);

    /**
     * Solve with chunk-based output (similar to original solver format)
     * @param demand The demand matrix
     * @param chunkSize The chunk size in bytes
     * @return ChunkBasedResult with precondition, postcondition, and transfer events
     */
    ChunkBasedResult solveWithChunks(const std::vector<std::vector<DataSize>>& demand,
                                      DataSize chunkSize);

    /**
     * Get the last solve result (for detailed analysis)
     */
    const SolverResult& getLastResult() const { return lastResult_; }

    /**
     * Compare different routing strategies
     * @param demand The demand matrix
     */
    void compareStrategies(const std::vector<std::vector<DataSize>>& demand);

    /**
     * Get NPU count
     */
    int npusCount() const { return npusCount_; }

private:
    std::vector<int> shape_;
    bool isTorus_;
    int npusCount_;
    Bandwidth bandwidth_;
    Latency latency_;
    DataSize chunkSize_;
    std::shared_ptr<Topology> topology_;  // For switch-based topologies

    SolverConfig config_;
    SolverResult lastResult_;

    std::unique_ptr<UnifiedSolver> solver_;

    // Helper to convert flow-based result to chunk-based result
    ChunkBasedResult convertToChunkBased(const SolverResult& result,
                                          const std::vector<std::vector<DataSize>>& demand,
                                          DataSize chunkSize);
};

} // namespace tacos
