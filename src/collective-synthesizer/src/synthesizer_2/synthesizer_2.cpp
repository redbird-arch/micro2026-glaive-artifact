/*
# File name  :    synthesizer_2.cpp
# Author     :    Chen Chen
# Time       :    2025/01/06
# Description:    Implementation of Synthesizer2 interface
*/

#include "synthesizer_2.h"
#include <algorithm>
#include <numeric>

namespace tacos {

Synthesizer2::Synthesizer2(const std::vector<int>& shape, bool isTorus,
                           Bandwidth bandwidth, Latency latency)
    : shape_(shape), isTorus_(isTorus),
      bandwidth_(bandwidth), latency_(latency), chunkSize_(4096), topology_(nullptr) {

    npusCount_ = 1;
    for (int d : shape) {
        npusCount_ *= d;
    }

    // Create the unified solver
    solver_ = std::make_unique<UnifiedSolver>(shape, isTorus, bandwidth, latency);

    // Default configuration
    config_.verbose = true;
    config_.printSchedule = false;
    config_.enableDiffusion = true;
    config_.routingStrategy = RoutingStrategy::ADAPTIVE;
    config_.schedulingPolicy = SchedulingPolicy::GREEDY_LOAD_BALANCE;
}

Synthesizer2::Synthesizer2(std::shared_ptr<Topology> topology, const std::vector<int>& shape,
                           Bandwidth bandwidth, Latency latency)
    : shape_(shape), isTorus_(false),
      bandwidth_(bandwidth), latency_(latency), chunkSize_(4096), topology_(topology) {

    npusCount_ = 1;
    for (int d : shape) {
        npusCount_ *= d;
    }

    // For switch topologies, we still create the unified solver with shape
    // but it will use topology_ for path computation if needed
    // Note: UnifiedSolver may need modification to support topology-based routing
    solver_ = std::make_unique<UnifiedSolver>(shape, false, bandwidth, latency);

    // Default configuration
    config_.verbose = true;
    config_.printSchedule = false;
    config_.enableDiffusion = true;
    config_.routingStrategy = RoutingStrategy::ADAPTIVE;
    config_.schedulingPolicy = SchedulingPolicy::GREEDY_LOAD_BALANCE;
}

void Synthesizer2::setVerbose(bool verbose) {
    config_.verbose = verbose;
}

void Synthesizer2::setPrintSchedule(bool printSchedule) {
    config_.printSchedule = printSchedule;
}

void Synthesizer2::setEnableDiffusion(bool enable) {
    config_.enableDiffusion = enable;
}

void Synthesizer2::setRoutingStrategy(RoutingStrategy strategy) {
    config_.routingStrategy = strategy;
}

void Synthesizer2::setSchedulingPolicy(SchedulingPolicy policy) {
    config_.schedulingPolicy = policy;
}

void Synthesizer2::setChunkSize(DataSize chunkSize) {
    chunkSize_ = chunkSize;
}

Synthesizer2::Time Synthesizer2::solve(const std::vector<std::vector<DataSize>>& demand) {
    // Apply configuration
    solver_->setConfig(config_);

    // Run the solver
    lastResult_ = solver_->solve(demand);

    // Return makespan in microseconds
    return lastResult_.actualMakespan;
}

ChunkBasedResult Synthesizer2::solveWithChunks(const std::vector<std::vector<DataSize>>& demand,
                                                DataSize chunkSize) {
    // Apply configuration
    solver_->setConfig(config_);

    // Run the solver
    lastResult_ = solver_->solve(demand);

    // Convert to chunk-based result
    return convertToChunkBased(lastResult_, demand, chunkSize);
}

ChunkBasedResult Synthesizer2::convertToChunkBased(const SolverResult& result,
                                                    const std::vector<std::vector<DataSize>>& demand,
                                                    DataSize chunkSize) {
    ChunkBasedResult chunkResult;
    chunkResult.chunkSizeBytes = chunkSize;
    chunkResult.makespan = result.actualMakespan;

    // Initialize precondition and postcondition
    chunkResult.precondition.resize(npusCount_);
    chunkResult.postcondition.resize(npusCount_);

    // Step 1: Create chunks from demand matrix
    // IMPORTANT: We don't divide data equally into chunks
    // Instead, we use a single chunk per flow for optimal scheduling
    // This avoids unnecessary latency overhead from multiple transmissions
    int chunkId = 0;

    // Map from (src, dst) to list of chunk IDs
    std::map<std::pair<int, int>, std::vector<int>> flowToChunks;

    for (int src = 0; src < npusCount_; ++src) {
        for (int dst = 0; dst < npusCount_; ++dst) {
            if (src == dst || demand[src][dst] == 0) continue;

            DataSize totalBytes = demand[src][dst];

            // Create a single chunk for this flow (optimal strategy)
            ChunkInfo chunk;
            chunk.chunkId = chunkId;
            chunk.srcNpu = src;
            chunk.dstNpu = dst;
            chunk.byteSize = totalBytes;

            chunkResult.chunks.push_back(chunk);

            // Add to precondition (chunk starts at source)
            chunkResult.precondition[src].push_back(chunkId);

            // Add to postcondition (chunk ends at destination)
            chunkResult.postcondition[dst].push_back(chunkId);

            // Track which chunks belong to which flow
            flowToChunks[{src, dst}].push_back(chunkId);

            chunkId++;
        }
    }

    chunkResult.totalChunks = chunkId;

    // Step 2: Generate transfer events from the schedule
    // We need to map the flow-based events to chunk-based events

    // Process schedule events and create chunk transfer events
    const auto& scheduleEvents = result.scheduleResult.events;

    // Group events by commodity (src, dst)
    std::map<std::pair<int, int>, std::vector<TransferEvent>> eventsByFlow;
    for (const auto& event : scheduleEvents) {
        eventsByFlow[{event.commoditySrc, event.commodityDst}].push_back(event);
    }

    // Build path information for each flow
    std::map<std::pair<int, int>, std::vector<int>> flowPaths;
    for (const auto& [flow, events] : eventsByFlow) {
        // Sort events by start time to get the path order
        std::vector<TransferEvent> sortedEvents = events;
        std::sort(sortedEvents.begin(), sortedEvents.end(),
            [](const TransferEvent& a, const TransferEvent& b) {
                return a.startTime < b.startTime;
            });

        // Extract path from events
        std::vector<int> path;
        if (!sortedEvents.empty()) {
            path.push_back(sortedEvents[0].edgeSrc);
            for (const auto& event : sortedEvents) {
                path.push_back(event.edgeDst);
            }
        }
        flowPaths[flow] = path;
    }

    // Calculate transfer time for one chunk using alpha-beta model
    double alpha = latency_ / 1000.0;  // ns to us
    double bandwidthBytesPerUs = bandwidth_ * (1 << 30) / 1e6;  // GB/s to bytes/us

    // Track the earliest available time for each edge
    // CRITICAL: GPU protocol - one link can only send one chunk at a time
    // NOTE: A GPU can have multiple links (NVLINK, PCIE), each link is independent
    //       So a GPU can send/receive multiple chunks simultaneously on different links
    std::map<std::pair<int, int>, double> edgeAvailableTime;

    // Track when each chunk arrives at each node (for store-and-forward)
    // chunkArrivalTime[chunkId][nodeId] = time when chunk arrives at node
    std::map<int, std::map<int, double>> chunkArrivalTime;

    // Initialize: all chunks start at their source at time 0
    for (const auto& [flow, chunks] : flowToChunks) {
        int srcNode = flow.first;
        for (int chunkId : chunks) {
            chunkArrivalTime[chunkId][srcNode] = 0.0;
        }
    }

    // Event-driven scheduling with GPU protocol constraints
    // IMPORTANT: We only need to respect ONE constraint:
    // 1. Each link (edge) can only transfer one chunk at a time
    // A GPU can use multiple links simultaneously!

    // Track which hop each chunk is currently at
    std::map<int, int> chunkCurrentHop;
    for (const auto& [flow, chunks] : flowToChunks) {
        for (int chunkId : chunks) {
            chunkCurrentHop[chunkId] = 0;  // All chunks start at hop 0
        }
    }

    // Track which chunks have completed all hops
    std::set<int> completedChunks;
    int totalChunks = chunkResult.totalChunks;

    // Event-driven simulation
    double currentTime = 0.0;
    const double TIME_EPSILON = 0.001;  // 1 nanosecond precision

    while (completedChunks.size() < static_cast<size_t>(totalChunks)) {
        // Find all chunks that are ready to transfer at current time
        std::vector<std::tuple<int, int, int, double>> readyTransfers;  // (chunkId, srcNode, dstNode, transferTime)

        for (const auto& [flow, chunks] : flowToChunks) {
            const auto& path = flowPaths[flow];
            if (path.size() < 2) continue;

            for (int chunkId : chunks) {
                if (completedChunks.count(chunkId)) continue;

                int hopIdx = chunkCurrentHop[chunkId];
                if (hopIdx + 1 >= static_cast<int>(path.size())) {
                    completedChunks.insert(chunkId);
                    continue;
                }

                int srcNode = path[hopIdx];
                int dstNode = path[hopIdx + 1];
                std::pair<int, int> edge = {srcNode, dstNode};

                // Check if chunk is ready and edge is available
                double chunkReadyTime = chunkArrivalTime[chunkId][srcNode];
                if (chunkReadyTime > currentTime + TIME_EPSILON) continue;

                // Check if edge is available (only constraint!)
                if (edgeAvailableTime[edge] > currentTime + TIME_EPSILON) continue;

                // This transfer is ready
                DataSize chunkSize = chunkResult.chunks[chunkId].byteSize;
                double transferTime = alpha + (chunkSize / bandwidthBytesPerUs);
                readyTransfers.push_back({chunkId, srcNode, dstNode, transferTime});
            }
        }

        if (readyTransfers.empty()) {
            // No transfers ready at current time, advance to next event
            double nextEventTime = std::numeric_limits<double>::max();

            // Check when chunks will be ready
            for (const auto& [flow, chunks] : flowToChunks) {
                const auto& path = flowPaths[flow];
                for (int chunkId : chunks) {
                    if (completedChunks.count(chunkId)) continue;
                    int hopIdx = chunkCurrentHop[chunkId];
                    if (hopIdx >= static_cast<int>(path.size())) continue;

                    int srcNode = path[hopIdx];
                    double chunkReadyTime = chunkArrivalTime[chunkId][srcNode];
                    if (chunkReadyTime > currentTime) {
                        nextEventTime = std::min(nextEventTime, chunkReadyTime);
                    }
                }
            }

            // Check when edges will be available
            for (const auto& [edge, time] : edgeAvailableTime) {
                if (time > currentTime) {
                    nextEventTime = std::min(nextEventTime, time);
                }
            }

            if (nextEventTime == std::numeric_limits<double>::max()) {
                break;  // No more events
            }
            currentTime = nextEventTime;
            continue;
        }

        // Schedule ready transfers with better prioritization
        // Priority: 1) Longer paths first (to start them early)
        //           2) Larger chunks first (to maximize bandwidth utilization)
        std::sort(readyTransfers.begin(), readyTransfers.end(),
            [&](const auto& a, const auto& b) {
                int chunkA = std::get<0>(a);
                int chunkB = std::get<0>(b);

                // Find path lengths
                int pathLenA = 0, pathLenB = 0;
                for (const auto& [flow, chunks] : flowToChunks) {
                    for (int cid : chunks) {
                        if (cid == chunkA) pathLenA = flowPaths[flow].size();
                        if (cid == chunkB) pathLenB = flowPaths[flow].size();
                    }
                }

                // Prioritize longer paths
                if (pathLenA != pathLenB) {
                    return pathLenA > pathLenB;
                }

                // Then prioritize larger chunks
                return chunkResult.chunks[chunkA].byteSize > chunkResult.chunks[chunkB].byteSize;
            });

        // Schedule as many non-conflicting transfers as possible
        // Only constraint: each edge can only be used once
        std::set<std::pair<int, int>> usedEdges;

        for (const auto& [chunkId, srcNode, dstNode, transferTime] : readyTransfers) {
            std::pair<int, int> edge = {srcNode, dstNode};

            // Check if edge is still available (not used by earlier transfers in this batch)
            if (usedEdges.count(edge)) continue;

            // Schedule this transfer
            ChunkTransferEvent chunkEvent;
            chunkEvent.chunkId = chunkId;
            chunkEvent.fromNpu = srcNode;
            chunkEvent.toNpu = dstNode;
            chunkEvent.eventTime = currentTime;

            chunkResult.transferEvents.push_back(chunkEvent);

            // Update edge availability
            double transferEndTime = currentTime + transferTime;
            edgeAvailableTime[edge] = transferEndTime;

            // Mark edge as used in this batch
            usedEdges.insert(edge);

            // Update chunk state
            chunkArrivalTime[chunkId][dstNode] = transferEndTime;
            chunkCurrentHop[chunkId]++;
        }

        // Advance time slightly to process next batch
        currentTime += TIME_EPSILON;
    }

    // Sort transfer events by time
    std::sort(chunkResult.transferEvents.begin(), chunkResult.transferEvents.end());

    // Remove duplicate events (same chunk, same edge, same time)
    auto last = std::unique(chunkResult.transferEvents.begin(), chunkResult.transferEvents.end(),
        [](const ChunkTransferEvent& a, const ChunkTransferEvent& b) {
            return a.chunkId == b.chunkId && a.fromNpu == b.fromNpu &&
                   a.toNpu == b.toNpu && std::abs(a.eventTime - b.eventTime) < 0.001;
        });
    chunkResult.transferEvents.erase(last, chunkResult.transferEvents.end());

    // Calculate actual makespan from transfer events
    // The makespan is the maximum completion time across all chunks
    double actualMakespan = 0.0;

    // Recalculate alpha and bandwidth for makespan computation
    double alpha_makespan = latency_ / 1000.0;  // ns to us
    double bandwidthBytesPerUs_makespan = bandwidth_ * (1 << 30) / 1e6;  // GB/s to bytes/us

    int maxChunkId = -1;
    double maxStartTime = 0.0;
    for (const auto& event : chunkResult.transferEvents) {
        DataSize chunkSize = chunkResult.chunks[event.chunkId].byteSize;
        double transferTime = alpha_makespan + (chunkSize / bandwidthBytesPerUs_makespan);
        double completionTime = event.eventTime + transferTime;
        if (completionTime > actualMakespan) {
            actualMakespan = completionTime;
            maxChunkId = event.chunkId;
            maxStartTime = event.eventTime;
        }
    }

    if (config_.verbose) {
        std::cout << "\n[Makespan Calculation]" << std::endl;
        std::cout << "  Total transfer events: " << chunkResult.transferEvents.size() << std::endl;
        std::cout << "  Chunk with max completion: " << maxChunkId << std::endl;
        std::cout << "    Start time: " << maxStartTime << " us" << std::endl;
        std::cout << "    Size: " << (chunkResult.chunks[maxChunkId].byteSize / 1024.0) << " KB" << std::endl;
        std::cout << "    Completion: " << actualMakespan << " us" << std::endl;
    }

    chunkResult.makespan = actualMakespan;

    return chunkResult;
}

void Synthesizer2::compareStrategies(const std::vector<std::vector<DataSize>>& demand) {
    solver_->setConfig(config_);
    solver_->compareStrategies(demand);
}

} // namespace tacos
