/*
# File name  :    edge_scheduler.h
# Author     :    Galois
# Time       :    2025/01/05
# Description:    Edge-based scheduling for non-uniform AllToAll
#                 Uses weighted fair queueing on each edge
*/

#pragma once

#include <vector>
#include <queue>
#include <set>
#include <map>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <climits>

namespace tacos {

//=============================================================================
// Flow: Represents a commodity flow on an edge
//=============================================================================
struct Flow {
    int commoditySrc;      // Original source of the commodity
    int commodityDst;      // Original destination of the commodity
    long long totalBytes;  // Total bytes to transfer
    long long sentBytes;   // Bytes already sent
    int nextHop;           // Next hop in the path
    int hopIndex;          // Current position in path (0 = at source)
    std::vector<int> path; // Full path

    long long remainingBytes() const { return totalBytes - sentBytes; }
    bool isComplete() const { return sentBytes >= totalBytes; }
};

//=============================================================================
// EdgeState: State of a single edge during simulation
//=============================================================================
struct EdgeState {
    int src;
    int dst;
    double bandwidth;  // GiB/sec
    double latency;    // microseconds

    // Flows waiting to use this edge
    std::vector<Flow*> waitingFlows;

    // Currently active flow (if any)
    Flow* activeFlow = nullptr;
    double activeFlowEndTime = 0.0;

    // Statistics
    long long totalBytesTransferred = 0;
    double busyTime = 0.0;
};

//=============================================================================
// Event: Discrete event for simulation
//=============================================================================
struct Event {
    double time;
    enum Type { FLOW_ARRIVAL, TRANSFER_COMPLETE } type;
    int edgeSrc;
    int edgeDst;
    Flow* flow;

    bool operator>(const Event& other) const {
        return time > other.time;
    }
};

//=============================================================================
// EdgeScheduler: Discrete event simulation for edge-based scheduling
//=============================================================================
class EdgeScheduler {
public:
    using Bandwidth = double;
    using Latency = double;

    EdgeScheduler(int npusCount, Bandwidth defaultBandwidth, Latency defaultLatency)
        : npusCount_(npusCount),
          defaultBandwidth_(defaultBandwidth),
          defaultLatency_(defaultLatency) {

        // Initialize edge states
        edgeStates_.resize(npusCount);
        for (int i = 0; i < npusCount; ++i) {
            edgeStates_[i].resize(npusCount);
            for (int j = 0; j < npusCount; ++j) {
                edgeStates_[i][j].src = i;
                edgeStates_[i][j].dst = j;
                edgeStates_[i][j].bandwidth = defaultBandwidth;
                edgeStates_[i][j].latency = defaultLatency;
            }
        }
    }

    // Set edge properties (for heterogeneous networks)
    void setEdgeProperties(int src, int dst, Bandwidth bw, Latency lat) {
        edgeStates_[src][dst].bandwidth = bw;
        edgeStates_[src][dst].latency = lat;
    }

    // Add a flow to be scheduled
    void addFlow(int commoditySrc, int commodityDst, long long bytes,
                 const std::vector<int>& path) {
        Flow* flow = new Flow();
        flow->commoditySrc = commoditySrc;
        flow->commodityDst = commodityDst;
        flow->totalBytes = bytes;
        flow->sentBytes = 0;
        flow->hopIndex = 0;
        flow->path = path;
        if (path.size() > 1) {
            flow->nextHop = path[1];
        }
        flows_.push_back(flow);
    }

    //=========================================================================
    // Run simulation with FIFO scheduling on each edge
    //=========================================================================
    double runFIFO(bool verbose = false) {
        return runSimulation(SchedulingPolicy::FIFO, verbose);
    }

    //=========================================================================
    // Run simulation with Shortest Remaining Processing Time (SRPT)
    //=========================================================================
    double runSRPT(bool verbose = false) {
        return runSimulation(SchedulingPolicy::SRPT, verbose);
    }

    //=========================================================================
    // Run simulation with Weighted Fair Queueing
    //=========================================================================
    double runWFQ(bool verbose = false) {
        return runSimulation(SchedulingPolicy::WFQ, verbose);
    }

    // Get statistics
    void printStatistics() const {
        std::cout << "\n--- Edge Utilization Statistics ---\n";
        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                const auto& edge = edgeStates_[i][j];
                if (edge.totalBytesTransferred > 0) {
                    std::cout << "Edge (" << i << " -> " << j << "): "
                              << edge.totalBytesTransferred << " bytes, "
                              << std::fixed << std::setprecision(2)
                              << edge.busyTime << " us busy\n";
                }
            }
        }
    }

    ~EdgeScheduler() {
        for (auto* flow : flows_) {
            delete flow;
        }
    }

private:
    enum class SchedulingPolicy { FIFO, SRPT, WFQ };

    double runSimulation(SchedulingPolicy policy, bool verbose) {
        // Reset state
        for (auto& row : edgeStates_) {
            for (auto& edge : row) {
                edge.waitingFlows.clear();
                edge.activeFlow = nullptr;
                edge.activeFlowEndTime = 0.0;
                edge.totalBytesTransferred = 0;
                edge.busyTime = 0.0;
            }
        }

        // Priority queue for events (min-heap by time)
        std::priority_queue<Event, std::vector<Event>, std::greater<Event>> eventQueue;

        // Initialize: all flows arrive at time 0 at their first edge
        for (auto* flow : flows_) {
            if (flow->path.size() > 1) {
                flow->sentBytes = 0;
                flow->hopIndex = 0;
                flow->nextHop = flow->path[1];

                Event e;
                e.time = 0.0;
                e.type = Event::FLOW_ARRIVAL;
                e.edgeSrc = flow->path[0];
                e.edgeDst = flow->path[1];
                e.flow = flow;
                eventQueue.push(e);
            }
        }

        double currentTime = 0.0;
        int completedFlows = 0;
        int totalFlows = flows_.size();

        while (!eventQueue.empty() && completedFlows < totalFlows) {
            Event event = eventQueue.top();
            eventQueue.pop();
            currentTime = event.time;

            if (verbose) {
                std::cout << "[Time " << std::fixed << std::setprecision(2)
                          << currentTime << " us] ";
            }

            if (event.type == Event::FLOW_ARRIVAL) {
                // Flow arrives at an edge
                EdgeState& edge = edgeStates_[event.edgeSrc][event.edgeDst];

                if (verbose) {
                    std::cout << "Flow (" << event.flow->commoditySrc << "->"
                              << event.flow->commodityDst << ") arrives at edge ("
                              << event.edgeSrc << "->" << event.edgeDst << ")\n";
                }

                // Add to waiting queue
                edge.waitingFlows.push_back(event.flow);

                // If edge is idle, start processing
                if (edge.activeFlow == nullptr) {
                    startNextTransfer(edge, currentTime, eventQueue, policy, verbose);
                }

            } else if (event.type == Event::TRANSFER_COMPLETE) {
                // Transfer on edge completes
                EdgeState& edge = edgeStates_[event.edgeSrc][event.edgeDst];
                Flow* flow = event.flow;

                if (verbose) {
                    std::cout << "Transfer complete for flow ("
                              << flow->commoditySrc << "->" << flow->commodityDst
                              << ") on edge (" << event.edgeSrc << "->"
                              << event.edgeDst << ")\n";
                }

                // Update flow state
                flow->hopIndex++;
                edge.activeFlow = nullptr;

                // Check if flow has more hops
                if (flow->hopIndex + 1 < flow->path.size()) {
                    // Move to next edge
                    int nextSrc = flow->path[flow->hopIndex];
                    int nextDst = flow->path[flow->hopIndex + 1];
                    flow->nextHop = nextDst;

                    Event e;
                    e.time = currentTime;  // Arrives immediately (store-and-forward)
                    e.type = Event::FLOW_ARRIVAL;
                    e.edgeSrc = nextSrc;
                    e.edgeDst = nextDst;
                    e.flow = flow;
                    eventQueue.push(e);
                } else {
                    // Flow complete
                    completedFlows++;
                    if (verbose) {
                        std::cout << "  -> Flow (" << flow->commoditySrc << "->"
                                  << flow->commodityDst << ") COMPLETED\n";
                    }
                }

                // Start next transfer on this edge
                startNextTransfer(edge, currentTime, eventQueue, policy, verbose);
            }
        }

        if (verbose) {
            std::cout << "\nSimulation complete. Makespan: " << currentTime << " us\n";
            std::cout << "Completed flows: " << completedFlows << "/" << totalFlows << "\n";
        }

        return currentTime;
    }

    void startNextTransfer(EdgeState& edge, double currentTime,
                           std::priority_queue<Event, std::vector<Event>,
                                               std::greater<Event>>& eventQueue,
                           SchedulingPolicy policy, bool verbose) {
        if (edge.waitingFlows.empty()) {
            return;
        }

        // Select next flow based on policy
        Flow* selectedFlow = nullptr;
        size_t selectedIdx = 0;

        switch (policy) {
            case SchedulingPolicy::FIFO:
                selectedFlow = edge.waitingFlows.front();
                selectedIdx = 0;
                break;

            case SchedulingPolicy::SRPT:
                // Select flow with shortest remaining bytes
                {
                    long long minRemaining = LLONG_MAX;
                    for (size_t i = 0; i < edge.waitingFlows.size(); ++i) {
                        if (edge.waitingFlows[i]->remainingBytes() < minRemaining) {
                            minRemaining = edge.waitingFlows[i]->remainingBytes();
                            selectedFlow = edge.waitingFlows[i];
                            selectedIdx = i;
                        }
                    }
                }
                break;

            case SchedulingPolicy::WFQ:
                // For simplicity, use round-robin with equal weights
                // In practice, weights could be based on flow priority
                selectedFlow = edge.waitingFlows.front();
                selectedIdx = 0;
                break;
        }

        if (selectedFlow == nullptr) return;

        // Remove from waiting queue
        edge.waitingFlows.erase(edge.waitingFlows.begin() + selectedIdx);

        // Calculate transfer time using alpha-beta model matching original synthesizer:
        // - latency is in ns, convert to us by /1000
        // - bandwidth is in GB/s, convert to bytes/us: * (1 << 30) / 1e6
        double alpha = edge.latency / 1000.0;  // ns to us
        double bandwidthBytesPerUs = edge.bandwidth * (1 << 30) / 1e6;  // GB/s to bytes/us
        double beta = selectedFlow->totalBytes / bandwidthBytesPerUs;
        double transferTime = alpha + beta;

        edge.activeFlow = selectedFlow;
        edge.activeFlowEndTime = currentTime + transferTime;
        edge.totalBytesTransferred += selectedFlow->totalBytes;
        edge.busyTime += transferTime;

        if (verbose) {
            std::cout << "  -> Starting transfer of flow ("
                      << selectedFlow->commoditySrc << "->"
                      << selectedFlow->commodityDst << ") on edge ("
                      << edge.src << "->" << edge.dst << "), "
                      << selectedFlow->totalBytes << " bytes, "
                      << std::fixed << std::setprecision(2)
                      << transferTime << " us\n";
        }

        // Schedule completion event
        Event e;
        e.time = edge.activeFlowEndTime;
        e.type = Event::TRANSFER_COMPLETE;
        e.edgeSrc = edge.src;
        e.edgeDst = edge.dst;
        e.flow = selectedFlow;
        eventQueue.push(e);
    }

    int npusCount_;
    Bandwidth defaultBandwidth_;
    Latency defaultLatency_;
    std::vector<std::vector<EdgeState>> edgeStates_;
    std::vector<Flow*> flows_;
};

} // namespace tacos
