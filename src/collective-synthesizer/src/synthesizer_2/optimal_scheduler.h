/*
# File name  :    optimal_scheduler.h
# Author     :    Chen Chen
# Time       :    2025/01/06
# Description:    Optimal scheduling algorithms for minimizing time steps
#                 - Conflict-free scheduling using graph coloring
#                 - Latin Square based scheduling for uniform traffic
#                 - Greedy scheduling with priority queues
#                 - Discrete event simulation for accurate makespan
*/

#pragma once

#include <vector>
#include <queue>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

#include "congestion_aware_router.h"

namespace tacos {

//=============================================================================
// SchedulingPolicy: Different scheduling policies
//=============================================================================
enum class SchedulingPolicy {
    FIFO,                    // First-in-first-out
    SRPT,                    // Shortest remaining processing time
    LPT,                     // Longest processing time first
    CONFLICT_FREE,           // Graph coloring based
    LATIN_SQUARE,            // For uniform traffic
    GREEDY_LOAD_BALANCE      // Minimize max completion time
};

//=============================================================================
// TransferEvent: A single transfer event in the schedule
//=============================================================================
struct TransferEvent {
    int timeStep;            // Discrete time step
    double startTime;        // Continuous start time (us)
    double endTime;          // Continuous end time (us)
    int edgeSrc;
    int edgeDst;
    int commoditySrc;        // Original source of the commodity
    int commodityDst;        // Original destination of the commodity
    long long bytes;
    int pathIndex;           // Which path in multi-path routing

    bool operator<(const TransferEvent& other) const {
        if (timeStep != other.timeStep) return timeStep < other.timeStep;
        if (startTime != other.startTime) return startTime < other.startTime;
        return edgeSrc < other.edgeSrc;
    }
};

//=============================================================================
// ScheduleResult: Complete schedule with all transfer events
//=============================================================================
struct ScheduleResult {
    std::vector<TransferEvent> events;
    int totalTimeSteps;
    double makespan;         // Total time in microseconds
    double theoreticalLowerBound;

    // Per-edge statistics
    std::map<std::pair<int, int>, double> edgeUtilization;

    void print() const {
        std::cout << "\n========== Schedule Result ==========\n";
        std::cout << "Total Time Steps: " << totalTimeSteps << "\n";
        std::cout << "Makespan: " << std::fixed << std::setprecision(2) << makespan << " us\n";
        std::cout << "Theoretical Lower Bound: " << theoreticalLowerBound << " us\n";
        std::cout << "Efficiency: " << (theoreticalLowerBound / makespan * 100) << "%\n";
        std::cout << "Total Events: " << events.size() << "\n";

        // Print schedule by time step
        std::cout << "\n--- Schedule by Time Step ---\n";
        int currentStep = -1;
        for (const auto& e : events) {
            if (e.timeStep != currentStep) {
                currentStep = e.timeStep;
                std::cout << "\nTime Step " << currentStep << ":\n";
            }
            std::cout << "  [" << e.startTime << " - " << e.endTime << " us] "
                      << "Edge(" << e.edgeSrc << "->" << e.edgeDst << ") "
                      << "Commodity(" << e.commoditySrc << "->" << e.commodityDst << ") "
                      << e.bytes << " bytes\n";
        }
        std::cout << "=====================================\n";
    }

    void printSummary() const {
        std::cout << "\n--- Schedule Summary ---\n";
        std::cout << "Time Steps: " << totalTimeSteps << "\n";
        std::cout << "Makespan: " << std::fixed << std::setprecision(2) << makespan << " us\n";
        std::cout << "Lower Bound: " << theoreticalLowerBound << " us\n";
        std::cout << "Gap: " << ((makespan / theoreticalLowerBound - 1.0) * 100) << "%\n";
    }
};

//=============================================================================
// FlowState: State of a flow during scheduling
//=============================================================================
struct FlowState {
    int commoditySrc;
    int commodityDst;
    long long totalBytes;
    long long remainingBytes;
    int currentHop;          // Current position in path (0 = at source)
    std::vector<int> path;
    int pathIndex;           // For multi-path routing

    bool isComplete() const { return remainingBytes <= 0; }
    int currentNode() const { return path[currentHop]; }
    int nextNode() const { return (currentHop + 1 < static_cast<int>(path.size())) ? path[currentHop + 1] : -1; }
    bool hasMoreHops() const { return currentHop + 1 < static_cast<int>(path.size()); }
};

//=============================================================================
// EdgeSchedulerState: State of an edge during scheduling
//=============================================================================
struct EdgeSchedulerState {
    int src;
    int dst;
    double bandwidth;        // GiB/sec
    double latency;          // us
    std::vector<FlowState*> waitingFlows;
    FlowState* activeFlow = nullptr;
    double busyUntil = 0.0;
    long long totalBytesTransferred = 0;
};

//=============================================================================
// SimEvent: Simulation event for discrete event simulation
//=============================================================================
struct SimEvent {
    double time;
    enum Type { FLOW_ARRIVAL, TRANSFER_COMPLETE } type;
    int edgeSrc, edgeDst;
    FlowState* flow;

    bool operator>(const SimEvent& other) const {
        return time > other.time;
    }
};

//=============================================================================
// OptimalScheduler: Main scheduling class
//=============================================================================
class OptimalScheduler {
public:
    using DataSize = long long;
    using Bandwidth = double;
    using Latency = double;

    OptimalScheduler(int npusCount, Bandwidth bandwidth, Latency latency)
        : npusCount_(npusCount), bandwidth_(bandwidth), latency_(latency) {
        initializeEdgeStates();
    }

    //=========================================================================
    // Main scheduling function
    //=========================================================================
    ScheduleResult schedule(const RoutingPlan& routingPlan,
                           SchedulingPolicy policy = SchedulingPolicy::GREEDY_LOAD_BALANCE) {
        switch (policy) {
            case SchedulingPolicy::FIFO:
                return scheduleFIFO(routingPlan);
            case SchedulingPolicy::SRPT:
                return scheduleSRPT(routingPlan);
            case SchedulingPolicy::LPT:
                return scheduleLPT(routingPlan);
            case SchedulingPolicy::CONFLICT_FREE:
                return scheduleConflictFree(routingPlan);
            case SchedulingPolicy::GREEDY_LOAD_BALANCE:
                return scheduleGreedyLoadBalance(routingPlan);
            default:
                return scheduleFIFO(routingPlan);
        }
    }

    //=========================================================================
    // Compute theoretical lower bound
    //=========================================================================
    double computeLowerBound(const RoutingPlan& routingPlan,
                            const std::vector<std::vector<DataSize>>& demand) {
        // Use alpha-beta model matching original synthesizer:
        // - latency_ is in ns, convert to us by /1000
        // - bandwidth_ is in GB/s, convert to bytes/us: * (1 << 30) / 1e6
        double bandwidthBytesPerUs = bandwidth_ * (1 << 30) / 1e6;  // GB/s to bytes/us
        double latencyUs = latency_ / 1000.0;  // ns to us

        // LB1: Bottleneck edge (based on actual routing)
        double lb_edge = routingPlan.maxEdgeLoad / bandwidthBytesPerUs;

        // LB2: Max row sum (send constraint) - each node can only send on its outgoing links
        DataSize maxRowSum = 0;
        for (int i = 0; i < npusCount_; ++i) {
            DataSize rowSum = 0;
            for (int j = 0; j < npusCount_; ++j) {
                if (i != j) rowSum += demand[i][j];
            }
            maxRowSum = std::max(maxRowSum, rowSum);
        }
        double lb_send = maxRowSum / bandwidthBytesPerUs;

        // LB3: Max column sum (receive constraint) - each node can only receive on its incoming links
        DataSize maxColSum = 0;
        for (int j = 0; j < npusCount_; ++j) {
            DataSize colSum = 0;
            for (int i = 0; i < npusCount_; ++i) {
                if (i != j) colSum += demand[i][j];
            }
            maxColSum = std::max(maxColSum, colSum);
        }
        double lb_recv = maxColSum / bandwidthBytesPerUs;

        // LB4: Add minimum latency for the longest path
        int maxHops = 0;
        for (const auto& flow : routingPlan.flows) {
            for (const auto& path : flow.paths) {
                maxHops = std::max(maxHops, path.hopCount);
            }
        }
        double lb_latency = maxHops * latencyUs;

        // The lower bound is the max of bandwidth bounds plus latency
        double lb_bandwidth = std::max({lb_edge, lb_send, lb_recv});

        // For store-and-forward, we need at least the bandwidth time plus latency
        return lb_bandwidth + lb_latency;
    }

private:
    //=========================================================================
    // Initialize edge states
    //=========================================================================
    void initializeEdgeStates() {
        edgeStates_.resize(npusCount_);
        for (int i = 0; i < npusCount_; ++i) {
            edgeStates_[i].resize(npusCount_);
            for (int j = 0; j < npusCount_; ++j) {
                edgeStates_[i][j].src = i;
                edgeStates_[i][j].dst = j;
                edgeStates_[i][j].bandwidth = bandwidth_;
                edgeStates_[i][j].latency = latency_;
            }
        }
    }

    //=========================================================================
    // Reset edge states for new scheduling
    //=========================================================================
    void resetEdgeStates() {
        for (auto& row : edgeStates_) {
            for (auto& edge : row) {
                edge.waitingFlows.clear();
                edge.activeFlow = nullptr;
                edge.busyUntil = 0.0;
                edge.totalBytesTransferred = 0;
            }
        }
    }

    //=========================================================================
    // Create flow states from routing plan
    //=========================================================================
    std::vector<FlowState> createFlowStates(const RoutingPlan& routingPlan) {
        std::vector<FlowState> flows;

        for (const auto& commodity : routingPlan.flows) {
            for (size_t pathIdx = 0; pathIdx < commodity.paths.size(); ++pathIdx) {
                const auto& pathInfo = commodity.paths[pathIdx];

                FlowState flow;
                flow.commoditySrc = commodity.src;
                flow.commodityDst = commodity.dst;
                flow.totalBytes = pathInfo.bytes;
                flow.remainingBytes = pathInfo.bytes;
                flow.currentHop = 0;
                flow.path = pathInfo.nodes;
                flow.pathIndex = static_cast<int>(pathIdx);

                flows.push_back(flow);
            }
        }

        return flows;
    }

    //=========================================================================
    // FIFO Scheduling
    //=========================================================================
    ScheduleResult scheduleFIFO(const RoutingPlan& routingPlan) {
        return runDiscreteEventSimulation(routingPlan,
            [](const std::vector<FlowState*>& waiting) -> FlowState* {
                return waiting.empty() ? nullptr : waiting.front();
            });
    }

    //=========================================================================
    // SRPT Scheduling (Shortest Remaining Processing Time)
    //=========================================================================
    ScheduleResult scheduleSRPT(const RoutingPlan& routingPlan) {
        return runDiscreteEventSimulation(routingPlan,
            [](const std::vector<FlowState*>& waiting) -> FlowState* {
                if (waiting.empty()) return nullptr;
                return *std::min_element(waiting.begin(), waiting.end(),
                    [](FlowState* a, FlowState* b) {
                        return a->remainingBytes < b->remainingBytes;
                    });
            });
    }

    //=========================================================================
    // LPT Scheduling (Longest Processing Time First)
    //=========================================================================
    ScheduleResult scheduleLPT(const RoutingPlan& routingPlan) {
        return runDiscreteEventSimulation(routingPlan,
            [](const std::vector<FlowState*>& waiting) -> FlowState* {
                if (waiting.empty()) return nullptr;
                return *std::max_element(waiting.begin(), waiting.end(),
                    [](FlowState* a, FlowState* b) {
                        return a->remainingBytes < b->remainingBytes;
                    });
            });
    }

    //=========================================================================
    // Conflict-Free Scheduling using Graph Coloring
    //=========================================================================
    ScheduleResult scheduleConflictFree(const RoutingPlan& routingPlan) {
        ScheduleResult result;
        result.events.clear();

        auto flows = createFlowStates(routingPlan);

        // Build conflict graph
        // Two flows conflict if they share an edge at the same hop
        std::map<int, std::set<int>> conflictGraph;
        for (size_t i = 0; i < flows.size(); ++i) {
            conflictGraph[static_cast<int>(i)] = {};
        }

        for (size_t i = 0; i < flows.size(); ++i) {
            for (size_t j = i + 1; j < flows.size(); ++j) {
                // Check if flows share any edge
                std::set<std::pair<int, int>> edges_i, edges_j;
                for (size_t k = 0; k + 1 < flows[i].path.size(); ++k) {
                    edges_i.insert({flows[i].path[k], flows[i].path[k + 1]});
                }
                for (size_t k = 0; k + 1 < flows[j].path.size(); ++k) {
                    edges_j.insert({flows[j].path[k], flows[j].path[k + 1]});
                }

                // Check intersection
                for (const auto& e : edges_i) {
                    if (edges_j.count(e)) {
                        conflictGraph[static_cast<int>(i)].insert(static_cast<int>(j));
                        conflictGraph[static_cast<int>(j)].insert(static_cast<int>(i));
                        break;
                    }
                }
            }
        }

        // Graph coloring (greedy)
        std::vector<int> colors(flows.size(), -1);
        int numColors = 0;

        for (size_t i = 0; i < flows.size(); ++i) {
            std::set<int> usedColors;
            for (int neighbor : conflictGraph[static_cast<int>(i)]) {
                if (colors[neighbor] >= 0) {
                    usedColors.insert(colors[neighbor]);
                }
            }

            // Find smallest available color
            int color = 0;
            while (usedColors.count(color)) {
                color++;
            }
            colors[i] = color;
            numColors = std::max(numColors, color + 1);
        }

        // Schedule flows by color (time step)
        double currentTime = 0.0;
        int timeStep = 0;

        for (int color = 0; color < numColors; ++color) {
            double maxTransferTime = 0.0;

            for (size_t i = 0; i < flows.size(); ++i) {
                if (colors[i] != color) continue;

                const auto& flow = flows[i];

                // Schedule all hops of this flow
                double flowStartTime = currentTime;
                for (size_t hop = 0; hop + 1 < flow.path.size(); ++hop) {
                    TransferEvent event;
                    event.timeStep = timeStep + static_cast<int>(hop);
                    event.startTime = flowStartTime;
                    event.edgeSrc = flow.path[hop];
                    event.edgeDst = flow.path[hop + 1];
                    event.commoditySrc = flow.commoditySrc;
                    event.commodityDst = flow.commodityDst;
                    event.bytes = flow.totalBytes;
                    event.pathIndex = flow.pathIndex;

                    // Use alpha-beta model matching original synthesizer:
                    // - latency_ is in ns, convert to us by /1000
                    // - bandwidth_ is in GB/s, convert to bytes/us: * (1 << 30) / 1e6
                    double alpha = latency_ / 1000.0;  // ns to us
                    double bandwidthBytesPerUs = bandwidth_ * (1 << 30) / 1e6;  // GB/s to bytes/us
                    double beta = flow.totalBytes / bandwidthBytesPerUs;
                    double transferTime = alpha + beta;
                    event.endTime = flowStartTime + transferTime;

                    result.events.push_back(event);

                    flowStartTime = event.endTime;
                    maxTransferTime = std::max(maxTransferTime, event.endTime - currentTime);
                }
            }

            currentTime += maxTransferTime;
            timeStep++;
        }

        std::sort(result.events.begin(), result.events.end());

        result.totalTimeSteps = timeStep;
        result.makespan = currentTime;
        result.theoreticalLowerBound = 0;  // Will be set by caller

        return result;
    }

    //=========================================================================
    // Greedy Load Balance Scheduling
    //=========================================================================
    ScheduleResult scheduleGreedyLoadBalance(const RoutingPlan& routingPlan) {
        return runDiscreteEventSimulation(routingPlan,
            [this](const std::vector<FlowState*>& waiting) -> FlowState* {
                if (waiting.empty()) return nullptr;

                // Choose flow that would balance load best
                // Prioritize flows going to less loaded edges
                FlowState* best = nullptr;
                double bestScore = std::numeric_limits<double>::max();

                for (FlowState* flow : waiting) {
                    if (!flow->hasMoreHops()) continue;

                    int nextEdgeSrc = flow->currentNode();
                    int nextEdgeDst = flow->nextNode();

                    // Score based on current edge load
                    double score = static_cast<double>(edgeStates_[nextEdgeSrc][nextEdgeDst].totalBytesTransferred);

                    // Also consider remaining path length (prefer shorter)
                    score += (flow->path.size() - flow->currentHop - 1) * 1000;

                    if (score < bestScore) {
                        bestScore = score;
                        best = flow;
                    }
                }

                return best ? best : (waiting.empty() ? nullptr : waiting.front());
            });
    }

    //=========================================================================
    // Discrete Event Simulation
    //=========================================================================
    ScheduleResult runDiscreteEventSimulation(
        const RoutingPlan& routingPlan,
        std::function<FlowState*(const std::vector<FlowState*>&)> selectFlow) {

        ScheduleResult result;
        result.events.clear();
        resetEdgeStates();

        auto flows = createFlowStates(routingPlan);
        std::vector<FlowState*> flowPtrs;
        for (auto& f : flows) {
            flowPtrs.push_back(&f);
        }

        std::priority_queue<SimEvent, std::vector<SimEvent>, std::greater<SimEvent>> eventQueue;

        // Initialize: all flows arrive at time 0
        for (FlowState* flow : flowPtrs) {
            if (flow->hasMoreHops()) {
                SimEvent e;
                e.time = 0.0;
                e.type = SimEvent::FLOW_ARRIVAL;
                e.edgeSrc = flow->currentNode();
                e.edgeDst = flow->nextNode();
                e.flow = flow;
                eventQueue.push(e);
            }
        }

        double currentTime = 0.0;
        int completedFlows = 0;
        int totalFlows = static_cast<int>(flowPtrs.size());
        int timeStep = 0;

        while (!eventQueue.empty() && completedFlows < totalFlows) {
            SimEvent event = eventQueue.top();
            eventQueue.pop();
            currentTime = event.time;

            auto& edge = edgeStates_[event.edgeSrc][event.edgeDst];

            if (event.type == SimEvent::FLOW_ARRIVAL) {
                // Flow arrives at edge
                edge.waitingFlows.push_back(event.flow);

                // If edge is idle, start transfer
                if (edge.activeFlow == nullptr) {
                    startTransfer(edge, currentTime, eventQueue, selectFlow, result, timeStep);
                }

            } else if (event.type == SimEvent::TRANSFER_COMPLETE) {
                // Transfer complete
                FlowState* flow = event.flow;
                flow->currentHop++;
                edge.activeFlow = nullptr;

                // Check if flow has more hops
                if (flow->hasMoreHops()) {
                    // Move to next edge
                    SimEvent e;
                    e.time = currentTime;
                    e.type = SimEvent::FLOW_ARRIVAL;
                    e.edgeSrc = flow->currentNode();
                    e.edgeDst = flow->nextNode();
                    e.flow = flow;
                    eventQueue.push(e);
                } else {
                    // Flow complete
                    completedFlows++;
                }

                // Start next transfer on this edge
                startTransfer(edge, currentTime, eventQueue, selectFlow, result, timeStep);
            }
        }

        std::sort(result.events.begin(), result.events.end());

        result.totalTimeSteps = timeStep + 1;
        result.makespan = currentTime;
        result.theoreticalLowerBound = 0;  // Will be set by caller

        return result;
    }

    //=========================================================================
    // Start a transfer on an edge
    //=========================================================================
    void startTransfer(
        EdgeSchedulerState& edge,
        double currentTime,
        std::priority_queue<SimEvent, std::vector<SimEvent>, std::greater<SimEvent>>& eventQueue,
        std::function<FlowState*(const std::vector<FlowState*>&)> selectFlow,
        ScheduleResult& result,
        int& timeStep) {

        if (edge.waitingFlows.empty()) return;

        FlowState* selected = selectFlow(edge.waitingFlows);
        if (!selected) return;

        // Remove from waiting
        auto it = std::find(edge.waitingFlows.begin(), edge.waitingFlows.end(), selected);
        if (it != edge.waitingFlows.end()) {
            edge.waitingFlows.erase(it);
        }

        // Calculate transfer time using alpha-beta model matching original synthesizer:
        // - latency is in ns, convert to us by /1000
        // - bandwidth is in GB/s, convert to bytes/us: * (1 << 30) / 1e6
        double alpha = edge.latency / 1000.0;  // ns to us
        double bandwidthBytesPerUs = edge.bandwidth * (1 << 30) / 1e6;  // GB/s to bytes/us
        double beta = selected->totalBytes / bandwidthBytesPerUs;
        double transferTime = alpha + beta;

        edge.activeFlow = selected;
        edge.busyUntil = currentTime + transferTime;
        edge.totalBytesTransferred += selected->totalBytes;

        // Record event
        TransferEvent te;
        te.timeStep = timeStep;
        te.startTime = currentTime;
        te.endTime = currentTime + transferTime;
        te.edgeSrc = edge.src;
        te.edgeDst = edge.dst;
        te.commoditySrc = selected->commoditySrc;
        te.commodityDst = selected->commodityDst;
        te.bytes = selected->totalBytes;
        te.pathIndex = selected->pathIndex;
        result.events.push_back(te);

        // Schedule completion
        SimEvent e;
        e.time = edge.busyUntil;
        e.type = SimEvent::TRANSFER_COMPLETE;
        e.edgeSrc = edge.src;
        e.edgeDst = edge.dst;
        e.flow = selected;
        eventQueue.push(e);
    }

    int npusCount_;
    Bandwidth bandwidth_;
    Latency latency_;
    std::vector<std::vector<EdgeSchedulerState>> edgeStates_;
};

} // namespace tacos
