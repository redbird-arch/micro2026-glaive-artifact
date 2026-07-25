/*
# File name  :    unified_solver.h
# Author     :    Chen Chen
# Time       :    2025/01/06
# Description:    Unified solver that integrates all modules:
#                 - Matrix analysis
#                 - Congestion-aware routing
#                 - Optimal scheduling
#                 - Physics-based diffusion (when needed)
*/

#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

#include "matrix_analyzer.h"
#include "congestion_aware_router.h"
#include "optimal_scheduler.h"
#include "diffusion_model.h"

namespace tacos {

//=============================================================================
// SolverConfig: Configuration for the unified solver
//=============================================================================
struct SolverConfig {
    // Routing
    RoutingStrategy routingStrategy = RoutingStrategy::ADAPTIVE;
    bool enableMultiPath = true;
    int maxPaths = 2;

    // Scheduling
    SchedulingPolicy schedulingPolicy = SchedulingPolicy::GREEDY_LOAD_BALANCE;

    // Diffusion
    bool enableDiffusion = true;
    double diffusionThreshold = 0.4;  // Congestion score threshold
    DiffusionConfig diffusionConfig;

    // Tiered Algorithm Selection (NUS-based)
    bool enableTieredSelection = true;  // Use NUS to auto-select algorithm
    bool forceStrategy = false;         // If true, use routingStrategy regardless of NUS

    // Output
    bool verbose = true;
    bool printSchedule = false;
};

//=============================================================================
// SolverResult: Complete result from the unified solver
//=============================================================================
struct SolverResult {
    // Analysis
    MatrixAnalysisResult analysis;

    // NUS-based algorithm selection
    double nonUniformityScore;
    int selectedTier;
    std::string tierName;

    // Diffusion (if applied)
    bool diffusionApplied = false;
    DiffusionResult diffusionResult;

    // Routing
    RoutingPlan routingPlan;
    RoutingStrategy usedStrategy;

    // Scheduling
    ScheduleResult scheduleResult;

    // Summary
    double theoreticalLowerBound;
    double actualMakespan;
    double efficiency;  // LB / makespan
    int totalTimeSteps;

    void print() const {
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║              UNIFIED SOLVER RESULT SUMMARY                   ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Matrix Analysis:                                             ║\n";
        std::cout << "║   - NPUs: " << std::setw(5) << analysis.npusCount
                  << "                                               ║\n";
        std::cout << "║   - Total Demand: " << std::setw(12) << analysis.totalDemand
                  << " bytes                      ║\n";
        std::cout << "║   - Sparsity: " << std::fixed << std::setprecision(1)
                  << std::setw(5) << (analysis.sparsity * 100) << "%"
                  << "                                        ║\n";
        std::cout << "║   - Congestion Score: " << std::setprecision(2)
                  << std::setw(4) << analysis.congestion.congestionScore
                  << " (" << analysis.congestion.severityLevel() << ")"
                  << "                          ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ NUS-based Algorithm Selection:                               ║\n";
        std::cout << "║   - Non-Uniformity Score: " << std::setprecision(3)
                  << std::setw(5) << nonUniformityScore
                  << "                              ║\n";
        std::cout << "║   - Selected Tier: " << selectedTier << " (" << tierName << ")"
                  << std::string(std::max(0, 37 - (int)tierName.length()), ' ') << "║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";

        if (diffusionApplied) {
            std::cout << "║ Diffusion Applied:                                           ║\n";
            std::cout << "║   - Iterations: " << std::setw(4) << diffusionResult.iterationsUsed
                      << "                                          ║\n";
            std::cout << "║   - Imbalance: " << std::setprecision(2)
                      << std::setw(5) << diffusionResult.initialImbalance << " -> "
                      << std::setw(5) << diffusionResult.finalImbalance
                      << "                           ║\n";
            std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        }

        std::cout << "║ Routing:                                                     ║\n";
        std::cout << "║   - Strategy: " << std::setw(20) << strategyName(usedStrategy)
                  << "                       ║\n";
        std::cout << "║   - Max Edge Load: " << std::setw(12) << routingPlan.maxEdgeLoad
                  << " bytes                  ║\n";
        std::cout << "║   - Load Balance Ratio: " << std::setprecision(2)
                  << std::setw(5) << routingPlan.loadBalanceRatio
                  << "                              ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Scheduling:                                                  ║\n";
        std::cout << "║   - Time Steps: " << std::setw(6) << totalTimeSteps
                  << "                                       ║\n";
        std::cout << "║   - Theoretical LB: " << std::setprecision(2)
                  << std::setw(10) << theoreticalLowerBound << " us"
                  << "                        ║\n";
        std::cout << "║   - Actual Makespan: " << std::setprecision(2)
                  << std::setw(10) << actualMakespan << " us"
                  << "                       ║\n";
        std::cout << "║   - Efficiency: " << std::setprecision(1)
                  << std::setw(6) << (efficiency * 100) << "%"
                  << "                                    ║\n";
        std::cout << "║   - Gap: " << std::setprecision(1)
                  << std::setw(6) << ((1.0 / efficiency - 1.0) * 100) << "%"
                  << "                                         ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    }

private:
    std::string strategyName(RoutingStrategy strategy) const {
        switch (strategy) {
            case RoutingStrategy::DIMENSION_ORDER: return "DIMENSION_ORDER";
            case RoutingStrategy::LOAD_BALANCED: return "LOAD_BALANCED";
            case RoutingStrategy::MULTI_PATH: return "MULTI_PATH";
            case RoutingStrategy::ADAPTIVE: return "ADAPTIVE";
            case RoutingStrategy::DIMENSION_EXCHANGE: return "DIMENSION_EXCHANGE";
            case RoutingStrategy::LATIN_SQUARE: return "LATIN_SQUARE";
            default: return "UNKNOWN";
        }
    }
};

//=============================================================================
// UnifiedSolver: Main solver class
//=============================================================================
class UnifiedSolver {
public:
    using DataSize = long long;
    using Bandwidth = double;
    using Latency = double;

    UnifiedSolver(const std::vector<int>& shape, bool isTorus,
                  Bandwidth bandwidth, Latency latency)
        : shape_(shape), isTorus_(isTorus),
          bandwidth_(bandwidth), latency_(latency) {
        npusCount_ = 1;
        for (int d : shape) {
            npusCount_ *= d;
        }
    }

    void setConfig(const SolverConfig& config) {
        config_ = config;
    }

    //=========================================================================
    // Main solve function
    //=========================================================================
    SolverResult solve(const std::vector<std::vector<DataSize>>& demand) {
        SolverResult result;

        if (config_.verbose) {
            std::cout << "\n";
            std::cout << "========================================================\n";
            std::cout << "         UNIFIED SOLVER FOR NON-UNIFORM ALL-TO-ALL      \n";
            std::cout << "========================================================\n";
        }

        // Step 0: Compute and print theoretical lower bound FIRST
        if (config_.verbose) {
            std::cout << "\n";
            std::cout << "========================================================\n";
            std::cout << "[Step 0] Computing Theoretical Lower Bound\n";
            std::cout << "========================================================\n";

            // Compute theoretical lower bound based on demand matrix
            double theoreticalLB = computeTheoreticalLowerBound(demand);

            std::cout << "  Theoretical Lower Bound: " << std::fixed << std::setprecision(2)
                      << theoreticalLB << " us (" << (theoreticalLB / 1e6) << " s)\n";
            std::cout << "========================================================\n";
        }

        // Step 1: Analyze the demand matrix
        if (config_.verbose) {
            std::cout << "\n[Step 1] Analyzing demand matrix...\n";
        }

        MatrixAnalyzer analyzer(shape_, isTorus_);
        result.analysis = analyzer.analyze(demand);

        if (config_.verbose) {
            result.analysis.print();
        }

        // Step 1.5: NUS-based algorithm selection
        result.nonUniformityScore = result.analysis.nonUniformityScore;
        result.selectedTier = result.analysis.recommendedTier;
        result.tierName = getTierName(result.selectedTier);

        if (config_.verbose) {
            std::cout << "\n[Step 1.5] NUS-based Algorithm Selection\n";
            std::cout << "  Non-Uniformity Score (NUS): " << std::fixed << std::setprecision(3)
                      << result.nonUniformityScore << "\n";
            std::cout << "  Recommended Tier: " << result.selectedTier
                      << " (" << result.tierName << ")\n";
        }

        // Determine routing strategy based on tier (if tiered selection is enabled)
        RoutingStrategy selectedStrategy = config_.routingStrategy;
        if (config_.enableTieredSelection && !config_.forceStrategy) {
            selectedStrategy = tierToRoutingStrategy(result.selectedTier);
            if (config_.verbose) {
                std::cout << "  Selected Routing Strategy: " << strategyToString(selectedStrategy) << "\n";
            }
        }

        // Step 2: Apply diffusion if needed (only for Tier 4)
        std::vector<std::vector<DataSize>> workingDemand = demand;

        bool shouldApplyDiffusion = config_.enableDiffusion &&
            (result.selectedTier == 4 || result.analysis.congestion.needsDiffusion);

        if (shouldApplyDiffusion) {

            if (config_.verbose) {
                std::cout << "\n[Step 2] Applying diffusion for load balancing...\n";
            }

            DiffusionModel diffusion(shape_, isTorus_);
            diffusion.setConfig(config_.diffusionConfig);

            result.diffusionResult = diffusion.applyDiffusion(demand, result.analysis);
            result.diffusionApplied = true;

            if (config_.verbose) {
                result.diffusionResult.print();
            }

            workingDemand = result.diffusionResult.diffusedDemand;
        } else {
            if (config_.verbose) {
                std::cout << "\n[Step 2] Skipping diffusion (not needed)\n";
            }
            result.diffusionApplied = false;
        }

        // Step 3: Compute routing
        if (config_.verbose) {
            std::cout << "\n[Step 3] Computing routing...\n";
        }

        CongestionAwareRouter router(shape_, isTorus_);
        result.routingPlan = router.computeRouting(workingDemand, result.analysis,
                                                   selectedStrategy);
        result.usedStrategy = selectedStrategy;

        if (config_.verbose) {
            result.routingPlan.print(npusCount_);
        }

        // Step 4: Compute schedule
        if (config_.verbose) {
            std::cout << "\n[Step 4] Computing schedule...\n";
        }

        OptimalScheduler scheduler(npusCount_, bandwidth_, latency_);
        result.scheduleResult = scheduler.schedule(result.routingPlan,
                                                   config_.schedulingPolicy);

        // Compute lower bound
        result.theoreticalLowerBound = scheduler.computeLowerBound(result.routingPlan, demand);
        result.scheduleResult.theoreticalLowerBound = result.theoreticalLowerBound;

        result.actualMakespan = result.scheduleResult.makespan;
        result.totalTimeSteps = result.scheduleResult.totalTimeSteps;
        result.efficiency = result.theoreticalLowerBound / result.actualMakespan;

        if (config_.verbose) {
            result.scheduleResult.printSummary();
        }

        if (config_.printSchedule) {
            result.scheduleResult.print();
        }

        // Print final summary
        if (config_.verbose) {
            result.print();
        }

        return result;
    }

    //=========================================================================
    // Compare different strategies
    //=========================================================================
    void compareStrategies(const std::vector<std::vector<DataSize>>& demand) {
        std::cout << "\n";
        std::cout << "========================================================\n";
        std::cout << "              STRATEGY COMPARISON (All 5 Tiers)         \n";
        std::cout << "========================================================\n";

        MatrixAnalyzer analyzer(shape_, isTorus_);
        auto analysis = analyzer.analyze(demand);

        CongestionAwareRouter router(shape_, isTorus_);
        OptimalScheduler scheduler(npusCount_, bandwidth_, latency_);

        // All 5 routing strategies corresponding to Tier 0-4
        std::vector<RoutingStrategy> strategies = {
            RoutingStrategy::DIMENSION_ORDER,   // Tier 0: DOR XY
            RoutingStrategy::DOR_XY_BINPACK,    // Tier 1: DOR XY + Binpacking
            RoutingStrategy::LOAD_BALANCED,     // Tier 2: Load Balanced
            RoutingStrategy::MULTI_PATH,        // Tier 3: Multi-Path
            RoutingStrategy::ADAPTIVE           // Tier 4: Adaptive (with diffusion if needed)
        };

        std::vector<std::string> names = {
            "Tier0: DOR_XY",
            "Tier1: DOR_XY_BINPACK",
            "Tier2: LOAD_BALANCED",
            "Tier3: MULTI_PATH",
            "Tier4: ADAPTIVE"
        };

        std::cout << "\n";
        std::cout << std::setw(25) << "Strategy"
                  << std::setw(15) << "Max Edge Load"
                  << std::setw(15) << "Load Ratio"
                  << std::setw(15) << "Makespan (us)"
                  << std::setw(12) << "Efficiency"
                  << "\n";
        std::cout << std::string(82, '-') << "\n";

        double bestMakespan = std::numeric_limits<double>::max();
        std::string bestStrategy;

        for (size_t i = 0; i < strategies.size(); ++i) {
            // For ADAPTIVE strategy, apply diffusion if needed
            std::vector<std::vector<DataSize>> workingDemand = demand;
            bool diffusionApplied = false;

            if (strategies[i] == RoutingStrategy::ADAPTIVE && config_.enableDiffusion &&
                (analysis.congestion.needsDiffusion || analysis.nonUniformityScore >= 0.8)) {
                DiffusionModel diffusion(shape_, isTorus_);
                diffusion.setConfig(config_.diffusionConfig);
                auto diffResult = diffusion.applyDiffusion(demand, analysis);
                workingDemand = diffResult.diffusedDemand;
                diffusionApplied = true;
            }

            auto plan = router.computeRouting(workingDemand, analysis, strategies[i]);
            auto schedule = scheduler.schedule(plan, config_.schedulingPolicy);
            double lb = scheduler.computeLowerBound(plan, demand);
            double efficiency = lb / schedule.makespan;

            std::string displayName = names[i];
            if (diffusionApplied) {
                displayName += " +Diff";
            }

            std::cout << std::setw(25) << displayName
                      << std::setw(15) << plan.maxEdgeLoad
                      << std::setw(15) << std::fixed << std::setprecision(2) << plan.loadBalanceRatio
                      << std::setw(15) << std::setprecision(2) << schedule.makespan
                      << std::setw(11) << std::setprecision(1) << (efficiency * 100) << "%"
                      << "\n";

            if (schedule.makespan < bestMakespan) {
                bestMakespan = schedule.makespan;
                bestStrategy = displayName;
            }
        }

        std::cout << std::string(82, '-') << "\n";
        std::cout << "Best Strategy: " << bestStrategy << " (Makespan: " << bestMakespan << " us)\n";
        std::cout << "========================================================\n";
    }

private:
    //=========================================================================
    // Get tier name from tier number
    //=========================================================================
    std::string getTierName(int tier) const {
        switch (tier) {
            case 0: return "DOR_XY";
            case 1: return "DOR_XY_BINPACK";
            case 2: return "LOAD_BALANCED";
            case 3: return "MULTI_PATH";
            case 4: return "DIFFUSION_ADAPTIVE";
            default: return "UNKNOWN";
        }
    }

    //=========================================================================
    // Convert tier to routing strategy
    //=========================================================================
    RoutingStrategy tierToRoutingStrategy(int tier) const {
        switch (tier) {
            case 0: return RoutingStrategy::DIMENSION_ORDER;
            case 1: return RoutingStrategy::DOR_XY_BINPACK;  // Tier 1: Binpack optimization
            case 2: return RoutingStrategy::LOAD_BALANCED;
            case 3: return RoutingStrategy::MULTI_PATH;
            case 4: return RoutingStrategy::ADAPTIVE;  // Adaptive with diffusion
            default: return RoutingStrategy::ADAPTIVE;
        }
    }

    //=========================================================================
    // Convert routing strategy to string
    //=========================================================================
    std::string strategyToString(RoutingStrategy strategy) const {
        switch (strategy) {
            case RoutingStrategy::DIMENSION_ORDER: return "DIMENSION_ORDER";
            case RoutingStrategy::DOR_XY_BINPACK: return "DOR_XY_BINPACK";
            case RoutingStrategy::LOAD_BALANCED: return "LOAD_BALANCED";
            case RoutingStrategy::MULTI_PATH: return "MULTI_PATH";
            case RoutingStrategy::ADAPTIVE: return "ADAPTIVE";
            case RoutingStrategy::DIMENSION_EXCHANGE: return "DIMENSION_EXCHANGE";
            case RoutingStrategy::LATIN_SQUARE: return "LATIN_SQUARE";
            default: return "UNKNOWN";
        }
    }

    //=========================================================================
    // Compute node degree for a given node ID in mesh/torus topology
    //=========================================================================
    int computeNodeDegree(int nodeID) const {
        // Convert nodeID to coordinates (row-major order)
        // Similar to BaselineSolver::npuIDToCoordinate
        std::vector<int> coord(shape_.size());
        int remaining = nodeID;
        for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
            int stride = 1;
            for (int i = d + 1; i < static_cast<int>(shape_.size()); ++i) {
                stride *= shape_[i];
            }
            coord[d] = remaining / stride;
            remaining = remaining % stride;
        }
        
        // Calculate degree: count neighbors in each dimension
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

    //=========================================================================
    // Compute theoretical lower bound based on demand matrix
    //=========================================================================
    double computeTheoreticalLowerBound(const std::vector<std::vector<DataSize>>& demand) {
        // Use alpha-beta model matching original synthesizer:
        // - latency_ is in ns, convert to us by /1000
        // - bandwidth_ is in GB/s, convert to bytes/us: * (1 << 30) / 1e6
        double bandwidthBytesPerUs = bandwidth_ * (1 << 30) / 1e6;  // GB/s to bytes/us
        double latencyUs = latency_ / 1000.0;  // ns to us

        // Compute node degrees for all nodes (similar to Topology::degrees())
        std::vector<int> nodeDegrees(npusCount_);
        for (int i = 0; i < npusCount_; ++i) {
            nodeDegrees[i] = computeNodeDegree(i);
        }

        // LB1: Max row sum (send constraint) - each node can only send on its outgoing links
        // Use the actual node degree for each node
        double maxLbSend = 0.0;
        int maxRowNode = -1;
        for (int i = 0; i < npusCount_; ++i) {
            DataSize rowSum = 0;
            for (int j = 0; j < npusCount_; ++j) {
                if (i != j) rowSum += demand[i][j];
            }
            if (rowSum > 0) {
                double nodeBandwidth = bandwidthBytesPerUs * nodeDegrees[i];
                double lb_send_i = rowSum / nodeBandwidth;
                if (lb_send_i > maxLbSend) {
                    maxLbSend = lb_send_i;
                    maxRowNode = i;
                }
            }
        }
        double lb_send = maxLbSend;

        // LB2: Max column sum (receive constraint) - each node can only receive on its incoming links
        // Use the actual node degree for each node
        double maxLbRecv = 0.0;
        int maxColNode = -1;
        for (int j = 0; j < npusCount_; ++j) {
            DataSize colSum = 0;
            for (int i = 0; i < npusCount_; ++i) {
                if (i != j) colSum += demand[i][j];
            }
            if (colSum > 0) {
                double nodeBandwidth = bandwidthBytesPerUs * nodeDegrees[j];
                double lb_recv_j = colSum / nodeBandwidth;
                if (lb_recv_j > maxLbRecv) {
                    maxLbRecv = lb_recv_j;
                    maxColNode = j;
                }
            }
        }
        double lb_recv = maxLbRecv;

        // LB3: Estimate based on topology diameter (latency bound)
        // For mesh/torus, diameter is approximately sum of dimensions
        int diameter = 0;
        for (int d : shape_) {
            diameter += (d - 1);
        }
        if (isTorus_) {
            diameter = 0;
            for (int d : shape_) {
                diameter += (d / 2);
            }
        }
        double lb_latency = diameter * latencyUs;

        // Print detailed breakdown
        int dimensions = shape_.size();
        int minDegree = *std::min_element(nodeDegrees.begin(), nodeDegrees.end());
        int maxDegree = *std::max_element(nodeDegrees.begin(), nodeDegrees.end());
        int avgDegree = std::accumulate(nodeDegrees.begin(), nodeDegrees.end(), 0) / npusCount_;
        
        std::cout << "\n  Lower Bound Analysis:\n";
        std::cout << "    - Topology: " << (isTorus_ ? "Torus" : "Mesh")
                  << " (dimensions: " << dimensions << ")\n";
        std::cout << "    - Node degrees: min=" << minDegree << ", max=" << maxDegree
                  << ", avg=" << avgDegree << "\n";
        
        DataSize maxRowSum = 0;
        if (maxRowNode >= 0) {
            for (int j = 0; j < npusCount_; ++j) {
                if (maxRowNode != j) maxRowSum += demand[maxRowNode][j];
            }
        }
        
        DataSize maxColSum = 0;
        if (maxColNode >= 0) {
            for (int i = 0; i < npusCount_; ++i) {
                if (i != maxColNode) maxColSum += demand[i][maxColNode];
            }
        }
        
        std::cout << "    - Send Constraint (max row sum): " << std::fixed << std::setprecision(2)
                  << lb_send << " us";
        if (maxRowNode >= 0) {
            std::cout << " (bottleneck: node " << maxRowNode << ", degree=" << nodeDegrees[maxRowNode]
                      << ", " << (maxRowSum / (1024.0 * 1024.0)) << " MB)";
        }
        std::cout << "\n";

        std::cout << "    - Receive Constraint (max col sum): " << std::fixed << std::setprecision(2)
                  << lb_recv << " us";
        if (maxColNode >= 0) {
            std::cout << " (bottleneck: node " << maxColNode << ", degree=" << nodeDegrees[maxColNode]
                      << ", " << (maxColSum / (1024.0 * 1024.0)) << " MB)";
        }
        std::cout << "\n";

        std::cout << "    - Latency Bound (diameter): " << std::fixed << std::setprecision(2)
                  << lb_latency << " us (diameter: " << diameter << " hops)\n";

        // The lower bound is the max of bandwidth bounds plus latency
        double lb_bandwidth = std::max(lb_send, lb_recv);
        double totalLB = lb_bandwidth + lb_latency;

        std::cout << "    - Bandwidth Bound: " << std::fixed << std::setprecision(2)
                  << lb_bandwidth << " us\n";
        std::cout << "    - Total Lower Bound: " << std::fixed << std::setprecision(2)
                  << totalLB << " us\n";

        return totalLB;
    }

    std::vector<int> shape_;
    bool isTorus_;
    int npusCount_;
    Bandwidth bandwidth_;
    Latency latency_;
    SolverConfig config_;
};

} // namespace tacos
