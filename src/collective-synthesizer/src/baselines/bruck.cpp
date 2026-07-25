/*
# File name  :    bruck.cpp
# Author     :    Galois
# Time       :    2026/01/13 20:53:52
*/

#include <cassert>
#include <cmath>
#include <algorithm>
#include <queue>
#include <limits>
#include <iostream>
#include <tacos/baselines/bruck.h>
#include <tacos/thread_output.h>

using namespace tacos;

Bruck::Bruck(std::shared_ptr<Topology> topology, 
            const std::vector<int>& shape) noexcept
    : BaselineSolver(topology, shape) {
}

BaselineSolver::Result Bruck::solve(const std::vector<std::vector<long long>>& dataMatrix) {
    assert(static_cast<int>(dataMatrix.size()) == npusCount_);
    
    Result result;
    result.stepTransfers.clear();
    result.stepTimes.clear();
    
    // Calculate number of steps: log2(N)
    int numSteps = static_cast<int>(std::ceil(std::log2(npusCount_)));
    if (numSteps == 0) numSteps = 1;
    
    // Maintain a dynamic data matrix that gets updated after each step
    // In Bruck algorithm, after each step, sent data blocks are removed and
    // received data blocks replace them at the same rotated index positions
    std::vector<std::vector<long long>> currentDataMatrix = dataMatrix;
    
    // Structure to track link busy times (size = full topology for switch topologies with multi-hop paths)
    int topoSize = topology_->npusCount();
    std::vector<std::vector<BaselineSolver::Time>> linkBusyUntil(
        topoSize, std::vector<BaselineSolver::Time>(topoSize, -1.0));
    // Track total busy time per link for utilization analysis
    std::vector<std::vector<BaselineSolver::Time>> linkBusyTime(
        topoSize, std::vector<BaselineSolver::Time>(topoSize, 0.0));
    // Track busy intervals per link for detailed analysis
    std::vector<std::vector<std::vector<std::pair<BaselineSolver::Time, BaselineSolver::Time>>>> linkBusyIntervals(
        topoSize, std::vector<std::vector<std::pair<BaselineSolver::Time, BaselineSolver::Time>>>(topoSize));
    
    BaselineSolver::Time currentTime = 0.0;
    
    // Process each step
    for (int step = 0; step < numSteps; ++step) {
        BaselineSolver::Time stepStartTime = currentTime;
        std::vector<std::pair<BaselineSolver::NpuID, BaselineSolver::NpuID>> stepTransfers;
        
        // Calculate step offset: 2^k
        int offset = 1 << step;  // 2^step
        
        // Collect all transfers for this step
        struct HopInfo {
            BaselineSolver::NpuID hopSrc;
            BaselineSolver::NpuID hopDest;
            BaselineSolver::Time hopStart;
            BaselineSolver::Time hopEnd;
        };
        
        struct Transfer {
            BaselineSolver::NpuID src;
            BaselineSolver::NpuID dest;
            long long dataSize;
            std::vector<BaselineSolver::NpuID> path;
            std::vector<HopInfo> hopInfos;  // Store actual hop timing information
        };
        
        std::vector<Transfer> allTransfers;
        
        for (BaselineSolver::NpuID i = 0; i < npusCount_; ++i) {
            // Node i sends to node (i + 2^k) % N
            BaselineSolver::NpuID sendTo = (i + offset) % npusCount_;
            
            // Get data blocks to transfer in this step (using current data matrix)
            auto dataBlocks = getDataBlocksForStep(i, step, currentDataMatrix);
            
            for (const auto& [dest, dataSize] : dataBlocks) {
                if (dataSize > 0) {
                    auto path = getXYPath(i, dest);
                    Transfer transfer;
                    transfer.src = i;
                    transfer.dest = dest;
                    transfer.dataSize = dataSize;
                    transfer.path = path;
                    transfer.hopInfos.clear();  // Will be populated during simulation
                    allTransfers.push_back(transfer);
                }
            }
        }
        
        // Event-driven simulation: congestion-aware (wait for link if busy), store-and-forward per hop
        // (including switches for switch-based topologies; no cut-through).
        for (auto& transfer : allTransfers) {
            BaselineSolver::Time transferStartTime = stepStartTime;
            for (size_t hopIdx = 0; hopIdx < transfer.path.size() - 1; ++hopIdx) {
                BaselineSolver::NpuID hopSrc = transfer.path[hopIdx];
                BaselineSolver::NpuID hopDest = transfer.path[hopIdx + 1];
                if (!topology_->connected(hopSrc, hopDest)) continue;
                BaselineSolver::Time linkAvailableTime = (linkBusyUntil[hopSrc][hopDest] < 0)
                    ? transferStartTime : std::max(transferStartTime, linkBusyUntil[hopSrc][hopDest]);
                BaselineSolver::Time hopTime = calculateTransferTime(hopSrc, hopDest, transfer.dataSize);
                if (std::isnan(hopTime) || std::isinf(hopTime) || hopTime < 0) hopTime = 0.0;
                HopInfo hopInfo;
                hopInfo.hopSrc = hopSrc;
                hopInfo.hopDest = hopDest;
                hopInfo.hopStart = linkAvailableTime;
                hopInfo.hopEnd = linkAvailableTime + hopTime;
                transfer.hopInfos.push_back(hopInfo);
                linkBusyUntil[hopSrc][hopDest] = hopInfo.hopEnd;
                // Record busy interval and accumulate busy time for utilization (per link)
                linkBusyIntervals[hopSrc][hopDest].push_back({hopInfo.hopStart, hopInfo.hopEnd});
                linkBusyTime[hopSrc][hopDest] += (hopInfo.hopEnd - hopInfo.hopStart);
                transferStartTime = hopInfo.hopEnd;
            }
            stepTransfers.push_back({transfer.src, transfer.dest});
        }
        
        // Calculate step end time: maximum of all link busy times
        BaselineSolver::Time stepEndTime = stepStartTime;
        for (const auto& transfer : allTransfers) {
            for (size_t hopIdx = 0; hopIdx < transfer.path.size() - 1; ++hopIdx) {
                BaselineSolver::NpuID hopSrc = transfer.path[hopIdx];
                BaselineSolver::NpuID hopDest = transfer.path[hopIdx + 1];
                BaselineSolver::Time linkEndTime = linkBusyUntil[hopSrc][hopDest];
                
                // Check for invalid times
                if (std::isnan(linkEndTime) || std::isinf(linkEndTime)) {
                    std::cerr << "Warning: Invalid link end time for " << hopSrc << " -> " << hopDest << std::endl;
                    continue;
                }
                
                stepEndTime = std::max(stepEndTime, linkEndTime);
            }
        }
        
        currentTime = stepEndTime;
        BaselineSolver::Time stepTime = currentTime - stepStartTime;
        
        // Check for invalid step time
        if (std::isnan(stepTime) || std::isinf(stepTime)) {
            std::cerr << "Warning: Invalid step time calculated, using 0.0" << std::endl;
            stepTime = 0.0;
            currentTime = stepStartTime;
        }
        
        result.stepTimes.push_back(stepTime);
        result.stepTransfers.push_back(stepTransfers);
        
        // Print step information with detailed data sizes
        ThreadOutput::output("[Bruck Step ");
        ThreadOutput::output(step);
        ThreadOutput::output("] Time: ");
        ThreadOutput::output(stepTime);
        ThreadOutput::output(" us, Transfers: ");
        ThreadOutput::output(static_cast<int>(stepTransfers.size()));
        ThreadOutput::output(std::endl);
        
        // Print each transfer; path and hop use formatNodeName so switches show as layer+index (e.g. node_switch(node=0,sw=0))
        for (size_t tIdx = 0; tIdx < allTransfers.size(); ++tIdx) {
            const auto& transfer = allTransfers[tIdx];
            ThreadOutput::output("  ");
            ThreadOutput::output(topology_->formatNodeName(transfer.src));
            ThreadOutput::output(" -> ");
            ThreadOutput::output(topology_->formatNodeName(transfer.dest));
            ThreadOutput::output(" (");
            ThreadOutput::output(transfer.dataSize);
            ThreadOutput::output(" bytes");
            
            // Print path if multi-hop
            if (transfer.path.size() > 2) {
                ThreadOutput::output(", path: ");
                for (size_t i = 0; i < transfer.path.size(); ++i) {
                    ThreadOutput::output(topology_->formatNodeName(transfer.path[i]));
                    if (i < transfer.path.size() - 1) {
                        ThreadOutput::output("->");
                    }
                }
            }
            ThreadOutput::output(")");
            ThreadOutput::output(std::endl);
            
            // Print hop-by-hop details using stored timing information
            for (size_t hopIdx = 0; hopIdx < transfer.hopInfos.size(); ++hopIdx) {
                const auto& hopInfo = transfer.hopInfos[hopIdx];
                
                ThreadOutput::output("    Hop ");
                ThreadOutput::output(hopIdx);
                ThreadOutput::output(": ");
                ThreadOutput::output(topology_->formatNodeName(hopInfo.hopSrc));
                ThreadOutput::output(" -> ");
                ThreadOutput::output(topology_->formatNodeName(hopInfo.hopDest));
                ThreadOutput::output(" (");
                ThreadOutput::output(transfer.dataSize);
                ThreadOutput::output(" bytes, time: ");
                ThreadOutput::output(hopInfo.hopEnd-hopInfo.hopStart);
                ThreadOutput::output(" us, ");
                ThreadOutput::output(hopInfo.hopStart);
                ThreadOutput::output(" - ");
                ThreadOutput::output(hopInfo.hopEnd);
                ThreadOutput::output(" us, link busy until: ");
                ThreadOutput::output(hopInfo.hopEnd);
                ThreadOutput::output(" us)");
                ThreadOutput::output(std::endl);
            }
        }
        
        // Update currentDataMatrix after this step completes
        // In Bruck algorithm, after step k:
        // 1. Each node i sends all data blocks with rotated index j (bit k=1) to node (i+2^k)%N
        // 2. Each node i receives all data blocks with rotated index j (bit k=1) from node (i-2^k+N)%N
        // 3. The received data replaces the sent data at the same rotated index positions
        for (BaselineSolver::NpuID i = 0; i < npusCount_; ++i) {
            BaselineSolver::NpuID receiveFrom = (i - offset + npusCount_) % npusCount_;
            
            // For each rotated index j with bit k=1:
            // - Remove sent data: currentDataMatrix[i][originalDest] where originalDest = (j-i+N)%N
            // - Add received data: currentDataMatrix[receiveFrom][originalDest] where originalDest = (j-receiveFrom+N)%N
            for (BaselineSolver::NpuID rotatedIndex = 0; rotatedIndex < npusCount_; ++rotatedIndex) {
                if ((rotatedIndex >> step) & 1) {
                    // Original destination for sent data from node i
                    BaselineSolver::NpuID sentOriginalDest = (rotatedIndex + i) % npusCount_;
                    // Original destination for received data (from receiveFrom's perspective)
                    BaselineSolver::NpuID receivedOriginalDest = (rotatedIndex + receiveFrom) % npusCount_;
                    
                    // Remove sent data (set to 0)
                    currentDataMatrix[i][sentOriginalDest] = 0;
                    
                    // Add received data (from receiveFrom)
                    currentDataMatrix[i][sentOriginalDest] = currentDataMatrix[receiveFrom][receivedOriginalDest];
                }
            }
        }
        
        // Reset link busy times for next step (all links available at stepEndTime)
        for (int i = 0; i < topoSize; ++i) {
            for (int j = 0; j < topoSize; ++j) {
                if (linkBusyUntil[i][j] <= currentTime) {
                    linkBusyUntil[i][j] = -1.0;
                }
            }
        }
    }
    
    result.totalTime = currentTime;

    // Compute average link utilization over the whole run: (busy time / makespan) averaged over used links
    double avgUtil = 0.0;
    int linkCount = 0;
    if (result.totalTime > 0.0) {
        for (int i = 0; i < topoSize; ++i) {
            for (int j = 0; j < topoSize; ++j) {
                if (linkBusyTime[i][j] > 0.0) {
                    avgUtil += static_cast<double>(linkBusyTime[i][j]) / result.totalTime;
                    ++linkCount;
                }
            }
        }
        if (linkCount > 0) {
            avgUtil /= static_cast<double>(linkCount);
        }
    }
    result.averageUtilization = avgUtil;

    ThreadOutput::output("  Average Link Utilization: ");
    ThreadOutput::output(static_cast<double>(avgUtil * 100.0));
    ThreadOutput::output(" %");
    ThreadOutput::output(std::endl);

    // Print per-link busy intervals (in ns) and per-link utilization
    ThreadOutput::output("  Link Busy Intervals (ns) and Utilization:\n");
    for (int i = 0; i < topoSize; ++i) {
        for (int j = 0; j < topoSize; ++j) {
            const auto& intervals = linkBusyIntervals[i][j];
            if (intervals.empty()) continue;
            ThreadOutput::output("    Link(");
            ThreadOutput::output(topology_->formatNodeName(i));
            ThreadOutput::output("->");
            ThreadOutput::output(topology_->formatNodeName(j));
            ThreadOutput::output("): intervals=[");
            for (size_t k = 0; k < intervals.size(); ++k) {
                auto seg = intervals[k];
                long long startNs = static_cast<long long>(std::llround(seg.first * 1000.0));
                long long endNs = static_cast<long long>(std::llround(seg.second * 1000.0));
                ThreadOutput::output("[");
                ThreadOutput::output(startNs);
                ThreadOutput::output(", ");
                ThreadOutput::output(endNs);
                ThreadOutput::output("]");
                if (k + 1 < intervals.size()) {
                    ThreadOutput::output(", ");
                }
            }
            ThreadOutput::output("], utilization=");
            double util = (result.totalTime > 0.0)
                              ? static_cast<double>(linkBusyTime[i][j]) / result.totalTime
                              : 0.0;
            ThreadOutput::output(static_cast<double>(util * 100.0));
            ThreadOutput::output(" %");
            ThreadOutput::output(std::endl);
        }
    }

    return result;
}

std::vector<std::pair<BaselineSolver::NpuID, long long>> Bruck::getDataBlocksForStep(
    BaselineSolver::NpuID nodeID, int step, const std::vector<std::vector<long long>>& dataMatrix) const noexcept {
    
    std::vector<std::pair<BaselineSolver::NpuID, long long>> blocks;
    
    // Bruck algorithm with initial rotation:
    // After initial rotation at node i:
    //   - Original data block j (meant for destination j) becomes data block (j-i) % N
    //   - So rotated index j corresponds to original data block (j+i) % N
    //   - Original data block j was meant for destination j
    //   - Therefore, rotated index j corresponds to original destination (j+i) % N
    //
    // In step k, node i sends blocks whose rotated index j has bit k = 1 in its binary representation
    // The actual destination is (nodeID + 2^k) % N
    
    long long totalDataSize = 0;
    for (BaselineSolver::NpuID rotatedIndex = 0; rotatedIndex < npusCount_; ++rotatedIndex) {
        // Check if bit k is set in rotatedIndex's binary representation
        if ((rotatedIndex >> step) & 1) {
            // This rotated index j corresponds to original data block (j-i+N) % N
            // which was originally meant for destination (j-i+N) % N
            BaselineSolver::NpuID originalDest = (rotatedIndex + nodeID) % npusCount_;
            totalDataSize += dataMatrix[nodeID][originalDest];
        }
    }
    
    if (totalDataSize > 0) {
        // In Bruck, all blocks with bit k=1 are sent together in one transfer
        // The actual destination is (nodeID + 2^k) % N
        BaselineSolver::NpuID sendTo = (nodeID + (1 << step)) % npusCount_;
        blocks.push_back({sendTo, totalDataSize});
    }
    
    return blocks;
}
