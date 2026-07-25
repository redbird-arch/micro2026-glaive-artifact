/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <queue>
#include <set>
#include <tacos/synthesizer/profiler.h>
#include <tacos/collective/collective.h>
#include <tacos/thread_output.h>
#include <unordered_set>
#include <vector>
#include <iomanip>
#include <sstream>

using namespace tacos;

Profiler::Time Profiler::computeLowerBound(const std::shared_ptr<Topology>& topology,
                                            const std::shared_ptr<Collective>& collective,
                                            Profiler::ChunkSize chunkSize) noexcept {
    assert(topology != nullptr);
    assert(collective != nullptr);
    assert(chunkSize > 0);

    ThreadOutput::output("========================================================");
    ThreadOutput::output(std::endl);
    ThreadOutput::output("[Profiler] Computing Theoretical Lower Bound");
    ThreadOutput::output(std::endl);
    ThreadOutput::output("========================================================");
    ThreadOutput::output(std::endl);

    // Compute both lower bounds and take the maximum (most restrictive)
    Time degreeBound = computeDegreeBasedLowerBound(topology, collective, chunkSize);
    Time bisectionBound = computeBisectionBasedLowerBound(topology, collective, chunkSize);

    ThreadOutput::output("--------------------------------------------------------");
    ThreadOutput::output(std::endl);
    ThreadOutput::output("[Profiler] Summary:");
    ThreadOutput::output(std::endl);
    ThreadOutput::output("  Degree-Based Lower Bound: ");
    ThreadOutput::output(degreeBound);
    ThreadOutput::output(" us");
    ThreadOutput::output(std::endl);
    ThreadOutput::output("  Bisection-Based Lower Bound: ");
    ThreadOutput::output(bisectionBound);
    ThreadOutput::output(" us");
    ThreadOutput::output(std::endl);
    
    Time finalBound = std::max(degreeBound, bisectionBound);
    ThreadOutput::output("  Final Lower Bound (max): ");
    ThreadOutput::output(finalBound);
    ThreadOutput::output(" us (");
    ThreadOutput::output(finalBound / 1e6);
    ThreadOutput::output(" s)");
    ThreadOutput::output(std::endl);
    ThreadOutput::output("========================================================");
    ThreadOutput::output(std::endl);

    // Return the maximum (most restrictive) lower bound
    return finalBound;
}

Profiler::Time Profiler::computeDegreeBasedLowerBound(const std::shared_ptr<Topology>& topology,
                                                        const std::shared_ptr<Collective>& collective,
                                                        Profiler::ChunkSize chunkSize) noexcept {
    ThreadOutput::output("[Profiler] Computing Degree-Based Lower Bound...");
    ThreadOutput::output(std::endl);
    
    const int npusCount = topology->npusCount();
    const auto nodeDegrees = topology->degrees();
    const auto precondition = collective->getPrecondition();
    const auto postcondition = collective->getPostcondition();

    // Compute total data each node needs to send and receive
    // Important: exclude self-transfers (node doesn't need to send/receive to itself)
    std::vector<double> sendData(npusCount, 0.0);  // in bytes
    std::vector<double> recvData(npusCount, 0.0);   // in bytes

    // Count chunks sent from each node (excluding self-transfers)
    for (const auto& [chunk, src] : precondition) {
        auto it = postcondition.find(chunk);
        if (it == postcondition.end()) {
            continue;
        }
        const auto& dests = it->second;
        // Count destinations that are not the source itself
        int externalDests = 0;
        for (NpuID dest : dests) {
            if (dest != src) {
                externalDests++;
            }
        }
        sendData[src] += externalDests * chunkSize;
    }

    // Count chunks received by each node (excluding self-transfers)
    for (const auto& [chunk, dests] : postcondition) {
        auto it = precondition.find(chunk);
        if (it == precondition.end()) {
            continue;
        }
        NpuID src = it->second;
        for (NpuID dest : dests) {
            // Only count if source is different from destination
            if (dest != src) {
                recvData[dest] += chunkSize;
            }
        }
    }

    Profiler::Time maxTime = 0.0;
    int bottleneckNode = -1;
    std::string bottleneckReason = "";

    // For each node, compute time needed based on its degree and bandwidth
    for (int node = 0; node < npusCount; ++node) {
        int degree = nodeDegrees[node];
        if (degree == 0) {
            continue;  // Isolated node, skip
        }

        // Compute total bandwidth available from this node
        double totalBandwidth = 0.0;  // in GiB/s
        double minBandwidth = std::numeric_limits<double>::max();
        for (int neighbor = 0; neighbor < npusCount; ++neighbor) {
            if (topology->connected(node, neighbor)) {
                double bw = topology->bandwidth(node, neighbor);
                totalBandwidth += bw;
                minBandwidth = std::min(minBandwidth, bw);
            }
        }

        // Convert bandwidth from GiB/s to bytes/us
        // 1 GiB = 1024^3 bytes = 1073741824 bytes
        // 1 GiB/s = 1073741824 bytes/s = 1073741824 / 1e6 bytes/us = 1073.741824 bytes/us
        const double GiB_TO_BYTES_PER_US = 1073.741824;
        double totalBandwidthBytesPerUs = totalBandwidth * GiB_TO_BYTES_PER_US;

        Profiler::Time nodeTime = 0.0;
        std::string timeReason = "";

        // Time needed to send data: sendData / bandwidth
        if (sendData[node] > 0 && totalBandwidthBytesPerUs > 0) {
            Profiler::Time sendTime = sendData[node] / totalBandwidthBytesPerUs;
            if (sendTime > nodeTime) {
                nodeTime = sendTime;
                timeReason = "send";
            }
        }

        // Time needed to receive data: recvData / bandwidth
        if (recvData[node] > 0 && totalBandwidthBytesPerUs > 0) {
            Profiler::Time recvTime = recvData[node] / totalBandwidthBytesPerUs;
            if (recvTime > nodeTime) {
                nodeTime = recvTime;
                timeReason = "receive";
            }
        }

        // Also consider the bottleneck: if a node has low degree, it's constrained
        // by the number of parallel transfers it can make
        // For nodes with degree d, at most d transfers can happen simultaneously
        // So time >= max(sendData, recvData) / (d * minBandwidth)
        if (degree > 0 && minBandwidth < std::numeric_limits<double>::max()) {
            double maxData = std::max(sendData[node], recvData[node]);
            if (maxData > 0) {
                double degreeBandwidthBytesPerUs = degree * minBandwidth * GiB_TO_BYTES_PER_US;
                Profiler::Time degreeTime = maxData / degreeBandwidthBytesPerUs;
                if (degreeTime > nodeTime) {
                    nodeTime = degreeTime;
                    timeReason = "degree-constrained";
                }
            }
        }

        // Print node information
        ThreadOutput::output("  Node ");
        ThreadOutput::output(node);
        ThreadOutput::output(": degree=");
        ThreadOutput::output(degree);
        ThreadOutput::output(", send=");
        ThreadOutput::output(sendData[node] / (1024.0 * 1024.0));
        ThreadOutput::output(" MB, recv=");
        ThreadOutput::output(recvData[node] / (1024.0 * 1024.0));
        ThreadOutput::output(" MB, bandwidth=");
        ThreadOutput::output(totalBandwidth);
        ThreadOutput::output(" GiB/s, time=");
        ThreadOutput::output(nodeTime);
        ThreadOutput::output(" us (");
        ThreadOutput::output(timeReason);
        ThreadOutput::output(")");
        ThreadOutput::output(std::endl);

        if (nodeTime > maxTime) {
            maxTime = nodeTime;
            bottleneckNode = node;
            bottleneckReason = timeReason;
        }
    }

    ThreadOutput::output("[Profiler] Degree-Based Lower Bound: ");
    ThreadOutput::output(maxTime);
    ThreadOutput::output(" us");
    if (bottleneckNode >= 0) {
        ThreadOutput::output(" (bottleneck: node ");
        ThreadOutput::output(bottleneckNode);
        ThreadOutput::output(", reason: ");
        ThreadOutput::output(bottleneckReason);
        ThreadOutput::output(")");
    }
    ThreadOutput::output(std::endl);

    return maxTime;
}

Profiler::Time Profiler::computeBisectionBasedLowerBound(const std::shared_ptr<Topology>& topology,
                                                           const std::shared_ptr<Collective>& collective,
                                                           Profiler::ChunkSize chunkSize) noexcept {
    ThreadOutput::output("[Profiler] Computing Bisection-Based Lower Bound...");
    ThreadOutput::output(std::endl);
    
    const int npusCount = topology->npusCount();
    if (npusCount <= 1) {
        return 0.0;
    }

    // Try multiple bisection cuts and take the most restrictive one
    Profiler::Time maxBisectionTime = 0.0;
    int bestCutIndex = -1;
    std::string bestCutType = "";
    
    int cutIndex = 0;
    
    // Strategy 1: Try multiple balanced partitions using BFS from different starting points
    ThreadOutput::output("  Strategy 1: BFS-based partitions");
    ThreadOutput::output(std::endl);
    for (int startNode = 0; startNode < std::min(20, npusCount); ++startNode) {
        auto [partitionA, cutEdges] = findBisectionCutFromNode(topology, startNode);
        
        if (cutEdges.empty()) {
            continue;
        }

        // Compute total bandwidth across the cut
        double cutBandwidth = 0.0;  // in GiB/s
        std::set<std::pair<NpuID, NpuID>> processedEdges;
        for (const auto& [src, dest] : cutEdges) {
            auto edge = std::make_pair(std::min(src, dest), std::max(src, dest));
            if (processedEdges.find(edge) != processedEdges.end()) {
                continue;
            }
            processedEdges.insert(edge);
            
            if (topology->connected(src, dest)) {
                cutBandwidth += topology->bandwidth(src, dest);
            }
            if (topology->connected(dest, src)) {
                cutBandwidth += topology->bandwidth(dest, src);
            }
        }

        if (cutBandwidth <= 0) {
            continue;
        }

        // Compute total data volume that must cross the cut
        double crossCutData = computeCrossCutDataVolume(collective, partitionA, chunkSize);

        // Convert bandwidth from GiB/s to bytes/us
        const double GiB_TO_BYTES_PER_US = 1073.741824;
        double cutBandwidthBytesPerUs = cutBandwidth * GiB_TO_BYTES_PER_US;

        // Time = data volume / bandwidth
        if (cutBandwidthBytesPerUs > 0) {
            Profiler::Time bisectionTime = crossCutData / cutBandwidthBytesPerUs;
            
            // Count nodes in each partition
            int nodesInA = 0;
            for (bool inA : partitionA) {
                if (inA) nodesInA++;
            }
            
            ThreadOutput::output("    Cut #");
            ThreadOutput::output(cutIndex++);
            ThreadOutput::output(" (BFS from node ");
            ThreadOutput::output(startNode);
            ThreadOutput::output("): partitionA=");
            ThreadOutput::output(nodesInA);
            ThreadOutput::output(", partitionB=");
            ThreadOutput::output(npusCount - nodesInA);
            ThreadOutput::output(", cutEdges=");
            ThreadOutput::output(cutEdges.size());
            ThreadOutput::output(", cutBandwidth=");
            ThreadOutput::output(cutBandwidth);
            ThreadOutput::output(" GiB/s, crossCutData=");
            ThreadOutput::output(crossCutData / (1024.0 * 1024.0));
            ThreadOutput::output(" MB, time=");
            ThreadOutput::output(bisectionTime);
            ThreadOutput::output(" us");
            ThreadOutput::output(std::endl);
            
            if (bisectionTime > maxBisectionTime) {
                maxBisectionTime = bisectionTime;
                bestCutIndex = cutIndex - 1;
                bestCutType = "BFS";
            }
        }
    }

    // Strategy 2: Try simple balanced partitions (first half, middle split, etc.)
    ThreadOutput::output("  Strategy 2: Simple balanced partitions");
    ThreadOutput::output(std::endl);
    for (int splitPos = 1; splitPos < npusCount; ++splitPos) {
        std::vector<bool> partitionA(npusCount, false);
        for (int i = 0; i < splitPos; ++i) {
            partitionA[i] = true;
        }
        
        std::vector<std::pair<NpuID, NpuID>> cutEdges;
        double cutBandwidth = 0.0;
        std::set<std::pair<NpuID, NpuID>> processedEdges;
        
        for (int src = 0; src < npusCount; ++src) {
            for (int dest = 0; dest < npusCount; ++dest) {
                if (topology->connected(src, dest) && partitionA[src] != partitionA[dest]) {
                    auto edge = std::make_pair(std::min(src, dest), std::max(src, dest));
                    if (processedEdges.find(edge) == processedEdges.end()) {
                        processedEdges.insert(edge);
                        cutEdges.push_back({src, dest});
                        cutBandwidth += topology->bandwidth(src, dest);
                        if (topology->connected(dest, src)) {
                            cutBandwidth += topology->bandwidth(dest, src);
                        }
                    }
                }
            }
        }
        
        if (cutBandwidth > 0 && !cutEdges.empty()) {
            double crossCutData = computeCrossCutDataVolume(collective, partitionA, chunkSize);
            const double GiB_TO_BYTES_PER_US = 1073.741824;
            double cutBandwidthBytesPerUs = cutBandwidth * GiB_TO_BYTES_PER_US;
            if (cutBandwidthBytesPerUs > 0) {
                Profiler::Time bisectionTime = crossCutData / cutBandwidthBytesPerUs;
                
                ThreadOutput::output("    Cut #");
                ThreadOutput::output(cutIndex++);
                ThreadOutput::output(" (split at position ");
                ThreadOutput::output(splitPos);
                ThreadOutput::output("): partitionA=");
                ThreadOutput::output(splitPos);
                ThreadOutput::output(", partitionB=");
                ThreadOutput::output(npusCount - splitPos);
                ThreadOutput::output(", cutEdges=");
                ThreadOutput::output(cutEdges.size());
                ThreadOutput::output(", cutBandwidth=");
                ThreadOutput::output(cutBandwidth);
                ThreadOutput::output(" GiB/s, crossCutData=");
                ThreadOutput::output(crossCutData / (1024.0 * 1024.0));
                ThreadOutput::output(" MB, time=");
                ThreadOutput::output(bisectionTime);
                ThreadOutput::output(" us");
                ThreadOutput::output(std::endl);
                
                if (bisectionTime > maxBisectionTime) {
                    maxBisectionTime = bisectionTime;
                    bestCutIndex = cutIndex - 1;
                    bestCutType = "Split";
                }
            }
        }
    }

    ThreadOutput::output("[Profiler] Bisection-Based Lower Bound: ");
    ThreadOutput::output(maxBisectionTime);
    ThreadOutput::output(" us");
    if (bestCutIndex >= 0) {
        ThreadOutput::output(" (best cut: #");
        ThreadOutput::output(bestCutIndex);
        ThreadOutput::output(", type: ");
        ThreadOutput::output(bestCutType);
        ThreadOutput::output(")");
    }
    ThreadOutput::output(std::endl);

    return maxBisectionTime;
}

std::pair<std::vector<bool>, std::vector<std::pair<Profiler::NpuID, Profiler::NpuID>>>
Profiler::findBisectionCutFromNode(const std::shared_ptr<Topology>& topology, int startNode) noexcept {
    const int npusCount = topology->npusCount();
    if (npusCount <= 1) {
        return {{}, {}};
    }

    std::vector<bool> partitionA(npusCount, false);
    std::vector<bool> visited(npusCount, false);
    std::queue<int> q;

    // Start BFS from the given start node
    if (startNode >= npusCount) {
        startNode = 0;
    }
    q.push(startNode);
    visited[startNode] = true;
    partitionA[startNode] = true;
    int partitionSize = 1;
    int targetSize = npusCount / 2;

    while (!q.empty() && partitionSize < targetSize) {
        int node = q.front();
        q.pop();

        for (int neighbor = 0; neighbor < npusCount; ++neighbor) {
            if (topology->connected(node, neighbor) && !visited[neighbor]) {
                visited[neighbor] = true;
                partitionA[neighbor] = true;
                partitionSize++;
                q.push(neighbor);
                if (partitionSize >= targetSize) {
                    break;
                }
            }
        }
    }

    std::vector<std::pair<NpuID, NpuID>> cutEdges;
    for (int src = 0; src < npusCount; ++src) {
        for (int dest = 0; dest < npusCount; ++dest) {
            if (topology->connected(src, dest) && partitionA[src] != partitionA[dest]) {
                cutEdges.push_back({src, dest});
            }
        }
    }

    return {partitionA, cutEdges};
}

double Profiler::computeCrossCutDataVolume(const std::shared_ptr<Collective>& collective,
                                            const std::vector<bool>& partitionA,
                                            Profiler::ChunkSize chunkSize) noexcept {
    const auto precondition = collective->getPrecondition();
    const auto postcondition = collective->getPostcondition();

    // For AllGather-like collectives: each source node sends the same chunk to multiple destinations
    // We need to count how many unique chunks (from different sources) need to cross the cut
    // For AlltoAll-like collectives: each source-destination pair has a unique chunk
    
    // Strategy: For each source node in partition A, count how many destinations in partition B
    // For each source node in partition B, count how many destinations in partition A
    // But we need to be smart: if a chunk from src in A needs to go to multiple nodes in B,
    // we only need to send it once across the cut (then it can be forwarded within B)
    
    double totalData = 0.0;
    
    // Count nodes in each partition
    int nodesInA = 0;
    int nodesInB = 0;
    for (int i = 0; i < static_cast<int>(partitionA.size()); ++i) {
        if (partitionA[i]) {
            nodesInA++;
        } else {
            nodesInB++;
        }
    }
    
    // For each source node, determine if its chunks need to cross the cut
    // and how many unique chunks need to cross
    std::set<Collective::ChunkID> chunksFromAToB;  // Unique chunks that need to go from A to B
    std::set<Collective::ChunkID> chunksFromBToA;  // Unique chunks that need to go from B to A
    
    for (const auto& [chunk, src] : precondition) {
        auto it = postcondition.find(chunk);
        if (it == postcondition.end()) {
            continue;
        }
        const auto& dests = it->second;
        bool srcInA = partitionA[src];
        
        // Check if this chunk needs to cross the cut
        bool needsCrossCut = false;
        if (srcInA) {
            // Source in A: check if any destination is in B
            for (NpuID dest : dests) {
                if (dest != src && !partitionA[dest]) {  // dest in B and not self
                    needsCrossCut = true;
                    break;
                }
            }
            if (needsCrossCut) {
                chunksFromAToB.insert(chunk);
            }
        } else {
            // Source in B: check if any destination is in A
            for (NpuID dest : dests) {
                if (dest != src && partitionA[dest]) {  // dest in A and not self
                    needsCrossCut = true;
                    break;
                }
            }
            if (needsCrossCut) {
                chunksFromBToA.insert(chunk);
            }
        }
    }
    
    // Total data = number of unique chunks that need to cross * chunk size
    totalData = (chunksFromAToB.size() + chunksFromBToA.size()) * chunkSize;
    
    return totalData;
}

