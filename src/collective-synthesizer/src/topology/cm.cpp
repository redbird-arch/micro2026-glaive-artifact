/*
# File name  :    cm.cpp
# Author     :    Galois
# Time       :    2026/03/02
*/

#include <cassert>
#include <numeric>
#include <string>
#include <tacos/topology/cm.h>

using namespace tacos;

CM::CM(const std::vector<int>& shape,
       const std::vector<int>& switchShape,
       const std::vector<int>& linkCount,
       double directBandwidth,
       double directLatency,
       const std::vector<double>& latency,
       const std::vector<double>& bandwidth) noexcept
    : shape(shape), switchShape(switchShape), linkCount(linkCount) {

    assert(shape.size() >= 2);
    assert(switchShape.size() == 2u);
    assert(linkCount.size() == 2u);
    assert(latency.size() == 2u);
    assert(bandwidth.size() == 2u);

    // Shape is [gpusPerNode, numNodes] (lower hierarchy first)
    int gpusPerNode = shape[0];
    int numNodes = shape[1];
    int numGPUs = gpusPerNode * numNodes;

    int switchesPerNode = switchShape[0];  // also number of rails
    int switchesPerRail = switchShape[1];

    int numNodeSwitches = numNodes * switchesPerNode;
    int numRailSwitches = switchesPerNode * switchesPerRail;
    int totalSwitches = numNodeSwitches + numRailSwitches;
    int totalNodes = numGPUs + totalSwitches;

    setNpusCount(totalNodes);

    int switchBaseID = numGPUs;
    int nodeSwitchBaseID = switchBaseID;
    int railSwitchBaseID = switchBaseID + numNodeSwitches;

    // ========== Level 0: Direct links between odd-even adjacent cards within each node ==========
    for (int nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
        int nodeGPUBase = nodeIdx * gpusPerNode;
        for (int gpuOffset = 0; gpuOffset + 1 < gpusPerNode; gpuOffset += 2) {
            int gpuID0 = nodeGPUBase + gpuOffset;
            int gpuID1 = nodeGPUBase + gpuOffset + 1;
            connect(gpuID0, gpuID1, directLatency, directBandwidth, true);
        }
    }

    // ========== Level 1: Each GPU connects to each node switch in its node ==========
    double tier1Latency = latency[0];
    double tier1Bandwidth = bandwidth[0];
    int tier1LinksPerSwitch = linkCount[0] / switchesPerNode;

    for (int nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
        int nodeGPUBase = nodeIdx * gpusPerNode;
        int nodeSwitchBase = nodeSwitchBaseID + nodeIdx * switchesPerNode;
        for (int gpuOffset = 0; gpuOffset < gpusPerNode; ++gpuOffset) {
            int gpuID = nodeGPUBase + gpuOffset;
            for (int sw = 0; sw < switchesPerNode; ++sw) {
                int switchID = nodeSwitchBase + sw;
                for (int link = 0; link < tier1LinksPerSwitch; ++link) {
                    connect(gpuID, switchID, tier1Latency, tier1Bandwidth, true);
                }
            }
        }
    }

    // ========== Level 2: Node switches connect to rail switches (non-blocking) ==========
    // Each node switch (nodeIdx, railIdx) connects to all switchesPerRail rail switches of that rail.
    double tier2Latency = latency[1];
    double tier2Bandwidth = bandwidth[1];
    int tier2LinksPerRailSwitch = linkCount[1] / switchesPerRail;

    for (int nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
        for (int railIdx = 0; railIdx < switchesPerNode; ++railIdx) {
            int nodeSwitchID = nodeSwitchBaseID + nodeIdx * switchesPerNode + railIdx;
            int railSwitchBase = railSwitchBaseID + railIdx * switchesPerRail;
            for (int s = 0; s < switchesPerRail; ++s) {
                int railSwitchID = railSwitchBase + s;
                for (int link = 0; link < tier2LinksPerRailSwitch; ++link) {
                    connect(nodeSwitchID, railSwitchID, tier2Latency, tier2Bandwidth, true);
                }
            }
        }
    }
}

// Used in baseline/solver logs so path and hop lines show e.g. node_switch(node=0,sw=0) instead of raw switch ID.
std::string CM::formatNodeName(CM::NpuID id) const noexcept {
    int gpusPerNode = shape[0];
    int numNodes = shape[1];
    int numGPUs = gpusPerNode * numNodes;
    int switchesPerNode = switchShape[0];
    int switchesPerRail = switchShape[1];
    int numNodeSwitches = numNodes * switchesPerNode;
    int numRailSwitches = switchesPerNode * switchesPerRail;
    if (id < numGPUs)
        return std::to_string(id);
    if (id < numGPUs + numNodeSwitches) {
        int local = id - numGPUs;
        int nodeIdx = local / switchesPerNode;
        int sw = local % switchesPerNode;
        return "node_switch(node=" + std::to_string(nodeIdx) + ",sw=" + std::to_string(sw) + ")";
    }
    int local = id - numGPUs - numNodeSwitches;
    int railIdx = local / switchesPerRail;
    int sw = local % switchesPerRail;
    return "rail_switch(rail=" + std::to_string(railIdx) + ",sw=" + std::to_string(sw) + ")";
}
