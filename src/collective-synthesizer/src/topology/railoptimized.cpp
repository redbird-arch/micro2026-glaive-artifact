/*
# File name  :    railoptimized.cpp
# Author     :    Galois
# Time       :    2026/01/22 16:20:51
*/


#include <cassert>
#include <iostream>
#include <numeric>
#include <string>
#include <tacos/topology/railoptimized.h>

using namespace tacos;

RailOptimized::RailOptimized(const std::vector<int>& shape,
                             int switchDimension,
                             const std::vector<int>& switchShape,
                             const std::vector<int>& linkCount,
                             const std::vector<double>& latency,
                             const std::vector<double>& bandwidth) noexcept
    : shape(shape), switchDimension(switchDimension), switchShape(switchShape), linkCount(linkCount) {

    // Validate parameters
    assert(switchDimension == switchShape.size());
    assert(switchDimension == linkCount.size());
    assert(switchDimension == latency.size());
    assert(switchDimension == bandwidth.size());
    assert(switchDimension >= 1);

    // Compute total number of GPU nodes
    int numGPUs = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    
    // Shape is [gpusPerNode, numNodes] (lower hierarchy first)
    int gpusPerNode = shape[0];
    int numNodes = (shape.size() >= 2) ? shape[1] : 1;
    assert(numGPUs == gpusPerNode * numNodes);
    
    // Compute total number of switches
    // First dimension: scale-up switches (per node)
    int scaleUpSwitchesPerNode = switchShape[0];
    int totalScaleUpSwitches = numNodes * scaleUpSwitchesPerNode;
    
    // Remaining dimensions: scale-out switches (global)
    int totalScaleOutSwitches = 0;
    for (int d = 1; d < switchDimension; ++d) {
        totalScaleOutSwitches += switchShape[d];
    }
    
    int totalSwitches = totalScaleUpSwitches + totalScaleOutSwitches;
    int totalNodes = numGPUs + totalSwitches;
    
    // Set total number of nodes (GPUs + switches)
    setNpusCount(totalNodes);
    
    // Node ID mapping:
    // GPUs: 0 to numGPUs - 1
    // Switches: numGPUs to numGPUs + totalSwitches - 1
    
    int switchBaseID = numGPUs;
    
    // ========== Scale-up network (dimension 0) ==========
    // Each GPU connects to scale-up switches in its node
    // Same as fat-tree: each GPU connects to all scale-up switches in its node
    int scaleUpSwitchBaseID = switchBaseID;
    int scaleUpLatency = latency[0];
    double scaleUpBandwidth = bandwidth[0];
    int scaleUpLinksPerGPU = linkCount[0];
    int scaleUpLinksPerSwitch = scaleUpLinksPerGPU / scaleUpSwitchesPerNode;
    
    for (int nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
        int nodeGPUBase = nodeIdx * gpusPerNode;
        int nodeSwitchBase = scaleUpSwitchBaseID + nodeIdx * scaleUpSwitchesPerNode;
        
        // Connect each GPU in this node to each scale-up switch in this node
        for (int gpuOffset = 0; gpuOffset < gpusPerNode; ++gpuOffset) {
            int gpuID = nodeGPUBase + gpuOffset;
            for (int switchOffset = 0; switchOffset < scaleUpSwitchesPerNode; ++switchOffset) {
                int switchID = nodeSwitchBase + switchOffset;
                // Connect with multiple links per GPU-switch pair
                for (int link = 0; link < scaleUpLinksPerSwitch; ++link) {
                    connect(gpuID, switchID, scaleUpLatency, scaleUpBandwidth, true);
                }
            }
        }
    }
    
    // ========== Scale-out network (dimensions 1 and above) ==========
    if (switchDimension > 1) {
        // First layer of scale-out switches
        int scaleOutLayer1BaseID = switchBaseID + totalScaleUpSwitches;
        int scaleOutLayer1Count = switchShape[1];
        int scaleOutLayer1Latency = latency[1];
        double scaleOutLayer1Bandwidth = bandwidth[1];
        int scaleOutLayer1LinksPerGPU = linkCount[1];
        
        // In rail-optimized, each first-layer switch connects only to GPUs of the same rail
        // Rail is determined by the GPU's position within its node (gpuOffset)
        // Each rail has one first-layer switch
        for (int nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
            int nodeGPUBase = nodeIdx * gpusPerNode;
            for (int gpuOffset = 0; gpuOffset < gpusPerNode; ++gpuOffset) {
                int gpuID = nodeGPUBase + gpuOffset;
                // Each GPU connects to the switch corresponding to its rail (gpuOffset)
                int targetSwitch = scaleOutLayer1BaseID + gpuOffset;
                assert(targetSwitch < scaleOutLayer1BaseID + scaleOutLayer1Count);
                for (int link = 0; link < scaleOutLayer1LinksPerGPU; ++link) {
                    connect(gpuID, targetSwitch, scaleOutLayer1Latency, scaleOutLayer1Bandwidth, true);
                }
            }
        }
        
        // Connect scale-out layers: each layer N switch connects to all layer N+1 switches
        // Same as fat-tree: full connectivity between adjacent layers
        int currentLayerBaseID = scaleOutLayer1BaseID;
        int currentLayerCount = scaleOutLayer1Count;
        
        for (int layer = 2; layer < switchDimension; ++layer) {
            int nextLayerBaseID = currentLayerBaseID + currentLayerCount;
            int nextLayerCount = switchShape[layer];
            int layerLatency = latency[layer];
            double layerBandwidth = bandwidth[layer];
            int layerLinksPerSwitch = linkCount[layer];
            
            // Each switch in current layer connects to all switches in next layer
            for (int currentSwitch = 0; currentSwitch < currentLayerCount; ++currentSwitch) {
                int currentSwitchID = currentLayerBaseID + currentSwitch;
                for (int nextSwitch = 0; nextSwitch < nextLayerCount; ++nextSwitch) {
                    int nextSwitchID = nextLayerBaseID + nextSwitch;
                    for (int link = 0; link < layerLinksPerSwitch; ++link) {
                        connect(currentSwitchID, nextSwitchID, layerLatency, layerBandwidth, true);
                    }
                }
            }
            
            currentLayerBaseID = nextLayerBaseID;
            currentLayerCount = nextLayerCount;
        }
    }
}

std::string RailOptimized::formatNodeName(RailOptimized::NpuID id) const noexcept {
    int numGPUs = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    int numNodes = (shape.size() >= 2) ? shape[1] : 1;
    int scaleUpPerNode = switchShape[0];
    int totalScaleUpSwitches = numNodes * scaleUpPerNode;
    if (id < numGPUs)
        return std::to_string(id);
    if (id < numGPUs + totalScaleUpSwitches)
        return "scale_up_switch(" + std::to_string(id - numGPUs) + ")";
    return "scale_out_switch(" + std::to_string(id - numGPUs - totalScaleUpSwitches) + ")";
}

