/*
# File name  :    main.cpp
# Author     :    Galois
# Time       :    2025/09/24 21:32:50
*/

#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <filesystem>
#include <algorithm>
#include <sstream>
#include <numeric>
#include <set>
#include <cctype>
#include <cstring>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <nlohmann/json.hpp>
#include <tacos/collective/all_gather.h>
#include <tacos/collective/all_to_all.h>
#include <tacos/collective/all_to_all_v.h>
#include <tacos/collective/gather.h>
#include <tacos/event_queue/timer.h>
#include <tacos/synthesizer/synthesizer.h>
#include <tacos/topology/mesh.h>
#include <tacos/topology/torus.h>
#include <tacos/topology/hypercube.h>
#include <tacos/topology/fullmesh.h>
#include <tacos/topology/fattree.h>
#include <tacos/topology/railoptimized.h>
#include <tacos/topology/cm.h>
#include <tacos/baselines/bruck.h>
#include <tacos/baselines/spreadout.h>
#include <tacos/baselines/pairwise.h>
#include <tacos/baselines/biring.h>
#include <tacos/baselines/others.h>
#include "thread_output.h"
#include "synthesizer_2/synthesizer_2.h"
#include "synthesizer_3/synthesizer_3.h"
#include "synthesizer_4/synthesizer_4.h"
#include "synthesizer_standard/standard_synthesizer.h"

using namespace tacos;
using json = nlohmann::json;

// Structure to store link fault information
struct LinkFaultInfo {
    std::string coordStr;  // Original coordinate string from JSON, e.g., "[1,0]-[1,1]"
    int npu1;              // First node NpuID
    int npu2;              // Second node NpuID
};

// Structure to store node fault information
struct NodeFaultInfo {
    std::string coordStr;  // Original coordinate string from JSON, e.g., "[0,2]"
    int npuID;             // Converted NpuID
    double bandwidth;      // Custom bandwidth for this node
    double latency;        // Custom latency for this node
};

// Helper function to convert coordinate string like "[1,0]" to NpuID
// When useHierarchyOrder is true (fat-tree, rail-optimized, cm): shape is [gpusPerNode, numNodes],
//   coord is [gpuOffset, nodeIdx], npuID = nodeIdx * gpusPerNode + gpuOffset (lower hierarchy first).
// When useHierarchyOrder is false (mesh, torus, etc.): row-major,
//   [i, j, k, ...] maps to i*shape[1]*shape[2]*... + j*shape[2]*... + k*... + ...
int coordinateToNpuID(const std::vector<int>& shape, const std::vector<int>& coord, bool useHierarchyOrder = false) {
    if (coord.size() != shape.size()) {
        throw std::invalid_argument("Coordinate dimension mismatch with shape");
    }
    for (size_t d = 0; d < shape.size(); ++d) {
        if (coord[d] < 0 || coord[d] >= shape[d]) {
            throw std::invalid_argument("Coordinate out of bounds");
        }
    }
    if (useHierarchyOrder && shape.size() == 2) {
        // coord[0] = gpuOffset (lower hierarchy), coord[1] = nodeIdx (higher hierarchy)
        return coord[1] * shape[0] + coord[0];
    }
    // Row-major order
    int npuID = 0;
    int stride = 1;
    for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
        npuID += coord[d] * stride;
        stride *= shape[d];
    }
    return npuID;
}

// Helper function to parse coordinate string like "[1,0]" to vector<int>
std::vector<int> parseCoordinate(const std::string& coordStr) {
    std::vector<int> coord;
    std::string cleaned = coordStr;
    // Remove brackets and whitespace
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '['), cleaned.end());
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), ']'), cleaned.end());
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), ' '), cleaned.end());
    
    std::istringstream iss(cleaned);
    std::string token;
    while (std::getline(iss, token, ',')) {
        if (!token.empty()) {
            coord.push_back(std::stoi(token));
        }
    }
    return coord;
}

// Helper function to parse link fault string like "[1,0]-[1,1]"
std::pair<std::vector<int>, std::vector<int>> parseLinkFault(const std::string& linkStr) {
    // Find the separator (could be "-" or other)
    size_t sepPos = linkStr.find('-');
    if (sepPos == std::string::npos) {
        throw std::invalid_argument("Invalid link fault format: missing separator");
    }
    
    std::string coord1Str = linkStr.substr(0, sepPos);
    std::string coord2Str = linkStr.substr(sepPos + 1);
    
    auto coord1 = parseCoordinate(coord1Str);
    auto coord2 = parseCoordinate(coord2Str);
    
    return {coord1, coord2};
}

// Apply link faults to topology and return link fault information
std::vector<LinkFaultInfo> applyLinkFaults(std::shared_ptr<Topology> topology, 
                                           const json& config,
                                           const std::vector<int>& shape,
                                           bool shapeHierarchyOrder = false) {
    std::vector<LinkFaultInfo> linkFaultInfos;
    
    if (!config.contains("link_fault")) {
        return linkFaultInfos; // No link faults specified
    }
    
    const auto& linkFaults = config["link_fault"];
    if (!linkFaults.is_array()) {
        throw std::invalid_argument("link_fault must be an array");
    }
    
    for (const auto& linkFault : linkFaults) {
        if (!linkFault.is_string()) {
            throw std::invalid_argument("Each link_fault entry must be a string");
        }
        
        std::string linkStr = linkFault.get<std::string>();
        auto [coord1, coord2] = parseLinkFault(linkStr);
        
        // Validate coordinates
        if (coord1.size() != shape.size() || coord2.size() != shape.size()) {
            throw std::invalid_argument("Link fault coordinate dimension mismatch: " + linkStr);
        }
        
        for (size_t d = 0; d < shape.size(); ++d) {
            if (coord1[d] < 0 || coord1[d] >= shape[d] || 
                coord2[d] < 0 || coord2[d] >= shape[d]) {
                throw std::invalid_argument("Link fault coordinate out of bounds: " + linkStr);
            }
        }
        
        // Convert to NpuID
        int npu1 = coordinateToNpuID(shape, coord1, shapeHierarchyOrder);
        int npu2 = coordinateToNpuID(shape, coord2, shapeHierarchyOrder);
        
        // Validate that the link exists
        if (!topology->connected(npu1, npu2) && !topology->connected(npu2, npu1)) {
            throw std::invalid_argument("Link fault specified for non-existent link: " + linkStr);
        }
        
        // Store link fault information
        linkFaultInfos.push_back({linkStr, npu1, npu2});
        
        // Disconnect the link (bidirectional)
        topology->disconnect(npu1, npu2, true);
    }
    
    return linkFaultInfos;
}

// Helper function to format coordinate vector to string like "[0,2]"
std::string formatCoordinate(const std::vector<int>& coord) {
    std::string result = "[";
    for (size_t i = 0; i < coord.size(); ++i) {
        result += std::to_string(coord[i]);
        if (i + 1 < coord.size()) {
            result += ",";
        }
    }
    result += "]";
    return result;
}

// Apply node faults to topology and return node fault information
std::vector<NodeFaultInfo> applyNodeFaults(std::shared_ptr<Topology> topology,
                                            const json& config,
                                            const std::vector<int>& shape,
                                            bool shapeHierarchyOrder = false) {
    std::vector<NodeFaultInfo> nodeFaultInfos;
    
    if (!config.contains("node_fault")) {
        return nodeFaultInfos; // No node faults specified
    }
    
    const auto& nodeFaults = config["node_fault"];
    if (!nodeFaults.is_array()) {
        throw std::invalid_argument("node_fault must be an array");
    }
    
    for (const auto& nodeFault : nodeFaults) {
        if (!nodeFault.is_array()) {
            throw std::invalid_argument("Each node_fault entry must be an array");
        }
        
        auto coord = nodeFault.get<std::vector<int>>();
        
        // Validate coordinate dimension
        if (coord.size() != shape.size()) {
            throw std::invalid_argument("Node fault coordinate dimension mismatch: " + formatCoordinate(coord));
        }
        
        // Validate coordinate bounds
        for (size_t d = 0; d < shape.size(); ++d) {
            if (coord[d] < 0 || coord[d] >= shape[d]) {
                throw std::invalid_argument("Node fault coordinate out of bounds: " + formatCoordinate(coord));
            }
        }
        
        // Convert to NpuID
        int npuID = coordinateToNpuID(shape, coord, shapeHierarchyOrder);
        
        // Validate that the node exists
        if (npuID < 0 || npuID >= topology->npusCount()) {
            throw std::invalid_argument("Node fault specified for non-existent node: " + formatCoordinate(coord));
        }
        
        // Store node fault information
        nodeFaultInfos.push_back({formatCoordinate(coord), npuID});
        
        // Disconnect all links connected to this node
        for (int other = 0; other < topology->npusCount(); ++other) {
            if (other != npuID) {
                // Disconnect both directions
                if (topology->connected(npuID, other)) {
                    topology->disconnect(npuID, other, true);
                }
            }
        }
    }
    
    return nodeFaultInfos;
}

// Structure to store straggler information
struct StragglerInfo {
    std::string coordStr;  // Original coordinate string from JSON, e.g., "[0,1]"
    int npuID;             // Converted NpuID
    double bandwidth;      // Custom bandwidth for this node
    double latency;        // Custom latency for this node
};

struct DemandStats {
    int rows = 0;
    int cols = 0;
    std::size_t nonzeros = 0;
    long long totalBytes = 0;
    std::uint64_t fingerprint = 0;
};

namespace {
constexpr std::uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void fnvMixByte(std::uint64_t& hash, std::uint8_t byte) {
    hash ^= byte;
    hash *= kFnvPrime;
}

void fnvMixUint64(std::uint64_t& hash, std::uint64_t value) {
    for (int byte = 0; byte < 8; ++byte) {
        fnvMixByte(hash, static_cast<std::uint8_t>((value >> (byte * 8)) & 0xffU));
    }
}

void fnvMixInt64(std::uint64_t& hash, long long value) {
    fnvMixUint64(hash, static_cast<std::uint64_t>(value));
}

struct DemandFingerprintEntry {
    int src;
    int dst;
    long long bytes;
};

DemandStats finalizeDemandStats(int rows, int cols, std::vector<DemandFingerprintEntry> entries) {
    std::sort(entries.begin(), entries.end(),
              [](const DemandFingerprintEntry& lhs, const DemandFingerprintEntry& rhs) {
                  if (lhs.src != rhs.src) {
                      return lhs.src < rhs.src;
                  }
                  if (lhs.dst != rhs.dst) {
                      return lhs.dst < rhs.dst;
                  }
                  return lhs.bytes < rhs.bytes;
              });

    DemandStats stats;
    stats.rows = rows;
    stats.cols = cols;
    stats.nonzeros = entries.size();
    stats.fingerprint = kFnvOffsetBasis;
    fnvMixUint64(stats.fingerprint, static_cast<std::uint64_t>(rows));
    fnvMixUint64(stats.fingerprint, static_cast<std::uint64_t>(cols));
    fnvMixUint64(stats.fingerprint, static_cast<std::uint64_t>(entries.size()));

    for (const auto& entry : entries) {
        stats.totalBytes += entry.bytes;
        fnvMixUint64(stats.fingerprint, static_cast<std::uint64_t>(entry.src));
        fnvMixUint64(stats.fingerprint, static_cast<std::uint64_t>(entry.dst));
        fnvMixInt64(stats.fingerprint, entry.bytes);
    }
    fnvMixInt64(stats.fingerprint, stats.totalBytes);
    return stats;
}

DemandStats computeDenseDemandStats(const std::vector<std::vector<long long>>& demand) {
    std::vector<DemandFingerprintEntry> entries;
    std::size_t reserveCount = 0;
    for (const auto& row : demand) {
        reserveCount += static_cast<std::size_t>(
            std::count_if(row.begin(), row.end(), [](long long value) { return value > 0; }));
    }
    entries.reserve(reserveCount);

    int cols = demand.empty() ? 0 : static_cast<int>(demand.front().size());
    for (int src = 0; src < static_cast<int>(demand.size()); ++src) {
        cols = std::max(cols, static_cast<int>(demand[src].size()));
        for (int dst = 0; dst < static_cast<int>(demand[src].size()); ++dst) {
            if (src == dst || demand[src][dst] <= 0) {
                continue;
            }
            entries.push_back({src, dst, demand[src][dst]});
        }
    }
    return finalizeDemandStats(static_cast<int>(demand.size()), cols, std::move(entries));
}

DemandStats computeSparseDemandStats(const std::vector<DemandEntry3>& demand, int expectedNpus) {
    std::vector<DemandFingerprintEntry> entries;
    entries.reserve(demand.size());
    for (const auto& flow : demand) {
        if (flow.src == flow.dst || flow.bytes <= 0) {
            continue;
        }
        entries.push_back({flow.src, flow.dst, flow.bytes});
    }
    return finalizeDemandStats(expectedNpus, expectedNpus, std::move(entries));
}

void printDemandStats(const DemandStats& stats) {
    std::cout << "\n[Demand Summary]" << std::endl;
    std::cout << "  Demand Rows: " << stats.rows << std::endl;
    std::cout << "  Demand Cols: " << stats.cols << std::endl;
    std::cout << "  Demand Nonzeros: " << stats.nonzeros << std::endl;
    std::cout << "  Demand Total Bytes: " << stats.totalBytes << std::endl;
    std::cout << "  Demand Fingerprint: 0x" << std::hex << std::setw(16)
              << std::setfill('0') << stats.fingerprint << std::dec
              << std::setfill(' ') << std::endl;
}

struct HotColdTimeBreakdown {
    double hotOnlyUs = 0.0;
    double coldOnlyUs = 0.0;
    double overlapUs = 0.0;
    double activeUs = 0.0;
};

HotColdTimeBreakdown computeHotColdTimeBreakdown(const std::vector<StandardEvent>& events) {
    HotColdTimeBreakdown breakdown;
    if (events.empty()) {
        return breakdown;
    }

    std::vector<double> cuts;
    cuts.reserve(events.size() * 2);
    for (const auto& event : events) {
        if (event.endTime <= event.startTime) {
            continue;
        }
        cuts.push_back(event.startTime);
        cuts.push_back(event.endTime);
    }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end(),
                           [](double lhs, double rhs) {
                               return std::abs(lhs - rhs) <= 1e-9;
                           }),
               cuts.end());

    for (std::size_t idx = 0; idx + 1 < cuts.size(); ++idx) {
        const double start = cuts[idx];
        const double end = cuts[idx + 1];
        if (end <= start) {
            continue;
        }

        bool hasHot = false;
        bool hasCold = false;
        for (const auto& event : events) {
            if (event.startTime < end - 1e-9 && event.endTime > start + 1e-9) {
                if (event.isLatencyMatrix) {
                    hasCold = true;
                } else {
                    hasHot = true;
                }
            }
            if (hasHot && hasCold) {
                break;
            }
        }

        const double duration = end - start;
        if (hasHot || hasCold) {
            breakdown.activeUs += duration;
        }
        if (hasHot && hasCold) {
            breakdown.overlapUs += duration;
        } else if (hasHot) {
            breakdown.hotOnlyUs += duration;
        } else if (hasCold) {
            breakdown.coldOnlyUs += duration;
        }
    }
    return breakdown;
}

void printSpeedEvents(const std::vector<StandardEvent>& events) {
    for (const auto& event : events) {
        std::cout << "[Speed Event] class="
                  << (event.isLatencyMatrix ? "cold" : "hot")
                  << " link_src=" << event.src
                  << " link_dst=" << event.dst
                  << " flow_src=" << event.flowSrc
                  << " flow_dst=" << event.flowDst
                  << " chunk_id=" << event.chunkId
                  << " bytes=" << event.bytes
                  << std::fixed << std::setprecision(6)
                  << " start_us=" << event.startTime
                  << " end_us=" << event.endTime
                  << " path=";
        for (std::size_t idx = 0; idx < event.path.size(); ++idx) {
            if (idx > 0) {
                std::cout << ">";
            }
            std::cout << event.path[idx];
        }
        std::cout << std::endl;
    }
}
} // namespace

// Apply stragglers to topology (modify link properties for specific nodes)
std::vector<StragglerInfo> applyStragglers(std::shared_ptr<Topology> topology,
                                           const json& config,
                                           const std::vector<int>& shape,
                                           bool shapeHierarchyOrder = false) {
    std::vector<StragglerInfo> stragglerInfos;
    
    if (!config.contains("straggler")) {
        return stragglerInfos; // No stragglers specified
    }
    
    const auto& stragglers = config["straggler"];
    if (!stragglers.is_object()) {
        throw std::invalid_argument("straggler must be an object");
    }
    
    for (const auto& [key, value] : stragglers.items()) {
        // Parse coordinate from key (format: "[0,1]")
        auto coord = parseCoordinate(key);
        
        // Validate coordinate dimension
        if (coord.size() != shape.size()) {
            throw std::invalid_argument("Straggler coordinate dimension mismatch: " + key);
        }
        
        // Validate coordinate bounds
        for (size_t d = 0; d < shape.size(); ++d) {
            if (coord[d] < 0 || coord[d] >= shape[d]) {
                throw std::invalid_argument("Straggler coordinate out of bounds: " + key);
            }
        }
        
        // Convert to NpuID
        int npuID = coordinateToNpuID(shape, coord, shapeHierarchyOrder);
        
        // Validate that the node exists
        if (npuID < 0 || npuID >= topology->npusCount()) {
            throw std::invalid_argument("Straggler specified for non-existent node: " + key);
        }
        
        // Parse bandwidth and latency from value (format: [450, 100])
        if (!value.is_array() || value.size() != 2) {
            throw std::invalid_argument("Straggler value must be an array of [bandwidth, latency]: " + key);
        }
        
        double bandwidth = value[0].get<double>();
        double latency = value[1].get<double>();
        
        if (bandwidth <= 0) {
            throw std::invalid_argument("Straggler bandwidth must be positive: " + key);
        }
        if (latency < 0) {
            throw std::invalid_argument("Straggler latency must be non-negative: " + key);
        }
        
        // Store straggler information
        stragglerInfos.push_back({formatCoordinate(coord), npuID, bandwidth, latency});
        
        // Update all links connected to this node
        for (int other = 0; other < topology->npusCount(); ++other) {
            if (other != npuID) {
                // Update both directions if link exists
                if (topology->connected(npuID, other)) {
                    topology->updateLinkProperties(npuID, other, bandwidth, latency, true);
                }
            }
        }
    }
    
    return stragglerInfos;
}

// Create topology instance from JSON config and return fault information
struct TopologyFaultInfo {
    std::vector<LinkFaultInfo> linkFaultInfos;
    std::vector<NodeFaultInfo> nodeFaultInfos;
    std::vector<StragglerInfo> stragglerInfos;
};

std::pair<std::shared_ptr<Topology>, TopologyFaultInfo> createTopologyWithFaults(const json& config) {
    std::string topologyType = config["topology"].get<std::string>();
    std::shared_ptr<Topology> topology;
    std::vector<int> shape;
    TopologyFaultInfo faultInfo;

    if (topologyType == "mesh") {
        shape = config["shape"].get<std::vector<int>>();
        auto latency = config["latency"].get<std::vector<double>>();
        auto bandwidth = config["bandwidth"].get<std::vector<double>>();
        topology = std::make_shared<Mesh>(shape, latency, bandwidth);
    } else if (topologyType == "torus") {
        shape = config["shape"].get<std::vector<int>>();
        auto latency = config["latency"].get<std::vector<double>>();
        auto bandwidth = config["bandwidth"].get<std::vector<double>>();
        topology = std::make_shared<Torus>(shape, latency, bandwidth);
    } else if (topologyType == "hypercube") {
        int dimension = config["dimension"].get<int>();
        auto latency = config["latency"].get<std::vector<double>>();
        auto bandwidth = config["bandwidth"].get<std::vector<double>>();
        topology = std::make_shared<Hypercube>(dimension, latency, bandwidth);
        // Hypercube doesn't have shape, so link_fault, node_fault, and straggler are not supported for it
        if (config.contains("link_fault") || config.contains("node_fault") || config.contains("straggler")) {
            throw std::invalid_argument("link_fault, node_fault, and straggler are not supported for hypercube topology");
        }
        return {topology, faultInfo};
    } else if (topologyType == "fullmesh") {
        shape = config["shape"].get<std::vector<int>>();
        auto latency = config["latency"].get<std::vector<double>>();
        auto bandwidth = config["bandwidth"].get<std::vector<double>>();
        topology = std::make_shared<FullMesh>(shape, latency, bandwidth);
    } else if (topologyType == "fat-tree") {
        shape = config["shape"].get<std::vector<int>>();
        int switchDimension = config["switch-dimension"].get<int>();
        auto switchShape = config["switch-shape"].get<std::vector<int>>();
        auto linkCount = config["link-count"].get<std::vector<int>>();
        auto latency = config["latency"].get<std::vector<double>>();
        auto bandwidth = config["bandwidth"].get<std::vector<double>>();
        topology = std::make_shared<FatTree>(shape, switchDimension, switchShape, linkCount, latency, bandwidth);
    } else if (topologyType == "rail-optimized") {
        shape = config["shape"].get<std::vector<int>>();
        int switchDimension = config["switch-dimension"].get<int>();
        auto switchShape = config["switch-shape"].get<std::vector<int>>();
        auto linkCount = config["link-count"].get<std::vector<int>>();
        auto latency = config["latency"].get<std::vector<double>>();
        auto bandwidth = config["bandwidth"].get<std::vector<double>>();
        topology = std::make_shared<RailOptimized>(shape, switchDimension, switchShape, linkCount, latency, bandwidth);
    } else if (topologyType == "cm" || topologyType == "cm384") {
        shape = config["shape"].get<std::vector<int>>();
        auto switchShape = config["switch-shape"].get<std::vector<int>>();
        auto linkCount = config["link-count"].get<std::vector<int>>();
        double directBandwidth = config["direct-bandwidth"].get<double>();
        double directLatency = config["direct-latency"].get<double>();
        auto latency = config["latency"].get<std::vector<double>>();
        auto bandwidth = config["bandwidth"].get<std::vector<double>>();
        topology = std::make_shared<CM>(shape, switchShape, linkCount, directBandwidth, directLatency, latency, bandwidth);
    } else {
        throw std::invalid_argument("Unsupported topology type: " + topologyType);
    }

    // For switch topologies (fat-tree, rail-optimized, cm), shape is [gpusPerNode, numNodes] (lower hierarchy first)
    bool shapeHierarchyOrder = (topologyType == "fat-tree" || topologyType == "rail-optimized" || topologyType == "cm" || topologyType == "cm384");
    
    // Apply link faults if specified (before node faults, so node faults can disconnect links)
    faultInfo.linkFaultInfos = applyLinkFaults(topology, config, shape, shapeHierarchyOrder);
    
    // Apply node faults if specified
    faultInfo.nodeFaultInfos = applyNodeFaults(topology, config, shape, shapeHierarchyOrder);
    
    // Apply stragglers if specified (after all faults, so stragglers can override link properties)
    faultInfo.stragglerInfos = applyStragglers(topology, config, shape, shapeHierarchyOrder);
    
    return {topology, faultInfo};
}

// Create topology instance from JSON config (backward compatibility)
std::shared_ptr<Topology> createTopology(const json& config) {
    auto [topology, _] = createTopologyWithFaults(config);
    return topology;
}

// Apply node faults to collective (remove faulty nodes from postconditions)
void applyNodeFaultsToCollective(std::shared_ptr<Collective> collective,
                                  const std::vector<NodeFaultInfo>& nodeFaultInfos) {
    for (const auto& nodeFault : nodeFaultInfos) {
        collective->removeFaultyNode(nodeFault.npuID);
    }
}

// Helper function to calculate GCD (Greatest Common Divisor)
long long calculateGCD(long long a, long long b) {
    while (b != 0) {
        long long temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Helper function to calculate GCD of multiple numbers
long long calculateGCDMultiple(const std::vector<long long>& numbers) {
    if (numbers.empty()) {
        return 1;
    }
    long long result = numbers[0];
    for (size_t i = 1; i < numbers.size(); ++i) {
        result = calculateGCD(result, numbers[i]);
    }
    return result;
}

long long getBlockBytes(const json& config) {
    if (config.contains("block_bytes")) {
        return config["block_bytes"].get<long long>();
    }
    return 4LL * 1024LL;
}

std::vector<std::vector<long long>> parseRawCSVMatrix(const std::string& csvPath) {
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + csvPath);
    }

    std::vector<std::vector<long long>> dataMatrix;
    std::string line;
    size_t expectedColumns = 0;
    int row = 0;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::vector<long long> rowData;
        std::istringstream iss(line);
        std::string cell;

        while (std::getline(iss, cell, ',')) {
            cell.erase(std::remove_if(cell.begin(), cell.end(), ::isspace), cell.end());
            if (!cell.empty()) {
                rowData.push_back(std::stoll(cell));
            }
        }

        if (rowData.empty()) {
            continue;
        }

        if (expectedColumns == 0) {
            expectedColumns = rowData.size();
        } else if (rowData.size() != expectedColumns) {
            throw std::invalid_argument(
                "CSV file row " + std::to_string(row) +
                " has " + std::to_string(rowData.size()) +
                " columns, expected " + std::to_string(expectedColumns));
        }

        dataMatrix.push_back(std::move(rowData));
        ++row;
    }

    if (dataMatrix.empty()) {
        throw std::invalid_argument("CSV file is empty: " + csvPath);
    }

    return dataMatrix;
}

// Parse CSV file for variable datasize
// Returns a matrix where dataMatrix[i][j] = data size from node i to node j (in bytes)
std::vector<std::vector<long long>> parseVariableDatasizeCSV(
    const std::string& csvPath,
    int expectedNpus,
    long long blockBytes) {
    auto dataMatrix = parseRawCSVMatrix(csvPath);

    for (size_t row = 0; row < dataMatrix.size(); ++row) {
        if (static_cast<int>(dataMatrix[row].size()) != expectedNpus) {
            throw std::invalid_argument("CSV file row " + std::to_string(row) +
                                      " has " + std::to_string(dataMatrix[row].size()) +
                                      " columns, expected " + std::to_string(expectedNpus));
        }
        for (auto& value : dataMatrix[row]) {
            value *= blockBytes;
        }
    }

    if (dataMatrix.size() != expectedNpus) {
        throw std::invalid_argument("CSV file has " + std::to_string(dataMatrix.size()) +
                                  " rows, expected " + std::to_string(expectedNpus));
    }

    return dataMatrix;
}

std::vector<int> npuIDToCoordinate(
    const std::vector<int>& shape,
    int npuID,
    bool useHierarchyOrder = false) {
    std::vector<int> coord(shape.size(), 0);
    if (useHierarchyOrder && shape.size() == 2) {
        coord[0] = npuID % shape[0];
        coord[1] = npuID / shape[0];
        return coord;
    }

    int remaining = npuID;
    for (size_t d = 0; d < shape.size(); ++d) {
        int stride = 1;
        for (size_t next = d + 1; next < shape.size(); ++next) {
            stride *= shape[next];
        }
        coord[d] = remaining / stride;
        remaining %= stride;
    }
    return coord;
}

std::vector<DemandEntry3> generateSyntheticDemandFlows(
    const json& spec,
    const std::vector<int>& shape,
    const std::string& topologyType,
    int expectedNpus,
    long long blockBytes) {
    if (!spec.is_object()) {
        throw std::invalid_argument("synthetic_v_datasize must be an object");
    }

    const std::string pattern = spec.value("pattern", "far-end");

    std::vector<DemandEntry3> flows;
    if (pattern == "far-end") {
        const long long bytesPerFlow = spec.at("bytes-per-flow").get<long long>();
        const int fanout = spec.value("fanout", 1);
        if (bytesPerFlow <= 0) {
            throw std::invalid_argument("synthetic_v_datasize.bytes-per-flow must be positive");
        }
        if (fanout <= 0) {
            throw std::invalid_argument("synthetic_v_datasize.fanout must be positive");
        }

        const bool isTorus = topologyType == "torus";
        const bool useHierarchyOrder =
            topologyType == "fat-tree" || topologyType == "rail-optimized" ||
            topologyType == "cm" || topologyType == "cm384";

        flows.reserve(static_cast<size_t>(expectedNpus) * fanout);

        for (int src = 0; src < expectedNpus; ++src) {
            const auto srcCoord = npuIDToCoordinate(shape, src, useHierarchyOrder);
            std::set<int> usedDestinations;

            for (int replica = 0; replica < fanout; ++replica) {
                auto dstCoord = srcCoord;

                if (isTorus) {
                    for (size_t d = 0; d < shape.size(); ++d) {
                        const int offset = std::max(1, shape[d] / 2);
                        dstCoord[d] = (srcCoord[d] + offset) % shape[d];
                    }
                    if (fanout > 1 && !shape.empty()) {
                        const size_t dim = static_cast<size_t>(replica % static_cast<int>(shape.size()));
                        dstCoord[dim] = (dstCoord[dim] + replica) % shape[dim];
                    }
                } else {
                    for (size_t d = 0; d < shape.size(); ++d) {
                        dstCoord[d] = shape[d] - 1 - srcCoord[d];
                    }
                    if (fanout > 1 && !shape.empty()) {
                        const size_t dim = static_cast<size_t>(replica % static_cast<int>(shape.size()));
                        if (shape[dim] > 1) {
                            dstCoord[dim] = (dstCoord[dim] + replica) % shape[dim];
                        }
                    }
                }

                int dst = coordinateToNpuID(shape, dstCoord, useHierarchyOrder);
                if (dst == src || usedDestinations.count(dst) > 0) {
                    bool reassigned = false;
                    for (size_t dim = 0; dim < shape.size() && !reassigned; ++dim) {
                        if (shape[dim] <= 1) {
                            continue;
                        }

                        auto fallbackCoord = srcCoord;
                        if (isTorus) {
                            fallbackCoord[dim] = (fallbackCoord[dim] + replica + 1) % shape[dim];
                        } else {
                            const int forwardCandidate = fallbackCoord[dim] + replica + 1;
                            if (forwardCandidate < shape[dim]) {
                                fallbackCoord[dim] = forwardCandidate;
                            } else {
                                fallbackCoord[dim] =
                                    (fallbackCoord[dim] > 0) ? fallbackCoord[dim] - 1 : shape[dim] - 1;
                            }
                        }

                        dst = coordinateToNpuID(shape, fallbackCoord, useHierarchyOrder);
                        if (dst != src && usedDestinations.count(dst) == 0) {
                            reassigned = true;
                        }
                    }
                    if (!reassigned && (dst == src || usedDestinations.count(dst) > 0)) {
                        continue;
                    }
                }

                usedDestinations.insert(dst);
                flows.push_back({src, dst, bytesPerFlow});
            }
        }
        return flows;
    }

    if (pattern == "tiled-csv") {
        const std::string csvPath = spec.at("base_csv").get<std::string>();
        auto baseMatrix = parseRawCSVMatrix(csvPath);
        const int baseRows = static_cast<int>(baseMatrix.size());
        const int baseCols = static_cast<int>(baseMatrix.front().size());

        size_t nonZeroBaseEntries = 0;
        for (const auto& row : baseMatrix) {
            nonZeroBaseEntries += static_cast<size_t>(
                std::count_if(row.begin(), row.end(), [](long long value) { return value > 0; }));
        }
        const double nonZeroRatio =
            static_cast<double>(nonZeroBaseEntries) / static_cast<double>(baseRows * baseCols);
        const auto estimatedFlows =
            static_cast<size_t>(static_cast<double>(expectedNpus) * static_cast<double>(expectedNpus) * nonZeroRatio);
        flows.reserve(estimatedFlows);

        for (int src = 0; src < expectedNpus; ++src) {
            const auto& baseRow = baseMatrix[src % baseRows];
            for (int dst = 0; dst < expectedNpus; ++dst) {
                if (src == dst) {
                    continue;
                }
                const long long value = baseRow[dst % baseCols];
                if (value <= 0) {
                    continue;
                }
                flows.push_back({src, dst, value * blockBytes});
            }
        }
        return flows;
    }

    throw std::invalid_argument("Unsupported synthetic_v_datasize pattern: " + pattern);
}

// Create collective instance from JSON config and topology info
// Returns collective and chunk size
std::pair<std::shared_ptr<Collective>, long long> createCollectiveWithChunkSize(const json& config, int npusCount) {
    std::string collectiveType = config["collective"].get<std::string>();
    
    // Check if using variable datasize (v_datasize) or regular datasize
    bool useVariableDatasize = config.contains("v_datasize");
    
    if (useVariableDatasize) {
        // Variable datasize mode: read from CSV file
        std::string csvPath = config["v_datasize"].get<std::string>();
        long long blockBytes = getBlockBytes(config);
        
        // Parse CSV file
        auto dataMatrix = parseVariableDatasizeCSV(csvPath, npusCount, blockBytes);
        
        // Calculate GCD of all data sizes to determine chunk size
        std::vector<long long> allDataSizes;
        for (const auto& row : dataMatrix) {
            for (const auto& val : row) {
                if (val > 0) {
                    allDataSizes.push_back(val);
                }
            }
        }
        
        if (allDataSizes.empty()) {
            throw std::invalid_argument("CSV file contains no positive data sizes");
        }
        
        long long chunkSize = calculateGCDMultiple(allDataSizes);
        if (chunkSize == 0) {
            chunkSize = 1; // Fallback to 1 if GCD is 0
        }
        
        // Create AlltoAllV collective
        if (collectiveType == "alltoallv" || collectiveType == "alltoall") {
            auto collective = std::make_shared<AlltoAllV>(npusCount, dataMatrix, chunkSize);
            return {collective, chunkSize};
        } else {
            throw std::invalid_argument("v_datasize is only supported for alltoallv collective type");
        }
    } else {
        // Regular datasize mode
        if (!config.contains("datasize")) {
            throw std::invalid_argument("Either 'datasize' or 'v_datasize' must be specified");
        }
        
        long long datasize = config["datasize"].get<long long>();
        int chunkfactor = config["chunkfactor"].get<int>();
        const auto chunkSize = datasize / chunkfactor;

        // Use rank zero as the default root for multi-to-one collectives.
        int root = 0;
        std::shared_ptr<Collective> collective;
        
        if (collectiveType == "allgather") {
            collective = std::make_shared<AllGather>(npusCount, chunkfactor);
        } else if (collectiveType == "alltoall") {
            collective = std::make_shared<AlltoAll>(npusCount, chunkfactor);
        } else if (collectiveType == "gather") {
            collective = std::make_shared<Gather>(npusCount, chunkfactor, root);
        } else {
            throw std::invalid_argument("Unsupported collective type: " + collectiveType);
        }
        
        return {collective, chunkSize};
    }
}

// Backward compatibility wrapper
std::shared_ptr<Collective> createCollective(const json& config, int npusCount) {
    auto [collective, _] = createCollectiveWithChunkSize(config, npusCount);
    return collective;
}

// Structure to store thread results
struct ThreadResult {
    int thread_id;
    double solve_time;
    double collective_time;
    std::string filename;
    
    // Constructor for easy initialization
    ThreadResult(int id, double solve, double collective, const std::string& fname)
        : thread_id(id), solve_time(solve), collective_time(collective), filename(fname) {}
};

// Thread-safe output function
void threadSafeOutput(std::ofstream& log_file, const std::string& message) {
    static std::mutex output_mutex;
    std::lock_guard<std::mutex> lock(output_mutex);
    log_file << message << std::flush;
}

// Function to run synthesizer in a thread
void runSynthesizerThread(int thread_id, 
                         std::shared_ptr<Topology> topology, 
                         std::shared_ptr<Collective> collective, 
                         long long chunkSize,
                         const json& topoConfig,
                         const json& collConfig,
                         const std::string& output_dir,
                         const TopologyFaultInfo& faultInfo,
                         std::vector<ThreadResult>& results,
                         std::mutex& results_mutex,
                         std::atomic<int>& completed_threads) {
    
    // Create output file stream for this thread
    std::string filename = output_dir + "/" + std::to_string(thread_id + 1) + ".log";
    std::ofstream log_file(filename);
    if (!log_file) {
        std::cerr << "Error: Could not open log file " << filename << std::endl;
        completed_threads++;
        return;
    }
    
    // Set thread-local output stream
    tacos::ThreadOutput::setOutputStream(log_file);
    
    // Set precision for this file stream
    log_file << std::fixed;
    log_file.precision(2);
    
    try {
        // Use stringstream to build output messages
        std::stringstream ss;
        
        // print header
        ss << "[Glaive Collective Synthesizer - Thread " << (thread_id + 1) << "]" << std::endl;
        ss << "########################################################" << std::endl;

        // print topology info
        std::string topologyType = topoConfig["topology"].get<std::string>();
        int collectiveNpusCount;
        if (topologyType == "fat-tree" || topologyType == "rail-optimized" || topologyType == "cm" || topologyType == "cm384") {
            // For switch topologies, calculate GPU count from shape
            auto shape = topoConfig["shape"].get<std::vector<int>>();
            collectiveNpusCount = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        } else {
            // For direct-connect topologies, use total node count
            collectiveNpusCount = topology->npusCount();
        }
        const auto npusCount = topology->npusCount();
        ss << "[Topology Information]" << std::endl;
        ss << "\t- Topology: " << topologyType << std::endl;
        ss << "\t- NPUs Count: " << npusCount << std::endl;
        if (topoConfig.contains("shape")) {
            auto shape = topoConfig["shape"].get<std::vector<int>>();
            ss << "\t- Dimension: " << shape.size() << std::endl;
            ss << "\t- Shape: [";
            for (size_t i = 0; i < shape.size(); ++i) {
                ss << shape[i] << (i + 1 < shape.size() ? ", " : "]\n");
            }
        }
        if (topoConfig.contains("latency")) {
            auto latency = topoConfig["latency"].get<std::vector<double>>();
            ss << "\t- Latency per dimension (ns): [";
            for (size_t i = 0; i < latency.size(); ++i) {
                ss << latency[i] << (i + 1 < latency.size() ? ", " : "]\n");
            }
        }
        if (topoConfig.contains("bandwidth")) {
            auto bandwidth = topoConfig["bandwidth"].get<std::vector<double>>();
            ss << "\t- Bandwidth per dimension (GB/s): [";
            for (size_t i = 0; i < bandwidth.size(); ++i) {
                ss << bandwidth[i] << (i + 1 < bandwidth.size() ? ", " : "]\n");
            }
        }
        // Print link fault information
        if (!faultInfo.linkFaultInfos.empty()) {
            ss << "\t- Link Faults (coordinate format): ";
            for (size_t i = 0; i < faultInfo.linkFaultInfos.size(); ++i) {
                ss << faultInfo.linkFaultInfos[i].coordStr;
                if (i + 1 < faultInfo.linkFaultInfos.size()) {
                    ss << ", ";
                }
            }
            ss << std::endl;
            ss << "\t- Link Faults (NpuID format): ";
            for (size_t i = 0; i < faultInfo.linkFaultInfos.size(); ++i) {
                ss << faultInfo.linkFaultInfos[i].npu1 << "-" << faultInfo.linkFaultInfos[i].npu2;
                if (i + 1 < faultInfo.linkFaultInfos.size()) {
                    ss << ", ";
                }
            }
            ss << std::endl;
        }
        // Print node fault information
        if (!faultInfo.nodeFaultInfos.empty()) {
            ss << "\t- Node Faults (coordinate format): ";
            for (size_t i = 0; i < faultInfo.nodeFaultInfos.size(); ++i) {
                ss << faultInfo.nodeFaultInfos[i].coordStr;
                if (i + 1 < faultInfo.nodeFaultInfos.size()) {
                    ss << ", ";
                }
            }
            ss << std::endl;
            ss << "\t- Node Faults (NpuID format): ";
            for (size_t i = 0; i < faultInfo.nodeFaultInfos.size(); ++i) {
                ss << faultInfo.nodeFaultInfos[i].npuID;
                if (i + 1 < faultInfo.nodeFaultInfos.size()) {
                    ss << ", ";
                }
            }
            ss << std::endl;
        }

        // create collective object here, then its output can enter the log file of this thread
        auto [thread_collective, thread_chunkSize] = createCollectiveWithChunkSize(collConfig, collectiveNpusCount);
        
        // Apply node faults to collective (remove faulty nodes from postconditions)
        applyNodeFaultsToCollective(thread_collective, faultInfo.nodeFaultInfos);
        
        // get collective info
        int chunkfactor = thread_collective->getChunkFactor();
        const auto chunksCount = thread_collective->chunksCount();
        ss << "chunksCount = " << chunksCount << ", npusCount = " << npusCount << std::endl;
        const auto chunkSizeMB = thread_chunkSize / (1 << 20);

        // print initialization info
        ss << "########################################################" << std::endl;
        ss << "[Collective Information]" << std::endl;
        ss << "\t- Collective: " << collConfig["collective"].get<std::string>() << std::endl;
        ss << "\t- Chunks Count: " << chunksCount << std::endl;
        ss << "\t- Chunk Size: " << thread_chunkSize << " B";
        ss << " (" << chunkSizeMB << " MB)" << std::endl;
        ss << "\t- Chunk Factor = " << chunkfactor << std::endl;
        ss << "########################################################" << std::endl;

        // Write the header information to file
        threadSafeOutput(log_file, ss.str());
        ss.str(""); // Clear the stringstream

        // create timer and synthesizer
        auto synthesizerTimer = Timer();
        synthesizerTimer.start();
        
        // Create synthesizer and set its output stream
        auto synthesizer = Synthesizer();
        synthesizer.setThreadOutputStream(log_file);
        
        // Run synthesizer with thread-specific collective
        auto collectiveTime = synthesizer.solve(topology, thread_collective, thread_chunkSize);
        synthesizerTimer.stop();

        // print result
        auto time = synthesizerTimer.time();
        const auto timeSec = time / 1e6;
        ss << "Time to solve: " << time << " us";
        ss << " (" << timeSec << " s)" << std::endl;
        ss << "Collective Time: " << collectiveTime << " us" << std::endl;
        ss << "########################################################" << std::endl;
        ss << "[TACOS] Thread " << (thread_id + 1) << " Done!" << std::endl;

        // Write final results to file
        threadSafeOutput(log_file, ss.str());

        // Store result
        std::lock_guard<std::mutex> lock(results_mutex);
        results.push_back(ThreadResult(thread_id, time, collectiveTime, filename));
        
    } catch (const std::exception& e) {
        std::stringstream error_ss;
        error_ss << "Error in thread " << (thread_id + 1) << ": " << e.what() << std::endl;
        threadSafeOutput(log_file, error_ss.str());
    }
    
    log_file.close();
    completed_threads++;
}



// Function to parse log files and extract timing information
void parseLogFiles(const std::string& output_dir, std::vector<ThreadResult>& results, int thread_num) {
    for (int i = 1; i <= thread_num; ++i) {
        std::string filename = output_dir + "/" + std::to_string(i) + ".log";
        std::ifstream file(filename);
        if (!file) continue;
        
        double solve_time = 0.0;
        double collective_time = 0.0;
        std::string line;
        
        while (std::getline(file, line)) {
            if (line.find("Time to solve:") != std::string::npos) {
                // Extract the time value
                size_t pos = line.find(":");
                if (pos != std::string::npos) {
                    std::string time_str = line.substr(pos + 1);
                    size_t us_pos = time_str.find(" us");
                    if (us_pos != std::string::npos) {
                        time_str = time_str.substr(0, us_pos);
                        // Remove any non-numeric characters except decimal point
                        time_str.erase(std::remove_if(time_str.begin(), time_str.end(), 
                                        [](char c) { return !std::isdigit(c) && c != '.'; }), 
                                      time_str.end());
                        try {
                            solve_time = std::stod(time_str);
                        } catch (const std::exception& e) {
                            std::cerr << "Error parsing solve time from: " << time_str << std::endl;
                        }
                    }
                }
            } else if (line.find("Collective Time:") != std::string::npos) {
                // Extract the time value
                size_t pos = line.find(":");
                if (pos != std::string::npos) {
                    std::string time_str = line.substr(pos + 1);
                    size_t us_pos = time_str.find(" us");
                    if (us_pos != std::string::npos) {
                        time_str = time_str.substr(0, us_pos);
                        // Remove any non-numeric characters except decimal point
                        time_str.erase(std::remove_if(time_str.begin(), time_str.end(), 
                                        [](char c) { return !std::isdigit(c) && c != '.'; }), 
                                      time_str.end());
                        try {
                            collective_time = std::stod(time_str);
                        } catch (const std::exception& e) {
                            std::cerr << "Error parsing collective time from: " << time_str << std::endl;
                        }
                    }
                }
            }
        }
        
        if (solve_time > 0 && collective_time > 0) {
            results.push_back(ThreadResult(i-1, solve_time, collective_time, filename));
        }
        
        file.close();
    }
}

// Print usage information
void printUsage(const char* programName) {
    std::cerr << "Usage: " << programName << " <topology_config.json> <collective_config.json> [options]" << std::endl;
    std::cerr << "\nOptions:" << std::endl;
    std::cerr << "  --multithread <output_dir>  Run in multithread mode with 32 threads" << std::endl;
    std::cerr << "  --solver2                   Use Synthesizer2 (unified solver for non-uniform AllToAll)" << std::endl;
    std::cerr << "  --solver3                   Use Synthesizer3 (profiling-scheduling-fusion workflow)" << std::endl;
    std::cerr << "  --solver4                   Use Synthesizer4 (hotspot-based matrix decomposition)" << std::endl;
    std::cerr << "  --baselines                 Run baseline algorithms (Bruck, Spreadout, Pairwise, BiRing, and formula baselines)" << std::endl;
    std::cerr << "  --baseline-method <name>    Run only one baseline: biring, halfringdr, mpibaseline, pairwise, bruck, spreadout" << std::endl;
    std::cerr << "  --compare                   Compare different routing strategies (solver2 only)" << std::endl;
    std::cerr << "  --no-diffusion              Disable diffusion preprocessing (solver2 only)" << std::endl;
    std::cerr << "  --quiet                     Disable verbose output (solver2/solver3/solver4 only)" << std::endl;
    std::cerr << "  --print-schedule            Print detailed schedule (solver2/solver3/solver4 only)" << std::endl;
    std::cerr << "  mode=clean|complete|standard|speed Select condensed, complete, standard, or Speed output (solver3)" << std::endl;
    std::cerr << "  --standard-hot-cap <n>       Override Standard hot-flow cap; default is 4 * N" << std::endl;
    std::cerr << "  path_weight_sum=<w>         Override Standard Thrust NormSumLoad score weight; default 0.3" << std::endl;
    std::cerr << "  path_weight_max=<w>         Override Standard Thrust NormMaxLoad score weight; default 0.2" << std::endl;
    std::cerr << "  path_weight_data=<w>        Override Standard Thrust NormDataTransfer score weight; default 0.5" << std::endl;
    std::cerr << "  --strategy <name>           Force routing strategy (solver2 only):" << std::endl;
    std::cerr << "                              dimension_order, load_balanced, multi_path, adaptive" << std::endl;
    std::cerr << "\nExamples:" << std::endl;
    std::cerr << "  " << programName << " input/topology/mesh2d_1.json input/collective/allgather_1.json" << std::endl;
    std::cerr << "  " << programName << " input/topology/mesh2d_1.json input/collective/allgather_1.json --multithread results" << std::endl;
    std::cerr << "  " << programName << " input/topology/mesh2d_1.json input/collective/alltoallv.json --solver2" << std::endl;
    std::cerr << "  " << programName << " input/topology/mesh2d_1.json input/collective/alltoallv.json --solver2 --compare" << std::endl;
    std::cerr << "  " << programName << " input/topology/mesh2d_1.json input/collective/alltoallv.json --solver3" << std::endl;
    std::cerr << "  " << programName << " input/topology/mesh2d_1.json input/collective/alltoallv.json --solver4" << std::endl;
    std::cerr << "  " << programName << " input/topology/mesh2d_1.json input/collective/alltoallv.json --baselines" << std::endl;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    bool multithread = false;
    bool useSolver2 = false;
    bool useSolver3 = false;
    bool useSolver4 = false;
    bool useBaselines = false;
    bool compareMode = false;
    bool enableDiffusion = true;
    bool verbose = true;
    bool printSchedule = false;
    bool cleanMode = false;
    bool solver4CleanMode = false;  // mode=clean vs complete (default)
    bool standardMode = false;
    bool speedMode = false;
    int standardHotFlowCap = 0;
    double standardPathWeightSum = 0.3;
    double standardPathWeightMax = 0.2;
    double standardPathWeightData = 0.5;
    RoutingStrategy forcedStrategy = RoutingStrategy::ADAPTIVE;
    bool forceStrategy = false;
    std::string baselineMethod;
    std::string output_dir;
    std::string topoPath;
    std::string collPath;

    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    topoPath = argv[1];
    collPath = argv[2];

    // Parse options
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--multithread") == 0) {
            multithread = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                output_dir = argv[++i];
            } else {
                std::cerr << "Error: --multithread requires output directory" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "--solver2") == 0) {
            useSolver2 = true;
        } else if (strcmp(argv[i], "--solver3") == 0) {
            useSolver3 = true;
        } else if (strcmp(argv[i], "--solver4") == 0) {
            useSolver4 = true;
        } else if (strcmp(argv[i], "--baselines") == 0) {
            useBaselines = true;
        } else if (strcmp(argv[i], "--baseline-method") == 0 && i + 1 < argc) {
            useBaselines = true;
            baselineMethod = argv[++i];
        } else if (strcmp(argv[i], "--compare") == 0) {
            compareMode = true;
        } else if (strcmp(argv[i], "--no-diffusion") == 0) {
            enableDiffusion = false;
        } else if (strcmp(argv[i], "--quiet") == 0) {
            verbose = false;
        } else if (strcmp(argv[i], "--print-schedule") == 0) {
            printSchedule = true;
        } else if (strcmp(argv[i], "--standard-hot-cap") == 0 && i + 1 < argc) {
            standardHotFlowCap = std::stoi(argv[++i]);
            if (standardHotFlowCap <= 0) {
                std::cerr << "Error: --standard-hot-cap must be positive" << std::endl;
                return 1;
            }
        } else if (std::strncmp(argv[i], "hot_cap=", 8) == 0) {
            standardHotFlowCap = std::stoi(argv[i] + 8);
            if (standardHotFlowCap <= 0) {
                std::cerr << "Error: hot_cap must be positive" << std::endl;
                return 1;
            }
        } else if (std::strncmp(argv[i], "path_weight_sum=", 16) == 0) {
            standardPathWeightSum = std::stod(argv[i] + 16);
            if (standardPathWeightSum < 0.0) {
                std::cerr << "Error: path_weight_sum must be non-negative" << std::endl;
                return 1;
            }
        } else if (std::strncmp(argv[i], "path_weight_max=", 16) == 0) {
            standardPathWeightMax = std::stod(argv[i] + 16);
            if (standardPathWeightMax < 0.0) {
                std::cerr << "Error: path_weight_max must be non-negative" << std::endl;
                return 1;
            }
        } else if (std::strncmp(argv[i], "path_weight_data=", 17) == 0) {
            standardPathWeightData = std::stod(argv[i] + 17);
            if (standardPathWeightData < 0.0) {
                std::cerr << "Error: path_weight_data must be non-negative" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "--strategy") == 0 && i + 1 < argc) {
            forceStrategy = true;
            std::string strategyName = argv[++i];
            if (strategyName == "dimension_order") {
                forcedStrategy = RoutingStrategy::DIMENSION_ORDER;
            } else if (strategyName == "load_balanced") {
                forcedStrategy = RoutingStrategy::LOAD_BALANCED;
            } else if (strategyName == "multi_path") {
                forcedStrategy = RoutingStrategy::MULTI_PATH;
            } else if (strategyName == "adaptive") {
                forcedStrategy = RoutingStrategy::ADAPTIVE;
            } else {
                std::cerr << "Unknown strategy: " << strategyName << std::endl;
                return 1;
            }
        } else if (std::strncmp(argv[i], "mode=", 5) == 0) {
            const char* modeStr = argv[i] + 5;
            if (std::strcmp(modeStr, "clean") == 0) {
                cleanMode = true;
                solver4CleanMode = true;
                standardMode = false;
                speedMode = false;
            } else if (std::strcmp(modeStr, "complete") == 0) {
                cleanMode = false;
                solver4CleanMode = false;
                standardMode = false;
                speedMode = false;
            } else if (std::strcmp(modeStr, "standard") == 0) {
                cleanMode = true;
                solver4CleanMode = true;
                standardMode = true;
                speedMode = false;
            } else if (std::strcmp(modeStr, "speed") == 0) {
                cleanMode = true;
                solver4CleanMode = true;
                standardMode = true;
                speedMode = true;
            } else {
                std::cerr << "Unknown mode: " << modeStr << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "true") == 0 || strcmp(argv[i], "1") == 0) {
            // Legacy multithread option support
            multithread = true;
            if (i + 1 < argc) {
                output_dir = argv[++i];
            } else {
                std::cerr << "Error: Output directory must be specified when multithread is enabled" << std::endl;
                return 1;
            }
        } else if (argv[i][0] != '-' && multithread && output_dir.empty()) {
            output_dir = argv[i];
        }
    }

    if (multithread && output_dir.empty()) {
        std::cerr << "Error: Output directory must be specified when multithread is enabled" << std::endl;
        return 1;
    }
    if (standardPathWeightSum + standardPathWeightMax + standardPathWeightData <= 0.0) {
        std::cerr << "Error: at least one path score weight must be positive" << std::endl;
        return 1;
    }
    
    std::ifstream topoFile(topoPath);
    std::ifstream collFile(collPath);
    if (!topoFile || !collFile) {
        std::cerr << "Error: Could not open input config files." << std::endl;
        return 1;
    }

    // Get the config
    json topoConfig, collConfig;
    topoFile >> topoConfig;
    collFile >> collConfig;

    try {
        std::string topologyType = topoConfig["topology"].get<std::string>();
        const auto shape = topoConfig["shape"].get<std::vector<int>>();
        const bool useSyntheticSolver3Demand =
            useSolver3 && collConfig.contains("synthetic_v_datasize");
        const bool lightweightDirectSolver3 =
            useSyntheticSolver3Demand &&
            (topologyType == "mesh" || topologyType == "torus" || topologyType == "fullmesh");

        std::shared_ptr<Topology> topology;
        TopologyFaultInfo faultInfo;
        int collectiveNpusCount;
        std::shared_ptr<Collective> collective;
        long long chunkSize = 0;

        if (lightweightDirectSolver3) {
            collectiveNpusCount = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        } else {
            auto topologyWithFaults = createTopologyWithFaults(topoConfig);
            topology = topologyWithFaults.first;
            faultInfo = topologyWithFaults.second;

            if (topologyType == "fat-tree" || topologyType == "rail-optimized" ||
                topologyType == "cm" || topologyType == "cm384") {
                collectiveNpusCount = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
            } else {
                collectiveNpusCount = topology->npusCount();
            }

            if (!useSyntheticSolver3Demand) {
                auto collectiveWithChunkSize = createCollectiveWithChunkSize(collConfig, collectiveNpusCount);
                collective = collectiveWithChunkSize.first;
                chunkSize = collectiveWithChunkSize.second;

                applyNodeFaultsToCollective(collective, faultInfo.nodeFaultInfos);
            }
        }

        // ============================================================
        // Solver2 mode: Use Synthesizer2 (unified solver for non-uniform AllToAll)
        // ============================================================
        if (useSolver2) {
            std::string topologyType = topoConfig["topology"].get<std::string>();
            if (!collConfig.contains("v_datasize")) {
                std::cerr << "Error: Synthesizer2 requires v_datasize (CSV file) in collective config" << std::endl;
                return 1;
            }

            auto shape = topoConfig["shape"].get<std::vector<int>>();
            auto bandwidth = topoConfig["bandwidth"].get<std::vector<double>>();
            auto latency = topoConfig["latency"].get<std::vector<double>>();
            bool isTorus = (topologyType == "torus");

            // Parse demand matrix from CSV
            // For switch-based topologies, CSV only contains GPU nodes, not switches
            std::string csvPath = collConfig["v_datasize"].get<std::string>();
            int expectedNpus;
            if (topologyType == "fat-tree" || topologyType == "rail-optimized" || topologyType == "cm" || topologyType == "cm384") {
                // For switch topologies, calculate GPU count from shape
                expectedNpus = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
            } else {
                // For direct-connect topologies, use total node count
                expectedNpus = topology->npusCount();
            }
            auto dataMatrix = parseVariableDatasizeCSV(
                csvPath, expectedNpus, getBlockBytes(collConfig));

            // Create and configure Synthesizer2
            Synthesizer2 solver2 = (topologyType == "fat-tree" || topologyType == "rail-optimized" || topologyType == "cm" || topologyType == "cm384") ?
                Synthesizer2(topology, shape, bandwidth[0], latency[0]) :
                Synthesizer2(shape, isTorus, bandwidth[0], latency[0]);
            solver2.setVerbose(verbose);
            solver2.setPrintSchedule(printSchedule);
            solver2.setEnableDiffusion(enableDiffusion);
            if (forceStrategy) {
                solver2.setRoutingStrategy(forcedStrategy);
            }

            auto timer = Timer();

            if (compareMode) {
                // Compare different strategies
                timer.start();
                solver2.compareStrategies(dataMatrix);
                timer.stop();
                std::cout << "\nComparison Time: " << timer.time() << " us\n";
            } else {
                // Normal solve with chunk-based output (similar to original solver)
                timer.start();
                auto chunkResult = solver2.solveWithChunks(dataMatrix, chunkSize);
                timer.stop();

                // Print Precondition and Postcondition (similar to original solver)
                chunkResult.printPrecondition();
                chunkResult.printPostcondition();

                // Print topology and collective info header
                std::cout << "[Glaive Collective Synthesizer]" << std::endl;
                std::cout << "########################################################" << std::endl;
                std::cout << "[Topology Information]" << std::endl;
                std::cout << "\t- Topology: " << topologyType << std::endl;
                std::cout << "\t- NPUs Count: " << topology->npusCount() << std::endl;
                std::cout << "\t- Dimension: " << shape.size() << std::endl;
                std::cout << "\t- Shape: [";
                for (size_t i = 0; i < shape.size(); ++i) {
                    std::cout << shape[i] << (i + 1 < shape.size() ? ", " : "]\n");
                }
                std::cout << "\t- Latency per dimension (ns): [";
                for (size_t i = 0; i < latency.size(); ++i) {
                    std::cout << latency[i] << (i + 1 < latency.size() ? ", " : "]\n");
                }
                std::cout << "\t- Bandwidth per dimension (GB/s): [";
                for (size_t i = 0; i < bandwidth.size(); ++i) {
                    std::cout << bandwidth[i] << (i + 1 < bandwidth.size() ? ", " : "]\n");
                }
                std::cout << "########################################################" << std::endl;
                std::cout << "[Collective Information]" << std::endl;
                std::cout << "\t- Collective: " << collConfig["collective"].get<std::string>() << std::endl;
                std::cout << "\t- Chunks Count: " << chunkResult.totalChunks << std::endl;
                std::cout << "\t- Chunk Size: " << chunkSize << " B";
                std::cout << " (" << (chunkSize / (1 << 20)) << " MB)" << std::endl;
                int chunkfactor = collConfig.contains("chunkfactor") ? collConfig["chunkfactor"].get<int>() : 1;
                std::cout << "\t- Chunk Factor = " << chunkfactor << std::endl;
                std::cout << "########################################################" << std::endl;

                // Print postconditions list
                chunkResult.printPostconditionsList();

                // Print transfer events (similar to original solver format)
                for (const auto& event : chunkResult.transferEvents) {
                    // Find the chunk's destination
                    int chunkDst = chunkResult.chunks[event.chunkId].dstNpu;
                    std::cout << "Current [chunk, dest]: " << event.chunkId << ", " << chunkDst << std::endl;
                    std::cout << std::fixed << std::setprecision(6)
                              << "[EventTime " << event.eventTime << " us] Chunk "
                              << event.chunkId << ": " << event.fromNpu << " -> " << event.toNpu << std::endl;
                }

                std::cout << "\n[Performance Summary]" << std::endl;
                std::cout << "  Solver Time: " << std::fixed << std::setprecision(2)
                          << timer.time() << " us" << std::endl;
                std::cout << "  Makespan: " << chunkResult.makespan << " us" << std::endl;
            }

            std::cout << "\n[TACOS Solver2] Done!" << std::endl;

        // ============================================================
        // Solver3 mode: Use Synthesizer3 (profiling-scheduling-fusion workflow)
        // ============================================================
        } else if (useSolver3) {
            std::string topologyType = topoConfig["topology"].get<std::string>();
            const bool useSyntheticDemand = collConfig.contains("synthetic_v_datasize");
            if (!collConfig.contains("v_datasize") && !useSyntheticDemand) {
                std::cerr << "Error: Synthesizer3 requires either v_datasize (CSV file) or synthetic_v_datasize in collective config" << std::endl;
                return 1;
            }

            auto shape = topoConfig["shape"].get<std::vector<int>>();
            auto bandwidth = topoConfig["bandwidth"].get<std::vector<double>>();
            auto latency = topoConfig["latency"].get<std::vector<double>>();
            const bool isTorus = (topologyType == "torus");

            int expectedNpus;
            if (topologyType == "fat-tree" || topologyType == "rail-optimized" ||
                topologyType == "cm" || topologyType == "cm384") {
                expectedNpus = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
            } else {
                expectedNpus = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
            }

            std::vector<std::vector<long long>> dataMatrix;
            std::vector<DemandEntry3> sparseDemand;
            if (useSyntheticDemand) {
                sparseDemand = generateSyntheticDemandFlows(
                    collConfig["synthetic_v_datasize"], shape, topologyType, expectedNpus,
                    getBlockBytes(collConfig));
            } else {
                std::string csvPath = collConfig["v_datasize"].get<std::string>();
                dataMatrix = parseVariableDatasizeCSV(
                    csvPath, expectedNpus, getBlockBytes(collConfig));
            }
            const DemandStats demandStats = useSyntheticDemand
                ? computeSparseDemandStats(sparseDemand, expectedNpus)
                : computeDenseDemandStats(dataMatrix);
            printDemandStats(demandStats);

            auto createSolver3 = [&]() {
                if (topologyType == "mesh") {
                    return Synthesizer3(shape, Synthesizer3::DirectTopologyKind::Mesh, bandwidth[0], latency[0]);
                }
                if (topologyType == "torus") {
                    return Synthesizer3(shape, Synthesizer3::DirectTopologyKind::Torus, bandwidth[0], latency[0]);
                }
                if (topologyType == "fullmesh") {
                    return Synthesizer3(shape, Synthesizer3::DirectTopologyKind::FullMesh, bandwidth[0], latency[0]);
                }
                return Synthesizer3(topology, shape, bandwidth[0], latency[0]);
            };

            if (standardMode) {
                auto createStandardSolver = [&]() {
                    StandardSynthesizer solver = [&]() {
                    if (topologyType == "mesh") {
                        return StandardSynthesizer(shape, Synthesizer3::DirectTopologyKind::Mesh,
                                                 bandwidth[0], latency[0]);
                    }
                    if (topologyType == "torus") {
                        return StandardSynthesizer(shape, Synthesizer3::DirectTopologyKind::Torus,
                                                 bandwidth[0], latency[0]);
                    }
                    if (topologyType == "fullmesh") {
                        return StandardSynthesizer(shape, Synthesizer3::DirectTopologyKind::FullMesh,
                                                 bandwidth[0], latency[0]);
                    }
                    return StandardSynthesizer(topology, shape, bandwidth[0], latency[0]);
                    }();
                    solver.setPathScoreWeights(standardPathWeightSum,
                                               standardPathWeightMax,
                                               standardPathWeightData);
                    return solver;
                };

                auto cacheInitTimer = Timer();
                cacheInitTimer.start();
                StandardSynthesizer standardSolver = createStandardSolver();
                if (standardHotFlowCap > 0) {
                    standardSolver.setMaxBandwidthFlowsOverride(standardHotFlowCap);
                }
                cacheInitTimer.stop();
                const auto initialCacheStats = standardSolver.cacheStats();
                standardSolver.setCaptureSchedule(printSchedule || speedMode);

                auto printStandardCacheStats = [&](const char* phase,
                                                  const StandardCacheStats& stats,
                                                  double cacheInitTimeUs) {
                    std::cout << "[Standard Cache CSV] "
                              << "phase=" << phase
                              << ",init_time_us=" << std::fixed << std::setprecision(2) << cacheInitTimeUs
                              << ",gpu_nodes=" << stats.gpuNodeCount
                              << ",total_nodes=" << stats.totalNodeCount
                              << ",routing_kind=" << stats.routingKind
                              << ",all_small_switch_pairs_dominant=" << (stats.allSmallSwitchPairsDominant ? 1 : 0)
                              << ",connection_edges=" << stats.connectionEdges
                              << ",distance_matrix_entries=" << stats.distanceMatrixEntries
                              << ",link_cache_entries=" << stats.linkCacheEntries
                              << ",small_switch_candidate_pairs=" << stats.smallSwitchCandidatePairs
                              << ",small_switch_candidate_paths=" << stats.smallSwitchCandidatePaths
                              << ",small_switch_dominant_paths=" << stats.smallSwitchDominantPaths
                              << ",gpu_pair_min_chunk_entries=" << stats.gpuPairMinChunkEntries
                              << ",shortest_path_cache_entries=" << stats.shortestPathCacheEntries
                              << ",time_shortest_path_cache_entries=" << stats.timeShortestPathCacheEntries
                              << ",approx_cache_bytes=" << stats.approximateBytes
                              << std::endl;
                };
                printStandardCacheStats("initial", initialCacheStats, cacheInitTimer.time());

                auto timer = Timer();
                timer.start();
                auto standardResult = useSyntheticDemand ? standardSolver.solveSparse(sparseDemand)
                                                        : standardSolver.solve(dataMatrix);
                timer.stop();
                const auto postSolveCacheStats = standardSolver.cacheStats();
                printStandardCacheStats("post_solve", postSolveCacheStats, cacheInitTimer.time());

                if (printSchedule && !speedMode) {
                    standardSolver.printEvents(standardResult.events);
                }

                StandardProfileSummary standardProfile;
                double hotSolveTimeUs = 0.0;
                double coldSolveTimeUs = 0.0;
                StandardResult hotOnlyResult;
                StandardResult coldOnlyResult;
                HotColdTimeBreakdown hotColdBreakdown;
                if (speedMode) {
                    standardProfile = useSyntheticDemand
                        ? standardSolver.profileSparseSummary(sparseDemand)
                        : standardSolver.profileSummary(dataMatrix);

                    StandardSynthesizer hotSolver = createStandardSolver();
                    if (standardHotFlowCap > 0) {
                        hotSolver.setMaxBandwidthFlowsOverride(standardHotFlowCap);
                    }
                    auto hotTimer = Timer();
                    hotTimer.start();
                    hotOnlyResult = useSyntheticDemand ? hotSolver.solveSparseHotOnly(sparseDemand)
                                                       : hotSolver.solveHotOnly(dataMatrix);
                    hotTimer.stop();
                    hotSolveTimeUs = hotTimer.time();

                    StandardSynthesizer coldSolver = createStandardSolver();
                    if (standardHotFlowCap > 0) {
                        coldSolver.setMaxBandwidthFlowsOverride(standardHotFlowCap);
                    }
                    auto coldTimer = Timer();
                    coldTimer.start();
                    coldOnlyResult = useSyntheticDemand ? coldSolver.solveSparseColdOnly(sparseDemand)
                                                        : coldSolver.solveColdOnly(dataMatrix);
                    coldTimer.stop();
                    coldSolveTimeUs = coldTimer.time();

                    hotColdBreakdown = computeHotColdTimeBreakdown(standardResult.events);
                }

                std::cout << "\n[Performance Summary]" << std::endl;
                std::cout << "  Solver Time: " << std::fixed << std::setprecision(2)
                          << timer.time() << " us" << std::endl;
                std::cout << "  Total Makespan: " << standardResult.makespan << " us" << std::endl;
                std::cout << "  Standard Scheduled Chunks: " << standardResult.scheduledChunks << std::endl;
                std::cout << "  Standard Scheduled Events: " << standardResult.scheduledEvents << std::endl;

                if (speedMode) {
                    std::cout << "\n[Speed Summary]" << std::endl;
                    std::cout << "  Hot Threshold Bytes: " << std::fixed << std::setprecision(2)
                              << standardProfile.threshold << std::endl;
                    std::cout << "  Hot Flow Count: " << standardProfile.hotFlowCount << std::endl;
                    std::cout << "  Cold Flow Count: " << standardProfile.coldFlowCount << std::endl;
                    std::cout << "  Hot Bytes: " << standardProfile.hotBytes << std::endl;
                    std::cout << "  Cold Bytes: " << standardProfile.coldBytes << std::endl;
                    std::cout << "  Hot Solve Time: " << hotSolveTimeUs << " us" << std::endl;
                    std::cout << "  Cold Solve Time: " << coldSolveTimeUs << " us" << std::endl;
                    std::cout << "  Hot Only Makespan: " << hotOnlyResult.makespan << " us" << std::endl;
                    std::cout << "  Cold Only Makespan: " << coldOnlyResult.makespan << " us" << std::endl;
                    std::cout << "  Full Hot-Only Active Time: " << hotColdBreakdown.hotOnlyUs << " us" << std::endl;
                    std::cout << "  Full Cold-Only Active Time: " << hotColdBreakdown.coldOnlyUs << " us" << std::endl;
                    std::cout << "  Full Hot-Cold Overlap Time: " << hotColdBreakdown.overlapUs << " us" << std::endl;
                    std::cout << "  Full Active Time: " << hotColdBreakdown.activeUs << " us" << std::endl;
                    std::cout << "[Speed Summary CSV] "
                              << "threshold_bytes=" << standardProfile.threshold
                              << ",hot_flows=" << standardProfile.hotFlowCount
                              << ",cold_flows=" << standardProfile.coldFlowCount
                              << ",hot_bytes=" << standardProfile.hotBytes
                              << ",cold_bytes=" << standardProfile.coldBytes
                              << ",full_solver_time_us=" << timer.time()
                              << ",hot_solver_time_us=" << hotSolveTimeUs
                              << ",cold_solver_time_us=" << coldSolveTimeUs
                              << ",full_makespan_us=" << standardResult.makespan
                              << ",hot_only_makespan_us=" << hotOnlyResult.makespan
                              << ",cold_only_makespan_us=" << coldOnlyResult.makespan
                              << ",full_hot_only_active_us=" << hotColdBreakdown.hotOnlyUs
                              << ",full_cold_only_active_us=" << hotColdBreakdown.coldOnlyUs
                              << ",full_overlap_us=" << hotColdBreakdown.overlapUs
                              << ",full_active_us=" << hotColdBreakdown.activeUs
                              << std::endl;
                    printSpeedEvents(standardResult.events);
                }

                std::cout << "\n[TACOS Solver3] Done!" << std::endl;
                return 0;
            }

            Synthesizer3 solver3 = createSolver3();
            if (cleanMode) {
                solver3.setVerbose(false);
                solver3.setPrintSchedule(printSchedule);
                solver3.setCleanMode(true);
            } else {
                solver3.setVerbose(verbose);
                solver3.setPrintSchedule(printSchedule);
            }

            auto timer = Timer();

            timer.start();
            auto makespan = useSyntheticDemand ? solver3.solveSparse(sparseDemand)
                                               : solver3.solve(dataMatrix);
            timer.stop();

            const auto& profilingResult = solver3.getProfilingResult();
            const auto& scheduleResult = solver3.getScheduleResult();

            if (!cleanMode) {
                std::cout << "[Glaive Collective Synthesizer - Synthesizer3]" << std::endl;
                std::cout << "########################################################" << std::endl;
                std::cout << "[Topology Information]" << std::endl;
                std::cout << "\t- Topology: " << topologyType << std::endl;
                std::cout << "\t- NPUs Count: " << expectedNpus << std::endl;
                std::cout << "\t- Dimension: " << shape.size() << std::endl;
                std::cout << "\t- Shape: [";
                for (size_t i = 0; i < shape.size(); ++i) {
                    std::cout << shape[i] << (i + 1 < shape.size() ? ", " : "]\n");
                }
                std::cout << "\t- Latency per dimension (ns): [";
                for (size_t i = 0; i < latency.size(); ++i) {
                    std::cout << latency[i] << (i + 1 < latency.size() ? ", " : "]\n");
                }
                std::cout << "\t- Bandwidth per dimension (GB/s): [";
                for (size_t i = 0; i < bandwidth.size(); ++i) {
                    std::cout << bandwidth[i] << (i + 1 < bandwidth.size() ? ", " : "]\n");
                }
                std::cout << "########################################################" << std::endl;
                std::cout << "[Collective Information]" << std::endl;
                std::cout << "\t- Collective: " << collConfig["collective"].get<std::string>() << std::endl;
                int chunkfactor = collConfig.contains("chunkfactor") ? collConfig["chunkfactor"].get<int>() : 1;
                std::cout << "\t- Chunk Factor = " << chunkfactor << std::endl;
                std::cout << "########################################################" << std::endl;

                scheduleResult.printEvents();
            }

            std::cout << "\n[Performance Summary]" << std::endl;
            std::cout << "  Solver Time: " << std::fixed << std::setprecision(2)
                      << timer.time() << " us" << std::endl;
            if (!cleanMode) {
                std::cout << "  Latency Matrix Makespan: " << scheduleResult.latencyMatrixMakespan << " us" << std::endl;
                std::cout << "  Bandwidth Matrix Makespan: " << scheduleResult.bandwidthMatrixMakespan << " us" << std::endl;
                std::cout << "  Fused Makespan: " << scheduleResult.fusedMakespan << " us" << std::endl;
            }
            std::cout << "  Total Makespan: " << makespan << " us" << std::endl;

            std::cout << "\n[TACOS Solver3] Done!" << std::endl;

        // ============================================================
        // Solver4 mode: Use Synthesizer4 (hotspot-based matrix decomposition)
        // ============================================================
        } else if (useSolver4) {
            std::string topologyType = topoConfig["topology"].get<std::string>();
            if (!collConfig.contains("v_datasize")) {
                std::cerr << "Error: Synthesizer4 requires v_datasize (CSV file) in collective config" << std::endl;
                return 1;
            }

            auto shape = topoConfig["shape"].get<std::vector<int>>();
            auto bandwidth = topoConfig["bandwidth"].get<std::vector<double>>();
            auto latency = topoConfig["latency"].get<std::vector<double>>();
            bool isTorus = (topologyType == "torus");

            // Parse demand matrix from CSV
            // For switch-based topologies, CSV only contains GPU nodes, not switches
            std::string csvPath = collConfig["v_datasize"].get<std::string>();
            int expectedNpus;
            if (topologyType == "fat-tree" || topologyType == "rail-optimized" || topologyType == "cm" || topologyType == "cm384") {
                // For switch topologies, calculate GPU count from shape
                expectedNpus = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
            } else {
                // For direct-connect topologies, use total node count
                expectedNpus = topology->npusCount();
            }
            auto dataMatrix = parseVariableDatasizeCSV(
                csvPath, expectedNpus, getBlockBytes(collConfig));

            // Create and configure Synthesizer4
            Synthesizer4 solver4 = (topologyType == "fat-tree" || topologyType == "rail-optimized" || topologyType == "cm" || topologyType == "cm384") ?
                Synthesizer4(topology, shape, bandwidth[0], latency[0]) :
                Synthesizer4(shape, isTorus, bandwidth[0], latency[0]);
            // In clean mode, suppress internal verbose/schedule prints and enable clean workflow
            if (solver4CleanMode) {
                solver4.setVerbose(false);
                solver4.setPrintSchedule(false);
                solver4.setCleanMode(true);
            } else {
                solver4.setVerbose(verbose);
                solver4.setPrintSchedule(printSchedule);
            }

            auto timer = Timer();

            // Normal solve
            timer.start();
            auto makespan = solver4.solve(dataMatrix);
            timer.stop();

            // Get profiling and schedule results
            const auto& profilingResult = solver4.getProfilingResult();
            const auto& scheduleResult = solver4.getScheduleResult();

            if (!solver4CleanMode) {
                // Print topology and collective info header (complete mode)
                std::cout << "[Glaive Collective Synthesizer - Synthesizer4]" << std::endl;
                std::cout << "########################################################" << std::endl;
                std::cout << "[Topology Information]" << std::endl;
                std::cout << "\t- Topology: " << topologyType << std::endl;
                std::cout << "\t- NPUs Count: " << topology->npusCount() << std::endl;
                std::cout << "\t- Dimension: " << shape.size() << std::endl;
                std::cout << "\t- Shape: [";
                for (size_t i = 0; i < shape.size(); ++i) {
                    std::cout << shape[i] << (i + 1 < shape.size() ? ", " : "]\n");
                }
                std::cout << "\t- Latency per dimension (ns): [";
                for (size_t i = 0; i < latency.size(); ++i) {
                    std::cout << latency[i] << (i + 1 < latency.size() ? ", " : "]\n");
                }
                std::cout << "\t- Bandwidth per dimension (GB/s): [";
                for (size_t i = 0; i < bandwidth.size(); ++i) {
                    std::cout << bandwidth[i] << (i + 1 < bandwidth.size() ? ", " : "]\n");
                }
                std::cout << "########################################################" << std::endl;
                std::cout << "[Collective Information]" << std::endl;
                std::cout << "\t- Collective: " << collConfig["collective"].get<std::string>() << std::endl;
                int chunkfactor = collConfig.contains("chunkfactor") ? collConfig["chunkfactor"].get<int>() : 1;
                std::cout << "\t- Chunk Factor = " << chunkfactor << std::endl;
                std::cout << "########################################################" << std::endl;

                // Profiling result is already printed by solver4 if verbose
                // Schedule result summary is already printed by solver4 if verbose
            }

            // In both modes, print step-4 fused scheduling events (per-hop schedule)
            // scheduleResult.events has been set to fusedSchedulingEvents inside Synthesizer4::solve
            scheduleResult.printEvents();

            std::cout << "\n[Performance Summary]" << std::endl;
            std::cout << "  Solver Time: " << std::fixed << std::setprecision(2)
                      << timer.time() << " us" << std::endl;
            if (!solver4CleanMode) {
                std::cout << "  Regular Matrix Makespan: " << scheduleResult.regularMatrixMakespan << " us" << std::endl;
                std::cout << "  Hotspot Matrix Makespan: " << scheduleResult.hotspotMatrixMakespan << " us" << std::endl;
                std::cout << "  Fused Makespan: " << scheduleResult.fusedMakespan << " us" << std::endl;
            }
            std::cout << "  Total Makespan: " << makespan << " us" << std::endl;

            std::cout << "\n[TACOS Solver4] Done!" << std::endl;

        // ============================================================
        // Baselines mode: Run Bruck and Spreadout algorithms
        // ============================================================
        } else if (useBaselines) {
            std::string topologyType = topoConfig["topology"].get<std::string>();
            
            // Get shape for XY routing (if mesh/torus)
            std::vector<int> shape;
            if (topoConfig.contains("shape")) {
                shape = topoConfig["shape"].get<std::vector<int>>();
            }
            
            // Collective operates on GPU count only (switch topologies: shape product; else topology size)
            int collectiveNpusCount;
            if (topologyType == "fat-tree" || topologyType == "rail-optimized" || topologyType == "cm" || topologyType == "cm384") {
                collectiveNpusCount = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
            } else {
                collectiveNpusCount = topology->npusCount();
            }
            
            // Parse demand matrix from CSV or use regular datasize
            std::vector<std::vector<long long>> dataMatrix;
            if (collConfig.contains("v_datasize")) {
                std::string csvPath = collConfig["v_datasize"].get<std::string>();
                dataMatrix = parseVariableDatasizeCSV(
                    csvPath, collectiveNpusCount, getBlockBytes(collConfig));
            } else if (collConfig.contains("datasize")) {
                // For uniform alltoall, create uniform data matrix
                long long datasize = collConfig["datasize"].get<long long>();
                dataMatrix.resize(collectiveNpusCount);
                for (int i = 0; i < collectiveNpusCount; ++i) {
                    dataMatrix[i].resize(collectiveNpusCount, datasize);
                }
            } else {
                std::cerr << "Error: Baselines require either 'datasize' or 'v_datasize' in collective config" << std::endl;
                return 1;
            }
            printDemandStats(computeDenseDemandStats(dataMatrix));
            
            // Print header
            std::cout << "[Glaive Collective Synthesizer - Baselines]" << std::endl;
            std::cout << "########################################################" << std::endl;
            std::cout << "[Topology Information]" << std::endl;
            std::cout << "\t- Topology: " << topologyType << std::endl;
            std::cout << "\t- NPUs Count (collective): " << collectiveNpusCount << std::endl;
            if (!shape.empty()) {
                std::cout << "\t- Dimension: " << shape.size() << std::endl;
                std::cout << "\t- Shape: [";
                for (size_t i = 0; i < shape.size(); ++i) {
                    std::cout << shape[i] << (i + 1 < shape.size() ? ", " : "]\n");
                }
            }
            std::cout << "########################################################" << std::endl;
            std::cout << "[Collective Information]" << std::endl;
            std::cout << "\t- Collective: " << collConfig["collective"].get<std::string>() << std::endl;
            std::cout << "########################################################" << std::endl;

            auto printBaselineResult = [](const std::string& displayName,
                                          double algorithmTimeUs,
                                          const BaselineSolver::Result& baselineResult) {
                std::cout << "\n[" << displayName << " Results]" << std::endl;
                std::cout << "  Algorithm Time: " << std::fixed << std::setprecision(2)
                          << algorithmTimeUs << " us" << std::endl;
                std::cout << "  Makespan: " << baselineResult.totalTime << " us" << std::endl;
                std::cout << "  Number of Steps: " << baselineResult.stepTimes.size() << std::endl;
                for (size_t i = 0; i < baselineResult.stepTimes.size(); ++i) {
                    std::cout << "    Step " << i << ": " << baselineResult.stepTimes[i]
                              << " us (" << baselineResult.stepTransfers[i].size() << " transfers)" << std::endl;
                }
            };

            auto runSelectedBaseline = [&](const std::string& requestedMethod) -> int {
                std::string method = requestedMethod;
                std::transform(method.begin(), method.end(), method.begin(),
                               [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

                if (method == "biring") {
                    std::cout << "\n[Running BiRing Algorithm]" << std::endl;
                    std::cout << "########################################################" << std::endl;
                    auto timer = Timer();
                    timer.start();
                    BiRing solver(topology, shape);
                    auto result = solver.solve(dataMatrix);
                    timer.stop();
                    printBaselineResult("BiRing Algorithm", timer.time(), result);
                    std::cout << "########################################################" << std::endl;
                    std::cout << "[TACOS Baseline] Done!" << std::endl;
                    return 0;
                }

                if (method == "halfringdr" || method == "halfring" || method == "half-ring-dr" ||
                    method == "halfring+dimrotation") {
                    if (topologyType != "torus" || shape.empty()) {
                        std::cerr << "Error: HalfRing+DimRotation requires torus topology with shape information" << std::endl;
                        return 1;
                    }
                    std::cout << "\n[Running HalfRing+DimRotation Algorithm]" << std::endl;
                    std::cout << "########################################################" << std::endl;
                    auto timer = Timer();
                    timer.start();
                    HalfRingDimRotation solver(topology, shape);
                    auto result = solver.solve(dataMatrix);
                    timer.stop();
                    printBaselineResult("HalfRing+DimRotation Algorithm", timer.time(), result);
                    std::cout << "########################################################" << std::endl;
                    std::cout << "[TACOS Baseline] Done!" << std::endl;
                    return 0;
                }

                if (method == "mpibaseline" || method == "mpi" || method == "pairwise") {
                    const int npusCount = collectiveNpusCount;
                    const bool isPowerOf2 = (npusCount > 0) && ((npusCount & (npusCount - 1)) == 0);
                    if (!isPowerOf2) {
                        std::cerr << "Error: MPI baseline (pairwise exchange) requires number of nodes to be a power of 2" << std::endl;
                        return 1;
                    }
                    std::cout << "\n[Running MPI Baseline (Pairwise Exchange)]" << std::endl;
                    std::cout << "########################################################" << std::endl;
                    auto timer = Timer();
                    timer.start();
                    Pairwise solver(topology, shape);
                    auto result = solver.solve(dataMatrix);
                    timer.stop();
                    printBaselineResult("MPI Baseline (Pairwise Exchange)", timer.time(), result);
                    std::cout << "########################################################" << std::endl;
                    std::cout << "[TACOS Baseline] Done!" << std::endl;
                    return 0;
                }

                if (method == "bruck") {
                    std::cout << "\n[Running Bruck Algorithm]" << std::endl;
                    std::cout << "########################################################" << std::endl;
                    auto timer = Timer();
                    timer.start();
                    Bruck solver(topology, shape);
                    auto result = solver.solve(dataMatrix);
                    timer.stop();
                    printBaselineResult("Bruck Algorithm", timer.time(), result);
                    std::cout << "########################################################" << std::endl;
                    std::cout << "[TACOS Baseline] Done!" << std::endl;
                    return 0;
                }

                if (method == "spreadout") {
                    std::cout << "\n[Running Spreadout Algorithm]" << std::endl;
                    std::cout << "########################################################" << std::endl;
                    auto timer = Timer();
                    timer.start();
                    Spreadout solver(topology, shape);
                    auto result = solver.solve(dataMatrix);
                    timer.stop();
                    printBaselineResult("Spreadout Algorithm", timer.time(), result);
                    std::cout << "########################################################" << std::endl;
                    std::cout << "[TACOS Baseline] Done!" << std::endl;
                    return 0;
                }

                std::cerr << "Error: Unknown baseline method '" << requestedMethod
                          << "'. Supported values: biring, halfringdr, mpibaseline, pairwise, bruck, spreadout" << std::endl;
                return 1;
            };

            if (!baselineMethod.empty()) {
                return runSelectedBaseline(baselineMethod);
            }
            
            // Run Bruck algorithm
            std::cout << "\n[Running Bruck Algorithm]" << std::endl;
            std::cout << "########################################################" << std::endl;
            auto bruckTimer = Timer();
            bruckTimer.start();
            Bruck bruck(topology, shape);
            auto bruckResult = bruck.solve(dataMatrix);
            bruckTimer.stop();
            
            std::cout << "\n[Bruck Algorithm Results]" << std::endl;
            std::cout << "  Algorithm Time: " << std::fixed << std::setprecision(2)
                      << bruckTimer.time() << " us" << std::endl;
            std::cout << "  Makespan: " << bruckResult.totalTime << " us" << std::endl;
            std::cout << "  Number of Steps: " << bruckResult.stepTimes.size() << std::endl;
            for (size_t i = 0; i < bruckResult.stepTimes.size(); ++i) {
                std::cout << "    Step " << i << ": " << bruckResult.stepTimes[i] 
                          << " us (" << bruckResult.stepTransfers[i].size() << " transfers)" << std::endl;
            }
            
            // Run Spreadout algorithm
            std::cout << "\n[Running Spreadout Algorithm]" << std::endl;
            std::cout << "########################################################" << std::endl;
            auto spreadoutTimer = Timer();
            spreadoutTimer.start();
            Spreadout spreadout(topology, shape);
            auto spreadoutResult = spreadout.solve(dataMatrix);
            spreadoutTimer.stop();
            
            std::cout << "\n[Spreadout Algorithm Results]" << std::endl;
            std::cout << "  Algorithm Time: " << std::fixed << std::setprecision(2)
                      << spreadoutTimer.time() << " us" << std::endl;
            std::cout << "  Makespan: " << spreadoutResult.totalTime << " us" << std::endl;
            std::cout << "  Number of Steps: " << spreadoutResult.stepTimes.size() << std::endl;
            for (size_t i = 0; i < spreadoutResult.stepTimes.size(); ++i) {
                std::cout << "    Step " << i << ": " << spreadoutResult.stepTimes[i] 
                          << " us (" << spreadoutResult.stepTransfers[i].size() << " transfers)" << std::endl;
            }
            
            // Run Pairwise algorithm (only when number of nodes is a power of 2)
            BaselineSolver::Result pairwiseResult;
            bool pairwiseExecuted = false;
            // Use GPU count for power-of-2 check (collective size)
            int npusCount = collectiveNpusCount;
            // Check if npusCount is a power of 2: n > 0 && (n & (n - 1)) == 0
            bool isPowerOf2 = (npusCount > 0) && ((npusCount & (npusCount - 1)) == 0);
            
            if (isPowerOf2) {
                std::cout << "\n[Running Pairwise Algorithm]" << std::endl;
                std::cout << "########################################################" << std::endl;
                auto pairwiseTimer = Timer();
                pairwiseTimer.start();
                Pairwise pairwise(topology, shape);
                pairwiseResult = pairwise.solve(dataMatrix);
                pairwiseTimer.stop();
                pairwiseExecuted = true;
                
                std::cout << "\n[Pairwise Algorithm Results]" << std::endl;
                std::cout << "  Algorithm Time: " << std::fixed << std::setprecision(2)
                          << pairwiseTimer.time() << " us" << std::endl;
                std::cout << "  Makespan: " << pairwiseResult.totalTime << " us" << std::endl;
                std::cout << "  Number of Steps: " << pairwiseResult.stepTimes.size() << std::endl;
                for (size_t i = 0; i < pairwiseResult.stepTimes.size(); ++i) {
                    std::cout << "    Step " << i << ": " << pairwiseResult.stepTimes[i] 
                              << " us (" << pairwiseResult.stepTransfers[i].size() << " transfers)" << std::endl;
                }
            } else {
                std::cout << "\n[Pairwise Algorithm]" << std::endl;
                std::cout << "  Skipped: Pairwise algorithm requires number of nodes to be a power of 2" << std::endl;
                std::cout << "  Current number of nodes: " << npusCount << std::endl;
            }
            
            // Run BiRing algorithm (only when the number of nodes is even)
            BaselineSolver::Result biringResult;
            bool biringExecuted = false;
            if (npusCount % 2 == 0) {
                std::cout << "\n[Running BiRing Algorithm]" << std::endl;
                std::cout << "########################################################" << std::endl;
                auto biringTimer = Timer();
                biringTimer.start();
                BiRing biring(topology, shape);
                biringResult = biring.solve(dataMatrix);
                biringTimer.stop();
                biringExecuted = true;

                std::cout << "\n[BiRing Algorithm Results]" << std::endl;
                std::cout << "  Algorithm Time: " << std::fixed << std::setprecision(2)
                          << biringTimer.time() << " us" << std::endl;
                std::cout << "  Makespan: " << biringResult.totalTime << " us" << std::endl;
                std::cout << "  Number of Steps: " << biringResult.stepTimes.size() << std::endl;
                for (size_t i = 0; i < biringResult.stepTimes.size(); ++i) {
                    std::cout << "    Step " << i << ": " << biringResult.stepTimes[i]
                              << " us (" << biringResult.stepTransfers[i].size() << " transfers)" << std::endl;
                }
            } else {
                std::cout << "\n[BiRing Algorithm]" << std::endl;
                std::cout << "  Skipped: BiRing requires an even number of nodes" << std::endl;
                std::cout << "  Current number of nodes: " << npusCount << std::endl;
            }

            // Run HalfRing+DimRotation algorithm (only for Torus topology)
            BaselineSolver::Result halfRingResult;
            bool halfRingExecuted = false;
            if (topologyType == "torus" && !shape.empty()) {
                std::cout << "\n[Running HalfRing+DimRotation Algorithm]" << std::endl;
                std::cout << "########################################################" << std::endl;
                auto halfRingTimer = Timer();
                halfRingTimer.start();
                HalfRingDimRotation halfRing(topology, shape);
                halfRingResult = halfRing.solve(dataMatrix);
                halfRingTimer.stop();
                halfRingExecuted = true;
                
                std::cout << "\n[HalfRing+DimRotation Algorithm Results]" << std::endl;
                std::cout << "  Algorithm Time: " << std::fixed << std::setprecision(2)
                          << halfRingTimer.time() << " us" << std::endl;
                std::cout << "  Makespan: " << halfRingResult.totalTime << " us" << std::endl;
            } else if (topologyType == "torus" && shape.empty()) {
                std::cout << "\n[HalfRing+DimRotation Algorithm]" << std::endl;
                std::cout << "  Skipped: Shape information required for HalfRing+DimRotation" << std::endl;
            }
            
            // Run FoldedRing+DimRotation algorithm (only for Mesh topology)
            BaselineSolver::Result foldedRingResult;
            bool foldedRingExecuted = false;
            if (topologyType == "mesh" && !shape.empty()) {
                std::cout << "\n[Running FoldedRing+DimRotation Algorithm]" << std::endl;
                std::cout << "########################################################" << std::endl;
                auto foldedRingTimer = Timer();
                foldedRingTimer.start();
                FoldedRingDimRotation foldedRing(topology, shape);
                foldedRingResult = foldedRing.solve(dataMatrix);
                foldedRingTimer.stop();
                foldedRingExecuted = true;
                
                std::cout << "\n[FoldedRing+DimRotation Algorithm Results]" << std::endl;
                std::cout << "  Algorithm Time: " << std::fixed << std::setprecision(2)
                          << foldedRingTimer.time() << " us" << std::endl;
                std::cout << "  Makespan: " << foldedRingResult.totalTime << " us" << std::endl;
            } else if (topologyType == "mesh" && shape.empty()) {
                std::cout << "\n[FoldedRing+DimRotation Algorithm]" << std::endl;
                std::cout << "  Skipped: Shape information required for FoldedRing+DimRotation" << std::endl;
            }
            
            // Comparison summary
            std::cout << "\n[Baseline Comparison Summary]" << std::endl;
            std::cout << "########################################################" << std::endl;
            std::cout << "  Bruck Makespan:    " << std::fixed << std::setprecision(2)
                      << bruckResult.totalTime << " us" << std::endl;
            std::cout << "  Spreadout Makespan: " << spreadoutResult.totalTime << " us" << std::endl;
            if (pairwiseExecuted) {
                std::cout << "  Pairwise Makespan:  " << pairwiseResult.totalTime << " us" << std::endl;
            }
            if (biringExecuted) {
                std::cout << "  BiRing Makespan:    " << biringResult.totalTime << " us" << std::endl;
            }
            if (halfRingExecuted) {
                std::cout << "  HalfRing+DimRotation Makespan: " << halfRingResult.totalTime << " us" << std::endl;
            }
            if (foldedRingExecuted) {
                std::cout << "  FoldedRing+DimRotation Makespan: " << foldedRingResult.totalTime << " us" << std::endl;
            }
            
            // Track both the best overall baseline and the best classic MPI-style baseline.
            BaselineSolver::Time overallMinTime = std::min(bruckResult.totalTime, spreadoutResult.totalTime);
            std::string overallWinner = (bruckResult.totalTime < spreadoutResult.totalTime) ? "Bruck" : "Spreadout";
            BaselineSolver::Time mpiMinTime = overallMinTime;
            std::string mpiWinner = overallWinner;

            if (pairwiseExecuted && pairwiseResult.totalTime < overallMinTime) {
                overallMinTime = pairwiseResult.totalTime;
                overallWinner = "Pairwise";
            }
            if (pairwiseExecuted && pairwiseResult.totalTime < mpiMinTime) {
                mpiMinTime = pairwiseResult.totalTime;
                mpiWinner = "Pairwise";
            }
            if (biringExecuted && biringResult.totalTime < overallMinTime) {
                overallMinTime = biringResult.totalTime;
                overallWinner = "BiRing";
            }
            if (halfRingExecuted && halfRingResult.totalTime < overallMinTime) {
                overallMinTime = halfRingResult.totalTime;
                overallWinner = "HalfRing+DimRotation";
            }
            if (foldedRingExecuted && foldedRingResult.totalTime < overallMinTime) {
                overallMinTime = foldedRingResult.totalTime;
                overallWinner = "FoldedRing+DimRotation";
            }

            std::cout << "  Winner: " << overallWinner << " (makespan: " << overallMinTime << " us)" << std::endl;
            std::cout << "  Overall Baseline Makespan: " << overallMinTime << " us" << std::endl;
            std::cout << "  MPI Baseline Winner: " << mpiWinner << " (makespan: " << mpiMinTime << " us)" << std::endl;
            std::cout << "  MPI Baseline Makespan: " << mpiMinTime << " us" << std::endl;
            std::cout << "########################################################" << std::endl;
            std::cout << "[TACOS Baselines] Done!" << std::endl;

        // ============================================================
        // Multithread mode: Run original synthesizer with multiple threads
        // ============================================================
        } else if (multithread) {
            // Create output directory if it doesn't exist
            std::filesystem::create_directories(output_dir);

            int thread_num = 32;
            std::cout << "[TACOS] Running in multithread mode with " << thread_num << " threads" << std::endl;
            std::cout << "Output directory: " << output_dir << std::endl;

            std::vector<std::thread> threads;
            std::vector<ThreadResult> results;
            std::mutex results_mutex;
            std::atomic<int> completed_threads{0};
            auto main_timer = Timer();

            // Start timer and launch threads
            main_timer.start();

            for (int i = 0; i < thread_num; ++i) {
                threads.emplace_back(runSynthesizerThread, i, topology, collective, chunkSize,
                                   topoConfig, collConfig, output_dir, faultInfo,
                                   std::ref(results), std::ref(results_mutex), std::ref(completed_threads));
            }

            // Wait for all threads to complete
            for (auto& thread : threads) {
                thread.join();
            }
            main_timer.stop();

            std::cout << "All threads completed. Parsing log files for final results..." << std::endl;

            // Parse all log files to get complete results
            results.clear();
            parseLogFiles(output_dir, results, thread_num);

            // Find best results
            if (!results.empty()) {
                auto max_solve_time = std::max_element(results.begin(), results.end(),
                    [](const ThreadResult& a, const ThreadResult& b) { return a.solve_time < b.solve_time; });

                auto min_collective_time = std::min_element(results.begin(), results.end(),
                    [](const ThreadResult& a, const ThreadResult& b) { return a.collective_time < b.collective_time; });

                std::cout << "########################################################" << std::endl;
                std::cout << "[MULTITHREAD SUMMARY]" << std::endl;
                std::cout << "Total execution time: " << main_timer.time() << " us" << std::endl;
                std::cout << "Number of completed threads: " << results.size() << "/" << thread_num << std::endl;
                std::cout << "Longest solve time: " << max_solve_time->solve_time << " us" << std::endl;
                std::cout << "  (File: " << max_solve_time->filename << ", Thread: " << (max_solve_time->thread_id + 1) << ")" << std::endl;
                std::cout << "Best collective time: " << min_collective_time->collective_time << " us" << std::endl;
                std::cout << "  (File: " << min_collective_time->filename << ", Thread: " << (min_collective_time->thread_id + 1) << ")" << std::endl;
                std::cout << "########################################################" << std::endl;
            } else {
                std::cerr << "Error: No valid results found from any thread" << std::endl;
                return 1;
            }

        // ============================================================
        // Single-threaded mode: Original synthesizer behavior
        // ============================================================
        } else {
            // Single-threaded mode - original behavior
            auto synthesizerTimer = Timer();
            
            // Print header
            std::cout << "[Glaive Collective Synthesizer]" << std::endl;
            std::cout << "########################################################" << std::endl;
            
            // Print topology info
            const auto npusCount = topology->npusCount();
            std::cout << "[Topology Information]" << std::endl;
            std::cout << "\t- Topology: " << topoConfig["topology"].get<std::string>() << std::endl;
            std::cout << "\t- NPUs Count: " << topology->npusCount() << std::endl;
            if (topoConfig.contains("shape")) {
                auto shape = topoConfig["shape"].get<std::vector<int>>();
                std::cout << "\t- Dimension: " << shape.size() << std::endl;
                std::cout << "\t- Shape: [";
                for (size_t i = 0; i < shape.size(); ++i) {
                    std::cout << shape[i] << (i + 1 < shape.size() ? ", " : "]\n");
                }
            }
            if (topoConfig.contains("latency")) {
                auto latency = topoConfig["latency"].get<std::vector<double>>();
                std::cout << "\t- Latency per dimension (ns): [";
                for (size_t i = 0; i < latency.size(); ++i) {
                    std::cout << latency[i] << (i + 1 < latency.size() ? ", " : "]\n");
                }
            }
            if (topoConfig.contains("bandwidth")) {
                auto bandwidth = topoConfig["bandwidth"].get<std::vector<double>>();
                std::cout << "\t- Bandwidth per dimension (GB/s): [";
                for (size_t i = 0; i < bandwidth.size(); ++i) {
                    std::cout << bandwidth[i] << (i + 1 < bandwidth.size() ? ", " : "]\n");
                }
            }
            // Print link fault information
            if (!faultInfo.linkFaultInfos.empty()) {
                std::cout << "\t- Link Faults (coordinate format): ";
                for (size_t i = 0; i < faultInfo.linkFaultInfos.size(); ++i) {
                    std::cout << faultInfo.linkFaultInfos[i].coordStr;
                    if (i + 1 < faultInfo.linkFaultInfos.size()) {
                        std::cout << ", ";
                    }
                }
                std::cout << std::endl;
                std::cout << "\t- Link Faults (NpuID format): ";
                for (size_t i = 0; i < faultInfo.linkFaultInfos.size(); ++i) {
                    std::cout << faultInfo.linkFaultInfos[i].npu1 << "-" << faultInfo.linkFaultInfos[i].npu2;
                    if (i + 1 < faultInfo.linkFaultInfos.size()) {
                        std::cout << ", ";
                    }
                }
                std::cout << std::endl;
            }
            // Print node fault information
            if (!faultInfo.nodeFaultInfos.empty()) {
                std::cout << "\t- Node Faults (coordinate format): ";
                for (size_t i = 0; i < faultInfo.nodeFaultInfos.size(); ++i) {
                    std::cout << faultInfo.nodeFaultInfos[i].coordStr;
                    if (i + 1 < faultInfo.nodeFaultInfos.size()) {
                        std::cout << ", ";
                    }
                }
                std::cout << std::endl;
                std::cout << "\t- Node Faults (NpuID format): ";
                for (size_t i = 0; i < faultInfo.nodeFaultInfos.size(); ++i) {
                    std::cout << faultInfo.nodeFaultInfos[i].npuID;
                    if (i + 1 < faultInfo.nodeFaultInfos.size()) {
                        std::cout << ", ";
                    }
                }
                std::cout << std::endl;
            }
            // Print straggler information
            if (!faultInfo.stragglerInfos.empty()) {
                std::cout << "\t- Stragglers (coordinate format): ";
                for (size_t i = 0; i < faultInfo.stragglerInfos.size(); ++i) {
                    std::cout << faultInfo.stragglerInfos[i].coordStr << " [BW:" 
                              << faultInfo.stragglerInfos[i].bandwidth << ", Lat:" 
                              << faultInfo.stragglerInfos[i].latency << "]";
                    if (i + 1 < faultInfo.stragglerInfos.size()) {
                        std::cout << ", ";
                    }
                }
                std::cout << std::endl;
                std::cout << "\t- Stragglers (NpuID format): ";
                for (size_t i = 0; i < faultInfo.stragglerInfos.size(); ++i) {
                    std::cout << faultInfo.stragglerInfos[i].npuID << " [BW:" 
                              << faultInfo.stragglerInfos[i].bandwidth << ", Lat:" 
                              << faultInfo.stragglerInfos[i].latency << "]";
                    if (i + 1 < faultInfo.stragglerInfos.size()) {
                        std::cout << ", ";
                    }
                }
                std::cout << std::endl;
            }
            
            // Print collective info
            const auto chunkSizeMB = chunkSize / (1 << 20);
            std::cout << "########################################################" << std::endl;
            std::cout << "[Collective Information]" << std::endl;
            std::cout << "\t- Collective: " << collConfig["collective"].get<std::string>() << std::endl;
            std::cout << "\t- Chunks Count: " << collective->chunksCount() << std::endl;
            std::cout << "\t- Chunk Size: " << chunkSize << " B";
            std::cout << " (" << chunkSizeMB << " MB)" << std::endl;
            std::cout << "\t- Chunk Factor = " << collective->getChunkFactor() << std::endl;
            std::cout << "########################################################" << std::endl;
            if (collConfig.contains("v_datasize")) {
                int collectiveNpusForDemand;
                const std::string topologyTypeForDemand = topoConfig["topology"].get<std::string>();
                if (topologyTypeForDemand == "fat-tree" || topologyTypeForDemand == "rail-optimized" ||
                    topologyTypeForDemand == "cm" || topologyTypeForDemand == "cm384") {
                    auto demandShape = topoConfig["shape"].get<std::vector<int>>();
                    collectiveNpusForDemand = std::accumulate(demandShape.begin(), demandShape.end(), 1,
                                                              std::multiplies<int>());
                } else {
                    collectiveNpusForDemand = topology->npusCount();
                }
                auto demandMatrixForSummary = parseVariableDatasizeCSV(
                    collConfig["v_datasize"].get<std::string>(),
                    collectiveNpusForDemand,
                    getBlockBytes(collConfig));
                printDemandStats(computeDenseDemandStats(demandMatrixForSummary));
            }
            
            // Run synthesizer
            synthesizerTimer.start();
            auto synthesizer = Synthesizer();
            auto collectiveTime = synthesizer.solve(topology, collective, chunkSize);
            synthesizerTimer.stop();
            
            // Print results
            auto time = synthesizerTimer.time();
            const auto timeSec = time / 1e6;
            std::cout << "Time to solve: " << time << " us";
            std::cout << " (" << timeSec << " s)" << std::endl;
            std::cout << "Collective Time: " << collectiveTime << " us" << std::endl;
            std::cout << "########################################################" << std::endl;
            std::cout << "[TACOS] Done!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}



