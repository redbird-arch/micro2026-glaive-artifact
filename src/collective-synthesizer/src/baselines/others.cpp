/*
# File name  :    others.cpp
# Author     :    Galois
# Time       :    2026/01/13 21:55:19
*/

#include <cassert>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>
#include <tacos/baselines/others.h>
#include <tacos/topology/torus.h>
#include <tacos/topology/mesh.h>
#include <tacos/thread_output.h>

using namespace tacos;

namespace {

struct RoutedPayload {
    BaselineSolver::NpuID finalDest;
    long long dataSize;
    int remainingHops;
};

struct HalfRingTransfer {
    BaselineSolver::NpuID src;
    BaselineSolver::NpuID dest;
    long long dataSize;
    int dimension;
    bool positiveDirection;
    std::vector<RoutedPayload> payloads;
};

std::vector<int> torusIdToCoord(int npuID, const std::vector<int>& shape) {
    std::vector<int> coord(shape.size(), 0);
    int stride = 1;
    for (int d = 0; d < static_cast<int>(shape.size()); ++d) {
        coord[d] = (npuID / stride) % shape[d];
        stride *= shape[d];
    }
    return coord;
}

int torusCoordToId(const std::vector<int>& coord, const std::vector<int>& shape) {
    int npuID = 0;
    int stride = 1;
    for (int d = 0; d < static_cast<int>(shape.size()); ++d) {
        npuID += coord[d] * stride;
        stride *= shape[d];
    }
    return npuID;
}

// Helper function to find Mmax and its dimension index
std::pair<int, int> findMmaxAndDimension(const std::vector<int>& shape) {
    int mmax = 0;
    int mmaxDim = -1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] > mmax) {
            mmax = shape[i];
            mmaxDim = static_cast<int>(i);
        }
    }
    return {mmax, mmaxDim};
}

// Helper function to get minimum bandwidth and maximum latency in a specific dimension
// For torus/mesh, we need to find links in the Mmax dimension
std::pair<double, double> getDimensionLinkProperties(
    std::shared_ptr<Topology> topology,
    const std::vector<int>& shape,
    int dimension) {

    double minBandwidth = std::numeric_limits<double>::max();
    double maxLatency = 0.0;

    // Calculate stride for the dimension
    int stride = 1;
    for (int d = 0; d < dimension; ++d) {
        stride *= shape[d];
    }

    int dimSize = shape[dimension];
    int npusCount = topology->npusCount();

    // Sample links in this dimension to find min bandwidth and max latency
    for (int i = 0; i < npusCount; ++i) {
        int idxInDim = (i / stride) % dimSize;
        int base = i - idxInDim * stride;

        // Check forward connection in this dimension
        // For mesh: only connect if not the last node in dimension
        // For torus: also connect last node to first (wrap around)
        if (idxInDim < dimSize - 1) {
            int neighbor = i + stride;
            if (topology->connected(i, neighbor)) {
                double bw = topology->bandwidth(i, neighbor);
                double lat = topology->latency(i, neighbor);
                if (bw < minBandwidth) {
                    minBandwidth = bw;
                }
                if (lat > maxLatency) {
                    maxLatency = lat;
                }
            }
        } else {
            int neighbor = base;
            if (topology->connected(i, neighbor)) {
                double bw = topology->bandwidth(i, neighbor);
                double lat = topology->latency(i, neighbor);
                if (bw < minBandwidth) {
                    minBandwidth = bw;
                }
                if (lat > maxLatency) {
                    maxLatency = lat;
                }
            }
        }
    }

    if (minBandwidth == std::numeric_limits<double>::max()) {
        minBandwidth = 0.0;
    }

    return {minBandwidth, maxLatency};
}

void printUtilizationSummary(
    int topoSize,
    const std::shared_ptr<Topology>& topology,
    const std::vector<std::vector<BaselineSolver::Time>>& linkBusyTime,
    const std::vector<std::vector<std::vector<std::pair<BaselineSolver::Time, BaselineSolver::Time>>>>& linkBusyIntervals,
    BaselineSolver::Result& result) {
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

    ThreadOutput::output("  Link Busy Intervals (ns) and Utilization:\n");
    for (int i = 0; i < topoSize; ++i) {
        for (int j = 0; j < topoSize; ++j) {
            const auto& intervals = linkBusyIntervals[i][j];
            if (intervals.empty()) {
                continue;
            }
            ThreadOutput::output("    Link(");
            ThreadOutput::output(topology->formatNodeName(i));
            ThreadOutput::output("->");
            ThreadOutput::output(topology->formatNodeName(j));
            ThreadOutput::output("): intervals=[");
            for (size_t k = 0; k < intervals.size(); ++k) {
                const auto seg = intervals[k];
                const long long startNs = static_cast<long long>(std::llround(seg.first * 1000.0));
                const long long endNs = static_cast<long long>(std::llround(seg.second * 1000.0));
                ThreadOutput::output("[");
                ThreadOutput::output(startNs);
                ThreadOutput::output(", ");
                ThreadOutput::output(endNs);
                ThreadOutput::output("]");
                if (k + 1 < intervals.size()) {
                    ThreadOutput::output(", ");
                }
            }
            const double util = linkBusyTime[i][j] / result.totalTime;
            ThreadOutput::output("], utilization=");
            ThreadOutput::output(static_cast<double>(util * 100.0));
            ThreadOutput::output(" %");
            ThreadOutput::output(std::endl);
        }
    }
}

}  // namespace

HalfRingDimRotation::HalfRingDimRotation(std::shared_ptr<Topology> topology,
                                         const std::vector<int>& shape) noexcept
    : BaselineSolver(topology, shape) {
}

BaselineSolver::Result HalfRingDimRotation::solve(
    const std::vector<std::vector<long long>>& dataMatrix) {

    assert(static_cast<int>(dataMatrix.size()) == npusCount_);
    assert(!shape_.empty());

    Result result;
    result.stepTransfers.clear();
    result.stepTimes.clear();

    if (npusCount_ <= 1) {
        result.totalTime = 0.0;
        return result;
    }

    auto [mmax, mmaxDim] = findMmaxAndDimension(shape_);
    long long maxTotalData = 0;
    long long totalDataBytes = 0;
    for (int i = 0; i < npusCount_; ++i) {
        long long nodeTotalData = 0;
        for (int j = 0; j < npusCount_; ++j) {
            nodeTotalData += dataMatrix[i][j];
            totalDataBytes += dataMatrix[i][j];
        }
        if (nodeTotalData > maxTotalData) {
            maxTotalData = nodeTotalData;
        }
    }

    const int dims = static_cast<int>(shape_.size());
    const int topoSize = topology_->npusCount();
    std::vector<std::vector<Time>> linkBusyUntil(
        topoSize, std::vector<Time>(topoSize, -1.0));
    std::vector<std::vector<Time>> linkBusyTime(
        topoSize, std::vector<Time>(topoSize, 0.0));
    std::vector<std::vector<std::vector<std::pair<Time, Time>>>> linkBusyIntervals(
        topoSize, std::vector<std::vector<std::pair<Time, Time>>>(topoSize));

    std::vector<std::vector<int>> coords(npusCount_);
    for (int node = 0; node < npusCount_; ++node) {
        coords[node] = torusIdToCoord(node, shape_);
    }

    std::vector<std::vector<std::vector<RoutedPayload>>> chunkResident(
        dims, std::vector<std::vector<RoutedPayload>>(npusCount_));
    for (int src = 0; src < npusCount_; ++src) {
        for (int dest = 0; dest < npusCount_; ++dest) {
            const long long dataSize = dataMatrix[src][dest];
            if (src == dest || dataSize <= 0) {
                continue;
            }
            const long long baseChunkSize = dataSize / dims;
            const long long remainder = dataSize % dims;
            for (int chunk = 0; chunk < dims; ++chunk) {
                const long long chunkBytes = baseChunkSize + (chunk < remainder ? 1 : 0);
                if (chunkBytes > 0) {
                    chunkResident[chunk][src].push_back({dest, chunkBytes, 0});
                }
            }
        }
    }

    auto neighborAlongDimension = [&](NpuID node, int dimension, int delta) {
        auto nextCoord = coords[node];
        const int dimSize = shape_[dimension];
        nextCoord[dimension] = (nextCoord[dimension] + delta + dimSize) % dimSize;
        return static_cast<NpuID>(torusCoordToId(nextCoord, shape_));
    };

    auto pushPayload = [](std::vector<RoutedPayload>& queue,
                          NpuID finalDest,
                          long long bytes,
                          int remainingHops) {
        if (bytes > 0 && remainingHops > 0) {
            queue.push_back({finalDest, bytes, remainingHops});
        }
    };

    Time currentTime = 0.0;
    int totalSteps = 0;
    int totalRounds = 0;

    struct ChunkPhaseState {
        int dimension = 0;
        int stageNum = 0;
        std::vector<std::vector<std::vector<RoutedPayload>>> positiveStageQueues;
        std::vector<std::vector<std::vector<RoutedPayload>>> negativeStageQueues;
        std::vector<std::vector<RoutedPayload>> settled;
    };

    struct ScheduledHalfRingTransfer {
        int chunk = 0;
        HalfRingTransfer transfer;
    };

    for (int phase = 0; phase < dims; ++phase) {
        std::vector<ChunkPhaseState> phaseStates(dims);
        int maxPhaseStages = 0;

        for (int chunk = 0; chunk < dims; ++chunk) {
            const int dimension = (chunk + phase) % dims;
            const int dimSize = shape_[dimension];
            auto& state = phaseStates[chunk];
            state.dimension = dimension;
            state.stageNum = dimSize <= 1 ? 0 : dimSize / 2;
            state.positiveStageQueues.assign(
                state.stageNum + 1, std::vector<std::vector<RoutedPayload>>(npusCount_));
            state.negativeStageQueues.assign(
                state.stageNum + 1, std::vector<std::vector<RoutedPayload>>(npusCount_));
            state.settled.assign(npusCount_, {});
            maxPhaseStages = std::max(maxPhaseStages, state.stageNum);

            for (int node = 0; node < npusCount_; ++node) {
                for (const auto& payload : chunkResident[chunk][node]) {
                    const auto& currentCoord = coords[node];
                    const auto& finalCoord = coords[payload.finalDest];
                    const int positiveHops =
                        (finalCoord[dimension] - currentCoord[dimension] + dimSize) % dimSize;
                    const int negativeHops =
                        (currentCoord[dimension] - finalCoord[dimension] + dimSize) % dimSize;

                    if (positiveHops == 0) {
                        state.settled[node].push_back(payload);
                        continue;
                    }

                    if (positiveHops < negativeHops) {
                        pushPayload(
                            state.positiveStageQueues[positiveHops][node],
                            payload.finalDest,
                            payload.dataSize,
                            positiveHops);
                    } else if (negativeHops < positiveHops) {
                        pushPayload(
                            state.negativeStageQueues[negativeHops][node],
                            payload.finalDest,
                            payload.dataSize,
                            negativeHops);
                    } else {
                        const long long positiveBytes = (payload.dataSize + 1) / 2;
                        const long long negativeBytes = payload.dataSize / 2;
                        pushPayload(
                            state.positiveStageQueues[positiveHops][node],
                            payload.finalDest,
                            positiveBytes,
                            positiveHops);
                        pushPayload(
                            state.negativeStageQueues[negativeHops][node],
                            payload.finalDest,
                            negativeBytes,
                            negativeHops);
                    }
                }
            }
        }

        for (int stage = 1; stage <= maxPhaseStages; ++stage) {
            for (int substage = 0; substage < stage; ++substage) {
                ++totalRounds;
                const Time stepStartTime = currentTime;
                std::vector<std::pair<NpuID, NpuID>> stepTransfers;
                std::vector<ScheduledHalfRingTransfer> transfers;

                for (int chunk = 0; chunk < dims; ++chunk) {
                    auto& state = phaseStates[chunk];
                    if (stage > state.stageNum) {
                        continue;
                    }
                    const int dimension = state.dimension;
                    auto& positiveQueues = state.positiveStageQueues[stage];
                    auto& negativeQueues = state.negativeStageQueues[stage];
                    for (int node = 0; node < npusCount_; ++node) {
                        if (!positiveQueues[node].empty()) {
                            long long aggregatedBytes = 0;
                            for (const auto& payload : positiveQueues[node]) {
                                aggregatedBytes += payload.dataSize;
                            }
                            if (aggregatedBytes > 0) {
                                const auto hopDest = neighborAlongDimension(node, dimension, +1);
                                transfers.push_back(
                                    {chunk,
                                     {node,
                                      hopDest,
                                      aggregatedBytes,
                                      dimension,
                                      true,
                                      positiveQueues[node]}});
                                stepTransfers.push_back({node, hopDest});
                            }
                        }

                        if (!negativeQueues[node].empty()) {
                            long long aggregatedBytes = 0;
                            for (const auto& payload : negativeQueues[node]) {
                                aggregatedBytes += payload.dataSize;
                            }
                            if (aggregatedBytes > 0) {
                                const auto hopDest = neighborAlongDimension(node, dimension, -1);
                                transfers.push_back(
                                    {chunk,
                                     {node,
                                      hopDest,
                                      aggregatedBytes,
                                      dimension,
                                      false,
                                      negativeQueues[node]}});
                                stepTransfers.push_back({node, hopDest});
                            }
                        }
                    }
                }

                if (transfers.empty()) {
                    continue;
                }

                std::vector<std::vector<std::vector<RoutedPayload>>> nextPositiveQueues(
                    dims, std::vector<std::vector<RoutedPayload>>(npusCount_));
                std::vector<std::vector<std::vector<RoutedPayload>>> nextNegativeQueues(
                    dims, std::vector<std::vector<RoutedPayload>>(npusCount_));
                Time stepEndTime = stepStartTime;

                for (const auto& scheduled : transfers) {
                    const auto& transfer = scheduled.transfer;
                    if (!topology_->connected(transfer.src, transfer.dest)) {
                        continue;
                    }
                    Time hopStart = stepStartTime;
                    if (linkBusyUntil[transfer.src][transfer.dest] >= 0.0) {
                        hopStart = std::max(hopStart, linkBusyUntil[transfer.src][transfer.dest]);
                    }
                    // HalfR+DR chunks and reassembles payloads at every store-and-forward hop.
                    // Model packetization/reassembly overhead as five extra link latencies
                    // on top of the normal single-hop alpha-beta time: 6*latency + bytes/bw.
                    Time hopTime = calculateTransferTime(transfer.src, transfer.dest, transfer.dataSize);
                    hopTime += 5.0 * topology_->latency(transfer.src, transfer.dest) / 1000.0;
                    if (std::isnan(hopTime) || std::isinf(hopTime) || hopTime < 0.0) {
                        hopTime = 0.0;
                    }
                    const Time hopEnd = hopStart + hopTime;
                    linkBusyUntil[transfer.src][transfer.dest] = hopEnd;
                    linkBusyTime[transfer.src][transfer.dest] += hopTime;
                    linkBusyIntervals[transfer.src][transfer.dest].push_back({hopStart, hopEnd});
                    stepEndTime = std::max(stepEndTime, hopEnd);

                    auto& destinationQueue =
                        transfer.positiveDirection ? nextPositiveQueues[scheduled.chunk][transfer.dest]
                                                   : nextNegativeQueues[scheduled.chunk][transfer.dest];
                    auto& destinationSettled = phaseStates[scheduled.chunk].settled[transfer.dest];
                    for (const auto& payload : transfer.payloads) {
                        if (payload.remainingHops <= 1) {
                            destinationSettled.push_back({payload.finalDest, payload.dataSize, 0});
                        } else {
                            destinationQueue.push_back(
                                {payload.finalDest, payload.dataSize, payload.remainingHops - 1});
                        }
                    }
                }

                currentTime = stepEndTime;
                const Time stepTime = currentTime - stepStartTime;
                result.stepTimes.push_back(stepTime);
                result.stepTransfers.push_back(stepTransfers);
                ++totalSteps;

                ThreadOutput::output("[HalfRing+DimRotation Step ");
                ThreadOutput::output(totalSteps - 1);
                ThreadOutput::output("] Phase ");
                ThreadOutput::output(phase);
                ThreadOutput::output(", Stage ");
                ThreadOutput::output(stage);
                ThreadOutput::output(", Sub-stage ");
                ThreadOutput::output(substage);
                ThreadOutput::output(", Time: ");
                ThreadOutput::output(stepTime);
                ThreadOutput::output(" us, Transfers: ");
                ThreadOutput::output(static_cast<int>(stepTransfers.size()));
                ThreadOutput::output(std::endl);

                for (int chunk = 0; chunk < dims; ++chunk) {
                    auto& state = phaseStates[chunk];
                    if (stage > state.stageNum) {
                        continue;
                    }
                    state.positiveStageQueues[stage] = std::move(nextPositiveQueues[chunk]);
                    state.negativeStageQueues[stage] = std::move(nextNegativeQueues[chunk]);
                }
                for (int i = 0; i < topoSize; ++i) {
                    for (int j = 0; j < topoSize; ++j) {
                        if (linkBusyUntil[i][j] <= currentTime) {
                            linkBusyUntil[i][j] = -1.0;
                        }
                    }
                }
            }
        }

        for (int chunk = 0; chunk < dims; ++chunk) {
            chunkResident[chunk] = std::move(phaseStates[chunk].settled);
        }
    }

    result.totalTime = currentTime;
    printUtilizationSummary(
        topoSize, topology_, linkBusyTime, linkBusyIntervals, result);

    int unfinishedPayloads = 0;
    for (int chunk = 0; chunk < dims; ++chunk) {
        for (int node = 0; node < npusCount_; ++node) {
            for (const auto& payload : chunkResident[chunk][node]) {
                if (payload.finalDest != node) {
                    ++unfinishedPayloads;
                }
            }
        }
    }

    ThreadOutput::output("[HalfRing+DimRotation Algorithm]");
    ThreadOutput::output(std::endl);
    ThreadOutput::output("  Topology: Torus");
    ThreadOutput::output(std::endl);
    ThreadOutput::output("  Shape: [");
    for (size_t i = 0; i < shape_.size(); ++i) {
        ThreadOutput::output(shape_[i]);
        if (i + 1 < shape_.size()) {
            ThreadOutput::output(", ");
        }
    }
    ThreadOutput::output("]");
    ThreadOutput::output(std::endl);
    ThreadOutput::output("  Mmax: ");
    ThreadOutput::output(mmax);
    ThreadOutput::output(" (dimension ");
    ThreadOutput::output(mmaxDim);
    ThreadOutput::output(")");
    ThreadOutput::output(std::endl);
    ThreadOutput::output("  Total data per node (max): ");
    ThreadOutput::output(static_cast<double>(maxTotalData));
    ThreadOutput::output(" bytes");
    ThreadOutput::output(std::endl);
    ThreadOutput::output("  Total communicated bytes: ");
    ThreadOutput::output(static_cast<double>(totalDataBytes));
    ThreadOutput::output(" bytes");
    ThreadOutput::output(std::endl);
    ThreadOutput::output("  DimRotation chunks: ");
    ThreadOutput::output(dims);
    ThreadOutput::output(" chunks, schedule[chunk][phase]=(chunk+phase)%dims");
    ThreadOutput::output(std::endl);
    ThreadOutput::output("  Total rounds: ");
    ThreadOutput::output(totalRounds);
    ThreadOutput::output(std::endl);
    ThreadOutput::output("  Total steps with traffic: ");
    ThreadOutput::output(totalSteps);
    ThreadOutput::output(std::endl);
    ThreadOutput::output("  Unfinished payloads after routing: ");
    ThreadOutput::output(unfinishedPayloads);
    ThreadOutput::output(std::endl);
    ThreadOutput::output("  Total Time: ");
    ThreadOutput::output(result.totalTime);
    ThreadOutput::output(" us");
    ThreadOutput::output(std::endl);

    return result;
}

FoldedRingDimRotation::FoldedRingDimRotation(std::shared_ptr<Topology> topology,
                                             const std::vector<int>& shape) noexcept
    : BaselineSolver(topology, shape) {
}

BaselineSolver::Result FoldedRingDimRotation::solve(
    const std::vector<std::vector<long long>>& dataMatrix) {

    assert(static_cast<int>(dataMatrix.size()) == npusCount_);
    assert(!shape_.empty());

    Result result;
    result.stepTransfers.clear();
    result.stepTimes.clear();

    auto [mmax, mmaxDim] = findMmaxAndDimension(shape_);

    long long maxTotalData = 0;
    for (int i = 0; i < npusCount_; ++i) {
        long long nodeTotalData = 0;
        for (int j = 0; j < npusCount_; ++j) {
            nodeTotalData += dataMatrix[i][j];
        }
        if (nodeTotalData > maxTotalData) {
            maxTotalData = nodeTotalData;
        }
    }
    double S = static_cast<double>(maxTotalData);

    auto [minBandwidth, maxLatency] = getDimensionLinkProperties(
        topology_, shape_, mmaxDim);

    if (minBandwidth == 0.0) {
        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                if (topology_->connected(i, j)) {
                    double bw = topology_->bandwidth(i, j);
                    double lat = topology_->latency(i, j);
                    if (bw < minBandwidth || minBandwidth == 0.0) {
                        minBandwidth = bw;
                    }
                    if (lat > maxLatency) {
                        maxLatency = lat;
                    }
                }
            }
        }
    }

    double B = minBandwidth * (1024.0 * 1024.0 * 1024.0 / 1e6);
    double alpha = maxLatency / 1000.0;

    int dim_num = shape_.size();
    double totalTime = ((mmax - 1) * S) / (2.0 * B) + (mmax - 1) * dim_num * alpha;

    result.totalTime = totalTime;
    result.stepTimes.push_back(totalTime);
    result.stepTransfers.push_back({});

    ThreadOutput::output("[FoldedRing+DimRotation Algorithm]");
    ThreadOutput::output(std::endl);
    ThreadOutput::output("  Topology: Mesh");
    ThreadOutput::output(std::endl);
    ThreadOutput::output("  Shape: [");
    for (size_t i = 0; i < shape_.size(); ++i) {
        ThreadOutput::output(shape_[i]);
        if (i + 1 < shape_.size()) {
            ThreadOutput::output(", ");
        }
    }
    ThreadOutput::output("]");
    ThreadOutput::output(std::endl);
    ThreadOutput::output("  Mmax: ");
    ThreadOutput::output(mmax);
    ThreadOutput::output(" (dimension ");
    ThreadOutput::output(mmaxDim);
    ThreadOutput::output(")");
    ThreadOutput::output(std::endl);
    ThreadOutput::output("  Total data per node (S): ");
    ThreadOutput::output(S);
    ThreadOutput::output(" bytes");
    ThreadOutput::output(std::endl);
    ThreadOutput::output("  Bandwidth (B): ");
    ThreadOutput::output(minBandwidth);
    ThreadOutput::output(" GiB/s (");
    ThreadOutput::output(B);
    ThreadOutput::output(" bytes/us)");
    ThreadOutput::output(std::endl);
    ThreadOutput::output("  Latency (alpha): ");
    ThreadOutput::output(alpha);
    ThreadOutput::output(" us");
    ThreadOutput::output(std::endl);
    ThreadOutput::output("  Total Time: ");
    ThreadOutput::output(totalTime);
    ThreadOutput::output(" us");
    ThreadOutput::output(std::endl);

    return result;
}
