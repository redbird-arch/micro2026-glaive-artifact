/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <tacos/baselines/biring.h>
#include <tacos/thread_output.h>

using namespace tacos;

namespace {

struct RoutedMessage {
    int finalDest;
    long long dataSize;
    int remainingLogicalHops;
};

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
    std::vector<RoutedMessage> payloads;
    std::vector<HopInfo> hopInfos;
    bool clockwise;
};

void printTransfer(const std::shared_ptr<Topology>& topology, const Transfer& transfer) {
    ThreadOutput::output("  ");
    ThreadOutput::output(topology->formatNodeName(transfer.src));
    ThreadOutput::output(" -> ");
    ThreadOutput::output(topology->formatNodeName(transfer.dest));
    ThreadOutput::output(" (");
    ThreadOutput::output(transfer.dataSize);
    ThreadOutput::output(" bytes, ");
    ThreadOutput::output(static_cast<int>(transfer.payloads.size()));
    ThreadOutput::output(" forwarded messages, direction=");
    ThreadOutput::output(transfer.clockwise ? "clockwise" : "counter-clockwise");

    if (transfer.path.size() > 2) {
        ThreadOutput::output(", path: ");
        for (size_t i = 0; i < transfer.path.size(); ++i) {
            ThreadOutput::output(topology->formatNodeName(transfer.path[i]));
            if (i + 1 < transfer.path.size()) {
                ThreadOutput::output("->");
            }
        }
    }

    ThreadOutput::output(")");
    ThreadOutput::output(std::endl);

    for (size_t hopIdx = 0; hopIdx < transfer.hopInfos.size(); ++hopIdx) {
        const auto& hopInfo = transfer.hopInfos[hopIdx];
        ThreadOutput::output("    Hop ");
        ThreadOutput::output(hopIdx);
        ThreadOutput::output(": ");
        ThreadOutput::output(topology->formatNodeName(hopInfo.hopSrc));
        ThreadOutput::output(" -> ");
        ThreadOutput::output(topology->formatNodeName(hopInfo.hopDest));
        ThreadOutput::output(" (");
        ThreadOutput::output(transfer.dataSize);
        ThreadOutput::output(" bytes, time: ");
        ThreadOutput::output(hopInfo.hopEnd - hopInfo.hopStart);
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

BiRing::BiRing(std::shared_ptr<Topology> topology,
               const std::vector<int>& shape) noexcept
    : BaselineSolver(topology, shape) {
}

BaselineSolver::Result BiRing::solve(const std::vector<std::vector<long long>>& dataMatrix) {
    assert(static_cast<int>(dataMatrix.size()) == npusCount_);

    Result result;
    result.stepTransfers.clear();
    result.stepTimes.clear();

    if (npusCount_ <= 1) {
        result.totalTime = 0.0;
        return result;
    }

    const int topoSize = topology_->npusCount();
    std::vector<std::vector<BaselineSolver::Time>> linkBusyUntil(
        topoSize, std::vector<BaselineSolver::Time>(topoSize, -1.0));
    std::vector<std::vector<BaselineSolver::Time>> linkBusyTime(
        topoSize, std::vector<BaselineSolver::Time>(topoSize, 0.0));
    std::vector<std::vector<std::vector<std::pair<BaselineSolver::Time, BaselineSolver::Time>>>> linkBusyIntervals(
        topoSize, std::vector<std::vector<std::pair<BaselineSolver::Time, BaselineSolver::Time>>>(topoSize));

    std::vector<std::vector<RoutedMessage>> clockwiseQueues(npusCount_);
    std::vector<std::vector<RoutedMessage>> counterClockwiseQueues(npusCount_);
    int maxRounds = 0;

    for (BaselineSolver::NpuID src = 0; src < npusCount_; ++src) {
        for (BaselineSolver::NpuID dest = 0; dest < npusCount_; ++dest) {
            const long long dataSize = dataMatrix[src][dest];
            if (src == dest || dataSize <= 0) {
                continue;
            }

            const int clockwiseHops = (dest - src + npusCount_) % npusCount_;
            const int counterClockwiseHops = (src - dest + npusCount_) % npusCount_;
            const bool useClockwise =
                (clockwiseHops < counterClockwiseHops) ||
                (clockwiseHops == counterClockwiseHops && (src % 2 == 0));

            const int hops = useClockwise ? clockwiseHops : counterClockwiseHops;
            maxRounds = std::max(maxRounds, hops);

            RoutedMessage msg{dest, dataSize, hops};
            if (useClockwise) {
                clockwiseQueues[src].push_back(msg);
            } else {
                counterClockwiseQueues[src].push_back(msg);
            }
        }
    }

    BaselineSolver::Time currentTime = 0.0;

    for (int round = 0; round < maxRounds; ++round) {
        const BaselineSolver::Time stepStartTime = currentTime;
        std::vector<std::pair<BaselineSolver::NpuID, BaselineSolver::NpuID>> stepTransfers;
        std::vector<Transfer> allTransfers;

        for (BaselineSolver::NpuID node = 0; node < npusCount_; ++node) {
            if (!clockwiseQueues[node].empty()) {
                long long aggregatedBytes = 0;
                for (const auto& msg : clockwiseQueues[node]) {
                    aggregatedBytes += msg.dataSize;
                }
                if (aggregatedBytes > 0) {
                    const auto nextNode = static_cast<BaselineSolver::NpuID>((node + 1) % npusCount_);
                    Transfer transfer;
                    transfer.src = node;
                    transfer.dest = nextNode;
                    transfer.dataSize = aggregatedBytes;
                    transfer.path = getXYPath(node, nextNode);
                    transfer.payloads = clockwiseQueues[node];
                    transfer.clockwise = true;
                    allTransfers.push_back(transfer);
                    stepTransfers.push_back({transfer.src, transfer.dest});
                }
            }

            if (!counterClockwiseQueues[node].empty()) {
                long long aggregatedBytes = 0;
                for (const auto& msg : counterClockwiseQueues[node]) {
                    aggregatedBytes += msg.dataSize;
                }
                if (aggregatedBytes > 0) {
                    const auto nextNode =
                        static_cast<BaselineSolver::NpuID>((node - 1 + npusCount_) % npusCount_);
                    Transfer transfer;
                    transfer.src = node;
                    transfer.dest = nextNode;
                    transfer.dataSize = aggregatedBytes;
                    transfer.path = getXYPath(node, nextNode);
                    transfer.payloads = counterClockwiseQueues[node];
                    transfer.clockwise = false;
                    allTransfers.push_back(transfer);
                    stepTransfers.push_back({transfer.src, transfer.dest});
                }
            }
        }

        std::sort(allTransfers.begin(), allTransfers.end(), [](const Transfer& lhs, const Transfer& rhs) {
            if (lhs.src != rhs.src) {
                return lhs.src < rhs.src;
            }
            if (lhs.clockwise != rhs.clockwise) {
                return lhs.clockwise > rhs.clockwise;
            }
            return lhs.dest < rhs.dest;
        });

        for (auto& transfer : allTransfers) {
            BaselineSolver::Time transferStartTime = stepStartTime;
            for (size_t hopIdx = 0; hopIdx + 1 < transfer.path.size(); ++hopIdx) {
                const auto hopSrc = transfer.path[hopIdx];
                const auto hopDest = transfer.path[hopIdx + 1];
                if (!topology_->connected(hopSrc, hopDest)) {
                    continue;
                }
                BaselineSolver::Time linkAvailableTime = (linkBusyUntil[hopSrc][hopDest] < 0.0)
                    ? transferStartTime
                    : std::max(transferStartTime, linkBusyUntil[hopSrc][hopDest]);
                BaselineSolver::Time hopTime = calculateTransferTime(hopSrc, hopDest, transfer.dataSize);
                if (std::isnan(hopTime) || std::isinf(hopTime) || hopTime < 0.0) {
                    hopTime = 0.0;
                }
                transfer.hopInfos.push_back({hopSrc, hopDest, linkAvailableTime, linkAvailableTime + hopTime});
                linkBusyUntil[hopSrc][hopDest] = linkAvailableTime + hopTime;
                linkBusyIntervals[hopSrc][hopDest].push_back({linkAvailableTime, linkAvailableTime + hopTime});
                linkBusyTime[hopSrc][hopDest] += (hopTime);
                transferStartTime = linkAvailableTime + hopTime;
            }
        }

        BaselineSolver::Time stepEndTime = stepStartTime;
        for (const auto& transfer : allTransfers) {
            for (const auto& hopInfo : transfer.hopInfos) {
                if (std::isnan(hopInfo.hopEnd) || std::isinf(hopInfo.hopEnd)) {
                    continue;
                }
                stepEndTime = std::max(stepEndTime, hopInfo.hopEnd);
            }
        }

        currentTime = stepEndTime;
        BaselineSolver::Time stepTime = currentTime - stepStartTime;
        if (std::isnan(stepTime) || std::isinf(stepTime)) {
            stepTime = 0.0;
            currentTime = stepStartTime;
        }

        result.stepTimes.push_back(stepTime);
        result.stepTransfers.push_back(stepTransfers);

        ThreadOutput::output("[BiRing Step ");
        ThreadOutput::output(round);
        ThreadOutput::output("] Time: ");
        ThreadOutput::output(stepTime);
        ThreadOutput::output(" us, Transfers: ");
        ThreadOutput::output(static_cast<int>(stepTransfers.size()));
        ThreadOutput::output(std::endl);

        for (const auto& transfer : allTransfers) {
            printTransfer(topology_, transfer);
        }

        std::vector<std::vector<RoutedMessage>> nextClockwise(npusCount_);
        std::vector<std::vector<RoutedMessage>> nextCounterClockwise(npusCount_);
        for (const auto& transfer : allTransfers) {
            const auto nextNode = transfer.dest;
            for (const auto& payload : transfer.payloads) {
                if (payload.remainingLogicalHops <= 1) {
                    continue;
                }
                RoutedMessage forwarded = payload;
                forwarded.remainingLogicalHops--;
                if (transfer.clockwise) {
                    nextClockwise[nextNode].push_back(forwarded);
                } else {
                    nextCounterClockwise[nextNode].push_back(forwarded);
                }
            }
        }

        clockwiseQueues.swap(nextClockwise);
        counterClockwiseQueues.swap(nextCounterClockwise);

        for (int i = 0; i < topoSize; ++i) {
            for (int j = 0; j < topoSize; ++j) {
                if (linkBusyUntil[i][j] <= currentTime) {
                    linkBusyUntil[i][j] = -1.0;
                }
            }
        }
    }

    result.totalTime = currentTime;
    printUtilizationSummary(topoSize, topology_, linkBusyTime, linkBusyIntervals, result);
    return result;
}
