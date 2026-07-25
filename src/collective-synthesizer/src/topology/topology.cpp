/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include <algorithm>
#include <cassert>
#include <queue>
#include <tacos/topology/topology.h>

using namespace tacos;

Topology::Topology() noexcept = default;

void Topology::setNpusCount(const int npusCount) noexcept {
    assert(npusCount > 0);

    // set npusCount
    npusCount_ = npusCount;

    // allocate memory
    connected_ = decltype(connected_)(npusCount, std::vector<bool>(npusCount, false));
    latencies_ = decltype(latencies_)(npusCount, std::vector<Latency>(npusCount, 0));
    bandwidths_ = decltype(bandwidths_)(npusCount, std::vector<Bandwidth>(npusCount, 0));

    for (auto dest = 0; dest < npusCount; ++dest) {
        backtrackMap_[dest] = {};
    }
}

Topology::Bandwidth Topology::bandwidth(NpuID src, NpuID dest) const noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);
    assert(connected_[src][dest]);

    return bandwidths_[src][dest];
}

Topology::Latency Topology::latency(NpuID src, NpuID dest) const noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);
    assert(connected_[src][dest]);

    return latencies_[src][dest];
}

std::vector<Topology::NpuID> Topology::backtrack(const NpuID dest) const noexcept {
    assert(0 <= dest && dest < npusCount_);
    assert(backtrackMap_.size() == npusCount_);

    return backtrackMap_.at(dest);
}

void Topology::connect(const NpuID src,
                        const NpuID dest,
                        Latency latency,
                        Bandwidth bandwidth,
                        const bool bidirectional) noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);
    assert(bandwidth > 0);
    assert(latency >= 0);

    // connect src -> dest
    connected_[src][dest] = true;
    bandwidths_[src][dest] = bandwidth;
    latencies_[src][dest] = latency;
    backtrackMap_[dest].push_back(src);

    if (bidirectional) {
        // connect dest -> src (if bi-directional)
        connect(dest, src, latency, bandwidth, false);
    }
}

bool Topology::connected(const NpuID src, const NpuID dest) const noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);

    return connected_[src][dest];
}

void Topology::disconnect(const NpuID src, const NpuID dest, const bool bidirectional) noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);

    // disconnect src -> dest
    if (connected_[src][dest]) {
        connected_[src][dest] = false;
        bandwidths_[src][dest] = 0;
        latencies_[src][dest] = 0;
        
        // remove from backtrack map
        auto& backtrackList = backtrackMap_[dest];
        backtrackList.erase(
            std::remove(backtrackList.begin(), backtrackList.end(), src),
            backtrackList.end()
        );
    }

    if (bidirectional) {
        // disconnect dest -> src (if bi-directional)
        disconnect(dest, src, false);
    }
}

void Topology::updateLinkProperties(const NpuID src, const NpuID dest, const Bandwidth bandwidth, const Latency latency, const bool bidirectional) noexcept {
    assert(0 <= src && src < npusCount_);
    assert(0 <= dest && dest < npusCount_);
    assert(bandwidth > 0);
    assert(latency >= 0);

    // Update src -> dest link properties if it exists
    if (connected_[src][dest]) {
        bandwidths_[src][dest] = bandwidth;
        latencies_[src][dest] = latency;
    }

    if (bidirectional) {
        // Update dest -> src link properties (if bi-directional)
        updateLinkProperties(dest, src, bandwidth, latency, false);
    }
}

int Topology::npusCount() const noexcept {
    assert(npusCount_ > 0);

    return npusCount_;
}

int Topology::diameter() const noexcept {
    assert(npusCount_ > 0);
    auto maxDistance = 0;
    std::vector<int> distance(npusCount_, -1);
    std::queue<NpuID> q; // BFS queue

    for (auto start = 0; start < npusCount_; ++start) {
        std::fill(distance.begin(), distance.end(), -1);
        while (!q.empty()) {
            q.pop();
        }

        distance[start] = 0;
        q.push(start);

        while (!q.empty()) {
            const auto node = q.front();
            q.pop();

            for (auto dest = 0; dest < npusCount_; ++dest) {
                if (!connected_[node][dest] || distance[dest] != -1) {
                    continue;
                }

                distance[dest] = distance[node] + 1;
                maxDistance = std::max(maxDistance, distance[dest]);
                q.push(dest);
            }
        }
    }
    return maxDistance;
}

std::vector<int> Topology::degrees() const noexcept {
    assert(npusCount_ > 0);
    auto nodeDegrees = std::vector<int>(npusCount_, 0);
    for (auto src = 0; src < npusCount_; ++src) {
        auto degree = 0;
        for (auto dest = 0; dest < npusCount_; ++dest) {
            degree += connected_[src][dest] ? 1 : 0;
        }
        nodeDegrees[src] = degree;
    }
    return nodeDegrees;
}
