/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/
#include <cassert>
#include <limits>
#include <iostream>
#include <queue>
#include <map>
#include <tacos/synthesizer/synthesizer.h>
#include "thread_output.h"

using namespace tacos;

Synthesizer::Synthesizer() noexcept = default;

// Set thread-local output stream
void Synthesizer::setThreadOutputStream(std::ostream& os) {
    ThreadOutput::setOutputStream(os);
}

Synthesizer::Time Synthesizer::solve(const std::shared_ptr<Topology> topology,
                                     const std::shared_ptr<Collective> collective,
                                     ChunkSize chunkSize) noexcept {
    assert(chunkSize > 0);

    // initialize the synthesizer
    initialize_(topology, collective, chunkSize);

    // mark trivial initial case
    // that is, chunks in preconditions are already at their sources
    markPrecondition_();

    // then, repeat the link-chunk matching process
    while (!eventQueue_.empty()) {
        // get current event time
        currentTime_ = eventQueue_.pop();

        // first, filter out unsatisfied postconditions
        // this is required when choosing the chunk replacement candidates
        // during the expansion of the TEN
        auto postconditionMap = filterPostcondition_();
        mergeTempConditions_(&postconditionMap);

        // then, expand the TEN
        // this method will also process and update the arrival of chunks
        // at the current timestep, and will change the unsatisfied postconditions
        expandTenTimestep_(&postconditionMap);

        // after the expansion of the TEN, check if there are any unsatisfied postconditions
        auto postcondition = shufflePostcondition_(postconditionMap);

        if (postcondition.empty()) {
            // no unsatisfied postcondition left to map
            // if so, just proceed to the next event
            // until all chunks arrive at their destinations
            continue;
        }

        // Multi-sequence rescheduling: try multiple orders and pick the best
        // Save current state for simulation (base state)
        auto baseAvailable = std::vector<std::vector<bool>>(npusCount);
        for (auto src = 0; src < npusCount; ++src) {
            baseAvailable[src].resize(npusCount);
            for (auto dest = 0; dest < npusCount; ++dest) {
                baseAvailable[src][dest] = ten_->available(src, dest);
            }
        }
        const auto baseChunkMap = chunkMap_;
        const auto baseTempConditions = tempConditions_;

        // Generate candidate orders
        auto bestOrder = postcondition;
        auto bestUtilization = 0;
        int attempts = 0;

        // Try multiple orders
        while (attempts < maxRescheduleAttempts_) {
            // Reset state for this simulation
            auto availableCopy = baseAvailable;
            auto chunkMapCopy = baseChunkMap;
            auto tempConditionsCopy = baseTempConditions;

            // Generate order for this attempt
            std::vector<Condition> candidateOrder;
            if (attempts == 0) {
                // First attempt: use the original sorted order
                candidateOrder = postcondition;
            } else {
                // Subsequent attempts: shuffle within priority buckets
                // Group by (distance, degree) to maintain priority structure
                struct GroupKey {
                    int distance;
                    int degree;
                    bool operator<(const GroupKey& other) const {
                        if (distance != other.distance) return distance > other.distance;
                        return degree < other.degree;
                    }
                };
                std::map<GroupKey, std::vector<size_t>> groups;
                for (size_t i = 0; i < postcondition.size(); ++i) {
                    const auto& [chunk, dest] = postcondition[i];
                    const auto distance = computeConditionDistance_(chunk, dest);
                    const auto degree = (dest >= 0 && dest < static_cast<NpuID>(nodeDegrees_.size()))
                                            ? nodeDegrees_[dest]
                                            : std::numeric_limits<int>::max();
                    groups[{distance, degree}].push_back(i);
                }

                // Shuffle within each group
                for (auto& [key, indices] : groups) {
                    std::shuffle(indices.begin(), indices.end(), randomEngine);
                }

                // Reconstruct order
                for (const auto& [key, indices] : groups) {
                    for (const auto idx : indices) {
                        candidateOrder.push_back(postcondition[idx]);
                    }
                }
            }

            // Simulate this order
            const auto utilization = simulateLinkChunkMatching_(candidateOrder, availableCopy, chunkMapCopy, tempConditionsCopy);
            if (utilization > bestUtilization) {
                bestOrder = candidateOrder;
                bestUtilization = utilization;
            }
            attempts++;
        }

        // Execute the best order
        for (const auto [chunk, dest] : bestOrder) {
            // Execute link-chunk matching for both final and intermediate destinations
            linkChunkMatching_(chunk, dest);
        }
    }

    // all matching has been finished
    // return measured collective_ time
    assert(collectiveTime_ > 0);
    return collectiveTime_;
}

void Synthesizer::initialize_(const std::shared_ptr<Topology> topology,
                              const std::shared_ptr<Collective> collective,
                              const ChunkSize chunkSize) noexcept {
    // reset the event queue
    eventQueue_.reset();
    currentTime_ = 0;

    // set topology and collective
    topology_ = topology;
    collective_ = collective;

    // set variables
    npusCount = topology_->npusCount();
    chunksCount_ = collective_->chunksCount();
    diameter_ = topology_->diameter();
    nodeDegrees_ = topology_->degrees();
    tempConditions_.clear();

    // construct TEN from the topology
    ten_ = std::make_unique<TimeExpandedNetwork>(*topology_, chunkSize);

    // construct chunkMap_
    // Find the maximum chunk ID to ensure chunkMap_ is large enough
    // (chunks may not be consecutive after removing faulty nodes)
    int maxChunkID = 0;
    const auto preconditionMap = collective_->getPrecondition();
    for (const auto& [chunk, _] : preconditionMap) {
        maxChunkID = std::max(maxChunkID, chunk);
    }
    const auto postconditionMap = collective_->getPostcondition();
    for (const auto& [chunk, _] : postconditionMap) {
        maxChunkID = std::max(maxChunkID, chunk);
    }
    // Ensure chunkMap_ is large enough for all chunk IDs
    chunkMap_.assign(maxChunkID + 1, std::vector<bool>(npusCount, false));
}

void Synthesizer::markPrecondition_() noexcept {
    // for every chunk, mark its source NPU as true in the chunkMap_
    // Iterate over actual chunks in the collective (chunks may not be consecutive after removing faulty nodes)
    const auto preconditionMap = collective_->getPrecondition();
    for (const auto& [chunk, src] : preconditionMap) {
        if (chunk >= 0 && chunk < static_cast<int>(chunkMap_.size())) {
            chunkMap_[chunk][src] = true;
        }
    }
}

Synthesizer::PostconditionMap Synthesizer::filterPostcondition_() const noexcept {
    auto postconditionMap = PostconditionMap();

    // iterate over actual chunks in the collective (chunks may not be consecutive after removing faulty nodes)
    const auto postconditionMapAll = collective_->getPostcondition();
    for (const auto& [chunk, dests] : postconditionMapAll) {
        if (chunk >= 0 && chunk < static_cast<int>(chunkMap_.size())) {
            // check which destination NPUs have not yet received the chunk
            for (const auto dest : dests) {
                if (!chunkMap_[chunk][dest]) {
                    postconditionMap[dest].insert(chunk);
                }
            }
        }
    }

    return postconditionMap;
}

std::vector<Synthesizer::Condition> Synthesizer::shufflePostcondition_(
    const PostconditionMap& postconditionMap) noexcept {
    auto postcondition = std::vector<Condition>();

    struct ConditionPriority {
        Condition condition;
        int degree;
        int distance;
    };
    auto prioritized = std::vector<ConditionPriority>();

    for (const auto& [dest, chunks] : postconditionMap) {
        for (const auto chunk : chunks) {
            const auto distance = computeConditionDistance_(chunk, dest);
            const auto degree = (dest >= 0 && dest < static_cast<NpuID>(nodeDegrees_.size()))
                                    ? nodeDegrees_[dest]
                                    : std::numeric_limits<int>::max();
            prioritized.push_back({Condition{chunk, dest}, degree, distance});
        }
    }

    std::stable_sort(prioritized.begin(),
                     prioritized.end(),
                     [](const ConditionPriority& lhs, const ConditionPriority& rhs) {
                        if (lhs.distance != rhs.distance) {
                            return lhs.distance > rhs.distance;  // 2. farther first
                        }
                         if (lhs.degree != rhs.degree) {
                             return lhs.degree < rhs.degree;  // 1. smaller degree first
                         }
                         if (lhs.condition.second != rhs.condition.second) {
                             return lhs.condition.second < rhs.condition.second; // 3. smaller dest first
                         }
                         return lhs.condition.first < rhs.condition.first; // 4. smaller chunk first
                     });

    postcondition.reserve(prioritized.size());
    for (const auto& item : prioritized) {
        postcondition.push_back(item.condition);
    }

    auto finalConditions = std::vector<Condition>();
    auto intermediateConditions = std::vector<Condition>();
    for (const auto& item : postcondition) {
        if (isFinalDestination_(item.first, item.second)) {
            finalConditions.push_back(item);
        } else {
            intermediateConditions.push_back(item);
        }
    }

    // ThreadOutput::output("Postconditions (final) [chunk, dest]: ");
    // for (const auto& item : finalConditions) {
    //     ThreadOutput::output("[");
    //     ThreadOutput::output(item.first);
    //     ThreadOutput::output(", ");
    //     ThreadOutput::output(item.second);
    //     ThreadOutput::output("]");
    //     ThreadOutput::output(", ");
    // }
    // ThreadOutput::output(std::endl);

    // if (!intermediateConditions.empty()) {
    //     ThreadOutput::output("Postconditions (intermediate routing) [chunk, nextHop]: ");
    //     for (const auto& item : intermediateConditions) {
    //         ThreadOutput::output("[");
    //         ThreadOutput::output(item.first);
    //         ThreadOutput::output(", ");
    //         ThreadOutput::output(item.second);
    //         ThreadOutput::output("]");
    //         ThreadOutput::output(", ");
    //     }
    //     ThreadOutput::output(std::endl);
    // }

    return postcondition;
}
// std::vector<Synthesizer::Condition> Synthesizer::shufflePostcondition_(
//     const PostconditionMap& postconditionMap) noexcept {
//     auto postcondition = std::vector<Condition>();

//     struct ConditionPriority {
//         Condition condition;
//         int degree;
//         int distance;
//     };
//     auto prioritized = std::vector<ConditionPriority>();

//     for (const auto& [dest, chunks] : postconditionMap) {
//         for (const auto chunk : chunks) {
//             const auto distance = computeConditionDistance_(chunk, dest);
//             const auto degree = (dest >= 0 && dest < static_cast<NpuID>(nodeDegrees_.size()))
//                                     ? nodeDegrees_[dest]
//                                     : std::numeric_limits<int>::max();
//             prioritized.push_back({Condition{chunk, dest}, degree, distance});
//         }
//     }

//     std::stable_sort(prioritized.begin(),
//                      prioritized.end(),
//                      [](const ConditionPriority& lhs, const ConditionPriority& rhs) {
//                          if (lhs.degree != rhs.degree) {
//                              return lhs.degree < rhs.degree;  // 1. smaller degree first
//                          }
//                          if (lhs.distance != rhs.distance) {
//                              return lhs.distance > rhs.distance;  // 2. farther distance first
//                          }
//                          return false;  // equal priority; keep relative order for later shuffle
//                      });

//     // within each (degree, distance) bucket, randomly shuffle to break ties
//     auto begin = prioritized.begin();
//     while (begin != prioritized.end()) {
//         auto end = begin;
//         while (end != prioritized.end() &&
//                end->degree == begin->degree &&
//                end->distance == begin->distance) {
//             ++end;
//         }
//         std::shuffle(begin, end, randomEngine);
//         begin = end;
//     }

//     postcondition.reserve(prioritized.size());
//     for (const auto& item : prioritized) {
//         postcondition.push_back(item.condition);
//     }

//     return postcondition;
// }

void Synthesizer::mergeTempConditions_(PostconditionMap* const postconditionMap) const noexcept {
    assert(postconditionMap != nullptr);
    for (const auto& [dest, chunks] : tempConditions_) {
        auto& setRef = (*postconditionMap)[dest];
        setRef.insert(chunks.begin(), chunks.end());
    }
}

void Synthesizer::expandTenTimestep_(PostconditionMap* const postconditionMap) noexcept {
    // first, expand the TEN structure
    ten_->timestep(currentTime_);

    // this bool value is to track if any meaningful event happened during this timestep
    // e.g., chunk arrival or replacement
    // so that we can update the collective time
    auto eventHappened = false;

    // for every src-dest pairs
    for (auto src = 0; src < npusCount; src++) {
        for (auto dest = 0; dest < npusCount; dest++) {
            // if TEN is not available, skip
            // i.e., link doesn't exist or it is busy transferring a chunk
            if (!ten_->available(src, dest)) {
                continue;
            }

            // if a TEN link is available, there are two cases:
            // 1. the TEN link is indeed free, or
            // 2. it just become free by finishing a transfer
            // for case 2, we should mark this transfer as finished
            // and check for replacement possibilities

            // for case 1 (link is free), we can skip this
            auto chunk = ten_->chunk(src, dest);
            if (chunk < 0) {
                continue;
            }

            // for case 2, check if the chunk has already arrived at dest
            // by following other paths
            // and if so, check if we can replace this path with another chunk
            if (chunkMap_[chunk][dest]) {
                // dest has already received this chunk
                // so check the replacement candidates
                const auto replacementChunk = findReplacementChunk_(src, dest, postconditionMap);

                if (!replacementChunk.has_value()) {
                    // no replacement candidate found
                    // just mark this TEN link as available and skip
                    ten_->transferFinished(src, dest);
                    continue;
                }

                // replacement candidate found
                chunk = replacementChunk.value();
            }

            // a meaningful chunk (regardless of replacement) has arrived at dest
            eventHappened = true;

            // mark the chunk arrived at dest, and mark this TEN link as available
            chunkMap_[chunk][dest] = true;
            ten_->transferFinished(src, dest);

            // mark this postcondition as satisfied
            // i.e., remove this chunk from the postcondition map
            auto it = postconditionMap->find(dest);
            if (it != postconditionMap->end()) {
                it->second.erase(chunk);
                if (it->second.empty()) {
                    postconditionMap->erase(it);
                }
            }
            if (isFinalDestination_(chunk, dest)) {
                removeAllTempConditionsForChunk_(chunk);
            } else {
                removeTempCondition_(chunk, dest);
            }
            
        }
    }

    // at the end of the TEN expansion
    // if a meaningful event happened, we should update the collective time
    if (eventHappened) {
        // update collective time to current time
        collectiveTime_ = currentTime_;
    }
}

std::optional<Synthesizer::ChunkID> Synthesizer::findReplacementChunk_(
    const NpuID src, const NpuID dest, const PostconditionMap* const postconditionMap) noexcept {
    // trivial scenario: if dest has all postconditions satisfied,
    // there's no need for replacement
    if (postconditionMap->find(dest) == postconditionMap->end()) {
        return std::nullopt;
    }

    // check if any replacement candidate exists
    auto candidates = std::vector<ChunkID>();

    // iterate over all unsatisfied postcondition of this dest NPU
    for (const auto chunk : postconditionMap->at(dest)) {
        // if this chunk is available at src NPU
        // but has not yet arrived at dest NPU,
        // this chunk can be a replacement candidate
        if (chunkMap_[chunk][src] && !chunkMap_[chunk][dest]) {
            candidates.push_back(chunk);
        }
    }

    // if there's no candidate, return nullopt
    if (candidates.empty()) {
        return std::nullopt;
    }

    // if there's only one candidate, return it
    if (candidates.size() == 1) {
        return candidates[0];
    }

    // If there are multiple candidates, randomly select one.
    auto dist = std::uniform_int_distribution<>(0, candidates.size() - 1);
    const auto idx = dist(randomEngine);
    return candidates[idx];
}

void Synthesizer::linkChunkMatching_(const ChunkID chunk, const NpuID dest) noexcept {
    // backtrack source NPUs
    auto sources = ten_->backtrack(dest);

    // filter candidate link-chunk matching
    // that has the earliest estimated chunk arrival time
    auto arrivalTime = std::numeric_limits<Time>::max();
    auto candidates = std::vector<NpuID>();

    // iterate over all source NPUs
    for (const auto src : sources) {
        // if src does not have the chunk, skip

        // Allocate links in execution order. A later transmission may have
        // fewer viable shortest paths, so routing hints can trigger
        // backtracking and replacement when no direct match remains.
        if (!chunkMap_[chunk][src]) {
            continue;
        }

        // if source has the chunk, check the link transfer time
        const auto linkWeight = ten_->linkTransferTime(src, dest);
        const auto linkTime = currentTime_ + linkWeight;

        // if this link time is equal to the minimum time
        // add to the candidate links
        if (isEqual(linkTime, arrivalTime)) {
            candidates.emplace_back(src);
        }

        // if this link time is less than the minimum time,
        // update the minimum time and reset candidates
        if (linkTime < arrivalTime) {
            arrivalTime = linkTime;
            candidates.clear();
            candidates.emplace_back(src);
        }
    }

    // if candidates are empty, no match can be made
    if (candidates.empty()) {
        const auto routingHint = findRoutingHint_(chunk, dest);
        if (routingHint.has_value() && routingHint->distance > 1 &&
            routingHint->nextHop >= 0) {
            addTempCondition_(chunk, routingHint->nextHop);
            linkChunkMatching_(chunk, routingHint->nextHop);
        }
        return;
    }

    // randomly shuffle and select one source NPU to make link-chunk match
    std::shuffle(candidates.begin(), candidates.end(), randomEngine);
    const auto selectedSrc = candidates.front();

    // mark the TEN as occupied
    ten_->transferChunk(selectedSrc, dest, chunk, arrivalTime);

    // schedule an event when the matched chunk arrives
    eventQueue_.schedule(arrivalTime);

}

void Synthesizer::addTempCondition_(const ChunkID chunk, const NpuID dest) noexcept {
    assert(0 <= dest && dest < npusCount);
    if (chunkMap_[chunk][dest]) {
        return;
    }
    tempConditions_[dest].insert(chunk);
}

void Synthesizer::removeTempCondition_(const ChunkID chunk, const NpuID dest) noexcept {
    const auto it = tempConditions_.find(dest);
    if (it == tempConditions_.end()) {
        return;
    }
    it->second.erase(chunk);
    if (it->second.empty()) {
        tempConditions_.erase(it);
    }
}

// Use reverse BFS to find the nearest source and record its distance.
std::optional<Synthesizer::RoutingHint> Synthesizer::findRoutingHint_(const ChunkID chunk,
    const NpuID dest) noexcept {
    // Delegate to findRoutingHintWithTemp_ with current tempConditions_
    return findRoutingHintWithTemp_(chunk, dest, &tempConditions_);
}

// Find routing hint considering both chunkMap_ and tempConditions_
std::optional<Synthesizer::RoutingHint> Synthesizer::findRoutingHintWithTemp_(
    const ChunkID chunk, const NpuID dest, const PostconditionMap* tempConditions) const noexcept {
    assert(0 <= dest && dest < npusCount);
    const PostconditionMap* temp = tempConditions ? tempConditions : &tempConditions_;

    auto prev = std::vector<int>(npusCount, -1);
    auto distance = std::vector<int>(npusCount, -1);
    auto queue = std::queue<NpuID>();

    prev[dest] = dest;
    distance[dest] = 0;
    queue.push(dest);

    while (!queue.empty()) {
        const auto node = queue.front();
        queue.pop();

        if (distance[node] >= diameter_) {
            continue;
        }

        auto neighbors = topology_->backtrack(node);
        std::shuffle(neighbors.begin(), neighbors.end(), randomEngine);
        for (const auto src : neighbors) {
            if (distance[src] != -1) {
                continue;
            }

            distance[src] = distance[node] + 1;
            prev[src] = node;

            // Check both chunkMap_ and tempConditions_ for chunk availability
            bool hasChunk = chunkMap_[chunk][src];
            if (!hasChunk && temp) {
                const auto it = temp->find(src);
                if (it != temp->end() && it->second.find(chunk) != it->second.end()) {
                    hasChunk = true;
                }
            }

            if (hasChunk) {
                return RoutingHint{src, prev[src], distance[src]};
            }

            queue.push(src);
        }
    }

    return std::nullopt;
}
// std::optional<Synthesizer::RoutingHint> Synthesizer::findRoutingHint_(const ChunkID chunk,
//                                                                       const NpuID dest) noexcept {
//     assert(0 <= dest && dest < npusCount);

//     auto prev = std::vector<int>(npusCount, -1);
//     auto distance = std::vector<int>(npusCount, -1);
//     auto queue = std::queue<NpuID>();

//     prev[dest] = dest;
//     distance[dest] = 0;
//     queue.push(dest);

//     while (!queue.empty()) {
//         const auto node = queue.front();
//         queue.pop();

//         if (distance[node] >= diameter_) {
//             continue;
//         }

//         auto neighbors = topology_->backtrack(node);
//         std::shuffle(neighbors.begin(), neighbors.end(), randomEngine);
//         for (const auto src : neighbors) {
//             if (distance[src] != -1) {
//                 continue;
//             }

//             distance[src] = distance[node] + 1;
//             prev[src] = node;

//             if (chunkMap_[chunk][src]) {
//                 return RoutingHint{src, prev[src], distance[src]};
//             }

//             queue.push(src);
//         }
//     }

//     return std::nullopt;
// }

// Compute the routing-hint distance to the destination.
int Synthesizer::computeConditionDistance_(const ChunkID chunk, const NpuID dest) noexcept {
    // Use findRoutingHintWithTemp_ to consider tempConditions_ as well
    const auto hint = findRoutingHintWithTemp_(chunk, dest, &tempConditions_);
    if (hint.has_value()) {
        return hint->distance;
    }
    return std::numeric_limits<int>::max();
}

bool Synthesizer::isFinalDestination_(const ChunkID chunk, const NpuID dest) const noexcept {
    const auto dests = collective_->postcondition(chunk);
    return std::find(dests.begin(), dests.end(), dest) != dests.end();
}

void Synthesizer::removeAllTempConditionsForChunk_(const ChunkID chunk) noexcept {
    for (auto it = tempConditions_.begin(); it != tempConditions_.end();) {
        auto& chunks = it->second;
        chunks.erase(chunk);
        if (chunks.empty()) {
            it = tempConditions_.erase(it);
        } else {
            ++it;
        }
    }
}

int Synthesizer::simulateLinkChunkMatching_(
    const std::vector<Condition>& postcondition,
    std::vector<std::vector<bool>>& availableCopy,
    std::vector<std::vector<bool>>& chunkMapCopy,
    PostconditionMap& tempConditionsCopy) const noexcept {
    // Local references for simulation (need to modify)
    auto& available = availableCopy;
    auto& chunkMap = chunkMapCopy;
    auto& tempConditions = tempConditionsCopy;
    int linksUsed = 0;

    // Simulate processing each condition in order
    for (const auto [chunk, dest] : postcondition) {
        // Find available sources that have the chunk (simulate ten_->backtrack)
        auto sources = std::vector<NpuID>();
        for (const auto src : topology_->backtrack(dest)) {
            if (!available[src][dest]) {
                continue;
            }
            // Check if src has chunk (in chunkMap or tempConditions)
            bool hasChunk = chunkMap[chunk][src];
            if (!hasChunk) {
                const auto it = tempConditions.find(src);
                if (it != tempConditions.end() && it->second.find(chunk) != it->second.end()) {
                    hasChunk = true;
                }
            }
            if (hasChunk) {
                sources.push_back(src);
            }
        }

        if (!sources.empty()) {
            // Simulate using one link (pick first available)
            const auto selectedSrc = sources.front();
            available[selectedSrc][dest] = false;  // Mark as used
            linksUsed++;

            // Simulate chunk arrival at dest
            chunkMap[chunk][dest] = true;
            
            // If this is an intermediate hop, add to tempConditions
            if (!isFinalDestination_(chunk, dest)) {
                tempConditions[dest].insert(chunk);
            }
        } else {
            // Try to find routing hint for multi-hop
            const auto hint = findRoutingHintWithTemp_(chunk, dest, &tempConditions);
            if (hint.has_value() && hint->distance > 1 && hint->nextHop >= 0) {
                // Recursively simulate the next hop
                std::vector<Condition> nextHop = {{chunk, hint->nextHop}};
                linksUsed += simulateLinkChunkMatching_(nextHop, available, chunkMap, tempConditions);
            }
        }
    }

    return linksUsed;
}

bool Synthesizer::isEqual(const Time lhs, const Time rhs) noexcept {
    constexpr Time epsilon = 1e-9;
    return std::abs(lhs - rhs) < epsilon;
}
