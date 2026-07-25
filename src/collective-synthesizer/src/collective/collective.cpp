/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include <cassert>
#include <tacos/collective/collective.h>

using namespace tacos;

Collective::Collective() noexcept = default;

int Collective::chunksCount() const noexcept {
    assert(chunksCount_ > 0);
    return chunksCount_;
}

void Collective::updateChunkFactor(int Chunk_Factor) noexcept {
    ChunkFactor = Chunk_Factor;
}

int Collective::getChunkFactor() const noexcept {
    return ChunkFactor;
}

void Collective::chunk_(const NpuID src, std::unordered_set<NpuID> dests) noexcept {
    assert(src >= 0);
    assert(!dests.empty());

    // assert this chunk is not already registered
    assert(precondition_.find(newChunkID_) == precondition_.end());
    assert(postcondition_.find(newChunkID_) == postcondition_.end());

    // insert to precondition and postcondition
    precondition_[newChunkID_] = src;
    postcondition_[newChunkID_] = std::move(dests);
    chunksCount_++;

    // increment nw chunkID for the next chunk
    newChunkID_++;
}

Collective::NpuID Collective::precondition(const ChunkID chunk) const noexcept {
    assert(0 <= chunk && chunk < chunksCount_);
    assert(precondition_.find(chunk) != precondition_.end());

    return precondition_.at(chunk);
}

const std::unordered_set<Collective::NpuID>& Collective::postcondition(
    ChunkID chunk) const noexcept {
    assert(0 <= chunk && chunk < chunksCount_);
    assert(postcondition_.find(chunk) != postcondition_.end());

    return postcondition_.at(chunk);
}

std::unordered_map<Collective::ChunkID, Collective::NpuID>  Collective::getPrecondition() const noexcept {
    return precondition_;
}

std::unordered_map<Collective::ChunkID, std::unordered_set<Collective::NpuID>> Collective::getPostcondition() const noexcept {
    return postcondition_;
}

void Collective::removeFaultyNode(NpuID faultyNode) noexcept {
    // Remove faulty node from all postconditions
    for (auto& [chunk, dests] : postcondition_) {
        dests.erase(faultyNode);
    }
    
    // Also remove chunks where the source (precondition) is the faulty node
    // Since the faulty node should not send or receive data
    std::vector<ChunkID> chunksToRemove;
    for (const auto& [chunk, src] : precondition_) {
        if (src == faultyNode) {
            chunksToRemove.push_back(chunk);
        }
    }
    
    // Remove chunks with faulty node as source
    for (auto chunk : chunksToRemove) {
        precondition_.erase(chunk);
        postcondition_.erase(chunk);
        chunksCount_--;
    }
}
