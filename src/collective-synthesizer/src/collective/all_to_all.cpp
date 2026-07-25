/*
# File name  :    all_to_all.cpp
# Author     :    Galois
# Time       :    2025/11/20 19:55:42
*/

#include <cassert>
#include <iostream>
#include <tacos/collective/all_to_all.h>
#include "thread_output.h"

using namespace tacos;

AlltoAll::AlltoAll(const int npusCount, const int collectivesCount) noexcept : Collective() {
    assert(collectivesCount > 0);

    auto chunkId = 0;

    // For All-To-All: each NPU sends a different chunk to every other NPU
    // Each NPU will send (npusCount-1) chunks and receive (npusCount-1) chunks

    // register chunks for all source-destination pairs
    for (int c = 0; c < collectivesCount; ++c) {
        for (int src = 0; src < npusCount; ++src) {
            for (int dest = 0; dest < npusCount; ++dest) {
                auto dests = std::unordered_set<NpuID>{dest};
                chunk_(src, dests);
            }
        }
    }

    // update chunk factor
    int Chunk_Factor = collectivesCount;
    updateChunkFactor(Chunk_Factor);

    // print some info using thread-local output
    ThreadOutput::output("[AlltoAll] Precondition:");
    ThreadOutput::output(std::endl);

    std::unordered_map<NpuID, std::vector<ChunkID>> npuToChunks;
    for (const auto& [chunk, npu] : getPrecondition()) {
        npuToChunks[npu].push_back(chunk);
    }

    std::vector<NpuID> sortedNpus;
    for (const auto& [npu, chunks] : npuToChunks) {
        sortedNpus.push_back(npu);
    }
    std::sort(sortedNpus.begin(), sortedNpus.end());

    for (auto npu : sortedNpus) {
        ThreadOutput::output("\tNPU ");
        ThreadOutput::output(npu);
        ThreadOutput::output(": ");
        auto& chunks = npuToChunks[npu];
        std::sort(chunks.begin(), chunks.end());
        for (auto chunk : chunks) {
            ThreadOutput::output(chunk);
            ThreadOutput::output(" ");
        }
        ThreadOutput::output(std::endl);
    }

    ThreadOutput::output("[AlltoAll] Postcondition:");
    ThreadOutput::output(std::endl);

    std::unordered_map<NpuID, std::vector<ChunkID>> npuToChunksPost;
    for (const auto& [chunk, npus] : getPostcondition()) {
        for (auto npu : npus) {
            npuToChunksPost[npu].push_back(chunk);
        }
    }

    sortedNpus.clear();
    for (const auto& [npu, chunks] : npuToChunksPost) {
        sortedNpus.push_back(npu);
    }
    std::sort(sortedNpus.begin(), sortedNpus.end());

    for (auto npu : sortedNpus) {
        ThreadOutput::output("\tNPU ");
        ThreadOutput::output(npu);
        ThreadOutput::output(": ");
        auto& chunks = npuToChunksPost[npu];
        std::sort(chunks.begin(), chunks.end());
        for (auto chunk : chunks) {
            ThreadOutput::output(chunk);
            ThreadOutput::output(" ");
        }
        ThreadOutput::output(std::endl);
    }
}
