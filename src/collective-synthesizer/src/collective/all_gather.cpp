/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#include <cassert>
#include <iostream>
#include <tacos/collective/all_gather.h>
#include "thread_output.h"

using namespace tacos;

AllGather::AllGather(const int npusCount, const int collectivesCount) noexcept : Collective() {
    assert(collectivesCount > 0);

    auto chunkId = 0;

    // destination for All-Gather: all NPUs in the topology
    auto dests = std::unordered_set<NpuID>();
    for (auto dest = 0; dest < npusCount; ++dest) {
        dests.insert(dest);
    }

    // register chunks for all source NPUs
    for (int c = 0; c < collectivesCount; ++c) {
        for (int src = 0; src < npusCount; ++src) {
            chunk_(src, dests);
        }
    }

    // update chunk factor
    int Chunk_Factor = collectivesCount;
    updateChunkFactor(Chunk_Factor);

    // print some info using thread-local output
    // ThreadOutput::output("[AllGather] Precondition:");
    // ThreadOutput::output(std::endl);

    std::unordered_map<NpuID, std::vector<ChunkID>> npuToChunks;
    for (const auto& [chunk, npu] : getPrecondition()) {
        npuToChunks[npu].push_back(chunk);
    }

    std::vector<NpuID> sortedNpus;
    for (const auto& [npu, chunks] : npuToChunks) {
        sortedNpus.push_back(npu);
    }
    std::sort(sortedNpus.begin(), sortedNpus.end());

    // for (auto npu : sortedNpus) {
    //     ThreadOutput::output("\tNPU ");
    //     ThreadOutput::output(npu);
    //     ThreadOutput::output(": ");
    //     auto& chunks = npuToChunks[npu];
    //     std::sort(chunks.begin(), chunks.end());
    //     for (auto chunk : chunks) {
    //         ThreadOutput::output(chunk);
    //         ThreadOutput::output(" ");
    //     }
    //     ThreadOutput::output(std::endl);
    // }

    // ThreadOutput::output("[AllGather] Postcondition:");
    // ThreadOutput::output(std::endl);

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

    // for (auto npu : sortedNpus) {
    //     ThreadOutput::output("\tNPU ");
    //     ThreadOutput::output(npu);
    //     ThreadOutput::output(": ");
    //     auto& chunks = npuToChunksPost[npu];
    //     std::sort(chunks.begin(), chunks.end());
    //     for (auto chunk : chunks) {
    //         ThreadOutput::output(chunk);
    //         ThreadOutput::output(" ");
    //     }
    //     ThreadOutput::output(std::endl);
    // }
}
