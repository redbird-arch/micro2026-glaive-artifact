/*
# File name  :    all_to_all_v.cpp
# Author     :    Galois
# Time       :    2025/12/25 20:28:45
*/

#include <cassert>
#include <iostream>
#include <tacos/collective/all_to_all_v.h>
#include "thread_output.h"

using namespace tacos;

AlltoAllV::AlltoAllV(const int npusCount, 
                     const std::vector<std::vector<long long>>& dataMatrix,
                     const long long chunkSize) noexcept : Collective() {
    assert(npusCount > 0);
    assert(static_cast<int>(dataMatrix.size()) == npusCount);
    assert(chunkSize > 0);

    auto chunkId = 0;

    // For All-To-All-V: each NPU sends variable-sized data to every other NPU
    // The number of chunks for each source-destination pair is determined by dataMatrix[i][j] / chunkSize
    
    // register chunks for all source-destination pairs based on data sizes
    for (int src = 0; src < npusCount; ++src) {
        assert(static_cast<int>(dataMatrix[src].size()) == npusCount);
        for (int dest = 0; dest < npusCount; ++dest) {
            long long dataSize = dataMatrix[src][dest];
            if (dataSize > 0) {
                // Calculate number of chunks for this source-destination pair
                int numChunks = static_cast<int>(dataSize / chunkSize);
                assert(numChunks > 0); // dataSize should be at least chunkSize
                
                // Create chunks for this source-destination pair
                auto dests = std::unordered_set<NpuID>{dest};
                for (int c = 0; c < numChunks; ++c) {
                    chunk_(src, dests);
                }
            }
            // If dataSize is 0, no chunks are created for this pair
        }
    }

    // update chunk factor (for AlltoAllV, chunk factor is 1 as chunk size is determined by GCD)
    updateChunkFactor(1);

    // // print some info using thread-local output
    // ThreadOutput::output("[AlltoAllV] Precondition:");
    // ThreadOutput::output(std::endl);

    // std::unordered_map<NpuID, std::vector<ChunkID>> npuToChunks;
    // for (const auto& [chunk, npu] : getPrecondition()) {
    //     npuToChunks[npu].push_back(chunk);
    // }

    // std::vector<NpuID> sortedNpus;
    // for (const auto& [npu, chunks] : npuToChunks) {
    //     sortedNpus.push_back(npu);
    // }
    // std::sort(sortedNpus.begin(), sortedNpus.end());

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

    // ThreadOutput::output("[AlltoAllV] Postcondition:");
    // ThreadOutput::output(std::endl);

    // std::unordered_map<NpuID, std::vector<ChunkID>> npuToChunksPost;
    // for (const auto& [chunk, npus] : getPostcondition()) {
    //     for (auto npu : npus) {
    //         npuToChunksPost[npu].push_back(chunk);
    //     }
    // }

    // sortedNpus.clear();
    // for (const auto& [npu, chunks] : npuToChunksPost) {
    //     sortedNpus.push_back(npu);
    // }
    // std::sort(sortedNpus.begin(), sortedNpus.end());

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

