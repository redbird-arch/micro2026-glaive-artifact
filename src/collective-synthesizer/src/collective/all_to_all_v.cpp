/*
# File name  :    all_to_all_v.cpp
# Author     :    Galois
# Time       :    2025/12/25 20:28:45
*/

#include <cassert>
#include <tacos/collective/all_to_all_v.h>

using namespace tacos;

AlltoAllV::AlltoAllV(const int npusCount, 
                     const std::vector<std::vector<long long>>& dataMatrix,
                     const long long chunkSize) noexcept : Collective() {
    assert(npusCount > 0);
    assert(static_cast<int>(dataMatrix.size()) == npusCount);
    assert(chunkSize > 0);

    auto chunkId = 0;

    // Register chunks for all source-destination pairs.
    for (int src = 0; src < npusCount; ++src) {
        assert(static_cast<int>(dataMatrix[src].size()) == npusCount);
        for (int dest = 0; dest < npusCount; ++dest) {
            long long dataSize = dataMatrix[src][dest];
            if (dataSize > 0) {
                int numChunks = static_cast<int>(dataSize / chunkSize);
                assert(numChunks > 0);
                
                auto dests = std::unordered_set<NpuID>{dest};
                for (int c = 0; c < numChunks; ++c) {
                    chunk_(src, dests);
                }
            }
        }
    }

    // AlltoAllV uses the chunk size supplied by its input.
    updateChunkFactor(1);
}
