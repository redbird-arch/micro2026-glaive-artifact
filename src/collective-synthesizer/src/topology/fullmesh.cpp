/*
# File name  :    fullmesh.cpp
# Author     :    Galois
# Time       :    2025/05/22 16:23:49
*/


#include <cassert>
#include <iostream>
#include <numeric>
#include <tacos/topology/fullmesh.h>

using namespace tacos;

FullMesh::FullMesh(const std::vector<int>& shape,
                   const std::vector<double>& latency,
                   const std::vector<double>& bandwidth) noexcept
    : shape(shape) {

    // get parameters
    const int dimensions = shape.size();
    assert(dimensions == latency.size());
    assert(dimensions == bandwidth.size());

    // compute total number of NPUs
    int npusCount = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    setNpusCount(npusCount);

    // build strides for each dimension
    std::vector<int> strides(dimensions);
    strides[0] = 1;
    for (int d = 1; d < dimensions; ++d) {
        strides[d] = strides[d - 1] * shape[d - 1];
    }

    // connect full mesh in each dimension
    for (int d = 0; d < dimensions; ++d) {
        int dimSize = shape[d];
        int stride = strides[d];
        int linkLatency = latency[d];
        double linkBandwidth = bandwidth[d];

        for (int i = 0; i < npusCount; ++i) {
            int coordInDim = (i / stride) % dimSize;
            int base = i - coordInDim * stride;

            for (int j = 0; j < dimSize; ++j) {
                if (j == coordInDim) continue;
                int neighbor = base + j * stride;
                if (!connected(i, neighbor)) {
                    connect(i, neighbor, linkLatency, linkBandwidth, true);
                }
            }
        }
    }
}
