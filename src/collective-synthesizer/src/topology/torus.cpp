/*
# File name  :    torus.cpp
# Author     :    Galois
# Time       :    2025/05/22 15:59:06
*/

#include <cassert>
#include <tacos/topology/torus.h>

using namespace tacos;

Torus::Torus(const std::vector<int>& shape,
             const std::vector<double>& latency,
             const std::vector<double>& bandwidth) noexcept
    : shape(shape) {

    const int dimensions = shape.size();
    assert(dimensions == latency.size());
    assert(dimensions == bandwidth.size());

    int npusCount = 1;
    for (int dim : shape) {
        assert(dim > 0);
        npusCount *= dim;
    }

    setNpusCount(npusCount);

    int stride = 1;
    for (int d = 0; d < dimensions; ++d) {
        int dimSize = shape[d];
        int linkLatency = latency[d];
        double linkBandwidth = bandwidth[d];

        for (int i = 0; i < npusCount; ++i) {
            int src = i;

            int idxInDim = (i / stride) % dimSize;
            int base = i - idxInDim * stride;
            int dest;

            // forward connection
            if (idxInDim < dimSize - 1)
                dest = src + stride;
            else
                dest = base; // wrap around

            connect(src, dest, linkLatency, linkBandwidth, true);
        }

        stride *= dimSize;
    }
}