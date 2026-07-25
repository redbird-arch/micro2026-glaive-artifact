/*
# File name  :    mesh.cpp
# Author     :    Galois
# Time       :    2025/05/22 15:57:01
*/

#include <cassert>
#include <tacos/topology/mesh.h>

using namespace tacos;

Mesh::Mesh(const std::vector<int>& shape,
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
            int neighbor = src + stride;

            if ((i / stride) % dimSize < dimSize - 1) {
                connect(src, neighbor, linkLatency, linkBandwidth, true);
            }
        }

        stride *= dimSize;
    }
}
