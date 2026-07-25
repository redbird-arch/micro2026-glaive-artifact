/*
# File name  :    hypercube.cpp
# Author     :    Galois
# Time       :    2025/05/22 15:56:49
*/

#include <cassert>
#include <tacos/topology/hypercube.h>

using namespace tacos;

Hypercube::Hypercube(int dimension,
                     const std::vector<double>& latency,
                     const std::vector<double>& bandwidth) noexcept
    : dimension(dimension) {
    
    assert(dimension > 0);
    assert(latency.size() == dimension);
    assert(bandwidth.size() == dimension);

    int npusCount = 1 << dimension; // 2^dimension
    setNpusCount(npusCount);

    // Connect all edges differing by one bit
    for (int i = 0; i < npusCount; ++i) {
        for (int d = 0; d < dimension; ++d) {
            int neighbor = i ^ (1 << d);  // Flip d-th bit
            if (i < neighbor) {           // Avoid duplicate
                connect(i, neighbor, latency[d], bandwidth[d], true);
            }
        }
    }
}