/*
# File name  :    hypercube.h
# Author     :    Galois
# Time       :    2025/05/22 15:56:19
*/

#pragma once

#include <vector>
#include <tacos/topology/topology.h>

namespace tacos {

class Hypercube : public Topology {
public:
    Hypercube(int dimension,
              const std::vector<double>& latency,
              const std::vector<double>& bandwidth) noexcept;

private:
    int dimension;
};

} // namespace tacos