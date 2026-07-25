/*
# File name  :    torus.h
# Author     :    Galois
# Time       :    2025/05/22 15:59:22
*/

#pragma once

#include <vector>
#include <tacos/topology/topology.h>

namespace tacos {

class Torus : public Topology {
public:
    Torus(const std::vector<int>& shape,
          const std::vector<double>& latency,
          const std::vector<double>& bandwidth) noexcept;

private:
    std::vector<int> shape;
};

} // namespace tacos