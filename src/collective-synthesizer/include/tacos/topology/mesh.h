/*
# File name  :    mesh.h
# Author     :    Galois
# Time       :    2025/05/22 15:57:13
*/

#pragma once

#include <vector>
#include <tacos/topology/topology.h>

namespace tacos {

class Mesh : public Topology {
public:
    Mesh(const std::vector<int>& shape,
         const std::vector<double>& latency,
         const std::vector<double>& bandwidth) noexcept;

private:
    std::vector<int> shape;
};

} // namespace tacos