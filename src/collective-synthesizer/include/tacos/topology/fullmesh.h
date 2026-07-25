/*
# File name  :    fullmesh.h
# Author     :    Galois
# Time       :    2025/05/22 16:23:16
*/

#pragma once

#include <vector>
#include <tacos/topology/topology.h>

namespace tacos {

class FullMesh : public Topology {
public:
    FullMesh(const std::vector<int>& shape,
             const std::vector<double>& latency,
             const std::vector<double>& bandwidth) noexcept;

private:
    std::vector<int> shape;
};

}  // namespace tacos