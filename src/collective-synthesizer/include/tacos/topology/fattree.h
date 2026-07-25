/*
# File name  :    fattree.h
# Author     :    Galois
# Time       :    2026/01/22 16:21:33
*/

#pragma once

#include <vector>
#include <tacos/topology/topology.h>

namespace tacos {

class FatTree : public Topology {
public:
    FatTree(const std::vector<int>& shape,
            int switchDimension,
            const std::vector<int>& switchShape,
            const std::vector<int>& linkCount,
            const std::vector<double>& latency,
            const std::vector<double>& bandwidth) noexcept;

    std::string formatNodeName(NpuID id) const noexcept override;

private:
    std::vector<int> shape;
    int switchDimension;
    std::vector<int> switchShape;
    std::vector<int> linkCount;
};

}  // namespace tacos

