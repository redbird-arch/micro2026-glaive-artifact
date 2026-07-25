/*
# File name  :    cm.h
# Author     :    Galois
# Time       :    2026/03/02
#
# CM topology (e.g. CM384): 3-level hierarchy
# - Level 0: Intra-node direct links between odd-even adjacent cards (e.g. card 0-1, 2-3, ...)
# - Level 1: Card to node switch (like fat-tree scale-up), each card connects to each switch in its node
# - Level 2: Node switches connect to rail switches (not cards); non-blocking between node and rail layer
#   Each node has switchesPerNode switches (one per rail). Each rail has switchesPerRail switches.
#   Each node switch connects to all switchesPerRail rail switches of its rail.
#   Each rail switch connects to all numNodes node switches of that rail.
*/

#pragma once

#include <vector>
#include <tacos/topology/topology.h>

namespace tacos {

class CM : public Topology {
public:
    CM(const std::vector<int>& shape,
       const std::vector<int>& switchShape,
       const std::vector<int>& linkCount,
       double directBandwidth,
       double directLatency,
       const std::vector<double>& latency,
       const std::vector<double>& bandwidth) noexcept;

    std::string formatNodeName(NpuID id) const noexcept override;

private:
    std::vector<int> shape;
    std::vector<int> switchShape;
    std::vector<int> linkCount;
};

}  // namespace tacos
