/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#pragma once

#include <tacos/topology/topology.h>
#include <unordered_map>
#include <unordered_set>

namespace tacos {

/// @brief Abstract base class for collective communication patterns
class Collective {
  public:
    // data types
    /// @brief Chunk ID
    using ChunkID = int;

    /// @brief chunk size: in bytes
    using ChunkSize = int64_t;

    /// @brief chunk factor: how many slices for given original data
    int ChunkFactor;

    using NpuID = Topology::NpuID;

    /// @brief Base class constructor for collective pattern
    Collective() noexcept;

    /// @brief Return the source NPU for a given chunk
    /// @param chunk chunk ID
    /// @return source NPU of the chunk
    [[nodiscard]] NpuID precondition(ChunkID chunk) const noexcept;

    /// @brief Return the destination NPUs for a given chunk
    /// @param chunk chunk ID
    /// @return destination NPUs of the chunk
    [[nodiscard]] const std::unordered_set<NpuID>& postcondition(ChunkID chunk) const noexcept;

    /// @brief Get the number of chunks in this collective
    /// @return number of chunks in the collective pattern
    [[nodiscard]] int chunksCount() const noexcept;

    /// @brief Get the number of chunk factor in this collective
    /// @return number of chunk slices in the collective pattern
    [[nodiscard]] int getChunkFactor() const noexcept;

    /// @brief Get the precondition in this collective
    /// @return precondition
    [[nodiscard]] std::unordered_map<ChunkID, NpuID> getPrecondition() const noexcept;

    /// @brief Get the postcondition in this collective
    /// @return postcondition
    [[nodiscard]] std::unordered_map<ChunkID, std::unordered_set<NpuID>> getPostcondition() const noexcept;

    /// @brief Remove a faulty node from all postconditions
    /// @param faultyNode NpuID of the faulty node to remove
    void removeFaultyNode(NpuID faultyNode) noexcept;

  protected:
    /// @brief Number of chunks in the collective
    int chunksCount_ = 0;

    /// @brief Insert new precondition and postcondition for a chunk
    void chunk_(NpuID src, std::unordered_set<NpuID> dests) noexcept;

    /// @brief Update chunk factor during the initialization of specific collective function
    void updateChunkFactor(int Chunk_Factor) noexcept;

  private:
    /// @brief New Chunk ID to be used
    int newChunkID_ = 0;

    /// @brief Precondition: maps each chunk to the (single) source NPU
    std::unordered_map<ChunkID, NpuID> precondition_ = {};

    /// @brief Postcondition: maps each chunk to (multiple) destination NPUs
    std::unordered_map<ChunkID, std::unordered_set<NpuID>> postcondition_ = {};
};
}  // namespace tacos
