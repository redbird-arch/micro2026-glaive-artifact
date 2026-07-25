/*
# File name  :    diffusion_model.h
# Author     :    Chen Chen
# Time       :    2025/01/06
# Description:    Physics-inspired diffusion models for traffic redistribution
#                 - Heat diffusion model (Laplacian smoothing)
#                 - Ising model for load balancing
#                 - Gradient descent optimization
#                 - Used to pre-process non-uniform traffic before routing
*/

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>
#include <functional>

#include "matrix_analyzer.h"

namespace tacos {

//=============================================================================
// DiffusionConfig: Configuration for diffusion algorithms
//=============================================================================
struct DiffusionConfig {
    // Heat diffusion parameters
    double diffusionCoefficient = 0.1;   // Alpha in heat equation
    int maxIterations = 100;
    double convergenceThreshold = 0.01;

    // Ising model parameters
    double temperature = 1.0;            // T in Boltzmann distribution
    double couplingStrength = 1.0;       // J in Ising Hamiltonian
    double externalField = 0.0;          // h in Ising Hamiltonian

    // Convergence bound
    double loadImbalanceBound = 1.5;     // Stop when max/avg < this value

    // Constraints
    bool preserveTotalDemand = true;     // Total demand must be conserved
    bool preserveRowSums = false;        // Row sums must be conserved
    bool preserveColSums = false;        // Column sums must be conserved

    // 2D Mesh/Torus specific parameters
    bool enable2DOptimization = true;    // Use 2D-specific optimizations
    double localityWeight = 0.7;         // Weight for local neighbors (vs diagonal)
    bool adaptiveCoefficient = true;     // Adapt diffusion coefficient based on imbalance
};

//=============================================================================
// DiffusionResult: Result of diffusion process
//=============================================================================
struct DiffusionResult {
    std::vector<std::vector<long long>> originalDemand;
    std::vector<std::vector<long long>> diffusedDemand;
    int iterationsUsed;
    double initialImbalance;
    double finalImbalance;
    bool converged;

    // Redistribution plan: how traffic was moved
    // redistribution[i][j][k][l] = bytes moved from (i,j) to (k,l)
    std::vector<std::vector<std::vector<std::vector<long long>>>> redistribution;

    void print() const {
        std::cout << "\n========== Diffusion Result ==========\n";
        std::cout << "Iterations: " << iterationsUsed << "\n";
        std::cout << "Initial Imbalance: " << initialImbalance << "\n";
        std::cout << "Final Imbalance: " << finalImbalance << "\n";
        std::cout << "Converged: " << (converged ? "YES" : "NO") << "\n";
        std::cout << "Improvement: " << ((1.0 - finalImbalance / initialImbalance) * 100) << "%\n";
        std::cout << "======================================\n";
    }
};

//=============================================================================
// DiffusionModel: Physics-inspired traffic diffusion
//=============================================================================
class DiffusionModel {
public:
    using DataSize = long long;

    DiffusionModel(const std::vector<int>& shape, bool isTorus)
        : shape_(shape), isTorus_(isTorus), config_() {
        npusCount_ = 1;
        for (int d : shape) {
            npusCount_ *= d;
        }
        dimensions_ = shape.size();

        // Compute strides
        strides_.resize(dimensions_);
        int stride = 1;
        for (int d = dimensions_ - 1; d >= 0; --d) {
            strides_[d] = stride;
            stride *= shape_[d];
        }

        // Build adjacency for Laplacian
        buildAdjacency();
    }

    void setConfig(const DiffusionConfig& config) {
        config_ = config;
    }

    //=========================================================================
    // Heat Diffusion Model
    // Solves: dD/dt = alpha * Laplacian(D)
    // Smooths out hotspots by diffusing traffic to neighbors
    //=========================================================================
    DiffusionResult applyHeatDiffusion(const std::vector<std::vector<DataSize>>& demand) {
        DiffusionResult result;
        result.originalDemand = demand;
        result.iterationsUsed = 0;
        result.converged = false;

        // Convert to double for computation
        std::vector<std::vector<double>> D(npusCount_, std::vector<double>(npusCount_, 0));
        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                D[i][j] = demand[i][j];
            }
        }

        result.initialImbalance = computeImbalance(D);

        // Iterative diffusion
        for (int iter = 0; iter < config_.maxIterations; ++iter) {
            auto D_new = D;

            // Apply Laplacian smoothing to each element
            for (int i = 0; i < npusCount_; ++i) {
                for (int j = 0; j < npusCount_; ++j) {
                    if (i == j) continue;

                    // Compute Laplacian contribution from neighbors
                    double laplacian = 0;
                    int neighborCount = 0;

                    // Neighbors in source dimension
                    for (int ni : getNeighbors(i)) {
                        laplacian += D[ni][j] - D[i][j];
                        neighborCount++;
                    }

                    // Neighbors in destination dimension
                    for (int nj : getNeighbors(j)) {
                        laplacian += D[i][nj] - D[i][j];
                        neighborCount++;
                    }

                    if (neighborCount > 0) {
                        D_new[i][j] = D[i][j] + config_.diffusionCoefficient * laplacian / neighborCount;
                        D_new[i][j] = std::max(0.0, D_new[i][j]);  // Non-negative
                    }
                }
            }

            // Normalize to preserve total demand if required
            if (config_.preserveTotalDemand) {
                double originalTotal = 0, newTotal = 0;
                for (int i = 0; i < npusCount_; ++i) {
                    for (int j = 0; j < npusCount_; ++j) {
                        if (i != j) {
                            originalTotal += D[i][j];
                            newTotal += D_new[i][j];
                        }
                    }
                }
                if (newTotal > 0) {
                    double scale = originalTotal / newTotal;
                    for (int i = 0; i < npusCount_; ++i) {
                        for (int j = 0; j < npusCount_; ++j) {
                            D_new[i][j] *= scale;
                        }
                    }
                }
            }

            // Check convergence
            double maxChange = 0;
            for (int i = 0; i < npusCount_; ++i) {
                for (int j = 0; j < npusCount_; ++j) {
                    maxChange = std::max(maxChange, std::abs(D_new[i][j] - D[i][j]));
                }
            }

            D = D_new;
            result.iterationsUsed = iter + 1;

            double currentImbalance = computeImbalance(D);
            if (currentImbalance < config_.loadImbalanceBound || maxChange < config_.convergenceThreshold) {
                result.converged = true;
                break;
            }
        }

        result.finalImbalance = computeImbalance(D);

        // Convert back to integer
        result.diffusedDemand.resize(npusCount_, std::vector<DataSize>(npusCount_, 0));
        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                result.diffusedDemand[i][j] = static_cast<DataSize>(std::round(D[i][j]));
            }
        }

        return result;
    }

    //=========================================================================
    // Ising Model for Load Balancing
    // Uses Metropolis-Hastings to find low-energy (balanced) configuration
    // Energy = sum of squared deviations from mean
    //=========================================================================
    DiffusionResult applyIsingModel(const std::vector<std::vector<DataSize>>& demand) {
        DiffusionResult result;
        result.originalDemand = demand;
        result.iterationsUsed = 0;
        result.converged = false;

        // Convert to double
        std::vector<std::vector<double>> D(npusCount_, std::vector<double>(npusCount_, 0));
        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                D[i][j] = demand[i][j];
            }
        }

        result.initialImbalance = computeImbalance(D);

        // Random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> uniform(0.0, 1.0);
        std::uniform_int_distribution<> nodeDistSrc(0, npusCount_ - 1);
        std::uniform_int_distribution<> nodeDistDst(0, npusCount_ - 1);

        double currentEnergy = computeEnergy(D);

        // Simulated annealing with Ising-like dynamics
        double temperature = config_.temperature;

        for (int iter = 0; iter < config_.maxIterations; ++iter) {
            // Propose a move: transfer some demand between neighboring pairs
            int i1 = nodeDistSrc(gen);
            int j1 = nodeDistDst(gen);
            if (i1 == j1 || D[i1][j1] <= 0) continue;

            // Find a neighbor to transfer to
            auto neighbors_i = getNeighbors(i1);
            auto neighbors_j = getNeighbors(j1);

            if (neighbors_i.empty() && neighbors_j.empty()) continue;

            // Choose randomly between source neighbor or dest neighbor
            int i2 = i1, j2 = j1;
            if (!neighbors_i.empty() && (neighbors_j.empty() || uniform(gen) < 0.5)) {
                i2 = neighbors_i[std::uniform_int_distribution<>(0, neighbors_i.size() - 1)(gen)];
            } else if (!neighbors_j.empty()) {
                j2 = neighbors_j[std::uniform_int_distribution<>(0, neighbors_j.size() - 1)(gen)];
            }

            if (i2 == j2) continue;

            // Propose transfer amount (small fraction)
            double transferAmount = D[i1][j1] * 0.1;
            if (transferAmount < 1) transferAmount = 1;

            // Compute energy change
            auto D_new = D;
            D_new[i1][j1] -= transferAmount;
            D_new[i2][j2] += transferAmount;

            if (D_new[i1][j1] < 0) continue;

            double newEnergy = computeEnergy(D_new);
            double deltaE = newEnergy - currentEnergy;

            // Metropolis criterion
            bool accept = false;
            if (deltaE < 0) {
                accept = true;
            } else {
                double prob = std::exp(-deltaE / temperature);
                accept = (uniform(gen) < prob);
            }

            if (accept) {
                D = D_new;
                currentEnergy = newEnergy;
            }

            // Cooling schedule
            temperature *= 0.99;

            result.iterationsUsed = iter + 1;

            // Check convergence
            double currentImbalance = computeImbalance(D);
            if (currentImbalance < config_.loadImbalanceBound) {
                result.converged = true;
                break;
            }
        }

        result.finalImbalance = computeImbalance(D);

        // Convert back to integer
        result.diffusedDemand.resize(npusCount_, std::vector<DataSize>(npusCount_, 0));
        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                result.diffusedDemand[i][j] = static_cast<DataSize>(std::round(D[i][j]));
            }
        }

        return result;
    }

    //=========================================================================
    // Gradient Descent Optimization
    // Minimizes load imbalance using gradient descent
    //=========================================================================
    DiffusionResult applyGradientDescent(const std::vector<std::vector<DataSize>>& demand) {
        DiffusionResult result;
        result.originalDemand = demand;
        result.iterationsUsed = 0;
        result.converged = false;

        // Convert to double
        std::vector<std::vector<double>> D(npusCount_, std::vector<double>(npusCount_, 0));
        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                D[i][j] = demand[i][j];
            }
        }

        result.initialImbalance = computeImbalance(D);

        double learningRate = 0.01;

        for (int iter = 0; iter < config_.maxIterations; ++iter) {
            // Compute edge loads with current demand
            auto edgeLoads = computeEdgeLoads(D);

            // Find max and average edge load
            double maxLoad = 0, totalLoad = 0;
            int edgeCount = 0;
            for (int i = 0; i < npusCount_; ++i) {
                for (int j = 0; j < npusCount_; ++j) {
                    if (edgeLoads[i][j] > 0) {
                        maxLoad = std::max(maxLoad, edgeLoads[i][j]);
                        totalLoad += edgeLoads[i][j];
                        edgeCount++;
                    }
                }
            }
            double avgLoad = (edgeCount > 0) ? totalLoad / edgeCount : 0;

            // Gradient: reduce demand that contributes to overloaded edges
            auto D_new = D;
            for (int src = 0; src < npusCount_; ++src) {
                for (int dst = 0; dst < npusCount_; ++dst) {
                    if (src == dst || D[src][dst] <= 0) continue;

                    // Get path for this demand
                    auto path = dimensionOrderPath(src, dst);

                    // Check if path uses overloaded edges
                    double maxPathLoad = 0;
                    for (size_t k = 0; k + 1 < path.size(); ++k) {
                        maxPathLoad = std::max(maxPathLoad, edgeLoads[path[k]][path[k + 1]]);
                    }

                    // If path is overloaded, reduce demand and redistribute
                    if (maxPathLoad > avgLoad * 1.2) {
                        double reduction = D[src][dst] * learningRate * (maxPathLoad / avgLoad - 1.0);
                        reduction = std::min(reduction, D[src][dst] * 0.5);  // Don't reduce too much

                        D_new[src][dst] -= reduction;

                        // Redistribute to neighbors
                        auto neighbors_src = getNeighbors(src);
                        auto neighbors_dst = getNeighbors(dst);

                        if (!neighbors_src.empty()) {
                            double perNeighbor = reduction / neighbors_src.size();
                            for (int ns : neighbors_src) {
                                if (ns != dst) {
                                    D_new[ns][dst] += perNeighbor;
                                }
                            }
                        }
                    }
                }
            }

            // Ensure non-negative
            for (int i = 0; i < npusCount_; ++i) {
                for (int j = 0; j < npusCount_; ++j) {
                    D_new[i][j] = std::max(0.0, D_new[i][j]);
                }
            }

            // Normalize if needed
            if (config_.preserveTotalDemand) {
                double originalTotal = 0, newTotal = 0;
                for (int i = 0; i < npusCount_; ++i) {
                    for (int j = 0; j < npusCount_; ++j) {
                        if (i != j) {
                            originalTotal += D[i][j];
                            newTotal += D_new[i][j];
                        }
                    }
                }
                if (newTotal > 0) {
                    double scale = originalTotal / newTotal;
                    for (int i = 0; i < npusCount_; ++i) {
                        for (int j = 0; j < npusCount_; ++j) {
                            D_new[i][j] *= scale;
                        }
                    }
                }
            }

            D = D_new;
            result.iterationsUsed = iter + 1;

            double currentImbalance = computeImbalance(D);
            if (currentImbalance < config_.loadImbalanceBound) {
                result.converged = true;
                break;
            }
        }

        result.finalImbalance = computeImbalance(D);

        // Convert back to integer
        result.diffusedDemand.resize(npusCount_, std::vector<DataSize>(npusCount_, 0));
        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                result.diffusedDemand[i][j] = static_cast<DataSize>(std::round(D[i][j]));
            }
        }

        return result;
    }

    //=========================================================================
    // Automatic diffusion method selection
    // Enhanced with 2D Mesh/Torus optimizations
    //=========================================================================
    DiffusionResult applyDiffusion(const std::vector<std::vector<DataSize>>& demand,
                                   const MatrixAnalysisResult& analysis) {
        // For 2D topologies with optimization enabled, use specialized method
        if (config_.enable2DOptimization && dimensions_ == 2) {
            std::cout << "Using 2D-optimized heat diffusion for Mesh/Torus\n";
            return apply2DOptimizedDiffusion(demand, analysis);
        }

        // Choose method based on analysis
        if (analysis.congestion.congestionScore > 0.7) {
            // Severe congestion: use Ising model (more aggressive)
            std::cout << "Using Ising model for severe congestion\n";
            return applyIsingModel(demand);
        } else if (analysis.hotspotCount > npusCount_ * 0.2) {
            // Many hotspots: use gradient descent
            std::cout << "Using gradient descent for hotspot reduction\n";
            return applyGradientDescent(demand);
        } else {
            // Default: heat diffusion (gentle smoothing)
            std::cout << "Using heat diffusion for load balancing\n";
            return applyHeatDiffusion(demand);
        }
    }

    //=========================================================================
    // 2D-Optimized Heat Diffusion for Mesh/Torus
    // Uses adaptive coefficient and locality-aware smoothing
    //=========================================================================
    DiffusionResult apply2DOptimizedDiffusion(const std::vector<std::vector<DataSize>>& demand,
                                               const MatrixAnalysisResult& analysis) {
        DiffusionResult result;
        result.originalDemand = demand;
        result.iterationsUsed = 0;
        result.converged = false;

        // Convert to double for computation
        std::vector<std::vector<double>> D(npusCount_, std::vector<double>(npusCount_, 0));
        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                D[i][j] = demand[i][j];
            }
        }

        result.initialImbalance = computeImbalance(D);

        // Adaptive diffusion coefficient based on initial imbalance
        double alpha = config_.diffusionCoefficient;
        if (config_.adaptiveCoefficient) {
            // Higher imbalance -> higher diffusion coefficient (faster convergence)
            alpha = config_.diffusionCoefficient * std::min(2.0, result.initialImbalance / 2.0);
        }

        // Precompute 2D neighbor structure
        std::vector<std::vector<int>> directNeighbors(npusCount_);
        std::vector<std::vector<int>> diagonalNeighbors(npusCount_);
        for (int i = 0; i < npusCount_; ++i) {
            auto coord = idToCoord(i);

            // Direct neighbors (up, down, left, right)
            for (int d = 0; d < 2; ++d) {
                auto newCoord = coord;
                // Positive direction
                if (isTorus_) {
                    newCoord[d] = (coord[d] + 1) % shape_[d];
                    directNeighbors[i].push_back(coordToId(newCoord));
                } else if (coord[d] + 1 < shape_[d]) {
                    newCoord[d] = coord[d] + 1;
                    directNeighbors[i].push_back(coordToId(newCoord));
                }
                // Negative direction
                newCoord = coord;
                if (isTorus_) {
                    newCoord[d] = (coord[d] - 1 + shape_[d]) % shape_[d];
                    directNeighbors[i].push_back(coordToId(newCoord));
                } else if (coord[d] > 0) {
                    newCoord[d] = coord[d] - 1;
                    directNeighbors[i].push_back(coordToId(newCoord));
                }
            }
        }

        // Iterative diffusion with 2D optimization
        for (int iter = 0; iter < config_.maxIterations; ++iter) {
            auto D_new = D;
            double maxChange = 0;

            // Apply 2D-aware Laplacian smoothing
            for (int i = 0; i < npusCount_; ++i) {
                for (int j = 0; j < npusCount_; ++j) {
                    if (i == j) continue;

                    // Compute weighted Laplacian
                    double laplacian = 0;
                    double totalWeight = 0;

                    // Direct neighbors in source dimension (higher weight)
                    for (int ni : directNeighbors[i]) {
                        double weight = config_.localityWeight;
                        laplacian += weight * (D[ni][j] - D[i][j]);
                        totalWeight += weight;
                    }

                    // Direct neighbors in destination dimension
                    for (int nj : directNeighbors[j]) {
                        double weight = config_.localityWeight;
                        laplacian += weight * (D[i][nj] - D[i][j]);
                        totalWeight += weight;
                    }

                    if (totalWeight > 0) {
                        double delta = alpha * laplacian / totalWeight;
                        D_new[i][j] = std::max(0.0, D[i][j] + delta);
                        maxChange = std::max(maxChange, std::abs(delta));
                    }
                }
            }

            // Normalize to preserve total demand
            if (config_.preserveTotalDemand) {
                double originalTotal = 0, newTotal = 0;
                for (int i = 0; i < npusCount_; ++i) {
                    for (int j = 0; j < npusCount_; ++j) {
                        if (i != j) {
                            originalTotal += D[i][j];
                            newTotal += D_new[i][j];
                        }
                    }
                }
                if (newTotal > 0) {
                    double scale = originalTotal / newTotal;
                    for (int i = 0; i < npusCount_; ++i) {
                        for (int j = 0; j < npusCount_; ++j) {
                            D_new[i][j] *= scale;
                        }
                    }
                }
            }

            D = D_new;
            result.iterationsUsed = iter + 1;

            // Check convergence
            double currentImbalance = computeImbalance(D);
            if (currentImbalance < config_.loadImbalanceBound ||
                maxChange < config_.convergenceThreshold) {
                result.converged = true;
                break;
            }

            // Adaptive coefficient decay for stability
            if (config_.adaptiveCoefficient && iter > 0 && iter % 20 == 0) {
                alpha *= 0.9;  // Gradually reduce for fine-tuning
            }
        }

        result.finalImbalance = computeImbalance(D);

        // Convert back to integer
        result.diffusedDemand.resize(npusCount_, std::vector<DataSize>(npusCount_, 0));
        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                result.diffusedDemand[i][j] = static_cast<DataSize>(std::round(D[i][j]));
            }
        }

        return result;
    }

private:
    //=========================================================================
    // Build adjacency list
    //=========================================================================
    void buildAdjacency() {
        adjacency_.resize(npusCount_);
        for (int i = 0; i < npusCount_; ++i) {
            adjacency_[i] = getNeighbors(i);
        }
    }

    //=========================================================================
    // Get neighbors of a node
    //=========================================================================
    std::vector<int> getNeighbors(int nodeId) const {
        std::vector<int> neighbors;
        auto coord = idToCoord(nodeId);

        for (int d = 0; d < dimensions_; ++d) {
            auto newCoord = coord;

            // Positive direction
            if (isTorus_) {
                newCoord[d] = (coord[d] + 1) % shape_[d];
                neighbors.push_back(coordToId(newCoord));
            } else if (coord[d] + 1 < shape_[d]) {
                newCoord[d] = coord[d] + 1;
                neighbors.push_back(coordToId(newCoord));
            }

            // Negative direction
            newCoord = coord;
            if (isTorus_) {
                newCoord[d] = (coord[d] - 1 + shape_[d]) % shape_[d];
                neighbors.push_back(coordToId(newCoord));
            } else if (coord[d] > 0) {
                newCoord[d] = coord[d] - 1;
                neighbors.push_back(coordToId(newCoord));
            }
        }

        return neighbors;
    }

    //=========================================================================
    // Compute load imbalance (max/avg ratio)
    //=========================================================================
    double computeImbalance(const std::vector<std::vector<double>>& D) const {
        auto edgeLoads = computeEdgeLoads(D);

        double maxLoad = 0, totalLoad = 0;
        int edgeCount = 0;

        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                if (edgeLoads[i][j] > 0) {
                    maxLoad = std::max(maxLoad, edgeLoads[i][j]);
                    totalLoad += edgeLoads[i][j];
                    edgeCount++;
                }
            }
        }

        double avgLoad = (edgeCount > 0) ? totalLoad / edgeCount : 0;
        return (avgLoad > 0) ? maxLoad / avgLoad : 1.0;
    }

    //=========================================================================
    // Compute edge loads using dimension-order routing
    //=========================================================================
    std::vector<std::vector<double>> computeEdgeLoads(const std::vector<std::vector<double>>& D) const {
        std::vector<std::vector<double>> loads(npusCount_, std::vector<double>(npusCount_, 0));

        for (int src = 0; src < npusCount_; ++src) {
            for (int dst = 0; dst < npusCount_; ++dst) {
                if (src == dst || D[src][dst] <= 0) continue;

                auto path = dimensionOrderPath(src, dst);
                for (size_t k = 0; k + 1 < path.size(); ++k) {
                    loads[path[k]][path[k + 1]] += D[src][dst];
                }
            }
        }

        return loads;
    }

    //=========================================================================
    // Compute energy for Ising model (sum of squared deviations)
    //=========================================================================
    double computeEnergy(const std::vector<std::vector<double>>& D) const {
        auto edgeLoads = computeEdgeLoads(D);

        double totalLoad = 0;
        int edgeCount = 0;
        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                if (edgeLoads[i][j] > 0) {
                    totalLoad += edgeLoads[i][j];
                    edgeCount++;
                }
            }
        }

        double avgLoad = (edgeCount > 0) ? totalLoad / edgeCount : 0;

        // Energy = sum of squared deviations from mean
        double energy = 0;
        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                if (edgeLoads[i][j] > 0) {
                    double deviation = edgeLoads[i][j] - avgLoad;
                    energy += deviation * deviation;
                }
            }
        }

        return energy;
    }

    //=========================================================================
    // Dimension-order routing path
    //=========================================================================
    std::vector<int> dimensionOrderPath(int src, int dst) const {
        std::vector<int> path;
        path.push_back(src);

        if (src == dst) return path;

        auto srcCoord = idToCoord(src);
        auto dstCoord = idToCoord(dst);
        auto currentCoord = srcCoord;

        for (int d = 0; d < dimensions_; ++d) {
            while (currentCoord[d] != dstCoord[d]) {
                int diff = dstCoord[d] - currentCoord[d];

                if (isTorus_) {
                    int posDist = (dstCoord[d] - currentCoord[d] + shape_[d]) % shape_[d];
                    int negDist = (currentCoord[d] - dstCoord[d] + shape_[d]) % shape_[d];

                    if (posDist <= negDist) {
                        currentCoord[d] = (currentCoord[d] + 1) % shape_[d];
                    } else {
                        currentCoord[d] = (currentCoord[d] - 1 + shape_[d]) % shape_[d];
                    }
                } else {
                    if (diff > 0) {
                        currentCoord[d]++;
                    } else {
                        currentCoord[d]--;
                    }
                }

                path.push_back(coordToId(currentCoord));
            }
        }

        return path;
    }

    //=========================================================================
    // Coordinate conversion
    //=========================================================================
    std::vector<int> idToCoord(int id) const {
        std::vector<int> coord(dimensions_);
        for (int d = 0; d < dimensions_; ++d) {
            coord[d] = (id / strides_[d]) % shape_[d];
        }
        return coord;
    }

    int coordToId(const std::vector<int>& coord) const {
        int id = 0;
        for (int d = 0; d < dimensions_; ++d) {
            id += coord[d] * strides_[d];
        }
        return id;
    }

    std::vector<int> shape_;
    std::vector<int> strides_;
    int dimensions_;
    int npusCount_;
    bool isTorus_;
    DiffusionConfig config_;
    std::vector<std::vector<int>> adjacency_;
};

} // namespace tacos
