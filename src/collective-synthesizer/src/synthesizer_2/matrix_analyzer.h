/*
# File name  :    matrix_analyzer.h
# Author     :    Chen Chen
# Time       :    2025/01/06
# Description:    Matrix analysis module for non-uniform AllToAll
#                 - Sparsity analysis
#                 - Hotspot distribution analysis
#                 - Symmetry detection (uniform/axis-symmetric/center-symmetric)
#                 - Congestion prediction
*/

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <unordered_map>

namespace tacos {

//=============================================================================
// MatrixSymmetryType: Classification of demand matrix symmetry
//=============================================================================
enum class MatrixSymmetryType {
    UNIFORM,           // All non-diagonal elements are equal
    AXIS_SYMMETRIC,    // D[i][j] = D[j][i] (symmetric matrix)
    CENTER_SYMMETRIC,  // D[i][j] = D[P-1-i][P-1-j]
    ROW_UNIFORM,       // Each row has same sum
    COL_UNIFORM,       // Each column has same sum
    SPARSE,            // Most elements are zero
    GENERAL            // No special structure
};

//=============================================================================
// HotspotInfo: Information about traffic hotspots
//=============================================================================
struct HotspotInfo {
    int nodeId;
    long long outgoingTraffic;  // Total bytes this node sends
    long long incomingTraffic;  // Total bytes this node receives
    double outgoingRatio;       // Ratio to average
    double incomingRatio;       // Ratio to average

    bool isHotspot(double threshold = 2.0) const {
        return outgoingRatio > threshold || incomingRatio > threshold;
    }
};

//=============================================================================
// CongestionPrediction: Predicted congestion level
//=============================================================================
struct CongestionPrediction {
    double congestionScore;      // 0.0 (no congestion) to 1.0 (severe)
    bool needsDiffusion;         // Whether to apply diffusion algorithm
    std::vector<std::pair<int, int>> hotEdges;  // Predicted hot edges

    std::string severityLevel() const {
        if (congestionScore < 0.3) return "LOW";
        if (congestionScore < 0.6) return "MEDIUM";
        return "HIGH";
    }
};

//=============================================================================
// MatrixAnalysisResult: Complete analysis result
//=============================================================================
struct MatrixAnalysisResult {
    // Basic statistics
    int npusCount;
    long long totalDemand;
    long long maxDemand;
    long long minNonZeroDemand;
    double avgDemand;
    double stdDemand;

    // Sparsity
    double sparsity;             // Fraction of zero elements
    int nonZeroCount;

    // Row/Column statistics
    long long maxRowSum;
    long long minRowSum;
    long long maxColSum;
    long long minColSum;
    double rowSumVariance;
    double colSumVariance;

    // Symmetry
    std::vector<MatrixSymmetryType> symmetryTypes;
    double symmetryScore;        // 0.0 (asymmetric) to 1.0 (perfectly symmetric)

    // Hotspots
    std::vector<HotspotInfo> hotspots;
    int hotspotCount;
    double hotspotConcentration; // Gini coefficient of traffic distribution

    // Congestion prediction
    CongestionPrediction congestion;

    // Non-Uniformity Score (NUS) for algorithm selection
    double nonUniformityScore;       // 0.0 (uniform) to 1.0 (standardly non-uniform)
    double sparsityScore;            // Normalized sparsity component
    double hotspotScore;             // Normalized hotspot component
    double giniScore;                // Normalized Gini component
    double edgeImbalanceScore;       // Normalized edge load imbalance component
    int recommendedTier;             // 0-4, recommended algorithm tier

    void print() const {
        std::cout << "\n========== Matrix Analysis Result ==========\n";
        std::cout << "NPUs: " << npusCount << "\n";
        std::cout << "Total Demand: " << totalDemand << " bytes\n";
        std::cout << "Max/Min/Avg Demand: " << maxDemand << "/"
                  << minNonZeroDemand << "/" << std::fixed << std::setprecision(2)
                  << avgDemand << "\n";
        std::cout << "Std Deviation: " << stdDemand << "\n";
        std::cout << "\n--- Sparsity ---\n";
        std::cout << "Sparsity: " << sparsity * 100 << "%\n";
        std::cout << "Non-zero elements: " << nonZeroCount << "/"
                  << (npusCount * npusCount - npusCount) << "\n";
        std::cout << "\n--- Row/Column Balance ---\n";
        std::cout << "Max/Min Row Sum: " << maxRowSum << "/" << minRowSum << "\n";
        std::cout << "Max/Min Col Sum: " << maxColSum << "/" << minColSum << "\n";
        std::cout << "Row Sum Variance: " << rowSumVariance << "\n";
        std::cout << "Col Sum Variance: " << colSumVariance << "\n";
        std::cout << "\n--- Symmetry ---\n";
        std::cout << "Symmetry Score: " << symmetryScore << "\n";
        std::cout << "Detected Types: ";
        for (auto type : symmetryTypes) {
            switch (type) {
                case MatrixSymmetryType::UNIFORM: std::cout << "UNIFORM "; break;
                case MatrixSymmetryType::AXIS_SYMMETRIC: std::cout << "AXIS_SYMMETRIC "; break;
                case MatrixSymmetryType::CENTER_SYMMETRIC: std::cout << "CENTER_SYMMETRIC "; break;
                case MatrixSymmetryType::ROW_UNIFORM: std::cout << "ROW_UNIFORM "; break;
                case MatrixSymmetryType::COL_UNIFORM: std::cout << "COL_UNIFORM "; break;
                case MatrixSymmetryType::SPARSE: std::cout << "SPARSE "; break;
                case MatrixSymmetryType::GENERAL: std::cout << "GENERAL "; break;
            }
        }
        std::cout << "\n";
        std::cout << "\n--- Hotspots ---\n";
        std::cout << "Hotspot Count: " << hotspotCount << "\n";
        std::cout << "Hotspot Concentration (Gini): " << hotspotConcentration << "\n";
        if (!hotspots.empty()) {
            std::cout << "Top Hotspots:\n";
            int count = std::min(5, (int)hotspots.size());
            for (int i = 0; i < count; ++i) {
                const auto& h = hotspots[i];
                std::cout << "  Node " << h.nodeId << ": out=" << h.outgoingTraffic
                          << " (" << h.outgoingRatio << "x), in=" << h.incomingTraffic
                          << " (" << h.incomingRatio << "x)\n";
            }
        }
        std::cout << "\n--- Congestion Prediction ---\n";
        std::cout << "Congestion Score: " << congestion.congestionScore << "\n";
        std::cout << "Severity: " << congestion.severityLevel() << "\n";
        std::cout << "Needs Diffusion: " << (congestion.needsDiffusion ? "YES" : "NO") << "\n";
        std::cout << "\n--- Non-Uniformity Score (NUS) ---\n";
        std::cout << "NUS: " << nonUniformityScore << "\n";
        std::cout << "  Sparsity Score: " << sparsityScore << "\n";
        std::cout << "  Hotspot Score: " << hotspotScore << "\n";
        std::cout << "  Gini Score: " << giniScore << "\n";
        std::cout << "  Edge Imbalance Score: " << edgeImbalanceScore << "\n";
        std::cout << "Recommended Tier: " << recommendedTier << " (";
        switch (recommendedTier) {
            case 0: std::cout << "DOR_XY"; break;
            case 1: std::cout << "DOR_XY_BINPACK"; break;
            case 2: std::cout << "LOAD_BALANCED"; break;
            case 3: std::cout << "MULTI_PATH"; break;
            case 4: std::cout << "DIFFUSION_ADAPTIVE"; break;
        }
        std::cout << ")\n";
        std::cout << "=============================================\n";
    }
};

//=============================================================================
// MatrixAnalyzer: Analyzes demand matrix properties
//=============================================================================
class MatrixAnalyzer {
public:
    using DataSize = long long;

    MatrixAnalyzer(const std::vector<int>& shape, bool isTorus)
        : shape_(shape), isTorus_(isTorus) {
        npusCount_ = 1;
        for (int d : shape) {
            npusCount_ *= d;
        }
        dimensions_ = shape.size();

        // Compute strides for coordinate conversion
        strides_.resize(dimensions_);
        int stride = 1;
        for (int d = dimensions_ - 1; d >= 0; --d) {
            strides_[d] = stride;
            stride *= shape_[d];
        }
    }

    //=========================================================================
    // Main analysis function
    //=========================================================================
    MatrixAnalysisResult analyze(const std::vector<std::vector<DataSize>>& demand) {
        MatrixAnalysisResult result;
        result.npusCount = npusCount_;

        // Basic statistics
        computeBasicStats(demand, result);

        // Sparsity analysis
        computeSparsity(demand, result);

        // Row/Column statistics
        computeRowColStats(demand, result);

        // Symmetry detection
        detectSymmetry(demand, result);

        // Hotspot analysis
        analyzeHotspots(demand, result);

        // Congestion prediction
        predictCongestion(demand, result);

        // Compute Non-Uniformity Score (NUS)
        computeNUS(demand, result);

        return result;
    }

private:
    //=========================================================================
    // Basic statistics
    //=========================================================================
    void computeBasicStats(const std::vector<std::vector<DataSize>>& demand,
                           MatrixAnalysisResult& result) {
        DataSize total = 0;
        DataSize maxVal = 0;
        DataSize minNonZero = std::numeric_limits<DataSize>::max();
        int count = 0;
        std::vector<DataSize> values;

        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                if (i == j) continue;
                DataSize val = demand[i][j];
                total += val;
                maxVal = std::max(maxVal, val);
                if (val > 0) {
                    minNonZero = std::min(minNonZero, val);
                    values.push_back(val);
                }
                count++;
            }
        }

        result.totalDemand = total;
        result.maxDemand = maxVal;
        result.minNonZeroDemand = (minNonZero == std::numeric_limits<DataSize>::max()) ? 0 : minNonZero;
        result.avgDemand = (count > 0) ? (double)total / count : 0;

        // Standard deviation
        double variance = 0;
        for (DataSize val : values) {
            double diff = val - result.avgDemand;
            variance += diff * diff;
        }
        result.stdDemand = (values.size() > 0) ? std::sqrt(variance / values.size()) : 0;
    }

    //=========================================================================
    // Sparsity analysis
    //=========================================================================
    void computeSparsity(const std::vector<std::vector<DataSize>>& demand,
                         MatrixAnalysisResult& result) {
        int zeroCount = 0;
        int nonZeroCount = 0;
        int totalElements = npusCount_ * npusCount_ - npusCount_; // Exclude diagonal

        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                if (i == j) continue;
                if (demand[i][j] == 0) {
                    zeroCount++;
                } else {
                    nonZeroCount++;
                }
            }
        }

        result.sparsity = (totalElements > 0) ? (double)zeroCount / totalElements : 0;
        result.nonZeroCount = nonZeroCount;
    }

    //=========================================================================
    // Row/Column statistics
    //=========================================================================
    void computeRowColStats(const std::vector<std::vector<DataSize>>& demand,
                            MatrixAnalysisResult& result) {
        std::vector<DataSize> rowSums(npusCount_, 0);
        std::vector<DataSize> colSums(npusCount_, 0);

        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                if (i == j) continue;
                rowSums[i] += demand[i][j];
                colSums[j] += demand[i][j];
            }
        }

        result.maxRowSum = *std::max_element(rowSums.begin(), rowSums.end());
        result.minRowSum = *std::min_element(rowSums.begin(), rowSums.end());
        result.maxColSum = *std::max_element(colSums.begin(), colSums.end());
        result.minColSum = *std::min_element(colSums.begin(), colSums.end());

        // Variance
        double avgRow = (double)result.totalDemand / npusCount_;
        double avgCol = avgRow;

        double rowVar = 0, colVar = 0;
        for (int i = 0; i < npusCount_; ++i) {
            rowVar += (rowSums[i] - avgRow) * (rowSums[i] - avgRow);
            colVar += (colSums[i] - avgCol) * (colSums[i] - avgCol);
        }
        result.rowSumVariance = rowVar / npusCount_;
        result.colSumVariance = colVar / npusCount_;
    }

    //=========================================================================
    // Symmetry detection
    //=========================================================================
    void detectSymmetry(const std::vector<std::vector<DataSize>>& demand,
                        MatrixAnalysisResult& result) {
        result.symmetryTypes.clear();

        // Check for uniform
        bool isUniform = true;
        DataSize firstVal = -1;
        for (int i = 0; i < npusCount_ && isUniform; ++i) {
            for (int j = 0; j < npusCount_ && isUniform; ++j) {
                if (i == j) continue;
                if (firstVal < 0) {
                    firstVal = demand[i][j];
                } else if (demand[i][j] != firstVal) {
                    isUniform = false;
                }
            }
        }
        if (isUniform) {
            result.symmetryTypes.push_back(MatrixSymmetryType::UNIFORM);
        }

        // Check for axis symmetry: D[i][j] = D[j][i]
        bool isAxisSymmetric = true;
        double axisSymmetryError = 0;
        int axisCount = 0;
        for (int i = 0; i < npusCount_; ++i) {
            for (int j = i + 1; j < npusCount_; ++j) {
                DataSize diff = std::abs(demand[i][j] - demand[j][i]);
                DataSize maxVal = std::max(demand[i][j], demand[j][i]);
                if (maxVal > 0) {
                    axisSymmetryError += (double)diff / maxVal;
                    axisCount++;
                }
                if (demand[i][j] != demand[j][i]) {
                    isAxisSymmetric = false;
                }
            }
        }
        if (isAxisSymmetric) {
            result.symmetryTypes.push_back(MatrixSymmetryType::AXIS_SYMMETRIC);
        }

        // Check for center symmetry: D[i][j] = D[P-1-i][P-1-j]
        bool isCenterSymmetric = true;
        for (int i = 0; i < npusCount_ && isCenterSymmetric; ++i) {
            for (int j = 0; j < npusCount_ && isCenterSymmetric; ++j) {
                if (i == j) continue;
                int ci = npusCount_ - 1 - i;
                int cj = npusCount_ - 1 - j;
                if (demand[i][j] != demand[ci][cj]) {
                    isCenterSymmetric = false;
                }
            }
        }
        if (isCenterSymmetric) {
            result.symmetryTypes.push_back(MatrixSymmetryType::CENTER_SYMMETRIC);
        }

        // Check for row uniformity
        bool isRowUniform = true;
        DataSize firstRowSum = -1;
        for (int i = 0; i < npusCount_; ++i) {
            DataSize rowSum = 0;
            for (int j = 0; j < npusCount_; ++j) {
                if (i != j) rowSum += demand[i][j];
            }
            if (firstRowSum < 0) {
                firstRowSum = rowSum;
            } else if (rowSum != firstRowSum) {
                isRowUniform = false;
                break;
            }
        }
        if (isRowUniform) {
            result.symmetryTypes.push_back(MatrixSymmetryType::ROW_UNIFORM);
        }

        // Check for column uniformity
        bool isColUniform = true;
        DataSize firstColSum = -1;
        for (int j = 0; j < npusCount_; ++j) {
            DataSize colSum = 0;
            for (int i = 0; i < npusCount_; ++i) {
                if (i != j) colSum += demand[i][j];
            }
            if (firstColSum < 0) {
                firstColSum = colSum;
            } else if (colSum != firstColSum) {
                isColUniform = false;
                break;
            }
        }
        if (isColUniform) {
            result.symmetryTypes.push_back(MatrixSymmetryType::COL_UNIFORM);
        }

        // Check for sparsity
        if (result.sparsity > 0.5) {
            result.symmetryTypes.push_back(MatrixSymmetryType::SPARSE);
        }

        // If no special structure found
        if (result.symmetryTypes.empty()) {
            result.symmetryTypes.push_back(MatrixSymmetryType::GENERAL);
        }

        // Compute overall symmetry score (based on axis symmetry)
        result.symmetryScore = (axisCount > 0) ? 1.0 - (axisSymmetryError / axisCount) : 1.0;
    }

    //=========================================================================
    // Hotspot analysis
    //=========================================================================
    void analyzeHotspots(const std::vector<std::vector<DataSize>>& demand,
                         MatrixAnalysisResult& result) {
        std::vector<DataSize> outgoing(npusCount_, 0);
        std::vector<DataSize> incoming(npusCount_, 0);

        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                if (i == j) continue;
                outgoing[i] += demand[i][j];
                incoming[j] += demand[i][j];
            }
        }

        double avgOut = (double)result.totalDemand / npusCount_;
        double avgIn = avgOut;

        result.hotspots.clear();
        for (int i = 0; i < npusCount_; ++i) {
            HotspotInfo info;
            info.nodeId = i;
            info.outgoingTraffic = outgoing[i];
            info.incomingTraffic = incoming[i];
            info.outgoingRatio = (avgOut > 0) ? outgoing[i] / avgOut : 0;
            info.incomingRatio = (avgIn > 0) ? incoming[i] / avgIn : 0;
            result.hotspots.push_back(info);
        }

        // Sort by max ratio
        std::sort(result.hotspots.begin(), result.hotspots.end(),
            [](const HotspotInfo& a, const HotspotInfo& b) {
                return std::max(a.outgoingRatio, a.incomingRatio) >
                       std::max(b.outgoingRatio, b.incomingRatio);
            });

        // Count hotspots (ratio > 2.0)
        result.hotspotCount = 0;
        for (const auto& h : result.hotspots) {
            if (h.isHotspot(2.0)) {
                result.hotspotCount++;
            }
        }

        // Compute Gini coefficient for traffic concentration
        result.hotspotConcentration = computeGiniCoefficient(outgoing, incoming);
    }

    //=========================================================================
    // Gini coefficient computation
    //=========================================================================
    double computeGiniCoefficient(const std::vector<DataSize>& outgoing,
                                  const std::vector<DataSize>& incoming) {
        // Combine outgoing and incoming traffic
        std::vector<DataSize> traffic(npusCount_);
        for (int i = 0; i < npusCount_; ++i) {
            traffic[i] = outgoing[i] + incoming[i];
        }

        std::sort(traffic.begin(), traffic.end());

        DataSize totalTraffic = 0;
        for (DataSize t : traffic) {
            totalTraffic += t;
        }

        if (totalTraffic == 0) return 0;

        // Gini = 1 - 2 * (area under Lorenz curve)
        double cumulativeSum = 0;
        double giniSum = 0;
        for (int i = 0; i < npusCount_; ++i) {
            cumulativeSum += traffic[i];
            giniSum += cumulativeSum;
        }

        double gini = 1.0 - 2.0 * giniSum / (npusCount_ * totalTraffic) + 1.0 / npusCount_;
        return std::max(0.0, std::min(1.0, gini));
    }

    //=========================================================================
    // Congestion prediction
    //=========================================================================
    void predictCongestion(const std::vector<std::vector<DataSize>>& demand,
                           MatrixAnalysisResult& result) {
        // Predict edge loads using dimension-order routing
        std::vector<std::vector<DataSize>> edgeLoads(npusCount_,
            std::vector<DataSize>(npusCount_, 0));

        for (int src = 0; src < npusCount_; ++src) {
            for (int dst = 0; dst < npusCount_; ++dst) {
                if (src == dst || demand[src][dst] == 0) continue;

                // Simulate dimension-order routing
                std::vector<int> path = dimensionOrderRoute(src, dst);
                for (size_t i = 0; i + 1 < path.size(); ++i) {
                    edgeLoads[path[i]][path[i + 1]] += demand[src][dst];
                }
            }
        }

        // Find max edge load
        DataSize maxEdgeLoad = 0;
        DataSize totalEdgeLoad = 0;
        int edgeCount = 0;

        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                if (edgeLoads[i][j] > 0) {
                    maxEdgeLoad = std::max(maxEdgeLoad, edgeLoads[i][j]);
                    totalEdgeLoad += edgeLoads[i][j];
                    edgeCount++;
                }
            }
        }

        double avgEdgeLoad = (edgeCount > 0) ? (double)totalEdgeLoad / edgeCount : 0;

        // Congestion score based on load imbalance
        double loadImbalance = (avgEdgeLoad > 0) ? (double)maxEdgeLoad / avgEdgeLoad : 1.0;

        // Also consider row/column imbalance
        double rowImbalance = (result.minRowSum > 0) ?
            (double)result.maxRowSum / result.minRowSum : 1.0;
        double colImbalance = (result.minColSum > 0) ?
            (double)result.maxColSum / result.minColSum : 1.0;

        // Combined congestion score
        double score = 0.0;
        score += 0.4 * std::min(1.0, (loadImbalance - 1.0) / 5.0);  // Edge load imbalance
        score += 0.2 * std::min(1.0, (rowImbalance - 1.0) / 5.0);   // Row imbalance
        score += 0.2 * std::min(1.0, (colImbalance - 1.0) / 5.0);   // Column imbalance
        score += 0.2 * result.hotspotConcentration;                  // Hotspot concentration

        result.congestion.congestionScore = std::min(1.0, score);

        // Determine if diffusion is needed
        // Threshold: congestion score > 0.4 or hotspot count > 20% of nodes
        result.congestion.needsDiffusion =
            (result.congestion.congestionScore > 0.4) ||
            (result.hotspotCount > npusCount_ * 0.2);

        // Find hot edges (load > 2x average)
        result.congestion.hotEdges.clear();
        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                if (edgeLoads[i][j] > 2 * avgEdgeLoad) {
                    result.congestion.hotEdges.push_back({i, j});
                }
            }
        }
    }

    //=========================================================================
    // Dimension-order routing (for congestion prediction)
    //=========================================================================
    std::vector<int> dimensionOrderRoute(int src, int dst) const {
        std::vector<int> path;
        path.push_back(src);

        if (src == dst) return path;

        std::vector<int> srcCoord = idToCoord(src);
        std::vector<int> dstCoord = idToCoord(dst);
        std::vector<int> currentCoord = srcCoord;

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

    //=========================================================================
    // NUS Configuration - OLMoE percentile thresholds
    //=========================================================================
    struct NUSConfig {
        // Weights for NUS components
        double sparsityWeight = 0.25;
        double hotspotWeight = 0.35;
        double giniWeight = 0.25;
        double edgeImbalanceWeight = 0.15;

        // Sparsity percentile thresholds (from OLMoE analysis)
        double sparsityP20 = 0.14;
        double sparsityP40 = 0.40;
        double sparsityP60 = 0.60;
        double sparsityP80 = 0.76;

        // Hotspot ratio percentile thresholds
        double hotspotP20 = 2.94;
        double hotspotP40 = 3.58;
        double hotspotP60 = 3.90;
        double hotspotP80 = 4.18;

        // Gini coefficient percentile thresholds
        double giniP20 = 0.22;
        double giniP40 = 0.27;
        double giniP60 = 0.33;
        double giniP80 = 0.38;

        // Tier thresholds
        double tier0Max = 0.2;
        double tier1Max = 0.4;
        double tier2Max = 0.6;
        double tier3GiniThreshold = 0.35;
    };

    NUSConfig nusConfig_;

    //=========================================================================
    // Compute Non-Uniformity Score (NUS)
    //=========================================================================
    void computeNUS(const std::vector<std::vector<DataSize>>& demand,
                    MatrixAnalysisResult& result) {
        // 1. Compute normalized sparsity score
        result.sparsityScore = normalizeByPercentile(
            result.sparsity,
            nusConfig_.sparsityP20, nusConfig_.sparsityP40,
            nusConfig_.sparsityP60, nusConfig_.sparsityP80);

        // 2. Compute normalized hotspot score
        double maxHotspotRatio = 1.0;
        if (!result.hotspots.empty()) {
            maxHotspotRatio = std::max(result.hotspots[0].outgoingRatio,
                                       result.hotspots[0].incomingRatio);
        }
        result.hotspotScore = normalizeHotspotByPercentile(
            maxHotspotRatio,
            nusConfig_.hotspotP20, nusConfig_.hotspotP40,
            nusConfig_.hotspotP60, nusConfig_.hotspotP80);

        // 3. Compute normalized Gini score
        result.giniScore = normalizeByPercentile(
            result.hotspotConcentration,
            nusConfig_.giniP20, nusConfig_.giniP40,
            nusConfig_.giniP60, nusConfig_.giniP80);

        // 4. Compute edge imbalance score (from congestion prediction)
        result.edgeImbalanceScore = computeEdgeImbalanceScore(demand);

        // 5. Compute weighted NUS
        result.nonUniformityScore =
            nusConfig_.sparsityWeight * result.sparsityScore +
            nusConfig_.hotspotWeight * result.hotspotScore +
            nusConfig_.giniWeight * result.giniScore +
            nusConfig_.edgeImbalanceWeight * result.edgeImbalanceScore;

        // Clamp to [0, 1]
        result.nonUniformityScore = std::max(0.0, std::min(1.0, result.nonUniformityScore));

        // 6. Determine recommended tier
        result.recommendedTier = determineRecommendedTier(result);
    }

    //=========================================================================
    // Normalize value by percentile thresholds
    //=========================================================================
    double normalizeByPercentile(double value, double p20, double p40,
                                  double p60, double p80) const {
        if (value < p20) {
            return value / p20 * 0.2;
        } else if (value < p40) {
            return 0.2 + (value - p20) / (p40 - p20) * 0.2;
        } else if (value < p60) {
            return 0.4 + (value - p40) / (p60 - p40) * 0.2;
        } else if (value < p80) {
            return 0.6 + (value - p60) / (p80 - p60) * 0.2;
        } else {
            // Cap at 1.0
            return std::min(1.0, 0.8 + (value - p80) / (1.0 - p80) * 0.2);
        }
    }

    //=========================================================================
    // Normalize hotspot ratio (starts from 1.0, not 0)
    //=========================================================================
    double normalizeHotspotByPercentile(double ratio, double p20, double p40,
                                         double p60, double p80) const {
        // Hotspot ratio starts from 1.0 (uniform)
        if (ratio < p20) {
            return (ratio - 1.0) / (p20 - 1.0) * 0.2;
        } else if (ratio < p40) {
            return 0.2 + (ratio - p20) / (p40 - p20) * 0.2;
        } else if (ratio < p60) {
            return 0.4 + (ratio - p40) / (p60 - p40) * 0.2;
        } else if (ratio < p80) {
            return 0.6 + (ratio - p60) / (p80 - p60) * 0.2;
        } else {
            // Cap at 1.0
            return std::min(1.0, 0.8 + (ratio - p80) / (p80 - p60) * 0.2);
        }
    }

    //=========================================================================
    // Compute edge load imbalance score
    //=========================================================================
    double computeEdgeImbalanceScore(const std::vector<std::vector<DataSize>>& demand) const {
        // Predict edge loads using dimension-order routing
        std::vector<std::vector<DataSize>> edgeLoads(npusCount_,
            std::vector<DataSize>(npusCount_, 0));

        for (int src = 0; src < npusCount_; ++src) {
            for (int dst = 0; dst < npusCount_; ++dst) {
                if (src == dst || demand[src][dst] == 0) continue;

                std::vector<int> path = dimensionOrderRoute(src, dst);
                for (size_t i = 0; i + 1 < path.size(); ++i) {
                    edgeLoads[path[i]][path[i + 1]] += demand[src][dst];
                }
            }
        }

        // Find max and average edge load
        DataSize maxEdgeLoad = 0;
        DataSize totalEdgeLoad = 0;
        int edgeCount = 0;

        for (int i = 0; i < npusCount_; ++i) {
            for (int j = 0; j < npusCount_; ++j) {
                if (edgeLoads[i][j] > 0) {
                    maxEdgeLoad = std::max(maxEdgeLoad, edgeLoads[i][j]);
                    totalEdgeLoad += edgeLoads[i][j];
                    edgeCount++;
                }
            }
        }

        if (edgeCount == 0 || totalEdgeLoad == 0) return 0.0;

        double avgEdgeLoad = (double)totalEdgeLoad / edgeCount;
        double imbalanceRatio = (double)maxEdgeLoad / avgEdgeLoad;

        // Normalize: 1.0 = uniform, 5.0+ = standardly imbalanced
        return std::min(1.0, (imbalanceRatio - 1.0) / 4.0);
    }

    //=========================================================================
    // Determine recommended algorithm tier based on NUS
    //=========================================================================
    int determineRecommendedTier(const MatrixAnalysisResult& result) const {
        double nus = result.nonUniformityScore;

        if (nus < nusConfig_.tier0Max) {
            return 0;  // DOR_XY
        } else if (nus < nusConfig_.tier1Max) {
            return 1;  // DOR_XY_BINPACK
        } else if (nus < nusConfig_.tier2Max) {
            return 2;  // LOAD_BALANCED
        } else {
            // For high NUS, check Gini to decide between multi-path and diffusion
            if (result.hotspotConcentration < nusConfig_.tier3GiniThreshold) {
                return 3;  // MULTI_PATH
            } else {
                return 4;  // DIFFUSION_ADAPTIVE
            }
        }
    }
};

} // namespace tacos
