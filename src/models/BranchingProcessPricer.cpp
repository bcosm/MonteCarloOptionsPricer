#include "../../include/models/BranchingProcessPricer.h"
#include "../../include/core/common.h"  
#ifdef _OPENMP
#include <omp.h>
#endif
#include <random>
#include <limits>
#include <cmath>
#include <stdexcept>
#include <algorithm>

double BranchingProcesses::PredictOptionPrice(
    const std::vector<std::vector<double>>& pricePaths,
    double r,
    double strike,
    double maturity,
    double dt,
    bool isCall,
    int numBranches,
    const std::vector<int>& exerciseTimes)
{
    if (pricePaths.empty() || pricePaths[0].empty()) {
        throw std::runtime_error("BranchingProcesses: Empty pricePaths.");
    }
    if (exerciseTimes.empty()) {
        throw std::runtime_error("BranchingProcesses: No exercise times.");
    }
    if (strike <= 0.0) {
        throw std::runtime_error("BranchingProcesses: Strike must be positive.");
    }

    this->maturity_ = maturity;

    double lowerEstimate = ComputeLowerBound(pricePaths, r, strike, dt, isCall, exerciseTimes);

    double upperEstimate = ComputeUpperBound(pricePaths, r, strike, dt, isCall, numBranches, exerciseTimes);

    return 0.5 * (lowerEstimate + upperEstimate);
}

double BranchingProcesses::ComputeLowerBound(
    const std::vector<std::vector<double>>& pricePaths,
    double r,
    double strike,
    double dt,
    bool isCall,
    const std::vector<int>& exerciseTimes)
{
    int N = static_cast<int>(pricePaths.size());
    double sumPayoffs = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:sumPayoffs)
#endif
    for (int i = 0; i < N; ++i) {
        double bestValue = 0.0;
        for (int tIdx : exerciseTimes) {
            double t = tIdx * dt;
            if (t > this->maturity_) {
                break;
            }
            double payoff = PayoffFunction(isCall, pricePaths[i][tIdx], strike);
            double discPayoff = std::exp(-r * t) * payoff;
            if (discPayoff > bestValue) {
                bestValue = discPayoff;
                break;
            }
        }
        sumPayoffs += bestValue;
    }
    return sumPayoffs / N;
}

double BranchingProcesses::ComputeUpperBound(
    const std::vector<std::vector<double>>& pricePaths,
    double r,
    double strike,
    double dt,
    bool isCall,
    int numBranches,
    const std::vector<int>& exerciseTimes)
{
    int N = static_cast<int>(pricePaths.size());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> pathDist(0, N - 1);

    double sumPayoffs = 0.0;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:sumPayoffs)
#endif
    for (int i = 0; i < N; ++i) {
        double bestValue = 0.0;
        for (int tIdx : exerciseTimes) {
            double t = tIdx * dt;
            if (t > this->maturity_) {
                break;
            }

            double payoff = PayoffFunction(isCall, pricePaths[i][tIdx], strike);
            double discNow = std::exp(-r * t) * payoff;

            double continuation = 0.0;
            if (tIdx < exerciseTimes.back()) {
                double sumFuture = 0.0;
                for (int b = 0; b < numBranches; ++b) {
                    int rp = pathDist(gen);
                    double bestFut = 0.0;
                    for (int k = tIdx + 1; k < (int)pricePaths[rp].size(); ++k) {
                        double tk = k * dt;
                        if (tk > this->maturity_) {
                            break;
                        }
                        double pf = PayoffFunction(isCall, pricePaths[rp][k], strike);
                        double disc = std::exp(-r * (tk - t)) * pf;
                        if (disc > bestFut) {
                            bestFut = disc;
                        }
                    }
                    sumFuture += bestFut;
                }
                continuation = (sumFuture / numBranches) * std::exp(-r * t);
            }

            double better = std::max(discNow, continuation);
            if (better > bestValue) {
                bestValue = better;
            }
        }
        sumPayoffs += bestValue;
    }
    return sumPayoffs / N;
}
