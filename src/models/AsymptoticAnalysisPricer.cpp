#include "../../include/models/AsymptoticAnalysisPricer.h"
#include "../../include/core/common.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

static double AsymptoticBoundaryPut(double t, double T, double K, double r, double D, double sigma)
{
    double eps = T - t;
    if (eps < 1e-10) return K;

    double c0 = 0.5 * sigma * std::sqrt(eps * std::log(1.0 / eps));
    double boundary = K + c0;

    double eps0 = 0.01;
    if (eps < eps0) {
        boundary -= 0.5 * (r - D) * eps;
    }
    return boundary;
}

static double AsymptoticBoundaryCall(double t, double T, double K, double r, double D, double sigma)
{
    double eps = T - t;
    if (eps < 1e-10) return K;

    double c0 = 0.5 * sigma * std::sqrt(eps * std::log(1.0 / eps));
    double boundary = K - c0;

    double eps0 = 0.01;
    if (eps < eps0) {
        boundary += 0.5 * (D - r) * eps;
    }
    return boundary;
}

double AsymptoticAnalysis::PredictOptionPrice(
    const std::vector<std::vector<double>>& pricePaths,
    double r,
    double strike,
    double maturity,
    double dt,
    bool isCall,
    double sigma,
    double dividend)
{
    if (pricePaths.empty() || pricePaths[0].empty()) {
        return 0.0;
    }

    int N = static_cast<int>(pricePaths.size());
    int M = static_cast<int>(pricePaths[0].size());

    for(int i = 0; i < N; ++i) {
        if(pricePaths[i].size() != size_t(M)) {
            return 0.0;
        }
    }

    std::vector<double> bestPayoffs(N, 0.0);

    try {
        for (int i = 0; i < N; ++i) {
            double pathBest = 0.0;
            for (int j = 0; j < M; ++j) {
                double t = j * dt;
                if (t > maturity) break;

                double S = pricePaths[i][j];
                if (std::isnan(S) || std::isinf(S)) continue;

                double boundary = (isCall)
                    ? AsymptoticBoundaryCall(t, maturity, strike, r, dividend, sigma)
                    : AsymptoticBoundaryPut(t, maturity, strike, r, dividend, sigma);

                bool inExerciseRegion = false;
                if (isCall) {
                    if (S > boundary) inExerciseRegion = true;
                } else {
                    if (S < boundary) inExerciseRegion = true;
                }

                if (inExerciseRegion) {
                    double payoff = PayoffFunction(isCall, S, strike);
                    if (std::isnan(payoff) || std::isinf(payoff)) continue;
                    double discounted = std::exp(-r * t) * payoff;
                    if (discounted > pathBest) {
                        pathBest = discounted;
                    }
                }
            }
            bestPayoffs[i] = pathBest;
        }

        double sumPayoffs = 0.0;
        int validPaths = 0;
        for(double payoff : bestPayoffs) {
            if (!std::isnan(payoff) && !std::isinf(payoff)) {
                sumPayoffs += payoff;
                validPaths++;
            }
        }
        
        return validPaths > 0 ? sumPayoffs / validPaths : 0.0;
    }
    catch(...) {
        return 0.0;  // Return 0 instead of propagating exceptions
    }
}
