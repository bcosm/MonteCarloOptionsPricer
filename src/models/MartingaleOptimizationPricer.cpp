// Martingale optimization approach for American option pricing

#include "../../include/models/MartingaleOptimizationPricer.h"
#include "../../include/core/common.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <algorithm>

static Eigen::VectorXd PolyBasis(double x, int polyOrder)
{
    Eigen::VectorXd v(polyOrder + 1);
    v[0] = 1.0;
    for (int i = 1; i <= polyOrder; ++i) {
        v[i] = v[i - 1] * x;
    }
    return v;
}

double MartingaleOptimization::PredictOptionPrice(
    const std::vector<std::vector<double>>& pricePaths,
    double r,
    double strike,
    double maturity,
    double dt,
    bool isCall,
    int polyOrder,
    int maxIterations)
{
    if (pricePaths.empty() || pricePaths[0].empty()) {
        throw std::runtime_error("MartingaleOptimization: Empty pricePaths.");
    }

    pPaths_      = &pricePaths;
    r_           = r;
    strike_      = strike;
    maturity_    = maturity;
    dt_          = dt;
    isCall_      = isCall;
    polyOrder_   = polyOrder;

    int N = static_cast<int>(pricePaths.size());

    Mcoeff_.assign(polyOrder_ + 1, 0.0);
    offset_ = 0.0;
    pathStop_.assign(N, 0);
    pathPrimal_.assign(N, 0.0);
    pathDual_.assign(N, 0.0);

    double finalLower = 0.0, finalUpper = 0.0;

    for (int iter = 1; iter <= maxIterations; ++iter) {
        auto [primal, dual] = DoIteration(pricePaths);
        UpdateMartingale(pricePaths);
        finalLower = primal;
        finalUpper = dual;
    }

    return 0.5 * (finalLower + finalUpper);
}

std::pair<double, double> MartingaleOptimization::DoIteration(
    const std::vector<std::vector<double>>& pricePaths)
{
    int N = static_cast<int>(pricePaths.size());
    int M = static_cast<int>(pricePaths[0].size());

    double sumPrimal = 0.0;
    for (int i = 0; i < N; ++i) {
        double bestVal = 0.0;
        int bestIdx = 0;
        for (int j = 0; j < M; ++j) {
            double t = j * dt_;
            if (t > maturity_) {
                break;
            }
            double S = pricePaths[i][j];
            double payoff = PayoffFunction(isCall_, S, strike_);
            double discPayoff = payoff * PathDiscountFactor(j);

            if (discPayoff > bestVal) {
                bestVal = discPayoff;
                bestIdx = j;
            }
        }
        pathStop_[i]   = bestIdx;
        pathPrimal_[i] = bestVal;
        sumPrimal     += bestVal;
    }
    double primalVal = sumPrimal / N;

    double sumDual = 0.0;
    for (int i = 0; i < N; ++i) {
        double bestVal = 0.0;
        for (int j = 0; j < M; ++j) {
            double t = j * dt_;
            if (t > maturity_) {
                break;
            }
            double S = pricePaths[i][j];
            double payoff = PayoffFunction(isCall_, S, strike_);
            double discPayoff = payoff * PathDiscountFactor(j);

            double Mj = EvaluateMartingale(S) - offset_;
            double candidate = discPayoff - Mj;
            if (candidate > bestVal) {
                bestVal = candidate;
            }
        }
        pathDual_[i] = bestVal;
        sumDual     += bestVal;
    }
    double dualVal = sumDual / N;

    return {primalVal, dualVal};
}

void MartingaleOptimization::UpdateMartingale(const std::vector<std::vector<double>>& pricePaths)
{
    int N = static_cast<int>(pricePaths.size());
    int M = static_cast<int>(pricePaths[0].size());

    std::vector<double> Xvals;
    std::vector<double> Yvals;
    Xvals.reserve(2 * N);
    Yvals.reserve(2 * N);

    for (int i = 0; i < N; ++i) {
        int jStop = pathStop_[i];
        double Sstop = (*pPaths_)[i][jStop];
        double payoffStop = PayoffFunction(isCall_, Sstop, strike_);
        double discPayoffStop = payoffStop * PathDiscountFactor(jStop);

        double target = 0.5 * discPayoffStop;
        Xvals.push_back(Sstop);
        Yvals.push_back(target);

        int jOther = (jStop + M / 2) % M;
        double Sother = (*pPaths_)[i][jOther];
        double payoffOther = PayoffFunction(isCall_, Sother, strike_);
        double discPayoffOther = payoffOther * PathDiscountFactor(jOther);
        double target2 = 0.2 * discPayoffOther;
        Xvals.push_back(Sother);
        Yvals.push_back(target2);
    }

    int sampleSize = static_cast<int>(Xvals.size());
    if (sampleSize < (polyOrder_ + 1)) {
        return;
    }

    Eigen::MatrixXd A(sampleSize, polyOrder_ + 1);
    Eigen::VectorXd b(sampleSize);

    for (int i = 0; i < sampleSize; ++i) {
        double x = Xvals[i];
        Eigen::VectorXd pb = PolyBasis(x, polyOrder_);
        A.row(i) = pb;
        b(i) = Yvals[i];
    }

    Eigen::VectorXd c = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    Mcoeff_.resize(polyOrder_ + 1);
    for (int k = 0; k <= polyOrder_; ++k) {
        Mcoeff_[k] = c(k);
    }

    double sumM0 = 0.0;
    for (int i = 0; i < N; ++i) {
        double S0 = (*pPaths_)[i][0];
        sumM0 += EvaluateMartingale(S0);
    }
    offset_ = sumM0 / N;
}

double MartingaleOptimization::EvaluateMartingale(double S) const
{
    double val = 0.0;
    double power = 1.0;
    for (int k = 0; k <= polyOrder_; ++k) {
        val += Mcoeff_[k] * power;
        power *= S;
    }
    return val;
}