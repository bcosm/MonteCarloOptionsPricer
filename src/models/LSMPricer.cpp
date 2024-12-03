#include "../../include/models/LSMPricer.h"
#include "../../include/core/common.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>

static Eigen::VectorXd PolynomialBasis(double x, int polyOrder)
{
    Eigen::VectorXd v(polyOrder + 1);
    v[0] = 1.0;
    for (int i = 1; i <= polyOrder; ++i) {
        v[i] = v[i - 1] * x;
    }
    return v;
}

double LSM::PredictOptionPrice(
    const std::vector<std::vector<double>>& pricePaths,
    double r,
    double strike,
    double maturity,
    double dt,
    bool isCall,
    int polyOrder)
{
    if (pricePaths.empty() || pricePaths[0].empty()) {
        throw std::runtime_error("LSM::PredictOptionPrice: Empty pricePaths.");
    }

    int N = static_cast<int>(pricePaths.size());
    int M = static_cast<int>(pricePaths[0].size());

    std::vector<std::vector<double>> Values(N, std::vector<double>(M, 0.0));

    for (int i = 0; i < N; ++i) {
        double payoff = PayoffFunction(isCall, pricePaths[i][M - 1], strike);
        Values[i][M - 1] = payoff;
    }

    for (int j = M - 2; j >= 0; --j) {
        double thisTime = j * dt;
        if (thisTime > maturity) {
            for (int i = 0; i < N; ++i) {
                Values[i][j] = Values[i][j + 1] * std::exp(-r * dt);
            }
            continue;
        }

        std::vector<int> itmIndices;
        itmIndices.reserve(N);
        for (int i = 0; i < N; ++i) {
            double payoff = PayoffFunction(isCall, pricePaths[i][j], strike);
            if (payoff > 1e-14) {
                itmIndices.push_back(i);
            }
        }

        if (!itmIndices.empty()) {
            Eigen::MatrixXd A(itmIndices.size(), polyOrder + 1);
            Eigen::VectorXd b(itmIndices.size());

            for (size_t k = 0; k < itmIndices.size(); ++k) {
                int i = itmIndices[k];
                if (i >= N || i < 0) {
                    throw std::runtime_error("LSM: Invalid path index in regression");
                }
                double discVal = Values[i][j + 1] * std::exp(-r * dt);
                b(k) = discVal;

                double S = pricePaths[i][j];
                A.row(k) = PolynomialBasis(S, polyOrder);
            }

            Eigen::VectorXd c = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

            for (int i : itmIndices) {
                double S = pricePaths[i][j];
                double immediate = PayoffFunction(isCall, S, strike);

                Eigen::VectorXd basis = PolynomialBasis(S, polyOrder);
                double cont = basis.dot(c);

                Values[i][j] = std::max(immediate, cont);
            }
        }

        for (int i = 0; i < N; ++i) {
            double payoff = PayoffFunction(isCall, pricePaths[i][j], strike);
            if (payoff < 1e-14) {
                Values[i][j] = Values[i][j + 1] * std::exp(-r * dt);
            }
        }
    }

    double sumVal = 0.0;
    for (int i = 0; i < N; ++i) {
        sumVal += Values[i][0];
    }
    return sumVal / N;
}
