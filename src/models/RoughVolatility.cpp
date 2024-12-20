// Rough volatility model using fractional Brownian motion for price path generation

#include "../../include/models/RoughVolatility.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <complex>
#include <stdexcept>

#define _USE_MATH_DEFINES
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double mean(const std::vector<double>& v) {
    double s = std::accumulate(v.begin(), v.end(), 0.0);
    return v.empty() ? 0.0 : s / v.size();
}

static double variance(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double m = mean(v), var = 0.0;
    for (auto x : v) {
        double d = x - m;
        var += d * d;
    }
    return var / (v.size() - 1);
}

static double covariance(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 2) return 0.0;
    double mx = mean(x), my = mean(y), c = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        c += (x[i] - mx) * (y[i] - my);
    }
    return c / (x.size() - 1);
}

static void detrendSegment(std::vector<double>& segment) {
    size_t n = segment.size();
    if (n < 2) return;

    std::vector<double> t(n, 0.0);
    for (size_t i = 0; i < n; i++) {
        t[i] = static_cast<double>(i + 1);
    }

    double tm = mean(t);
    double ym = mean(segment);
    double num = 0.0, den = 0.0;

    for (size_t i = 0; i < n; i++) {
        num += (t[i] - tm) * (segment[i] - ym);
        den += (t[i] - tm) * (t[i] - tm);
    }

    if (std::abs(den) < 1e-14) return;

    double slope = num / den;
    double intercept = ym - slope * tm;

    for (size_t i = 0; i < n; i++) {
        segment[i] -= (slope * t[i] + intercept);
    }
}

static double hurstExponentDFA(const std::vector<double>& data_in) {
    std::vector<double> data = data_in;
    if (data.size() < 2) return 0.5;

    double m = mean(data);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] -= m;
    }
    for (size_t i = 1; i < data.size(); i++) {
        data[i] += data[i - 1];
    }

    std::vector<double> logWindowSize, logFluctuation;
    size_t minWindowSize = 4;
    size_t maxWindowSize = data.size() / 4;
    size_t w = minWindowSize;

    while (w <= maxWindowSize) {
        std::vector<double> fluctuations;
        for (size_t start = 0; start + w <= data.size(); start += w) {
            std::vector<double> segment(data.begin() + start, data.begin() + start + w);
            detrendSegment(segment);
            double sum_sq = 0.0;
            for (auto val : segment) {
                sum_sq += val * val;
            }
            double rms = std::sqrt(sum_sq / w);
            fluctuations.push_back(rms);
        }
        double mf = mean(fluctuations);
        if (mf > 0.0) {
            logWindowSize.push_back(std::log((double)w));
            logFluctuation.push_back(std::log(mf));
        }
        w *= 2;
    }

    size_t n = logWindowSize.size();
    if (n < 2) return 0.5;

    double sumX = 0.0, sumY = 0.0, sumXX = 0.0, sumXY = 0.0;
    for (size_t i = 0; i < n; i++) {
        sumX  += logWindowSize[i];
        sumY  += logFluctuation[i];
        sumXX += logWindowSize[i] * logWindowSize[i];
        sumXY += logWindowSize[i] * logFluctuation[i];
    }

    double slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    return slope;
}

RoughVolatility::RoughVolatility() {}

std::vector<double> RoughVolatility::logReturns(const std::vector<double>& prices) {
    std::vector<double> rets;
    rets.reserve(prices.size() > 1 ? prices.size() - 1 : 0);
    for (size_t i = 1; i < prices.size(); i++) {
        rets.push_back(std::log(prices[i] / prices[i - 1]));
    }
    return rets;
}

double RoughVolatility::estimateR(const std::vector<double>& logrets, double dt_yr) {
    double mu = mean(logrets);
    double annual_mu = mu / dt_yr;
    return annual_mu;
}

double RoughVolatility::estimateXi(const std::vector<double>& logrets, double dt_yr) {
    double var_r = variance(logrets);
    double annual_var = var_r / dt_yr;
    return annual_var;
}

double RoughVolatility::estimateH(const std::vector<double>& logrets) {
    return hurstExponentDFA(logrets);
}

double RoughVolatility::estimateEta(const std::vector<double>& logrets, double H) {
    (void)H;
    double stdev = std::sqrt(variance(logrets));
    return stdev * 2.0;
}

double RoughVolatility::estimateRho(const std::vector<double>& logrets) {
    std::vector<double> sq;
    sq.reserve(logrets.size());
    for (auto r : logrets) {
        sq.push_back(r * r);
    }
    double c = covariance(logrets, sq);
    double rho = c / (std::sqrt(variance(logrets) * variance(sq)));
    if (rho > 0.0) {
        rho = -0.3;
    }
    return rho;
}

void RoughVolatility::fft(std::vector<std::complex<double>>& a, int inv) {
    size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = 2 * M_PI / len * (inv < 0 ? -1 : 1);
        std::complex<double> wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (size_t j = 0; j < len / 2; j++) {
                std::complex<double> u = a[i + j];
                std::complex<double> v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inv < 0) {
        for (size_t i = 0; i < n; i++) {
            a[i] /= n;
        }
    }
}

size_t RoughVolatility::nextPowerOfTwo(size_t n) {
    size_t p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

std::vector<std::complex<double>> RoughVolatility::rbergomiPhi(
    const std::vector<double>& lambda, double H
) {
    (void)H;
    size_t N = lambda.size();
    size_t M = nextPowerOfTwo(N);

    std::vector<std::complex<double>> phi(M, std::complex<double>(0.0, 0.0));
    for (size_t i = 0; i < N; i++) {
        phi[i] = std::complex<double>(lambda[i], 0.0);
    }
    fft(phi, 1);
    return phi;
}

std::vector<double> RoughVolatility::rbergomiLambda(
    const std::vector<double>& timeGrid, double H
) {
    size_t N = timeGrid.size();
    std::vector<double> lambda(N, 0.0);
    for (size_t i = 0; i < N; i++) {
        lambda[i] = 0.5 * (std::pow(timeGrid[i], 2 * H));
    }
    return lambda;
}

std::vector<std::complex<double>> RoughVolatility::genComplexGaussians(size_t N) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::normal_distribution<double> d(0.0, 1.0);

    std::vector<std::complex<double>> Z(N, std::complex<double>(0.0, 0.0));
    for (size_t i = 0; i < N; i++) {
        double re = d(g);
        double im = d(g);
        Z[i] = std::complex<double>(re, im);
    }
    return Z;
}

std::vector<double> RoughVolatility::gaussians(size_t N) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::normal_distribution<double> d(0.0, 1.0);

    std::vector<double> G(N, 0.0);
    for (size_t i = 0; i < N; i++) {
        G[i] = d(g);
    }
    return G;
}

std::vector<double> RoughVolatility::fractionalGaussian(
    const std::vector<std::complex<double>>& phi,
    const std::vector<std::complex<double>>& Z,
    double H, double eta
) {
    size_t N = Z.size();
    size_t M = nextPowerOfTwo(N);

    std::vector<std::complex<double>> A(M, std::complex<double>(0.0, 0.0));
    for (size_t i = 0; i < N; i++) {
        A[i] = phi[i] * Z[i];
    }

    fft(A, -1);

    std::vector<double> res(M, 0.0);
    for (size_t i = 0; i < M; i++) {
        res[i] = A[i].real();
    }

    double scale = std::sqrt(2 * H) * eta;
    std::vector<double> X(M, 0.0);
    for (size_t i = 0; i < M; i++) {
        X[i] = scale * res[i];
    }

    X.resize(N);
    return X;
}

std::vector<double> RoughVolatility::forwardVariance(
    const std::vector<double>& X,
    const std::vector<double>& tGrid,
    double xi,
    double H,
    double eta
) {
    size_t N = X.size();
    std::vector<double> v(N, 0.0);
    for (size_t i = 0; i < N; i++) {
        double t = tGrid[i];
        double ma = -0.5 * eta * eta * std::pow(t, 2 * H);
        v[i] = xi * std::exp(X[i] + ma);
    }
    return v;
}

// Main path generation method using rough Bergomi model with estimated parameters
std::vector<std::vector<double>> RoughVolatility::GenerateStockPricePaths(
    const std::vector<double>& historical_prices,
    int forward_steps,
    int path_num
) {
    if (historical_prices.size() < 2) {
        throw std::runtime_error("Historical prices vector too small.");
    }

    double dt_yr = 1.0 / 252.0;
    double dt = dt_yr;

    std::vector<double> rets = logReturns(historical_prices);

    double r   = 0.04;
    double xi  = estimateXi(rets, dt_yr);
    double H   = estimateH(rets);
    double eta = estimateEta(rets, H);
    double rho = estimateRho(rets);
    double S0  = historical_prices.back();

    int num_paths = path_num;
    int num_steps = forward_steps;
    double T      = num_steps * dt;

    std::vector<double> timeGrid(num_steps + 1, 0.0);
    for (size_t i = 0; i <= (size_t)num_steps; i++) {
        timeGrid[i] = i * dt;
    }

    std::vector<double> lambda = rbergomiLambda(timeGrid, H);
    std::vector<std::complex<double>> phi = rbergomiPhi(lambda, H);
    std::vector<std::vector<double>> paths(num_paths, std::vector<double>(num_steps + 1, 0.0));

    for (size_t i = 0; i < (size_t)num_paths; i++) {
        std::vector<std::complex<double>> Z = genComplexGaussians(num_steps);
        std::vector<double> X = fractionalGaussian(phi, Z, H, eta);
        std::vector<double> v = forwardVariance(X, timeGrid, xi, H, eta);

        std::vector<double> W1 = gaussians(num_steps);
        std::vector<double> W2 = gaussians(num_steps);

        paths[i][0] = S0;
        for (size_t j = 1; j <= (size_t)num_steps; j++) {
            double dw1 = std::sqrt(dt) * W1[j - 1];
            double dw2 = std::sqrt(dt) * W2[j - 1];
            double dW  = rho * dw1 + std::sqrt(1.0 - rho * rho) * dw2;

            double vt = v[j - 1];
            double drift = (r - 0.5 * vt) * dt;
            double diff  = std::sqrt(std::max(0.0, vt)) * dW;
            paths[i][j] = paths[i][j - 1] * std::exp(drift + diff);
        }
    }

    return paths;
}
