#ifndef ROUGHVOLATILITY_H
#define ROUGHVOLATILITY_H

#include <vector>
#include <complex>

// Implements a rough volatility model for path generation
class RoughVolatility {
public:
    // Default constructor
    RoughVolatility();

    // Generates simulated paths based on past price data
    std::vector<std::vector<double>> GenerateStockPricePaths(
        const std::vector<double>& historical_prices,
        int forward_steps,
        int path_num
    );

private:
    // Estimates drift term
    double estimateR(
        const std::vector<double>& logrets, double dt_yr=1.0/252.0
    );

    double estimateXi(const std::vector<double>& logrets, double dt_yr=1.0/252.0);
    double estimateH(const std::vector<double>& logrets);
    double estimateEta(const std::vector<double>& logrets, double H);
    double estimateRho(const std::vector<double>& logrets);
    std::vector<double> logReturns(const std::vector<double>& prices);

    // FFT transform utility
    void fft(std::vector<std::complex<double>>& a, int inv);
    static size_t nextPowerOfTwo(size_t n);

    // Helper methods for rough Bergomi model
    std::vector<std::complex<double>> rbergomiPhi(const std::vector<double>& lambda, double H);
    std::vector<double> rbergomiLambda(const std::vector<double>& timeGrid, double H);
    std::vector<std::complex<double>> genComplexGaussians(size_t N);
    std::vector<double> gaussians(size_t N);
    std::vector<double> fractionalGaussian(
        const std::vector<std::complex<double>>& phi,
        const std::vector<std::complex<double>>& Z,
        double H, double eta
    );
    std::vector<double> forwardVariance(
        const std::vector<double>& X,
        const std::vector<double>& tGrid,
        double xi,
        double H,
        double eta
    );
};

#endif
