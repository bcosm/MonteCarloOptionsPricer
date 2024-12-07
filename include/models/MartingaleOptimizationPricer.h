// Martingale optimization approach for American option pricing
#pragma once
#include <vector>
#include <cmath>


class MartingaleOptimization
{
public:
    double PredictOptionPrice(
        const std::vector<std::vector<double>>& pricePaths,
        double r,
        double strike,
        double maturity,
        double dt,
        bool isCall,
        int polyOrder,
        int maxIterations = 5);

private:

    double maturity_{0.0};
    double r_{0.0};
    double strike_{0.0};
    double dt_{0.0};
    bool   isCall_{false};
    int    polyOrder_{2};


    const std::vector<std::vector<double>>* pPaths_{nullptr};


    std::vector<int>   pathStop_;
    std::vector<double> pathPrimal_;
    std::vector<double> pathDual_;
    std::vector<double> Mcoeff_;
    double offset_{0.0};


    std::pair<double,double> DoIteration(const std::vector<std::vector<double>>& pricePaths);
    void UpdateMartingale(const std::vector<std::vector<double>>& pricePaths);


    double EvaluateMartingale(double S) const;


    inline double PathDiscountFactor(int j) const
    {
        double t = j * dt_;
        if (t > maturity_) t = maturity_;
        return std::exp(-r_ * t);
    }
};