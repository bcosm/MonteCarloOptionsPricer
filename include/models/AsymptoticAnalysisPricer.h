#pragma once
#include <vector>

// Provides an asymptotic expansion approach for option pricing
class AsymptoticAnalysis
{
public:
    double PredictOptionPrice(const std::vector<std::vector<double>>& pricePaths,
                              double r,
                              double strike,
                              double maturity,
                              double dt,
                              bool isCall,
                              double sigma,
                              double dividend);
};
