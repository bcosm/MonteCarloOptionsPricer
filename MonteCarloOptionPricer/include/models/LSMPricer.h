#pragma once
#include <vector>

// Applies Longstaff-Schwartz method for American option pricing
class LSM
{
public:
    double PredictOptionPrice(const std::vector<std::vector<double>>& pricePaths,
                              double r,
                              double strike,
                              double maturity,
                              double dt,
                              bool isCall,
                              int polyOrder);
};
