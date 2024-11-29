// Common utility functions for Monte Carlo option pricing models

#pragma once
#include <algorithm>
#include <cmath>

// Calculate option payoff given call/put type, stock price, and strike price
inline double PayoffFunction(bool isCall, double stockPrice, double strike) {
    if (isCall) {
        return std::max(0.0, stockPrice - strike);
    } else {
        return std::max(0.0, strike - stockPrice);
    }
}
