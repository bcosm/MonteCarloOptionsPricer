#pragma once
#include <vector>

// Utilizes branching processes for grabbing lower/upper bounds on option prices
class BranchingProcesses
{
public:
    double PredictOptionPrice(
        const std::vector<std::vector<double>>& pricePaths,
        double r,
        double strike,
        double maturity,
        double dt,
        bool isCall,
        int numBranches,
        const std::vector<int>& exerciseTimes);

private:
    double ComputeLowerBound(
        const std::vector<std::vector<double>>& pricePaths,
        double r,
        double strike,
        double dt,
        bool isCall,
        const std::vector<int>& exerciseTimes);

    double ComputeUpperBound(
        const std::vector<std::vector<double>>& pricePaths,
        double r,
        double strike,
        double dt,
        bool isCall,
        int numBranches,
        const std::vector<int>& exerciseTimes);

    // Used to store maturity for internal calculations
    double maturity_{0.0};
};
