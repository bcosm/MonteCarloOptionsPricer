// Evaluation script for trained Bayesian neural network model

#include "../../include/core/BayesianNN.h"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>

int main() {

    try {

        std::string test_file = "test_data.csv";
        std::vector<std::string> inputColumns = {
            "underlying_last", "dte", "strike_distance_pct", "delta", "gamma",
            "vega", "theta", "rho", "iv", "volume", "dividend",
            "asymptotic_prediction", "branching_prediction", "lsm_prediction",
            "martingale_prediction", "twenty_day_vol", "twenty_day_momentum"
        };
        std::string targetColumn = "last";

        std::vector<std::vector<float>> X_test;
        std::vector<float> Y_test;
        ReadCSV(test_file, X_test, Y_test, inputColumns, targetColumn);

        int inputDim = static_cast<int>(inputColumns.size());
        int hiddenDim = 64;

        BayesianTrainer evaluator(inputDim, hiddenDim);

        std::string model_file = "bayesian_model.pt";
        evaluator.LoadModel(model_file);


        int nSamples = 100;
        double stds = 3.0;
        double sumError = 0.0, sumSquaredError = 0.0;
        int coverageCount = 0;

        std::ofstream detailOut("evaluation_results.csv");
        detailOut << "Index,Actual,Mean,Lower,Upper,Error,InsideInterval\n";

        size_t totalSamples = X_test.size();
        size_t currentSample = 0;
        auto startTime = std::chrono::steady_clock::now();

        for(size_t i = 0; i < X_test.size(); ++i) {
            auto sample = X_test[i];
            auto actual = Y_test[i];
            auto [meanVal, lowerVal, upperVal] =
                evaluator.MetaModelPrediction(sample, nSamples, stds);

            double error = std::abs(meanVal - actual);
            bool inside = (actual >= lowerVal && actual <= upperVal);

            sumError += error;
            sumSquaredError += (error * error);
            coverageCount += (inside ? 1 : 0);

            detailOut << i << "," << actual << "," << meanVal << ","
                      << lowerVal << "," << upperVal << "," << error << ","
                      << (inside ? "1" : "0") << "\n";

            currentSample++;
            double progress = static_cast<double>(currentSample) / totalSamples;
            int barWidth = 50;

            std::cout << "[";
            int pos = static_cast<int>(barWidth * progress);
            for (int j = 0; j < barWidth; ++j) {
                if (j < pos) std::cout << "=";
                else if (j == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << std::fixed << std::setprecision(1) << (progress * 100.0) << "% ";

            auto currentTime = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = currentTime - startTime;
            double eta = (elapsed.count() / progress) - elapsed.count();
            std::cout << "ETA: " << std::fixed << std::setprecision(1) << eta << "s    \r" << std::flush;
        }

        std::cout << std::endl;

        double avgError = sumError / X_test.size();
        double rmse = std::sqrt(sumSquaredError / X_test.size());
        double coverage = static_cast<double>(coverageCount) / X_test.size() * 100.0;

        std::cout << "\n=== EVALUATION RESULTS ===\n"
                  << "Total Samples: " << X_test.size() << "\n"
                  << "Mean Absolute Error (MAE): " << std::fixed << std::setprecision(4) << avgError << "\n"
                  << "Root Mean Squared Error (RMSE): " << rmse << "\n"
                  << "Coverage (" << stds
                  << " std dev): " << coverage << "%\n"
                  << "Detailed results saved in 'evaluation_results.csv'\n";

    } catch(const std::exception& e) {

        std::cerr << "Evaluation error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}