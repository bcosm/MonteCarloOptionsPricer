// Training script for Bayesian neural network model

#include "../../include/core/BayesianNN.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <unordered_map>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <csignal>

namespace fs = std::filesystem;

int main()
{
    try {
        ::signal(SIGINT, signal_handler);

        std::string train_file = "train_data.csv";
        std::string valid_file = "valid_data.csv";
        std::string test_file  = "test_data.csv";
        std::string model_file = "bayesian_model.pt";
        std::string checkpoint_file = "checkpoint.pt";

        std::vector<std::string> inputColumns = {
            "underlying_last", "dte", "strike_distance_pct", "delta", "gamma",
            "vega", "theta", "rho", "iv", "volume", "dividend",
            "asymptotic_prediction", "branching_prediction", "lsm_prediction",
            "martingale_prediction", "twenty_day_vol", "twenty_day_momentum"
        };
        std::string targetColumn = "last";

        std::vector<std::vector<float>> X_train;
        std::vector<float> Y_train;

        std::vector<std::vector<float>> X_valid;
        std::vector<float> Y_valid;

        std::vector<std::vector<float>> X_test;
        std::vector<float> Y_test;

        std::cout << "Reading training data..." << std::endl;
        ReadCSV(train_file, X_train, Y_train, inputColumns, targetColumn);

        std::cout << "Reading validation data..." << std::endl;
        ReadCSV(valid_file, X_valid, Y_valid, inputColumns, targetColumn);

        std::cout << "Reading test data..." << std::endl;
        ReadCSV(test_file, X_test, Y_test, inputColumns, targetColumn);

        int inputDim = inputColumns.size();
        int hiddenDim = 64;

        BayesianTrainer trainer(inputDim, hiddenDim);

        int numEpochs = 100;
        int batchSize = 256;
        double learningRate = 3e-4;

        trainer.model_->to(trainer.device_);
        bool on_gpu = trainer.model_->parameters()[0].device().is_cuda();
        std::cout << "Model successfully moved to GPU: " << (on_gpu ? "Yes" : "No") << std::endl;

        auto dataTensor   = trainer.Vector2DToTensor(X_train).to(trainer.device_);
        auto targetTensor = trainer.Vector1DToTensor(Y_train).unsqueeze(1).to(trainer.device_);

        std::cout << "\nStarting training..." << std::endl;
        trainer.TrainModel(X_train, Y_train, numEpochs, batchSize, learningRate, checkpoint_file);

        trainer.SaveModel(model_file);

        BayesianTrainer loaded_trainer(inputDim, hiddenDim);
        loaded_trainer.LoadModel(model_file);

        std::vector<float> newFeatures = X_test[0];
        auto [predMean, ciLower, ciUpper] = loaded_trainer.MetaModelPrediction(newFeatures, 1);
        std::cout << "\nSingle Prediction for first test sample: " << predMean << std::endl;
        std::cout << "Actual 'last' value: " << Y_test[0] << std::endl;

        int mcSamples = 100;
        auto [mcMean, mcLower, mcUpper] = loaded_trainer.MetaModelPrediction(newFeatures, mcSamples);
        std::cout << mcSamples << "x MC-Dropout Prediction: " << mcMean
                  << " (95% CI: [" << mcLower << ", " << mcUpper << "])" << std::endl;

        {
            loaded_trainer.LoadModel(model_file);
            loaded_trainer.MetaModelPrediction(newFeatures, 1);
            loaded_trainer.MetaModelPrediction(newFeatures, 1);

            double sumVals = 0.0;
            double sumSq   = 0.0;
            for(int64_t i=0; i<mcSamples; ++i){
                auto [tmp, lower, upper] = loaded_trainer.MetaModelPrediction(newFeatures, 1);
                sumVals += tmp;
                sumSq   += (tmp * tmp);
            }
            double meanVal = sumVals / mcSamples;
            double varVal  = (sumSq / mcSamples) - (meanVal * meanVal);
            double stdVal  = (varVal > 0 ? std::sqrt(varVal) : 0.0);
            std::cout << "MC Dropout mean = " << meanVal << ", stdDev = " << stdVal << std::endl;
        }

        std::cout << "\nEvaluating on validation set..." << std::endl;
        double valMSE = 0.0;
        for(size_t i=0; i<X_valid.size(); ++i){
            auto [pred, lower, upper] = loaded_trainer.MetaModelPrediction(X_valid[i], 1);
            double error = pred - Y_valid[i];
            valMSE += error * error;
        }
        valMSE /= X_valid.size();
        std::cout << "Validation MSE: " << valMSE << std::endl;

        std::cout << "\nEvaluating on test set..." << std::endl;
        double testMSE = 0.0;
        for(int64_t i=0; i<X_test.size(); ++i){
            auto [pred, lower, upper] = loaded_trainer.MetaModelPrediction(X_test[i], 1);
            double error = pred - Y_test[i];
            testMSE += error * error;
        }
        testMSE /= X_test.size();
        std::cout << "Test MSE: " << testMSE << std::endl;

        std::cout << "\nTraining and evaluation complete." << std::endl;
    } catch(const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
}