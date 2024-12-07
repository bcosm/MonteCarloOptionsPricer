// CSV file reading utilities for training data

#include "../../include/core/BayesianNN.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

void ReadCSV(const std::string& filename,
             std::vector<std::vector<float>>& X,
             std::vector<float>& Y,
             const std::vector<std::string>& inputColumns,
             const std::string& targetColumn)
{

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("Failed to read header from " + filename);
    }
    std::stringstream ss(line);
    std::vector<std::string> headers;
    std::string column;
    while (std::getline(ss, column, ',')) {
        headers.push_back(column);
    }

    std::vector<int> inputIndices;
    for (const auto& col : inputColumns) {
        auto it = std::find(headers.begin(), headers.end(), col);
        if (it != headers.end()) {
            inputIndices.push_back(std::distance(headers.begin(), it));
        }
        else {
            throw std::runtime_error("Input column " + col + " not found in " + filename);
        }
    }
    auto targetIt = std::find(headers.begin(), headers.end(), targetColumn);
    if (targetIt == headers.end()) {
        throw std::runtime_error("Target column " + targetColumn + " not found in " + filename);
    }
    int targetIndex = std::distance(headers.begin(), targetIt);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<std::string> tokens;
        std::string token;
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        std::vector<float> inputs;
        for (int idx : inputIndices) {
            inputs.push_back(std::stof(tokens[idx]));
        }
        X.push_back(inputs);

        Y.push_back(std::stof(tokens[targetIndex]));
    }
}