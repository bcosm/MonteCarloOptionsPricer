/*************************************************************
 * BayesianNN.h
 *************************************************************/
#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <csignal>

// Loads data from CSV into feature matrices and labels
void ReadCSV(const std::string& filename, 
            std::vector<std::vector<float>>& X, 
            std::vector<float>& Y, 
            const std::vector<std::string>& inputColumns, 
            const std::string& targetColumn);

// RealNVP-like normalizing flow
struct RealNVPFlowImpl : public torch::nn::Module
{
    torch::nn::Linear sLayer{nullptr}, tLayer{nullptr};

    RealNVPFlowImpl(int dim)
    {
        sLayer = register_module("sLayer", torch::nn::Linear(dim, dim));
        tLayer = register_module("tLayer", torch::nn::Linear(dim, dim));
    }

    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& x)
    {
        auto s = sLayer->forward(x);
        auto t = tLayer->forward(x);
        auto z = (x * torch::exp(s)) + t; 
        auto logdetJ = s.sum(/*dim=*/1);
        return {z, logdetJ};
    }
};
TORCH_MODULE(RealNVPFlow);

// Bayesian feedforward NN with dropout-based uncertainty
struct BayesianMetaModelNNImpl : public torch::nn::Module
{
public:
    BayesianMetaModelNNImpl(int inputDim, int hiddenDim);

    torch::nn::Linear fcOut{nullptr};

    // Normalization layers
    torch::nn::InstanceNorm1d bn1{nullptr}, bn2{nullptr}, bn3{nullptr},
                              bn4{nullptr}, bn5{nullptr};

private:
    // Main fully connected layers
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr},
                      fc5{nullptr}, fc6{nullptr};

    // Shortcuts for skip connections
    torch::nn::Linear fcSkip1{nullptr}, fcSkip2{nullptr}, 
                      fcSkip3{nullptr}, fcSkip4{nullptr};

    torch::nn::Dropout drop1{nullptr}, drop2{nullptr}, drop3{nullptr},
                       drop4{nullptr}, drop5{nullptr};

    // Additional gating for nonlinearity
    torch::nn::Linear fcGate{nullptr};

    // Optional attention mechanism
    torch::nn::MultiheadAttention attn{nullptr};

    // Mixture density network head
    torch::nn::Linear fcMDN{nullptr};

    // Flow transformations
    torch::nn::ModuleList flowTransforms{nullptr};

public:
    torch::Tensor swish(const torch::Tensor& x);
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor mixtureParams(const torch::Tensor& x);
    torch::Tensor applyFlows(const torch::Tensor& x);
};
TORCH_MODULE(BayesianMetaModelNN);

// Trains and manages a BayesianMetaModelNN
class BayesianTrainer
{
public:
    BayesianTrainer(int inputDim, int hiddenDim);
    void TrainModel(const std::vector<std::vector<float>>& X,
                   const std::vector<float>& Y,
                   int numEpochs = 50,
                   int batchSize = 32,
                   double lr = 0.001,
                   const std::string& checkpointPath = "checkpoint.pt");
    void SaveModel(const std::string& filename);
    void LoadModel(const std::string& filename);
    void SaveCheckpoint(const std::string& filename, int epoch, double epochLoss);
    bool LoadCheckpoint(const std::string& filename, int& epoch, double& epochLoss);
    std::tuple<double, double, double> MetaModelPrediction(
        const std::vector<float>& inputFeatures,
        int nSamples = 100,
        double stds = 3.0
    );
    int GetCurrentEpoch() const { return current_epoch_; }

    BayesianMetaModelNN model_;
    torch::Device device_;

    torch::Tensor Vector2DToTensor(const std::vector<std::vector<float>>& data);
    torch::Tensor Vector1DToTensor(const std::vector<float>& data);

private:
    torch::optim::Adam optimizer_;
    int current_epoch_;
};

extern void signal_handler(int signal);
