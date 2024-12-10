// Implementation of Bayesian neural network with normalizing flows for options pricing

#include "../../include/core/BayesianNN.h"
#include <torch/torch.h>
#include <torch/cuda.h>
#ifdef _WIN32
#include <windows.h>
#include <C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA\v12.4/include/cuda_runtime.h>
#include <C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include/cuda.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include <csignal>
#include <atomic>
#include <filesystem>
#include <tuple>
#include <iomanip>
namespace fs = std::filesystem;


std::atomic<bool> stop_training(false);


void signal_handler(int signal) {

    if (signal == SIGINT) {
        std::cout << "\nInterrupt signal (" << signal << ") received. Stopping training gracefully..." << std::endl;
        stop_training = true;
    }
}



BayesianMetaModelNNImpl::BayesianMetaModelNNImpl(int inputDim, int hiddenDim)
{

    const int h1 = 512;
    const int h2 = 256;
    const int h3 = 128;
    const int h4 = 64;
    const int h5 = 32;
    const int h6 = 16;


    fc1 = register_module("fc1", torch::nn::Linear(inputDim, h1));
    fc2 = register_module("fc2", torch::nn::Linear(h1, h2));
    fc3 = register_module("fc3", torch::nn::Linear(h2, h3));
    fc4 = register_module("fc4", torch::nn::Linear(h3, h4));
    fc5 = register_module("fc5", torch::nn::Linear(h4, h5));
    fc6 = register_module("fc6", torch::nn::Linear(h5, h6));
    fcOut = register_module("fcOut", torch::nn::Linear(h6, 1));


    fcSkip1 = register_module("fcSkip1", torch::nn::Linear(h1, h4));
    fcSkip2 = register_module("fcSkip2", torch::nn::Linear(h2, h4));
    fcSkip3 = register_module("fcSkip3", torch::nn::Linear(h3, h5));
    fcSkip4 = register_module("fcSkip4", torch::nn::Linear(h4, h5));


    bn1 = register_module("bn1", torch::nn::InstanceNorm1d(h1));
    bn2 = register_module("bn2", torch::nn::InstanceNorm1d(h2));
    bn3 = register_module("bn3", torch::nn::InstanceNorm1d(h3));
    bn4 = register_module("bn4", torch::nn::InstanceNorm1d(h4));
    bn5 = register_module("bn5", torch::nn::InstanceNorm1d(h5));


    drop1 = register_module("drop1", torch::nn::Dropout(0.3));
    drop2 = register_module("drop2", torch::nn::Dropout(0.3));
    drop3 = register_module("drop3", torch::nn::Dropout(0.2));
    drop4 = register_module("drop4", torch::nn::Dropout(0.2));
    drop5 = register_module("drop5", torch::nn::Dropout(0.1));


    fcGate = register_module("fcGate", torch::nn::Linear(h6, h6));


    attn = register_module("attn", torch::nn::MultiheadAttention(h3, 4));



    fcMDN = register_module("fcMDN", torch::nn::Linear(h6, 15));


    flowTransforms = register_module("flowTransforms", torch::nn::ModuleList());


    auto flow = std::make_shared<RealNVPFlowImpl>(h6);
    flowTransforms->push_back(flow);


    auto init_weights = [](torch::nn::Linear& layer) {
        torch::nn::init::kaiming_normal_(
            layer->weight,
            std::sqrt(5.0),
            torch::kFanIn,
            torch::kLeakyReLU
        );
        if (layer->bias.defined()) {
            double fan_in = layer->weight.size(1);
            double bound = 1.0 / std::sqrt(fan_in);
            torch::nn::init::uniform_(layer->bias, -bound, bound);
        }
    };


    init_weights(fc1);
    init_weights(fc2);
    init_weights(fc3);
    init_weights(fc4);
    init_weights(fc5);
    init_weights(fc6);
    init_weights(fcOut);
    init_weights(fcSkip1);
    init_weights(fcSkip2);
    init_weights(fcSkip3);
    init_weights(fcSkip4);
    init_weights(fcGate);
    init_weights(fcMDN);
}

torch::Tensor BayesianMetaModelNNImpl::swish(const torch::Tensor& x)
{

    return x * torch::sigmoid(x);
}

torch::Tensor BayesianMetaModelNNImpl::forward(torch::Tensor x)
{


    auto out1 = torch::relu(bn1(fc1->forward(x)));
    out1 = drop1(out1);
    auto skip1 = fcSkip1->forward(out1);


    auto out2 = torch::relu(bn2(fc2->forward(out1)));
    out2 = drop2(out2);
    auto skip2 = fcSkip2->forward(out2);


    auto out3 = torch::relu(bn3(fc3->forward(out2)));
    out3 = drop3(out3);


    auto out4 = torch::relu(bn4(fc4->forward(out3)));
    out4 = drop4(out4);


    auto out5 = torch::relu(bn5(fc5->forward(out4)));
    out5 = drop5(out5);


    auto skip3 = fcSkip3->forward(out3);
    auto skip4 = fcSkip4->forward(out4);
    out5 = out5 + skip3 + skip4;


    auto out6 = torch::relu(fc6->forward(out5));


    auto gatedOut = out6 * torch::sigmoid(fcGate->forward(out6));




    auto attnInput = out3.unsqueeze(0);
    attnInput = attnInput.transpose(0, 1);
    auto attnOutput = std::get<0>(attn->forward(attnInput, attnInput, attnInput));
    attnOutput = attnOutput.transpose(0, 1).squeeze(0);


    std::vector<torch::Tensor> to_cat = {gatedOut, attnOutput};
    auto combined = torch::cat(to_cat, 1);



    auto outFinal = combined.narrow(1, 0, 16);
    outFinal = swish(outFinal);


    auto z = applyFlows(outFinal);


    auto mdnOut = mixtureParams(z);







    return mdnOut;
}


torch::Tensor BayesianMetaModelNNImpl::mixtureParams(const torch::Tensor& x) {
    auto rawOut = fcMDN->forward(x);
    auto splits = rawOut.split(5, 1);


    auto means = splits[0];
    auto logVars = splits[1].clamp(-10, 2);
    auto mixLogits = splits[2];
    auto mixingCoefs = torch::softmax(mixLogits, 1);

    return torch::cat({means, logVars, mixingCoefs}, 1);
}


torch::Tensor BayesianMetaModelNNImpl::applyFlows(const torch::Tensor& x)
{
    auto z = x;
    for (const auto& module : *flowTransforms) {
        auto flow = std::dynamic_pointer_cast<RealNVPFlowImpl>(module);
        if (flow) {
            auto outPair = flow->forward(z);
            z = outPair.first;
        }
    }
    return z;
}



BayesianTrainer::BayesianTrainer(int inputDim, int hiddenDim)
    : device_(torch::kCPU),
      model_(inputDim, hiddenDim),
      optimizer_(model_->parameters(), torch::optim::AdamOptions(0.001)),
      current_epoch_(0)
{


    #ifdef _WIN32
    HMODULE torchCudaDll = LoadLibraryA("torch_cuda.dll");
    if (torchCudaDll == NULL) {
        std::cerr << "Failed to load torch_cuda.dll. Error code: " << GetLastError() << std::endl;
    }
    else {
        std::cout << "Successfully loaded torch_cuda.dll" << std::endl;
    }
    #endif


    std::cout << "\nCUDA Diagnostics:" << std::endl;
    std::cout << "CUDA available: " << (torch::cuda::is_available() ? "Yes" : "No") << std::endl;
    std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;

    if(torch::cuda::is_available()) {
        try {

            device_ = torch::Device(torch::kCUDA, 0);
            std::cout << "Selected GPU device: 0" << std::endl;


            torch::Tensor testTensor = torch::ones({1}, device_);
            torch::cuda::synchronize();


            model_->to(device_);
            bool on_gpu = model_->parameters()[0].device().is_cuda();
            std::cout << "Model successfully moved to GPU: " << (on_gpu ? "Yes" : "No") << std::endl;

        } catch (const c10::Error& e) {
            std::cerr << "CUDA initialization error: " << e.msg() << std::endl;
            std::cerr << "Falling back to CPU" << std::endl;
            device_ = torch::Device(torch::kCPU);
        }
    } else {
        std::cout << "\nWARNING: CUDA is not available. Common reasons:" << std::endl;
        std::cout << "1. NVIDIA GPU driver not installed" << std::endl;
        std::cout << "2. LibTorch CUDA version mismatch with installed CUDA" << std::endl;
        std::cout << "3. LibTorch built without CUDA support" << std::endl;
        std::cout << "\nFalling back to CPU training (will be much slower)..." << std::endl;
        device_ = torch::Device(torch::kCPU);
    }


    model_->to(device_);


    optimizer_.zero_grad();
}

bool BayesianTrainer::LoadCheckpoint(const std::string& filename, int& epoch, double& epochLoss)
{

    try {
        torch::serialize::InputArchive archive;
        archive.load_from(filename);


        model_->load(archive);
        model_->to(device_);


        optimizer_.load(archive);


        torch::Tensor epT, lsT;
        archive.read("epoch", epT);
        archive.read("loss", lsT);
        epoch = epT.item<int>();
        epochLoss = lsT.item<double>();

        current_epoch_ = epoch;

        std::cout << "Loaded checkpoint from '" << filename << "' at epoch " << epoch
                  << " with loss " << epochLoss << "." << std::endl;

        return true;
    }
    catch(const c10::Error& e){
        std::cerr << "Error loading checkpoint: " << e.what() << std::endl;
        return false;
    }
    catch(const std::exception& e){
        std::cerr << "General error loading checkpoint: " << e.what() << std::endl;
        return false;
    }
}

void BayesianTrainer::SaveCheckpoint(const std::string& filename, int epoch, double epochLoss)
{

    torch::serialize::OutputArchive archive;


    model_->save(archive);


    optimizer_.save(archive);


    archive.write("epoch", torch::tensor(static_cast<int>(epoch)));
    archive.write("loss", torch::tensor(static_cast<double>(epochLoss)));

    archive.save_to(filename);
    std::cout << "Checkpoint saved to '" << filename << "' at epoch " << epoch
              << " with loss " << epochLoss << "." << std::endl;
}


struct MyCustomTensorDataset : public torch::data::Dataset<MyCustomTensorDataset> {
    torch::Tensor data_, target_;
    MyCustomTensorDataset(const torch::Tensor& data, const torch::Tensor& target)
        : data_(data), target_(target) {}

    torch::data::Example<> get(size_t index) override {
        return { data_[index], target_[index] };
    }

    torch::optional<size_t> size() const override {
        return static_cast<size_t>(data_.size(0));
    }
};

// Main training loop with checkpointing and progress tracking
void BayesianTrainer::TrainModel(const std::vector<std::vector<float>>& X,
                                 const std::vector<float>& Y,
                                 int numEpochs,
                                 int batchSize,
                                 double lr,
                                 const std::string& checkpointPath)
{


    if (lr != 0.001) {
        for (auto& param_group : optimizer_.param_groups()) {
            auto &options = static_cast<torch::optim::AdamOptions&>(param_group.options());
            options.lr(lr);
        }
    }


    auto dataTensor   = Vector2DToTensor(X).to(device_);
    auto targetTensor = Vector1DToTensor(Y).unsqueeze(1).to(device_);


    auto dataset = MyCustomTensorDataset(dataTensor, targetTensor)
        .map(torch::data::transforms::Stack<>());


    auto dataLoader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(batchSize).workers(0)
    );


    int start_epoch = 1;
    double last_epoch_loss = 0.0;
    if(fs::exists(checkpointPath)){
        if(LoadCheckpoint(checkpointPath, start_epoch, last_epoch_loss)){
            start_epoch += 1;
        }
        else{
            std::cout << "Starting training from scratch." << std::endl;
        }
    }
    else{
        std::cout << "No checkpoint found. Starting training from scratch." << std::endl;
    }


    model_->train();


    auto total_start_time = std::chrono::steady_clock::now();

    int64_t datasetSize   = dataTensor.size(0);
    int64_t totalBatches  = (datasetSize + batchSize - 1) / batchSize;

    for(int epoch = start_epoch; epoch <= numEpochs; ++epoch) {
        if(stop_training){
            std::cout << "Training interrupted. Saving current checkpoint..." << std::endl;
            SaveCheckpoint(checkpointPath, epoch - 1, last_epoch_loss);
            std::cout << "Checkpoint saved. Exiting training loop." << std::endl;
            return;
        }

        auto epoch_start_time = std::chrono::steady_clock::now();

        double epochLoss = 0.0;
        int64_t batchCount = 0;


        auto epoch_timer_start = std::chrono::steady_clock::now();


        for(auto& batch : *dataLoader) {
            auto batch_start_time = std::chrono::steady_clock::now();

            auto inputs  = batch.data.to(device_);
            auto targets = batch.target.to(device_);

            optimizer_.zero_grad();


            torch::Tensor outputs;
            try {
                outputs = model_->forward(inputs);
            } catch (const c10::Error& e) {
                std::cout << "\nForward pass error: " << e.what() << std::endl;
                continue;
            }


            torch::Tensor loss;
            if (epoch <= 5) {
                auto means = outputs.narrow(1, 0, 5);

                auto mean_pred = means.mean(1).unsqueeze(1);
                loss = torch::mse_loss(mean_pred, targets);
            } else {
                try {

                    auto splits = outputs.split(5, 1);
                    auto means = splits[0];
                    auto logvars = splits[1].clamp(-10, 2);
                    auto mixing_coefs = torch::softmax(splits[2], 1);


                    auto vars = torch::exp(logvars) + 1e-6;
                    auto diff = (means - targets.expand({targets.size(0), 5})).pow(2);
                    auto log_probs = -0.5 * (diff / vars + logvars + torch::log(torch::tensor(2 * M_PI).to(device_)));
                    auto log_mixing_coefs = torch::log(mixing_coefs + 1e-6);
                    auto joint_log_probs = log_probs + log_mixing_coefs;


                    auto log_sum = torch::logsumexp(joint_log_probs, 1);
                    loss = -log_sum.mean();

                } catch (const c10::Error& e) {
                    std::cout << "\nMDN loss computation error: " << e.what() << std::endl;
                    continue;
                }
            }


            double l2_lambda = 1e-7;
            auto l2_reg = torch::zeros({1}, device_);
            for (const auto& p : model_->parameters()) {
                if (p.grad().defined()) {
                    l2_reg += p.pow(2).sum();
                }
            }
            loss = loss + l2_lambda * l2_reg;


            if (std::isnan(loss.item<double>())) {
                std::cout << "\nNaN loss detected! Skipping batch." << std::endl;
                continue;
            }


            try {
                loss.backward();
                torch::nn::utils::clip_grad_norm_(model_->parameters(), 1.0);
                optimizer_.step();
            } catch (const c10::Error& e) {
                std::cout << "\nBackward pass error: " << e.what() << std::endl;
                optimizer_.zero_grad();
                continue;
            }

            double batch_loss = loss.item<double>();
            epochLoss += batch_loss;
            batchCount++;


            double progress = static_cast<double>(batchCount) / totalBatches;


            auto batch_end_time = std::chrono::steady_clock::now();
            std::chrono::duration<double> batch_duration = batch_end_time - batch_start_time;
            double batch_time = batch_duration.count();


            double remaining_batches = totalBatches - batchCount;
            double estimated_remaining = remaining_batches * batch_time;


            if (batchCount % 100 == 0) {

                std::cout << "\r"
                          << "Epoch " << epoch << "/" << numEpochs << " | "
                          << "Batch " << batchCount << "/" << totalBatches << " | "
                          << "Loss: " << std::fixed << std::setprecision(4) << batch_loss << " | "
                          << "Elapsed: " << std::fixed << std::setprecision(2)
                          << std::chrono::duration<double>(batch_start_time - epoch_start_time).count() << "s | "
                          << "ETA: " << std::fixed << std::setprecision(2) << estimated_remaining << "s"
                          << std::flush;
            }
        }


        epochLoss /= batchCount;
        last_epoch_loss = epochLoss;


        auto epoch_end_time = std::chrono::steady_clock::now();
        std::chrono::duration<double> epoch_duration = epoch_end_time - epoch_start_time;


        SaveCheckpoint(checkpointPath, epoch, epochLoss);


        std::cout << "\n"
                  << "========================================" << "\n"
                  << "Epoch " << epoch << " Completed." << "\n"
                  << "Average Loss: " << std::fixed << std::setprecision(4) << epochLoss << "\n"
                  << "Epoch Time: " << std::fixed << std::setprecision(2) << epoch_duration.count() << " seconds" << "\n"
                  << "========================================" << "\n";
    }


    auto total_end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_duration = total_end_time - total_start_time;
    std::cout << "Total Training Time: " << std::fixed << std::setprecision(2)
              << total_duration.count() << " seconds." << std::endl;

    std::cout << "Training complete.\n";
}

void BayesianTrainer::SaveModel(const std::string& filename)
{

    model_->to(torch::kCPU);
    torch::save(model_, filename);
    std::cout << "Model saved to " << filename << std::endl;
}

void BayesianTrainer::LoadModel(const std::string& filename)
{

    torch::load(model_, filename);
    model_->to(device_);
    std::cout << "Model loaded from " << filename << std::endl;
}

std::tuple<double,double,double> BayesianTrainer::MetaModelPrediction(
    const std::vector<float>& inputFeatures, int nSamples, double stds)
{


    auto inputTensor = torch::tensor(inputFeatures)
        .reshape({1, static_cast<long>(inputFeatures.size())})
        .to(device_);

    double sumPred = 0.0, sumSq = 0.0;


    model_->eval();
    torch::NoGradGuard no_grad;

    if(nSamples > 1) {

        model_->train();
        model_->bn1->eval();
        model_->bn2->eval();
        model_->bn3->eval();

        for(int i=0; i<nSamples; ++i){
            auto out = model_->forward(inputTensor);
            double val = out[0][0].item<double>();
            sumPred += val;
            sumSq   += (val * val);
        }

        double mean = sumPred / nSamples;
        double var  = (sumSq / nSamples) - (mean * mean);
        double stdVal = (var > 0 ? std::sqrt(var) : 0.0);


        double margin = stds * stdVal;
        double lower  = mean - margin;
        double upper  = mean + margin;


        model_->eval();
        return std::make_tuple(mean, lower, upper);
    }
    else {

        auto out = model_->forward(inputTensor);
        double val = out[0][0].item<double>();
        return std::make_tuple(val, val, val);
    }
}

torch::Tensor BayesianTrainer::Vector2DToTensor(const std::vector<std::vector<float>>& data)
{
    if(data.empty()) {
        throw std::runtime_error("Vector2DToTensor: data is empty.");
    }
    int64_t N = data.size();
    int64_t D = data[0].size();

    auto result = torch::empty({N, D}, torch::kFloat32);
    for (int64_t i=0; i<N; ++i) {
        for(int64_t j=0; j<D; ++j) {
            result[i][j] = data[i][j];
        }
    }
    return result;
}


torch::Tensor BayesianTrainer::Vector1DToTensor(const std::vector<float>& data)
{
    int64_t N = data.size();
    auto result = torch::empty({N}, torch::kFloat32);
    for(int64_t i=0; i<N; ++i) {
        result[i] = data[i];
    }
    return result;
}