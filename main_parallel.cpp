#include "lstm.h"
#include "StockCSVParser.h"
#include "MultiFactorModel.h"
#include "RiskMetrics.h"
#include "PortfolioOptimizer.h"
#include "Utils.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <chrono>
#include <omp.h>
#include <thread>
#include <mutex>

// Thread-safe random number generator
class ThreadSafeRandom {
private:
    std::mt19937 gen;
    std::mutex mtx;
    
public:
    ThreadSafeRandom() : gen(std::random_device{}()) {}
    
    double operator()(std::uniform_real_distribution<double>& dist) {
        std::lock_guard<std::mutex> lock(mtx);
        return dist(gen);
    }
    
    int operator()(std::uniform_int_distribution<int>& dist) {
        std::lock_guard<std::mutex> lock(mtx);
        return dist(gen);
    }
};

// Parallel batch processing structure
struct BatchData {
    std::vector<std::vector<Eigen::VectorXd>> sequences;
    std::vector<double> targets;
    std::vector<double> predictions;
    std::vector<double> losses;
    
    BatchData(size_t size) : sequences(size), targets(size), predictions(size), losses(size) {}
};

// Parallel forward pass for a batch of sequences
std::vector<LSTMOutput> parallelForward(lstm& network, const std::vector<std::vector<Eigen::VectorXd>>& sequences) {
    std::vector<LSTMOutput> outputs(sequences.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < sequences.size(); ++i) {
        outputs[i] = network.forward(sequences[i]);
    }
    
    return outputs;
}

// Parallel gradient computation
void parallelGradientComputation(
    const std::vector<std::vector<Eigen::VectorXd>>& sequences,
    const std::vector<LSTMOutput>& outputs,
    const std::vector<double>& errors,
    const Eigen::MatrixXd& dense_W,
    std::vector<Eigen::MatrixXd>& dW_batch,
    std::vector<Eigen::VectorXd>& db_batch,
    std::vector<lstm>& network_copies
) {
    #pragma omp parallel for
    for (size_t i = 0; i < sequences.size(); ++i) {
        network_copies[i].zero_gradients();
        network_copies[i].backward(sequences[i], outputs[i], dense_W.transpose() * errors[i]);
        
        // Compute dense layer gradients
        auto hidden = outputs[i].H.back();
        dW_batch[i] = errors[i] * hidden.transpose();
        db_batch[i] = Eigen::VectorXd::Constant(1, errors[i]);
    }
}

int main()
{
    // Set number of threads
    int num_threads = std::thread::hardware_concurrency();
    omp_set_num_threads(num_threads);
    
    std::cout << "=== LSTM Multi-Factor Investment Model (Parallel) ===" << std::endl;
    std::cout << "Using " << num_threads << " threads" << std::endl;

    StockCSVParser parser;
    if (!parser.loadFromFile("./data/stock.csv")) 
    {
        std::cerr << "Failed to load CSV" << std::endl;
        return 1;
    }

    auto range = parser.getPriceRange();
    std::cout << "Price range: $" << std::fixed << std::setprecision(2)
              << range.first << " to $" << range.second << std::endl;

    int seq_len = 7;
    int features = 4;
    int neurons = 50;
    int epochs = 150;
    int inner_iters = 4;
    double lr = 5e-3;
    bool normalize = true;
    
    // Batch size for parallel processing
    int batch_size = 32;

    std::cout << "\nConfig:\nNeurons: " << neurons << "\nLR: " << std::scientific << lr
              << "\nEpochs: " << epochs << "\nSeq len: " << seq_len
              << "\nInner iters: " << inner_iters << "\nBatch size: " << batch_size << "\n" << std::endl;

    auto sequences = parser.createSequences(seq_len, features, normalize);
    auto targets = parser.getTargets(seq_len, normalize);
    if (sequences.empty())
     {
        std::cerr << "No sequences" << std::endl;
        return 1;
    }

    auto [train_seq, test_seq] = parser.createRandomSplit(sequences, 0.8);
    auto [train_targets, test_targets] = parser.createRandomTargetSplit(targets, 0.8);

    size_t max_train = std::min((size_t)1000, train_seq.size());
    train_seq.resize(max_train);
    train_targets.resize(max_train);

    std::cout << "Train samples: " << train_seq.size() << "\nTest samples: " << test_seq.size() << std::endl;

    auto [min_price, max_price] = parser.getPriceRange();
    min_price = std::max(5.0, min_price);
    max_price = std::min(500.0, max_price);

    lstm network(neurons, features);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> w_dist(0.0, 0.05), b_dist(0.0, 0.01);

    Eigen::MatrixXd dense_W(1, neurons);
    Eigen::VectorXd dense_b(1);
    for (int i = 0; i < neurons; i++) dense_W(0, i) = w_dist(gen);
    dense_b(0) = b_dist(gen);

    std::cout << "\n=== Training (Parallel) ===\nProgress: ";

    auto start = std::chrono::high_resolution_clock::now();

    double best_val_mse = std::numeric_limits<double>::max();
    int patience = 0;

    // Create network copies for parallel processing
    std::vector<lstm> network_copies(num_threads, lstm(neurons, features));
    for (int i = 0; i < num_threads; ++i) {
        network_copies[i] = network;
    }

    for (int epoch = 0; epoch < epochs; epoch++)
     {
        std::vector<double> preds, y_true;
        double loss_sum = 0.0;

        if (epoch > 0 && epoch % 40 == 0) lr *= 0.8;

        for (int iter = 0; iter < inner_iters; iter++) 
        {
            Eigen::MatrixXd dW = Eigen::MatrixXd::Zero(1, neurons);
            Eigen::VectorXd db = Eigen::VectorXd::Zero(1);
            network.zero_gradients();

            // Process in batches
            for (size_t batch_start = 0; batch_start < train_seq.size(); batch_start += batch_size) {
                size_t batch_end = std::min(batch_start + batch_size, train_seq.size());
                size_t actual_batch_size = batch_end - batch_start;
                
                // Create batch data
                std::vector<std::vector<Eigen::VectorXd>> batch_sequences(
                    train_seq.begin() + batch_start, 
                    train_seq.begin() + batch_end
                );
                std::vector<double> batch_targets(
                    train_targets.begin() + batch_start, 
                    train_targets.begin() + batch_end
                );
                
                // Parallel forward pass
                auto batch_outputs = parallelForward(network, batch_sequences);
                
                // Compute predictions and errors
                std::vector<double> batch_preds(actual_batch_size);
                std::vector<double> batch_errors(actual_batch_size);
                std::vector<Eigen::MatrixXd> dW_batch(actual_batch_size);
                std::vector<Eigen::VectorXd> db_batch(actual_batch_size);
                
                #pragma omp parallel for
                for (size_t i = 0; i < actual_batch_size; ++i) {
                    auto hidden = batch_outputs[i].H.back();
                    Eigen::VectorXd y_pred = dense_W * hidden + dense_b;
                    double pred = y_pred(0), target = batch_targets[i];
                    double err = pred - target;
                    double reg = 0.001 * (dense_W.squaredNorm() + dense_b.squaredNorm());
                    double loss = 0.5 * err * err + reg;
                    
                    batch_preds[i] = pred;
                    batch_errors[i] = err;
                    dW_batch[i] = err * hidden.transpose() + 0.001 * dense_W;
                    db_batch[i] = Eigen::VectorXd::Constant(1, err + 0.001 * dense_b(0));
                    
                    #pragma omp critical
                    {
                        loss_sum += loss;
                        preds.push_back(pred);
                        y_true.push_back(target);
                    }
                }
                
                // Parallel gradient computation
                parallelGradientComputation(
                    batch_sequences, batch_outputs, batch_errors,
                    dense_W, dW_batch, db_batch, network_copies
                );
                
                // Aggregate gradients
                for (size_t i = 0; i < actual_batch_size; ++i) {
                    dW += dW_batch[i];
                    db += db_batch[i];
                }
            }

            double norm = dW.norm();
            if (norm > 1.0) dW *= 1.0 / norm;

            double bs = static_cast<double>(train_seq.size());
            dense_W -= lr * dW / bs;
            dense_b -= lr * db / bs;
            network.updateParameters(-lr / bs);
        }

        if (epoch % 8 == 0) std::cout << "+";
        else std::cout << ".";
        std::cout.flush();

        if (epoch % 25 == 0 || epoch == epochs - 1)
         {
            std::cout << std::endl;
            double train_mse = calculateMSE(preds, y_true);
            std::vector<double> val_preds, val_y;
            int n_val = std::min(100, (int)test_seq.size());
            
            // Parallel validation
            #pragma omp parallel for
            for (int i = 0; i < n_val; i++) 
            {
                auto out = network.forward(test_seq[i]);
                Eigen::VectorXd h = out.H.back();
                double pred = (dense_W * h + dense_b)(0);
                
                #pragma omp critical
                {
                    val_preds.push_back(pred);
                    val_y.push_back(test_targets[i]);
                }
            }
            
            double val_mse = calculateMSE(val_preds, val_y);
            if (val_mse < best_val_mse)
            {
                best_val_mse = val_mse;
                patience = 0;
            }
            else
            {
                patience++;
            }
            auto minmax = std::minmax_element(preds.begin(), preds.end());
            std::cout << "Epoch " << (epoch + 1) << " | Train MSE: " << train_mse
                      << " | Val MSE: " << val_mse << " | LR: " << std::scientific << lr
                      << " | Range: [" << std::fixed << std::setprecision(4)
                      << *minmax.first << ", " << *minmax.second << "]" << std::endl;
            std::cout << "Progress: ";
            if (patience >= 3 && epoch > 75) 
            {
                std::cout << "\nEarly stopping\n";
                break;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << "\nTraining time: " << train_time.count() << "s\n";

    std::vector<double> preds, actuals;
    int test_count = std::min(500, (int)test_seq.size());
    
    // Parallel testing
    preds.resize(test_count);
    actuals.resize(test_count);
    
    #pragma omp parallel for
    for (int i = 0; i < test_count; i++) 
    {
        auto out = network.forward(test_seq[i]);
        auto h = out.H.back();
        preds[i] = (dense_W * h + dense_b)(0);
        actuals[i] = test_targets[i];
    }

    MultiFactorModel benchmark(features, 1);
    std::vector<Eigen::VectorXd> returns, factors;
    for (size_t i = 0; i < std::min((size_t)800, train_targets.size()); i++) 
    {
        Eigen::VectorXd r(1); r(0) = train_targets[i];
        returns.push_back(r);
        factors.push_back(train_seq[i].back());
    }
    benchmark.fit(returns, factors);
    double r2 = benchmark.calculateRSquared(returns, factors);

    std::vector<double> bench_preds;
    for (int i = 0; i < test_count; i++) 
    {
        bench_preds.push_back(benchmark.predict(factors[i]));
    }

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "LSTM MSE: " << calculateMSE(preds, actuals) << std::endl;
    std::cout << "LSTM R²: " << calculateRSquared(preds, actuals) << std::endl;
    std::cout << "Benchmark MSE: " << calculateMSE(bench_preds, actuals) << std::endl;
    std::cout << "Benchmark R²: " << r2 << std::endl;

    return 0;
}
