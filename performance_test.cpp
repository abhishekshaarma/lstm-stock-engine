#include "lstm.h"
#include "StockCSVParser.h"
#include "Utils.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <omp.h>
#include <thread>

// Performance comparison between serial and parallel versions
int main()
{
    std::cout << "=== LSTM Performance Comparison ===" << std::endl;
    
    // Load data
    StockCSVParser parser;
    if (!parser.loadFromFile("./data/stock.csv")) 
    {
        std::cerr << "Failed to load CSV" << std::endl;
        return 1;
    }

    // Configuration
    int seq_len = 7;
    int features = 4;
    int neurons = 50;
    int epochs = 30;  // Reduced for quick comparison
    double lr = 5e-3;
    bool normalize = true;

    // Create sequences
    auto sequences = parser.createSequences(seq_len, features, normalize);
    auto targets = parser.getTargets(seq_len, normalize);
    
    if (sequences.empty()) {
        std::cerr << "No sequences" << std::endl;
        return 1;
    }

    // Split data
    auto [train_seq, test_seq] = parser.createRandomSplit(sequences, 0.8);
    auto [train_targets, test_targets] = parser.createRandomTargetSplit(targets, 0.8);

    // Limit data for quick test
    size_t max_train = std::min((size_t)300, train_seq.size());
    train_seq.resize(max_train);
    train_targets.resize(max_train);

    std::cout << "Test configuration:" << std::endl;
    std::cout << "Train samples: " << train_seq.size() << std::endl;
    std::cout << "Test samples: " << test_seq.size() << std::endl;
    std::cout << "Epochs: " << epochs << std::endl;
    std::cout << "Neurons: " << neurons << std::endl;
    std::cout << std::endl;

    // Initialize network
    lstm network(neurons, features);

    // Initialize dense layer
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> w_dist(0.0, 0.05), b_dist(0.0, 0.01);

    Eigen::MatrixXd dense_W(1, neurons);
    Eigen::VectorXd dense_b(1);
    for (int i = 0; i < neurons; i++) dense_W(0, i) = w_dist(gen);
    dense_b(0) = b_dist(gen);

    // Test 1: Serial Training
    std::cout << "=== Serial Training ===" << std::endl;
    auto start_serial = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        std::vector<double> preds, y_true;
        double loss_sum = 0.0;

        // Serial forward pass
        for (size_t i = 0; i < train_seq.size(); i++) 
        {
            auto out = network.forward(train_seq[i]);
            auto hidden = out.H.back();

            Eigen::VectorXd y_pred = dense_W * hidden + dense_b;
            double pred = y_pred(0), target = train_targets[i];
            double err = pred - target;
            double reg = 0.001 * (dense_W.squaredNorm() + dense_b.squaredNorm());
            double loss = 0.5 * err * err + reg;
            
            preds.push_back(pred);
            y_true.push_back(target);
            loss_sum += loss;
        }

        // Serial gradient update
        Eigen::MatrixXd dW = Eigen::MatrixXd::Zero(1, neurons);
        Eigen::VectorXd db = Eigen::VectorXd::Zero(1);
        network.zero_gradients();

        for (size_t i = 0; i < train_seq.size(); i++) 
        {
            auto out = network.forward(train_seq[i]);
            auto hidden = out.H.back();
            double err = preds[i] - y_true[i];
            
            dW += err * hidden.transpose() + 0.001 * dense_W;
            db(0) += err + 0.001 * dense_b(0);
            network.backward(train_seq[i], out, dense_W.transpose() * err);
        }

        double norm = dW.norm();
        if (norm > 1.0) dW *= 1.0 / norm;

        double bs = static_cast<double>(train_seq.size());
        dense_W -= lr * dW / bs;
        dense_b -= lr * db / bs;
        network.updateParameters(-lr / bs);

        if (epoch % 10 == 0) {
            double train_mse = calculateMSE(preds, y_true);
            std::cout << "Epoch " << (epoch + 1) << " | Train MSE: " << train_mse << std::endl;
        }
    }
    
    auto end_serial = std::chrono::high_resolution_clock::now();
    auto serial_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_serial - start_serial);
    std::cout << "Serial training time: " << serial_time.count() << "ms" << std::endl;

    // Test 2: Parallel Training
    std::cout << "\n=== Parallel Training ===" << std::endl;
    int num_threads = std::thread::hardware_concurrency();
    omp_set_num_threads(num_threads);
    std::cout << "Using " << num_threads << " threads" << std::endl;
    
    auto start_parallel = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        std::vector<double> preds, y_true;
        double loss_sum = 0.0;

        // Pre-allocate vectors
        preds.resize(train_seq.size());
        y_true.resize(train_seq.size());

        // Parallel forward pass
        #pragma omp parallel for reduction(+:loss_sum)
        for (size_t i = 0; i < train_seq.size(); i++) 
        {
            auto out = network.forward(train_seq[i]);
            auto hidden = out.H.back();

            Eigen::VectorXd y_pred = dense_W * hidden + dense_b;
            double pred = y_pred(0), target = train_targets[i];
            double err = pred - target;
            double reg = 0.001 * (dense_W.squaredNorm() + dense_b.squaredNorm());
            double loss = 0.5 * err * err + reg;
            
            preds[i] = pred;
            y_true[i] = target;
            loss_sum += loss;
        }

        // Serial gradient update (for fair comparison)
        Eigen::MatrixXd dW = Eigen::MatrixXd::Zero(1, neurons);
        Eigen::VectorXd db = Eigen::VectorXd::Zero(1);
        network.zero_gradients();

        for (size_t i = 0; i < train_seq.size(); i++) 
        {
            auto out = network.forward(train_seq[i]);
            auto hidden = out.H.back();
            double err = preds[i] - y_true[i];
            
            dW += err * hidden.transpose() + 0.001 * dense_W;
            db(0) += err + 0.001 * dense_b(0);
            network.backward(train_seq[i], out, dense_W.transpose() * err);
        }

        double norm = dW.norm();
        if (norm > 1.0) dW *= 1.0 / norm;

        double bs = static_cast<double>(train_seq.size());
        dense_W -= lr * dW / bs;
        dense_b -= lr * db / bs;
        network.updateParameters(-lr / bs);

        if (epoch % 10 == 0) {
            double train_mse = calculateMSE(preds, y_true);
            std::cout << "Epoch " << (epoch + 1) << " | Train MSE: " << train_mse << std::endl;
        }
    }
    
    auto end_parallel = std::chrono::high_resolution_clock::now();
    auto parallel_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_parallel - start_parallel);
    std::cout << "Parallel training time: " << parallel_time.count() << "ms" << std::endl;

    // Performance comparison
    std::cout << "\n=== Performance Results ===" << std::endl;
    std::cout << "Serial time: " << serial_time.count() << "ms" << std::endl;
    std::cout << "Parallel time: " << parallel_time.count() << "ms" << std::endl;
    
    double speedup = static_cast<double>(serial_time.count()) / parallel_time.count();
    std::cout << "Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    
    double efficiency = speedup / num_threads * 100.0;
    std::cout << "Efficiency: " << std::fixed << std::setprecision(1) << efficiency << "%" << std::endl;
    
    if (speedup > 1.0) {
        std::cout << "✅ Parallel version is " << std::fixed << std::setprecision(1) 
                  << (speedup - 1.0) * 100.0 << "% faster!" << std::endl;
    } else {
        std::cout << "⚠️  Parallel overhead may be present for small datasets" << std::endl;
    }

    return 0;
}
