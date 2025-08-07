#include "lstm.h"
#include "StockCSVParser.h"
#include "MultiFactorModel.h"
#include "Utils.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <omp.h>
#include <thread>

// Simple parallel training demonstration
int main()
{
    // Set number of threads
    int num_threads = std::thread::hardware_concurrency();
    omp_set_num_threads(num_threads);
    
    std::cout << "=== LSTM Multi-Factor Investment Model (Parallel Training Demo) ===" << std::endl;
    std::cout << "Using " << num_threads << " threads" << std::endl;

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
    int epochs = 50;  // Reduced for demo
    double lr = 5e-3;
    bool normalize = true;

    std::cout << "\nConfig:\nNeurons: " << neurons << "\nLR: " << std::scientific << lr
              << "\nEpochs: " << epochs << "\nSeq len: " << seq_len << std::endl;

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

    // Limit training data for demo
    size_t max_train = std::min((size_t)500, train_seq.size());
    train_seq.resize(max_train);
    train_targets.resize(max_train);

    std::cout << "Train samples: " << train_seq.size() << "\nTest samples: " << test_seq.size() << std::endl;

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

    std::cout << "\n=== Training (Parallel) ===\nProgress: ";

    auto start = std::chrono::high_resolution_clock::now();

    // Training loop with parallel processing
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        std::vector<double> preds, y_true;
        double loss_sum = 0.0;

        if (epoch > 0 && epoch % 20 == 0) lr *= 0.8;

        // Parallel forward pass and loss computation
        preds.resize(train_seq.size());
        y_true.resize(train_seq.size());
        
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

        // Simple gradient update (serial for now)
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

        // Progress indicator
        if (epoch % 10 == 0) std::cout << "+";
        else std::cout << ".";
        std::cout.flush();

        // Validation every 25 epochs
        if (epoch % 25 == 0 || epoch == epochs - 1)
        {
            std::cout << std::endl;
            double train_mse = calculateMSE(preds, y_true);
            
            // Parallel validation
            std::vector<double> val_preds, val_y;
            int n_val = std::min(100, (int)test_seq.size());
            val_preds.resize(n_val);
            val_y.resize(n_val);
            
            #pragma omp parallel for
            for (int i = 0; i < n_val; i++) 
            {
                auto out = network.forward(test_seq[i]);
                Eigen::VectorXd h = out.H.back();
                val_preds[i] = (dense_W * h + dense_b)(0);
                val_y[i] = test_targets[i];
            }
            
            double val_mse = calculateMSE(val_preds, val_y);
            auto minmax = std::minmax_element(preds.begin(), preds.end());
            
            std::cout << "Epoch " << (epoch + 1) << " | Train MSE: " << train_mse
                      << " | Val MSE: " << val_mse << " | LR: " << std::scientific << lr
                      << " | Range: [" << std::fixed << std::setprecision(4)
                      << *minmax.first << ", " << *minmax.second << "]" << std::endl;
            std::cout << "Progress: ";
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto train_time = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << "\nTraining time: " << train_time.count() << "s\n";

    // Final evaluation
    std::vector<double> preds, actuals;
    int test_count = std::min(200, (int)test_seq.size());
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

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "LSTM MSE: " << calculateMSE(preds, actuals) << std::endl;
    std::cout << "LSTM R²: " << calculateRSquared(preds, actuals) << std::endl;

    return 0;
}
