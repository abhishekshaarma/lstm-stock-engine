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

int main()
{
    std::cout << "=== LSTM Multi-Factor Investment Model ===" << std::endl;

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

    std::cout << "\nConfig:\nNeurons: " << neurons << "\nLR: " << std::scientific << lr
              << "\nEpochs: " << epochs << "\nSeq len: " << seq_len
              << "\nInner iters: " << inner_iters << "\n" << std::endl;

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

    std::cout << "\n=== Training ===\nProgress: ";

    auto start = std::chrono::high_resolution_clock::now();

    double best_val_mse = std::numeric_limits<double>::max();
    int patience = 0;

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

            for (size_t i = 0; i < train_seq.size(); i++) 
            {
                auto out = network.forward(train_seq[i]);
                auto hidden = out.H.back();

                Eigen::VectorXd h = hidden;
                if (iter < inner_iters - 1)
                 {
                    for (int j = 0; j < h.size(); j++) 
                    {
                        if (gen() % 10 < 2) h(j) *= 0.5;
                    }
                }

                Eigen::VectorXd y_pred = dense_W * h + dense_b;
                double pred = y_pred(0), target = train_targets[i];
                double err = pred - target;
                double reg = 0.001 * (dense_W.squaredNorm() + dense_b.squaredNorm());
                double loss = 0.5 * err * err + reg;
                loss_sum += loss;

                preds.push_back(pred);
                y_true.push_back(target);

                dW += err * h.transpose() + 0.001 * dense_W;
                db(0) += err + 0.001 * dense_b(0);
                network.backward(train_seq[i], out, dense_W.transpose() * err);
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
            for (int i = 0; i < n_val; i++) 
            {
                auto out = network.forward(test_seq[i]);
                Eigen::VectorXd h = out.H.back();
                val_preds.push_back((dense_W * h + dense_b)(0));
                val_y.push_back(test_targets[i]);
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
    for (int i = 0; i < test_count; i++) 
    {
        auto out = network.forward(test_seq[i]);
        auto h = out.H.back();
        preds.push_back((dense_W * h + dense_b)(0));
        actuals.push_back(test_targets[i]);
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
        bench_preds.push_back(benchmark.predict(test_seq[i].back()));
    }

    double lstm_mse = calculateMSE(preds, actuals);
    double bench_mse = calculateMSE(bench_preds, actuals);
    double variation = *std::max_element(preds.begin(), preds.end()) - *std::min_element(preds.begin(), preds.end());

    int correct = 0;
    for (int i = 1; i < test_count; i++) 
    {
        bool up_lstm = preds[i] > preds[i-1];
        bool up_actual = actuals[i] > actuals[i-1];
        if (up_lstm == up_actual) correct++;
    }
    double direction = static_cast<double>(correct) / (test_count - 1) * 100;

    std::vector<double> prices, actual_prices, bench_prices;
    for (int i = 0; i < test_count; i++) 
    {
        prices.push_back(preds[i] * (max_price - min_price) + min_price);
        actual_prices.push_back(actuals[i] * (max_price - min_price) + min_price);
        bench_prices.push_back(bench_preds[i] * (max_price - min_price) + min_price);
    }

    std::vector<double> returns_pct;
    for (size_t i = 1; i < prices.size(); i++) 
    {
        double r = (prices[i] / prices[i - 1]) - 1.0;
        returns_pct.push_back(r);
    }

    double sharpe = computeSharpeRatio(returns_pct);
    double var_95 = computeVaR(returns_pct, 0.95);
    double max_dd = computeMaxDrawdown(prices);

    std::cout << "\n=== Evaluation ===" << std::endl;
    std::cout << "R² (Benchmark): " << std::fixed << std::setprecision(3) << r2 << std::endl;
    std::cout << "LSTM MSE: " << lstm_mse << "\nBenchmark MSE: " << bench_mse << std::endl;
    std::cout << "Prediction range: [" << *std::min_element(preds.begin(), preds.end()) << ", "
              << *std::max_element(preds.begin(), preds.end()) << "] (var: " << variation << ")" << std::endl;
    std::cout << "Direction accuracy: " << std::fixed << std::setprecision(1) << direction << "%" << std::endl;

    std::cout << "\n=== Risk Metrics ===" << std::endl;
    std::cout << "Sharpe Ratio: " << std::fixed << std::setprecision(2) << sharpe << std::endl;
    std::cout << "VaR (95%): " << std::fixed << std::setprecision(2) << var_95 << std::endl;
    std::cout << "Max Drawdown: " << std::fixed << std::setprecision(2) << max_dd << std::endl;

    return 0;
}
