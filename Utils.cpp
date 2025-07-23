// Utils.cpp - Implementation of utility functions
#include "Utils.h"
#include <cmath>
#include <algorithm>
#include <numeric>

double computeSharpeRatio(const std::vector<double>& returns) {
    if (returns.empty()) return 0.0;
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double sum_sq = 0.0;
    for (double r : returns) sum_sq += (r - mean) * (r - mean);
    double stddev = std::sqrt(sum_sq / (returns.size() - 1));
    return stddev > 0 ? mean / stddev : 0.0;
}

double computeVaR(const std::vector<double>& returns, double confidence) {
    if (returns.empty()) return 0.0;
    std::vector<double> sorted = returns;
    std::sort(sorted.begin(), sorted.end());
    size_t index = static_cast<size_t>((1.0 - confidence) * sorted.size());
    return sorted[index];
}

double computeMaxDrawdown(const std::vector<double>& cumulative_returns) {
    if (cumulative_returns.empty()) return 0.0;
    double max_drawdown = 0.0;
    double peak = cumulative_returns[0];
    for (double r : cumulative_returns) {
        if (r > peak) peak = r;
        double drawdown = (peak - r) / peak;
        if (drawdown > max_drawdown) max_drawdown = drawdown;
    }
    return max_drawdown;
}

// Simple loss function - Mean Squared Error
double calculateMSE(const std::vector<double>& predictions, const std::vector<double>& targets)
{
    if (predictions.size() != targets.size() || predictions.empty()) return 0.0;
    
    double mse = 0.0;
    for (size_t i = 0; i < predictions.size(); i++) {
        double error = predictions[i] - targets[i];
        mse += error * error;
    }
    return mse / predictions.size();
}

// Convert normalized predictions back to actual returns
std::vector<double> convertToReturns(const std::vector<double>& normalized_preds,
                                   const std::vector<double>& normalized_targets,
                                   double min_price, double max_price)
{
    std::vector<double> returns;
    
    for (size_t i = 0; i < normalized_preds.size(); i++)
    {
        double pred_price = normalized_preds[i] * (max_price - min_price) + min_price;
        double actual_price = normalized_targets[i] * (max_price - min_price) + min_price;
        
        // Calculate return as percentage change
        double return_val = (pred_price - actual_price) / actual_price;
        returns.push_back(return_val);
    }
    
    return returns;
}
