// Utils.h - Utility functions for LSTM training
#pragma once
#include <vector>
#include <cmath>

// Simple loss function - Mean Squared Error
double calculateMSE(const std::vector<double>& predictions, const std::vector<double>& targets);

// Convert normalized predictions back to actual returns
std::vector<double> convertToReturns(const std::vector<double>& normalized_preds,
                                   const std::vector<double>& normalized_targets,
                                   double min_price, double max_price);
double computeSharpeRatio(const std::vector<double>& returns);
double computeVaR(const std::vector<double>& returns, double confidence = 0.95);
double computeMaxDrawdown(const std::vector<double>& cumulative_returns);

