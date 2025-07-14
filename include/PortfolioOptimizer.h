#pragma once
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <iostream>

class PortfolioOptimizer {
private:
    int n_assets;
    double risk_free_rate;
    double max_weight;  // Maximum weight per asset
    
public:
    PortfolioOptimizer(int n_assets, double risk_free_rate = 0.02, double max_weight = 0.3)
        : n_assets(n_assets), risk_free_rate(risk_free_rate), max_weight(max_weight) {}
    
    // Factor selection based on Information Coefficient (IC)
    std::vector<int> selectFactors(const std::vector<std::vector<double>>& factor_data,
                                 const std::vector<double>& returns,
                                 double ic_threshold = 0.05) {
        
        std::cout << "\n=== Factor Selection (IC Analysis) ===" << std::endl;
        
        std::vector<int> selected_factors;
        int n_factors = factor_data.size();
        
        for (int f = 0; f < n_factors; f++) {
            double ic = calculateIC(factor_data[f], returns);
            
            std::cout << "Factor " << f << " IC: " << std::fixed << std::setprecision(4) << ic;
            
            if (std::abs(ic) > ic_threshold) {
                selected_factors.push_back(f);
                std::cout << " ✓ Selected";
            } else {
                std::cout << " ✗ Rejected";
            }
            std::cout << std::endl;
        }
        
        std::cout << "Selected " << selected_factors.size() << " factors out of " << n_factors << std::endl;
        return selected_factors;
    }
    
    // Calculate Information Coefficient (correlation between factor and future returns)
    double calculateIC(const std::vector<double>& factor_values,
                      const std::vector<double>& future_returns) {
        
        size_t n = std::min(factor_values.size(), future_returns.size());
        if (n < 2) return 0.0;
        
        // Calculate means
        double mean_factor = std::accumulate(factor_values.begin(), factor_values.begin() + n, 0.0) / n;
        double mean_return = std::accumulate(future_returns.begin(), future_returns.begin() + n, 0.0) / n;
        
        // Calculate correlation
        double numerator = 0.0, denom_factor = 0.0, denom_return = 0.0;
        
        for (size_t i = 0; i < n; i++) {
            double factor_dev = factor_values[i] - mean_factor;
            double return_dev = future_returns[i] - mean_return;
            
            numerator += factor_dev * return_dev;
            denom_factor += factor_dev * factor_dev;
            denom_return += return_dev * return_dev;
        }
        
        double denominator = std::sqrt(denom_factor * denom_return);
        return (denominator > 1e-8) ? numerator / denominator : 0.0;
    }
    
    // Simple mean-variance optimization (Markowitz-style)
    Eigen::VectorXd optimizePortfolio(const Eigen::VectorXd& expected_returns,
                                    const Eigen::MatrixXd& covariance_matrix,
                                    double risk_aversion = 1.0) {
        
        std::cout << "\n=== Portfolio Optimization ===" << std::endl;
        
        int n = expected_returns.size();
        Eigen::VectorXd weights = Eigen::VectorXd::Ones(n) / n;  // Equal weight starting point
        
        // Simplified optimization: maximize return - risk_aversion * risk
        // In practice, you'd use quadratic programming
        
        for (int iter = 0; iter < 100; iter++) {
            Eigen::VectorXd gradient = expected_returns - risk_aversion * covariance_matrix * weights;
            
            // Simple gradient ascent with constraints
            weights += 0.01 * gradient;
            
            // Apply constraints
            for (int i = 0; i < n; i++) {
                weights(i) = std::max(0.0, std::min(max_weight, weights(i)));
            }
            
            // Normalize to sum to 1
            double sum_weights = weights.sum();
            if (sum_weights > 1e-8) {
                weights /= sum_weights;
            }
        }
        
        // Print portfolio composition
        std::cout << "Optimized Portfolio Weights:" << std::endl;
        for (int i = 0; i < n; i++) {
            std::cout << "Asset " << i << ": " << std::fixed << std::setprecision(3) 
                      << weights(i) * 100 << "%" << std::endl;
        }
        
        // Calculate expected portfolio metrics
        double portfolio_return = weights.transpose() * expected_returns;
        double portfolio_risk = std::sqrt(weights.transpose() * covariance_matrix * weights);
        double sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk;
        
        std::cout << "\nPortfolio Metrics:" << std::endl;
        std::cout << "Expected Return: " << std::setprecision(4) << portfolio_return * 100 << "%" << std::endl;
        std::cout << "Expected Risk: " << portfolio_risk * 100 << "%" << std::endl;
        std::cout << "Expected Sharpe: " << std::setprecision(2) << sharpe_ratio << std::endl;
        
        return weights;
    }
    
    // Risk-constrained optimization with VaR limit
    Eigen::VectorXd optimizeWithVaRConstraint(const Eigen::VectorXd& expected_returns,
                                             const std::vector<std::vector<double>>& historical_returns,
                                             double max_var = 0.05) {
        
        std::cout << "\n=== VaR-Constrained Portfolio Optimization ===" << std::endl;
        
        int n = expected_returns.size();
        Eigen::VectorXd best_weights = Eigen::VectorXd::Ones(n) / n;
        double best_sharpe = -1e9;
        
        // Monte Carlo search for weights satisfying VaR constraint
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        for (int trial = 0; trial < 10000; trial++) {
            Eigen::VectorXd weights(n);
            double sum = 0.0;
            
            // Generate random weights
            for (int i = 0; i < n; i++) {
                weights(i) = dis(gen);
                sum += weights(i);
            }
            weights /= sum;  // Normalize
            
            // Apply max weight constraint
            bool valid = true;
            for (int i = 0; i < n; i++) {
                if (weights(i) > max_weight) {
                    valid = false;
                    break;
                }
            }
            if (!valid) continue;
            
            // Calculate portfolio returns
            std::vector<double> portfolio_returns;
            size_t min_periods = historical_returns[0].size();
            for (size_t t = 0; t < min_periods; t++) {
                double port_ret = 0.0;
                for (int i = 0; i < n; i++) {
                    if (t < historical_returns[i].size()) {
                        port_ret += weights(i) * historical_returns[i][t];
                    }
                }
                portfolio_returns.push_back(port_ret);
            }
            
            // Check VaR constraint
            double var = RiskMetrics::calculateVaR(portfolio_returns);
            if (var > max_var) continue;
            
            // Calculate Sharpe ratio
            double sharpe = RiskMetrics::calculateSharpeRatio(portfolio_returns);
            
            if (sharpe > best_sharpe) {
                best_sharpe = sharpe;
                best_weights = weights;
            }
        }
        
        std::cout << "Best Sharpe ratio found: " << std::fixed << std::setprecision(3) << best_sharpe << std::endl;
        std::cout << "VaR-constrained weights:" << std::endl;
        for (int i = 0; i < n; i++) {
            std::cout << "Asset " << i << ": " << std::setprecision(3) 
                      << best_weights(i) * 100 << "%" << std::endl;
        }
        
        return best_weights;
    }
    
    // Rebalance portfolio based on new LSTM predictions
    Eigen::VectorXd rebalancePortfolio(const Eigen::VectorXd& current_weights,
                                     const Eigen::VectorXd& new_predictions,
                                     double rebalance_threshold = 0.05) {
        
        std::cout << "\n=== Portfolio Rebalancing ===" << std::endl;
        
        // Calculate target weights based on predictions
        Eigen::VectorXd target_weights = new_predictions;
        target_weights = target_weights.cwiseMax(0.0);  // No short selling
        
        double sum = target_weights.sum();
        if (sum > 1e-8) {
            target_weights /= sum;
        }
        
        // Check if rebalancing is needed
        double max_deviation = (target_weights - current_weights).cwiseAbs().maxCoeff();
        
        if (max_deviation < rebalance_threshold) {
            std::cout << "No rebalancing needed (max deviation: " 
                      << std::fixed << std::setprecision(3) << max_deviation << ")" << std::endl;
            return current_weights;
        }
        
        std::cout << "Rebalancing triggered (max deviation: " 
                  << std::fixed << std::setprecision(3) << max_deviation << ")" << std::endl;
        
        std::cout << "Weight changes:" << std::endl;
        for (int i = 0; i < target_weights.size(); i++) {
            double change = target_weights(i) - current_weights(i);
            std::cout << "Asset " << i << ": " << std::setprecision(3) 
                      << current_weights(i) * 100 << "% → " 
                      << target_weights(i) * 100 << "% (" 
                      << std::showpos << change * 100 << "%)" << std::noshowpos << std::endl;
        }
        
        return target_weights;
    }
};
