#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <iomanip>

class RiskMetrics {
public:
    // Calculate Maximum Drawdown (key metric in paper)
    static double calculateMaxDrawdown(const std::vector<double>& returns) {
        if (returns.empty()) return 0.0;
        
        std::vector<double> cumulative_returns(returns.size());
        cumulative_returns[0] = 1.0 + returns[0];
        
        for (size_t i = 1; i < returns.size(); i++) {
            cumulative_returns[i] = cumulative_returns[i-1] * (1.0 + returns[i]);
        }
        
        double max_drawdown = 0.0;
        double peak = cumulative_returns[0];
        
        for (size_t i = 1; i < cumulative_returns.size(); i++) {
            if (cumulative_returns[i] > peak) {
                peak = cumulative_returns[i];
            }
            
            double drawdown = (peak - cumulative_returns[i]) / peak;
            max_drawdown = std::max(max_drawdown, drawdown);
        }
        
        return max_drawdown;
    }
    
    // Calculate Sharpe Ratio (risk-adjusted return)
    static double calculateSharpeRatio(const std::vector<double>& returns, 
                                     double risk_free_rate = 0.02) {
        if (returns.empty()) return 0.0;
        
        double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
        
        double variance = 0.0;
        for (double ret : returns) {
            variance += (ret - mean_return) * (ret - mean_return);
        }
        variance /= returns.size();
        double std_dev = std::sqrt(variance);
        
        if (std_dev == 0.0) return 0.0;
        
        // Annualized Sharpe ratio (assuming daily returns)
        double annualized_return = mean_return * 252;  // 252 trading days
        double annualized_std = std_dev * std::sqrt(252);
        
        return (annualized_return - risk_free_rate) / annualized_std;
    }
    
    // Calculate Value at Risk (VaR) at 95% confidence level
    static double calculateVaR(const std::vector<double>& returns, double confidence = 0.95) {
        if (returns.empty()) return 0.0;
        
        std::vector<double> sorted_returns = returns;
        std::sort(sorted_returns.begin(), sorted_returns.end());
        
        size_t var_index = static_cast<size_t>((1.0 - confidence) * sorted_returns.size());
        if (var_index >= sorted_returns.size()) var_index = sorted_returns.size() - 1;
        
        return -sorted_returns[var_index];  // Negative because VaR represents potential loss
    }
    
    // Calculate portfolio volatility (annualized)
    static double calculateVolatility(const std::vector<double>& returns) {
        if (returns.empty()) return 0.0;
        
        double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
        
        double variance = 0.0;
        for (double ret : returns) {
            variance += (ret - mean_return) * (ret - mean_return);
        }
        variance /= returns.size();
        
        // Annualized volatility
        return std::sqrt(variance * 252);
    }
    
    // Print comprehensive risk report (like Table 1 in paper)
    static void printRiskReport(const std::vector<double>& lstm_returns,
                              const std::vector<double>& benchmark_returns,
                              const std::string& model_name = "LSTM") {
        
        std::cout << "\n=== Risk Control Metrics Comparison (Paper Table 1) ===" << std::endl;
        std::cout << std::setw(25) << "Metric" 
                  << std::setw(15) << "Benchmark" 
                  << std::setw(15) << model_name 
                  << std::setw(15) << "Improvement" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        // Maximum Drawdown
        double bench_mdd = calculateMaxDrawdown(benchmark_returns) * 100;
        double lstm_mdd = calculateMaxDrawdown(lstm_returns) * 100;
        double mdd_improvement = (bench_mdd - lstm_mdd) / bench_mdd * 100;
        
        std::cout << std::setw(25) << "Max Drawdown (%)" 
                  << std::setw(15) << std::fixed << std::setprecision(1) << bench_mdd
                  << std::setw(15) << lstm_mdd
                  << std::setw(15) << mdd_improvement << "%" << std::endl;
        
        // Sharpe Ratio
        double bench_sharpe = calculateSharpeRatio(benchmark_returns);
        double lstm_sharpe = calculateSharpeRatio(lstm_returns);
        double sharpe_improvement = (lstm_sharpe - bench_sharpe) / bench_sharpe * 100;
        
        std::cout << std::setw(25) << "Sharpe Ratio" 
                  << std::setw(15) << std::setprecision(2) << bench_sharpe
                  << std::setw(15) << lstm_sharpe
                  << std::setw(15) << sharpe_improvement << "%" << std::endl;
        
        // VaR (95% confidence)
        double bench_var = calculateVaR(benchmark_returns) * 100;
        double lstm_var = calculateVaR(lstm_returns) * 100;
        double var_improvement = (bench_var - lstm_var) / bench_var * 100;
        
        std::cout << std::setw(25) << "VaR 95% (%)" 
                  << std::setw(15) << std::setprecision(2) << bench_var
                  << std::setw(15) << lstm_var
                  << std::setw(15) << var_improvement << "%" << std::endl;
        
        // Volatility
        double bench_vol = calculateVolatility(benchmark_returns) * 100;
        double lstm_vol = calculateVolatility(lstm_returns) * 100;
        double vol_improvement = (bench_vol - lstm_vol) / bench_vol * 100;
        
        std::cout << std::setw(25) << "Volatility (%)" 
                  << std::setw(15) << std::setprecision(1) << bench_vol
                  << std::setw(15) << lstm_vol
                  << std::setw(15) << vol_improvement << "%" << std::endl;
        
        std::cout << std::string(70, '-') << std::endl;
        
        // Summary like paper results
        std::cout << "\n📊 Paper Target Results:" << std::endl;
        std::cout << "• Max Drawdown reduction: ~29.6% (Paper: 12.5% → 8.8%)" << std::endl;
        std::cout << "• Sharpe Ratio improvement: ~32.4% (Paper: 0.68 → 0.90)" << std::endl;
        std::cout << "• VaR improvement: ~20.0% (Paper: -2.35% → -1.88%)" << std::endl;
    }
    
    // Portfolio performance tracking for different market conditions (Table 2)
    static void analyzeMarketConditions(const std::vector<double>& returns,
                                      const std::vector<int>& market_regime,
                                      const std::string& model_name) {
        
        std::cout << "\n=== Market Condition Analysis (Paper Table 2) ===" << std::endl;
        
        // Separate returns by market regime (0=bear, 1=sideways, 2=bull)
        std::vector<std::vector<double>> regime_returns(3);
        
        for (size_t i = 0; i < returns.size() && i < market_regime.size(); i++) {
            if (market_regime[i] >= 0 && market_regime[i] < 3) {
                regime_returns[market_regime[i]].push_back(returns[i]);
            }
        }
        
        std::vector<std::string> regime_names = {"Bear Market", "Sideways Market", "Bull Market"};
        
        for (int regime = 0; regime < 3; regime++) {
            if (regime_returns[regime].empty()) continue;
            
            std::cout << "\n" << regime_names[regime] << " (" << model_name << "):" << std::endl;
            
            double avg_return = std::accumulate(regime_returns[regime].begin(), 
                                              regime_returns[regime].end(), 0.0) 
                               / regime_returns[regime].size() * 100;
            
            double mdd = calculateMaxDrawdown(regime_returns[regime]) * 100;
            double sharpe = calculateSharpeRatio(regime_returns[regime]);
            double var = calculateVaR(regime_returns[regime]) * 100;
            
            std::cout << "  Average Daily Return: " << std::fixed << std::setprecision(3) << avg_return << "%" << std::endl;
            std::cout << "  Max Drawdown: " << std::setprecision(2) << mdd << "%" << std::endl;
            std::cout << "  Sharpe Ratio: " << std::setprecision(2) << sharpe << std::endl;
            std::cout << "  VaR (95%): " << std::setprecision(2) << var << "%" << std::endl;
        }
    }
};
