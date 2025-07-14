#pragma once
#include <Eigen/Dense>
#include <vector>
#include <iostream>

class MultiFactorModel {
private:
    Eigen::MatrixXd factor_betas;  // Factor loadings
    Eigen::VectorXd intercept;     // Alpha
    int n_factors;
    int n_assets;

public:
    MultiFactorModel(int n_factors, int n_assets) 
        : n_factors(n_factors), n_assets(n_assets) {
        factor_betas = Eigen::MatrixXd::Random(n_assets, n_factors) * 0.1;
        intercept = Eigen::VectorXd::Random(n_assets) * 0.01;
    }
    
    // Fit linear regression: R = alpha + beta1*F1 + beta2*F2 + ... + error
    void fit(const std::vector<Eigen::VectorXd>& returns,
             const std::vector<Eigen::VectorXd>& factors) {
        
        int T = returns.size();
        
        // Construct design matrix X = [1, F1, F2, ..., Fn]
        Eigen::MatrixXd X(T, n_factors + 1);
        Eigen::VectorXd y(T);
        
        for (int t = 0; t < T; t++) {
            X(t, 0) = 1.0;  // Intercept term
            for (int f = 0; f < n_factors; f++) {
                X(t, f + 1) = factors[t](f);
            }
            y(t) = returns[t](0);  // Assuming single asset for now
        }
        
        // Solve normal equations: (X'X)^-1 X'y
        Eigen::VectorXd coeffs = (X.transpose() * X).ldlt().solve(X.transpose() * y);
        
        intercept(0) = coeffs(0);
        for (int f = 0; f < n_factors; f++) {
            factor_betas(0, f) = coeffs(f + 1);
        }
        
        std::cout << "Multi-factor model fitted:" << std::endl;
        std::cout << "Alpha: " << intercept(0) << std::endl;
        std::cout << "Betas: ";
        for (int f = 0; f < n_factors; f++) {
            std::cout << factor_betas(0, f) << " ";
        }
        std::cout << std::endl;
    }
    
    // Predict returns using factor model
    double predict(const Eigen::VectorXd& factors) {
        double prediction = intercept(0);
        for (int f = 0; f < n_factors; f++) {
            prediction += factor_betas(0, f) * factors(f);
        }
        return prediction;
    }
    
    // Calculate R-squared
    double calculateRSquared(const std::vector<Eigen::VectorXd>& returns,
                           const std::vector<Eigen::VectorXd>& factors) {
        double ss_res = 0.0, ss_tot = 0.0;
        
        // Calculate mean return
        double mean_return = 0.0;
        for (const auto& ret : returns) {
            mean_return += ret(0);
        }
        mean_return /= returns.size();
        
        // Calculate explained variance
        for (size_t t = 0; t < returns.size(); t++) {
            double predicted = predict(factors[t]);
            double actual = returns[t](0);
            
            ss_res += (actual - predicted) * (actual - predicted);
            ss_tot += (actual - mean_return) * (actual - mean_return);
        }
        
        return 1.0 - (ss_res / ss_tot);
    }
};
