#pragma once
#include <random>
#include <cmath>
#include "LSTMOutput.h"
class lstm
{
private:
    int n_neurons;
    int input_size;
       
    // Forget gate parameters
    Eigen::MatrixXd Uf, Wf;
    Eigen::VectorXd bf;
    
    // Input gate parameters
    Eigen::MatrixXd Ui, Wi;
    Eigen::VectorXd bi;
    
    // Output gate parameters
    Eigen::MatrixXd Uo, Wo;
    Eigen::VectorXd bo;
    
    // Candidate values parameters (c tilde)
    Eigen::MatrixXd Ug, Wg;
    Eigen::VectorXd bg;
   
    
    // Random torXd bgnumber generator
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;

    
    // Activation functions
    Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x)
    {
        return (1.0 / (1.0 + (-x.array()).exp()));
    }
    
    Eigen::MatrixXd tanh_activation(const Eigen::MatrixXd& x)
    {
        return x.array().tanh();
    }

public:
//    lstm(int n_neurons, std::uniform_real_distribution<double> dis);
    LSTMOutput forward(const std::vector<Eigen::VectorXd>& X_t);
    lstm(int n_neurons, int input_size);
    // Getters
    const Eigen::MatrixXd& get_Uf() const { return Uf; }
    const Eigen::VectorXd& get_bf() const { return bf; }
    const Eigen::MatrixXd& get_Wf() const { return Wf; }
    
    const Eigen::MatrixXd& get_Ui() const { return Ui; }
    const Eigen::VectorXd& get_bi() const { return bi; }
    const Eigen::MatrixXd& get_Wi() const { return Wi; }
    
    const Eigen::MatrixXd& get_Uo() const { return Uo; }
    const Eigen::VectorXd& get_bo() const { return bo; }
    const Eigen::MatrixXd& get_Wo() const { return Wo; }
    
    // c tilde getters
    const Eigen::MatrixXd& get_Ug() const { return Ug; }
    const Eigen::VectorXd& get_bg() const { return bg; }
    const Eigen::MatrixXd& get_Wg() const { return Wg; }
    
    // Print parameters for debugging - DECLARATION
    void print_parameters() const;
};
