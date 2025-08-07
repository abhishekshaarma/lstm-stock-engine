#pragma once
#include <random>
#include <cmath>
#include <vector>
#include <mutex>
#include <omp.h>
#include "LSTMOutput.h"

class lstm_parallel
{
private:
    int n_neurons;
    int input_size;
    int num_threads;
       
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
   
    Eigen::MatrixXd dUf, dWf, dUi, dWi, dUo, dWo, dUg, dWg;
    Eigen::VectorXd dbf, dbi, dbo, dbg;
    
    // Thread-local storage for parallel operations
    std::vector<Eigen::MatrixXd> thread_dUf, thread_dWf, thread_dUi, thread_dWi;
    std::vector<Eigen::MatrixXd> thread_dUo, thread_dWo, thread_dUg, thread_dWg;
    std::vector<Eigen::VectorXd> thread_dbf, thread_dbi, thread_dbo, thread_dbg;
    
    // Random number generator
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;
    
    // Thread-safe random number generator
    std::vector<std::mt19937> thread_gens;
    std::mutex gen_mutex;

    // Activation functions
    Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x)
    {
        return (1.0 / (1.0 + (-x.array()).exp()));
    }
    
    Eigen::MatrixXd tanh_activation(const Eigen::MatrixXd& x)
    {
        return x.array().tanh();
    }

    Eigen::VectorXd sigmoid_derivative(const Eigen::VectorXd& sigmoid_output);
    Eigen::VectorXd tanh_derivative(const Eigen::VectorXd& tanh_output);
    
    // Initialize thread-local storage
    void initializeThreadStorage();

public:
    lstm_parallel(int n_neurons, int input_size, int num_threads = 0);
    
    // Parallel forward pass for multiple sequences
    std::vector<LSTMOutput> parallelForward(const std::vector<std::vector<Eigen::VectorXd>>& sequences);
    
    // Parallel backward pass for multiple sequences
    void parallelBackward(const std::vector<std::vector<Eigen::VectorXd>>& sequences,
                         const std::vector<LSTMOutput>& forward_outputs,
                         const std::vector<Eigen::VectorXd>& dvalues_final);
    
    // Single sequence operations (for compatibility)
    LSTMOutput forward(const std::vector<Eigen::VectorXd>& X_t);
    void backward(const std::vector<Eigen::VectorXd>& sequence_of_inputs, 
                  const LSTMOutput& forward_output, 
                  const Eigen::VectorXd& dvalues_final);
    
    void zero_gradients();
    void updateParameters(double learning_rate);
    
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
    const Eigen::MatrixXd& get_Ug() const { return Ug; }
    const Eigen::VectorXd& get_bg() const { return bg; }
    const Eigen::MatrixXd& get_Wg() const { return Wg; }

    // Print parameters for debugging
    void print_parameters() const;
    
    // Get number of threads
    int getNumThreads() const { return num_threads; }
};
