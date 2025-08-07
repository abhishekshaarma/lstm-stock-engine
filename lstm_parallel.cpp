#include "lstm_parallel.h"
#include <iostream>
#include <algorithm>

// Constructor implementation
lstm_parallel::lstm_parallel(int n_neurons, int input_size, int num_threads)
    : n_neurons(n_neurons), input_size(input_size), num_threads(num_threads), gen(rd()), dis(0.0, 0.1)
{
    if (num_threads <= 0) {
        this->num_threads = std::thread::hardware_concurrency();
    }
    
    double init_scale = 0.1;
    
    // Initialize forget gate parameters
    Uf = Eigen::MatrixXd::Random(n_neurons, input_size) * init_scale;
    bf = Eigen::VectorXd::Random(n_neurons) * init_scale;  
    Wf = Eigen::MatrixXd::Random(n_neurons, n_neurons) * init_scale;
        
    // Initialize input gate parameters
    Ui = Eigen::MatrixXd::Random(n_neurons, input_size) * init_scale;
    bi = Eigen::VectorXd::Random(n_neurons) * init_scale;  
    Wi = Eigen::MatrixXd::Random(n_neurons, n_neurons) * init_scale;
        
    // Initialize output gate parameters
    Uo = Eigen::MatrixXd::Random(n_neurons, input_size) * init_scale;
    bo = Eigen::VectorXd::Random(n_neurons) * init_scale;  
    Wo = Eigen::MatrixXd::Random(n_neurons, n_neurons) * init_scale;
        
    // Initialize candidate values parameters
    Ug = Eigen::MatrixXd::Random(n_neurons, input_size) * init_scale;
    bg = Eigen::VectorXd::Random(n_neurons) * init_scale;  
    Wg = Eigen::MatrixXd::Random(n_neurons, n_neurons) * init_scale;
    
    // Initialize thread-local storage
    initializeThreadStorage();
    
    // Initialize thread-local random generators
    thread_gens.resize(this->num_threads);
    for (int i = 0; i < this->num_threads; ++i) {
        thread_gens[i] = std::mt19937(rd());
    }
}

void lstm_parallel::initializeThreadStorage() {
    // Initialize thread-local gradient storage
    thread_dUf.resize(num_threads, Eigen::MatrixXd::Zero(n_neurons, input_size));
    thread_dWf.resize(num_threads, Eigen::MatrixXd::Zero(n_neurons, n_neurons));
    thread_dbf.resize(num_threads, Eigen::VectorXd::Zero(n_neurons));
    
    thread_dUi.resize(num_threads, Eigen::MatrixXd::Zero(n_neurons, input_size));
    thread_dWi.resize(num_threads, Eigen::MatrixXd::Zero(n_neurons, n_neurons));
    thread_dbi.resize(num_threads, Eigen::VectorXd::Zero(n_neurons));
    
    thread_dUo.resize(num_threads, Eigen::MatrixXd::Zero(n_neurons, input_size));
    thread_dWo.resize(num_threads, Eigen::MatrixXd::Zero(n_neurons, n_neurons));
    thread_dbo.resize(num_threads, Eigen::VectorXd::Zero(n_neurons));
    
    thread_dUg.resize(num_threads, Eigen::MatrixXd::Zero(n_neurons, input_size));
    thread_dWg.resize(num_threads, Eigen::MatrixXd::Zero(n_neurons, n_neurons));
    thread_dbg.resize(num_threads, Eigen::VectorXd::Zero(n_neurons));
}

void lstm_parallel::print_parameters() const
{
    std::cout << "Parallel LSTM Parameters:" << std::endl;
    std::cout << "Number of threads: " << num_threads << std::endl;
    std::cout << "Forget gate - Uf shape: " << get_Uf().rows() << "x" << get_Uf().cols() << std::endl;
    std::cout << "Forget gate - Wf shape: " << get_Wf().rows() << "x" << get_Wf().cols() << std::endl;
    std::cout << "Input gate - Ui shape: " << get_Ui().rows() << "x" << get_Ui().cols() << std::endl;
    std::cout << "Input gate - Wi shape: " << get_Wi().rows() << "x" << get_Wi().cols() << std::endl;
    std::cout << "Output gate - Uo shape: " << get_Uo().rows() << "x" << get_Uo().cols() << std::endl;
    std::cout << "Output gate - Wo shape: " << get_Wo().rows() << "x" << get_Wo().cols() << std::endl;
    std::cout << "Candidate - Ug shape: " << get_Ug().rows() << "x" << get_Ug().cols() << std::endl;
    std::cout << "Candidate - Wg shape: " << get_Wg().rows() << "x" << get_Wg().cols() << std::endl;
}

// Parallel forward pass for multiple sequences
std::vector<LSTMOutput> lstm_parallel::parallelForward(const std::vector<std::vector<Eigen::VectorXd>>& sequences) {
    std::vector<LSTMOutput> outputs(sequences.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < sequences.size(); ++i) {
        outputs[i] = forward(sequences[i]);
    }
    
    return outputs;
}

// Parallel backward pass for multiple sequences
void lstm_parallel::parallelBackward(const std::vector<std::vector<Eigen::VectorXd>>& sequences,
                                   const std::vector<LSTMOutput>& forward_outputs,
                                   const std::vector<Eigen::VectorXd>& dvalues_final) {
    
    // Zero all thread-local gradients
    #pragma omp parallel for
    for (int t = 0; t < num_threads; ++t) {
        thread_dUf[t].setZero();
        thread_dWf[t].setZero();
        thread_dbf[t].setZero();
        thread_dUi[t].setZero();
        thread_dWi[t].setZero();
        thread_dbi[t].setZero();
        thread_dUo[t].setZero();
        thread_dWo[t].setZero();
        thread_dbo[t].setZero();
        thread_dUg[t].setZero();
        thread_dWg[t].setZero();
        thread_dbg[t].setZero();
    }
    
    // Process sequences in parallel
    #pragma omp parallel for
    for (size_t i = 0; i < sequences.size(); ++i) {
        int thread_id = omp_get_thread_num();
        
        // Create temporary LSTM for this thread
        lstm_parallel temp_lstm(n_neurons, input_size, 1);
        temp_lstm.Uf = Uf; temp_lstm.Wf = Wf; temp_lstm.bf = bf;
        temp_lstm.Ui = Ui; temp_lstm.Wi = Wi; temp_lstm.bi = bi;
        temp_lstm.Uo = Uo; temp_lstm.Wo = Wo; temp_lstm.bo = bo;
        temp_lstm.Ug = Ug; temp_lstm.Wg = Wg; temp_lstm.bg = bg;
        
        // Perform backward pass
        temp_lstm.backward(sequences[i], forward_outputs[i], dvalues_final[i]);
        
        // Accumulate gradients to thread-local storage
        thread_dUf[thread_id] += temp_lstm.dUf;
        thread_dWf[thread_id] += temp_lstm.dWf;
        thread_dbf[thread_id] += temp_lstm.dbf;
        thread_dUi[thread_id] += temp_lstm.dUi;
        thread_dWi[thread_id] += temp_lstm.dWi;
        thread_dbi[thread_id] += temp_lstm.dbi;
        thread_dUo[thread_id] += temp_lstm.dUo;
        thread_dWo[thread_id] += temp_lstm.dWo;
        thread_dbo[thread_id] += temp_lstm.dbo;
        thread_dUg[thread_id] += temp_lstm.dUg;
        thread_dWg[thread_id] += temp_lstm.dWg;
        thread_dbg[thread_id] += temp_lstm.dbg;
    }
    
    // Aggregate gradients from all threads
    dUf.setZero(); dWf.setZero(); dbf.setZero();
    dUi.setZero(); dWi.setZero(); dbi.setZero();
    dUo.setZero(); dWo.setZero(); dbo.setZero();
    dUg.setZero(); dWg.setZero(); dbg.setZero();
    
    for (int t = 0; t < num_threads; ++t) {
        dUf += thread_dUf[t];
        dWf += thread_dWf[t];
        dbf += thread_dbf[t];
        dUi += thread_dUi[t];
        dWi += thread_dWi[t];
        dbi += thread_dbi[t];
        dUo += thread_dUo[t];
        dWo += thread_dWo[t];
        dbo += thread_dbo[t];
        dUg += thread_dUg[t];
        dWg += thread_dWg[t];
        dbg += thread_dbg[t];
    }
}

// Single sequence forward pass (same as original LSTM)
LSTMOutput lstm_parallel::forward(const std::vector<Eigen::VectorXd>& sequence_of_inputs)
{
    int T = sequence_of_inputs.size();
    
    // Initialize storage vectors
    std::vector<Eigen::VectorXd> H(T+1, Eigen::VectorXd::Zero(n_neurons));
    std::vector<Eigen::VectorXd> C(T+1, Eigen::VectorXd::Zero(n_neurons));
    std::vector<Eigen::VectorXd> C_tilde(T, Eigen::VectorXd::Zero(n_neurons));
    std::vector<Eigen::VectorXd> F(T, Eigen::VectorXd::Zero(n_neurons));
    std::vector<Eigen::VectorXd> O(T, Eigen::VectorXd::Zero(n_neurons));
    std::vector<Eigen::VectorXd> I(T, Eigen::VectorXd::Zero(n_neurons));
    
    // Initialize activation intermediate storage
    std::vector<Eigen::VectorXd> forget_gate_inputs(T, Eigen::VectorXd::Zero(n_neurons));
    std::vector<Eigen::VectorXd> input_gate_inputs(T, Eigen::VectorXd::Zero(n_neurons));
    std::vector<Eigen::VectorXd> output_gate_inputs(T, Eigen::VectorXd::Zero(n_neurons));
    std::vector<Eigen::VectorXd> candidate_inputs(T, Eigen::VectorXd::Zero(n_neurons));
    std::vector<Eigen::VectorXd> cell_state_tanh_inputs(T, Eigen::VectorXd::Zero(n_neurons));
    
    // Initialize gradients
    double init_scale = 0.1;
    dUf = Eigen::MatrixXd::Random(n_neurons, input_size) * init_scale;
    dWf = Eigen::MatrixXd::Random(n_neurons, n_neurons) * init_scale;
    dbf = Eigen::VectorXd::Random(n_neurons) * init_scale;
    
    dUi = Eigen::MatrixXd::Random(n_neurons, input_size) * init_scale;
    dWi = Eigen::MatrixXd::Random(n_neurons, n_neurons) * init_scale;
    dbi = Eigen::VectorXd::Random(n_neurons) * init_scale;
    
    dUo = Eigen::MatrixXd::Random(n_neurons, input_size) * init_scale;
    dWo = Eigen::MatrixXd::Random(n_neurons, n_neurons) * init_scale;
    dbo = Eigen::VectorXd::Random(n_neurons) * init_scale;
    
    dUg = Eigen::MatrixXd::Random(n_neurons, input_size) * init_scale;
    dWg = Eigen::MatrixXd::Random(n_neurons, n_neurons) * init_scale;
    dbg = Eigen::VectorXd::Random(n_neurons) * init_scale;
    
    // Initialize starting states
    Eigen::VectorXd ht = H[0];  
    Eigen::VectorXd ct = C[0];  
    
    for(int t = 0; t < T; t++)
    {
        Eigen::VectorXd xt = sequence_of_inputs[t];
        
        // Forget gate - store intermediate
        Eigen::VectorXd outf = Uf * xt + Wf * ht + bf;
        forget_gate_inputs[t] = outf;
        Eigen::VectorXd ft = sigmoid(outf);
        
        // Input gate - store intermediate
        Eigen::VectorXd outi = Ui * xt + Wi * ht + bi;
        input_gate_inputs[t] = outi;
        Eigen::VectorXd it = sigmoid(outi);
        
        // Output gate - store intermediate
        Eigen::VectorXd outo = Uo * xt + Wo * ht + bo;
        output_gate_inputs[t] = outo;
        Eigen::VectorXd ot = sigmoid(outo);
        
        // Candidate values - store intermediate
        Eigen::VectorXd outg = Ug * xt + Wg * ht + bg;
        candidate_inputs[t] = outg;
        Eigen::VectorXd c_tilde_t = tanh_activation(outg);
        
        // Cell state update
        Eigen::VectorXd ct_new = ft.cwiseProduct(ct) + it.cwiseProduct(c_tilde_t);
        cell_state_tanh_inputs[t] = ct_new;
        Eigen::VectorXd ct_tanh = tanh_activation(ct_new);
        
        // Hidden state update
        Eigen::VectorXd ht_new = ot.cwiseProduct(ct_tanh);
        
        // Store values
        F[t] = ft;
        I[t] = it;
        O[t] = ot;
        C_tilde[t] = c_tilde_t;
        C[t+1] = ct_new;
        H[t+1] = ht_new;
        
        // Update for next iteration
        ht = ht_new;
        ct = ct_new;
    }
    
    return LSTMOutput{H, C, C_tilde, F, O, I, forget_gate_inputs, input_gate_inputs, 
                     output_gate_inputs, candidate_inputs, cell_state_tanh_inputs};
}

// Single sequence backward pass (same as original LSTM)
void lstm_parallel::backward(const std::vector<Eigen::VectorXd>& sequence_of_inputs, 
                           const LSTMOutput& forward_output, 
                           const Eigen::VectorXd& dvalues_final)
{
    // Implementation would be the same as the original LSTM backward pass
    // This is a placeholder - you would copy the backward implementation from lstm.cpp
}

void lstm_parallel::zero_gradients()
{
    dUf.setZero(); dWf.setZero(); dbf.setZero();
    dUi.setZero(); dWi.setZero(); dbi.setZero();
    dUo.setZero(); dWo.setZero(); dbo.setZero();
    dUg.setZero(); dWg.setZero(); dbg.setZero();
}

void lstm_parallel::updateParameters(double learning_rate)
{
    Uf -= learning_rate * dUf;
    Wf -= learning_rate * dWf;
    bf -= learning_rate * dbf;
    
    Ui -= learning_rate * dUi;
    Wi -= learning_rate * dWi;
    bi -= learning_rate * dbi;
    
    Uo -= learning_rate * dUo;
    Wo -= learning_rate * dWo;
    bo -= learning_rate * dbo;
    
    Ug -= learning_rate * dUg;
    Wg -= learning_rate * dWg;
    bg -= learning_rate * dbg;
}

Eigen::VectorXd lstm_parallel::sigmoid_derivative(const Eigen::VectorXd& sigmoid_output)
{
    return sigmoid_output.cwiseProduct(Eigen::VectorXd::Ones(sigmoid_output.size()) - sigmoid_output);
}

Eigen::VectorXd lstm_parallel::tanh_derivative(const Eigen::VectorXd& tanh_output)
{
    return Eigen::VectorXd::Ones(tanh_output.size()) - tanh_output.cwiseProduct(tanh_output);
}
