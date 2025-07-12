#include "lstm.h"
#include "activations.h"
#include <iostream>
// Constructor implementation
lstm::lstm(int n_neurons, int input_size)  // Add input_size parameter
    : n_neurons(n_neurons), input_size(input_size), gen(rd()), dis(0.0, 0.1)  // Fix initialization
{
    // Initialize forget gate parameters
    Uf = Eigen::MatrixXd::Random(n_neurons, input_size) * 0.1;  // Use input_size
    bf = Eigen::VectorXd::Random(n_neurons) * 0.1;              // Remove the ,1
    Wf = Eigen::MatrixXd::Random(n_neurons, n_neurons) * 0.1;
        
    // Initialize input gate parameters
    Ui = Eigen::MatrixXd::Random(n_neurons, input_size) * 0.1;  // Use input_size
    bi = Eigen::VectorXd::Random(n_neurons) * 0.1;              // Remove the ,1
    Wi = Eigen::MatrixXd::Random(n_neurons, n_neurons) * 0.1;
        
    // Initialize output gate parameters
    Uo = Eigen::MatrixXd::Random(n_neurons, input_size) * 0.1;  // Use input_size
    bo = Eigen::VectorXd::Random(n_neurons) * 0.1;              // Remove the ,1
    Wo = Eigen::MatrixXd::Random(n_neurons, n_neurons) * 0.1;
        
    // Initialize candidate values parameters (c tilde)
    Ug = Eigen::MatrixXd::Random(n_neurons, input_size) * 0.1;  // Use input_size
    bg = Eigen::VectorXd::Random(n_neurons) * 0.1;              // Remove the ,1
    Wg = Eigen::MatrixXd::Random(n_neurons, n_neurons) * 0.1;
}
void lstm::print_parameters() const
{
    std::cout << "LSTM Parameters:" << std::endl;
    std::cout << "Forget gate - Uf shape: " << get_Uf().rows() << "x" << get_Uf().cols() << std::endl;
    std::cout << "Forget gate - Wf shape: " << get_Wf().rows() << "x" << get_Wf().cols() << std::endl;
    std::cout << "Input gate - Ui shape: " << get_Ui().rows() << "x" << get_Ui().cols() << std::endl;
    std::cout << "Input gate - Wi shape: " << get_Wi().rows() << "x" << get_Wi().cols() << std::endl;
    std::cout << "Output gate - Uo shape: " << get_Uo().rows() << "x" << get_Uo().cols() << std::endl;
    std::cout << "Output gate - Wo shape: " << get_Wo().rows() << "x" << get_Wo().cols() << std::endl;
    std::cout << "Candidate - Ug shape: " << get_Ug().rows() << "x" << get_Ug().cols() << std::endl;
    std::cout << "Candidate - Wg shape: " << get_Wg().rows() << "x" << get_Wg().cols() << std::endl;

}

LSTMOutput lstm::forward(const std::vector<Eigen::VectorXd>& sequence_of_inputs)
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
    
    // Initialize starting states
    Eigen::VectorXd ht = H[0];  
    Eigen::VectorXd ct = C[0];  
    
    for(int t = 0; t < T; t++)
    {
        Eigen::VectorXd xt = sequence_of_inputs[t];
        
        // Forget gate - store intermediate
        Eigen::VectorXd outf = Uf * xt + Wf * ht + bf;
        forget_gate_inputs[t] = outf;  // Store for backprop
        Eigen::VectorXd ft = sigmoid(outf);
        
        // Input gate - store intermediate
        Eigen::VectorXd outi = Ui * xt + Wi * ht + bi;
        input_gate_inputs[t] = outi;  // Store for backprop
        Eigen::VectorXd it = sigmoid(outi);
        
        // Output gate - store intermediate
        Eigen::VectorXd outo = Uo * xt + Wo * ht + bo;
        output_gate_inputs[t] = outo;  // Store for backprop
        Eigen::VectorXd ot = sigmoid(outo);
        
        // Candidate values - store intermediate
        Eigen::VectorXd outct_tilde = Ug * xt + Wg * ht + bg;
        candidate_inputs[t] = outct_tilde;  // Store for backprop
        Eigen::VectorXd ct_tilde = tanh_activation(outct_tilde);
        
        // Update cell state
        ct = ft.cwiseProduct(ct) + it.cwiseProduct(ct_tilde);
        cell_state_tanh_inputs[t] = ct;  // Store for backprop
        
        // Update hidden state  
        Eigen::VectorXd tanh_ct = tanh_activation(ct);
        ht = tanh_ct.cwiseProduct(ot);
        
        // Store results
        H[t+1] = ht;
        C[t+1] = ct;
        C_tilde[t] = ct_tilde;
        F[t] = ft;
        O[t] = ot;
        I[t] = it;
    }
    
    return {H, C, C_tilde, F, O, I, 
        forget_gate_inputs, input_gate_inputs, output_gate_inputs, 
        candidate_inputs, cell_state_tanh_inputs};
}

