#include "lstm.h"
#include <iostream>
// Constructor implementation
lstm::lstm(int n_neurons, int input_size)  // Add input_size parameter
    : n_neurons(n_neurons), input_size(input_size), gen(rd()), dis(0.0, 0.1)  // Fix initialization
{
    // Initialize forget gate parameters
    double weight_scale = std::sqrt(2.0 / (input_size + n_neurons));
    
    // Initialize forget gate parameters
    Uf = Eigen::MatrixXd::Random(n_neurons, input_size) * weight_scale;
    bf = Eigen::VectorXd::Zero(n_neurons);  // Start with zero bias
    Wf = Eigen::MatrixXd::Random(n_neurons, n_neurons) * weight_scale;
        
    // Initialize input gate parameters
    Ui = Eigen::MatrixXd::Random(n_neurons, input_size) * weight_scale;
    bi = Eigen::VectorXd::Zero(n_neurons);
    Wi = Eigen::MatrixXd::Random(n_neurons, n_neurons) * weight_scale;
        
    // Initialize output gate parameters
    Uo = Eigen::MatrixXd::Random(n_neurons, input_size) * weight_scale;
    bo = Eigen::VectorXd::Zero(n_neurons);
    Wo = Eigen::MatrixXd::Random(n_neurons, n_neurons) * weight_scale;
        
    // Initialize candidate values parameters
    Ug = Eigen::MatrixXd::Random(n_neurons, input_size) * weight_scale;
    bg = Eigen::VectorXd::Zero(n_neurons);
    Wg = Eigen::MatrixXd::Random(n_neurons, n_neurons) * weight_scale;
    
    // Initialize forget gate bias to 1 (helps with learning)
    bf = Eigen::VectorXd::Ones(n_neurons);
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

void lstm::zero_gradients()
{
    // Initialize gradients to zero
    dUf = Eigen::MatrixXd::Zero(n_neurons, input_size);
    dWf = Eigen::MatrixXd::Zero(n_neurons, n_neurons);
    dbf = Eigen::VectorXd::Zero(n_neurons);
    
    dUi = Eigen::MatrixXd::Zero(n_neurons, input_size);
    dWi = Eigen::MatrixXd::Zero(n_neurons, n_neurons);
    dbi = Eigen::VectorXd::Zero(n_neurons);
    
    dUo = Eigen::MatrixXd::Zero(n_neurons, input_size);
    dWo = Eigen::MatrixXd::Zero(n_neurons, n_neurons);
    dbo = Eigen::VectorXd::Zero(n_neurons);
    
    dUg = Eigen::MatrixXd::Zero(n_neurons, input_size);
    dWg = Eigen::MatrixXd::Zero(n_neurons, n_neurons);
    dbg = Eigen::VectorXd::Zero(n_neurons);
}

void lstm::updateParameters(double learning_rate)
{
    // Update forget gate parameters
    Uf -= learning_rate * dUf;
    Wf -= learning_rate * dWf;
    bf -= learning_rate * dbf;
    
    // Update input gate parameters
    Ui -= learning_rate * dUi;
    Wi -= learning_rate * dWi;
    bi -= learning_rate * dbi;
    
    // Update output gate parameters
    Uo -= learning_rate * dUo;
    Wo -= learning_rate * dWo;
    bo -= learning_rate * dbo;
    
    // Update candidate values parameters
    Ug -= learning_rate * dUg;
    Wg -= learning_rate * dWg;
    bg -= learning_rate * dbg;
}


void lstm::backward(const std::vector<Eigen::VectorXd>& sequence_of_inputs, 
                   const LSTMOutput& forward_output, 
                   const Eigen::VectorXd& dvalues_final)
{
    int T = sequence_of_inputs.size();
    
    // Extract forward pass results
    const auto& H = forward_output.H;
    const auto& C = forward_output.C;
    const auto& C_tilde = forward_output.C_tilde;
    const auto& F = forward_output.F;
    const auto& O = forward_output.O;
    const auto& I = forward_output.I;
    
    // Initialize gradients to zero
    zero_gradients();
    
        
    // Initialize gradient for hidden state
    Eigen::VectorXd dht = dvalues_final;
    Eigen::VectorXd dct = Eigen::VectorXd::Zero(n_neurons);
    
    // Backpropagate through time
    for(int t = T - 1; t >= 0; t--)
    {
        Eigen::VectorXd xt = sequence_of_inputs[t];
        
        // Get previous hidden and cell states
        Eigen::VectorXd ht_prev = (t > 0) ? H[t] : Eigen::VectorXd::Zero(n_neurons);
        Eigen::VectorXd ct_prev = (t > 0) ? C[t] : Eigen::VectorXd::Zero(n_neurons);
        
        // Current states
        Eigen::VectorXd ct_current = C[t+1];
        
        // Gradient of tanh(ct) w.r.t ct
        Eigen::VectorXd tanh_ct = tanh_activation(ct_current);
        Eigen::VectorXd dtanh_ct = tanh_derivative(tanh_ct);
        
        // Gradient from output gate multiplication: ht = tanh(ct) * ot
        Eigen::VectorXd dht_to_tanh_ct = dht.cwiseProduct(O[t]);
        Eigen::VectorXd dht_to_ot = dht.cwiseProduct(tanh_ct);
        
        // Add gradient from next timestep's cell state
        dct += dht_to_tanh_ct.cwiseProduct(dtanh_ct);
        
        // Output gate gradients
        Eigen::VectorXd dsigmo = dht_to_ot.cwiseProduct(sigmoid_derivative(O[t]));
        
        // Cell state gradients
        Eigen::VectorXd dct_to_ft = dct.cwiseProduct(ct_prev);
        Eigen::VectorXd dct_to_it = dct.cwiseProduct(C_tilde[t]);
        Eigen::VectorXd dct_to_ct_tilde = dct.cwiseProduct(I[t]);
        
        // Forget gate gradients
        Eigen::VectorXd dsigmf = dct_to_ft.cwiseProduct(sigmoid_derivative(F[t]));
        
        // Input gate gradients
        Eigen::VectorXd dsigmi = dct_to_it.cwiseProduct(sigmoid_derivative(I[t]));
        
        // Candidate values gradients
        Eigen::VectorXd dtanh1 = dct_to_ct_tilde.cwiseProduct(tanh_derivative(C_tilde[t]));
        
        // Accumulate parameter gradients
        
        // Forget gate parameter gradients
        dUf += dsigmf * xt.transpose();
        dWf += dsigmf * ht_prev.transpose();
        dbf += dsigmf;
        
        // Input gate parameter gradients
        dUi += dsigmi * xt.transpose();
        dWi += dsigmi * ht_prev.transpose();
        dbi += dsigmi;
        
        // Output gate parameter gradients
        dUo += dsigmo * xt.transpose();
        dWo += dsigmo * ht_prev.transpose();
        dbo += dsigmo;
        
        // Candidate values parameter gradients
        dUg += dtanh1 * xt.transpose();
        dWg += dtanh1 * ht_prev.transpose();
        dbg += dtanh1;
        
        // Compute gradients for previous timestep
        if(t > 0)
        {
            // Gradient w.r.t. previous hidden state
            dht = Wf.transpose() * dsigmf + 
                  Wi.transpose() * dsigmi + 
                  Wo.transpose() * dsigmo + 
                  Wg.transpose() * dtanh1;
            
            // Gradient w.r.t. previous cell state
            dct = dct.cwiseProduct(F[t]);
        }
    }
}

Eigen::VectorXd lstm::sigmoid_derivative(const Eigen::VectorXd& sigmoid_output)
{
    return sigmoid_output.cwiseProduct(Eigen::VectorXd::Ones(sigmoid_output.size()) - sigmoid_output);
}

Eigen::VectorXd lstm::tanh_derivative(const Eigen::VectorXd& tanh_output)
{
    return Eigen::VectorXd::Ones(tanh_output.size()) - tanh_output.cwiseProduct(tanh_output);
}
