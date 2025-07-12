#pragma once
#include <vector>
#include <Eigen/Dense>

struct LSTMOutput
{
    std::vector<Eigen::VectorXd> H;
    std::vector<Eigen::VectorXd> C;
    std::vector<Eigen::VectorXd> C_tilde;
    std::vector<Eigen::VectorXd> F;
    std::vector<Eigen::VectorXd> O;
    std::vector<Eigen::VectorXd> I;
    // Activation intermediate values (needed for backprop)
    std::vector<Eigen::VectorXd> forget_gate_inputs;     // outf values
    std::vector<Eigen::VectorXd> input_gate_inputs;      // outi values  
    std::vector<Eigen::VectorXd> output_gate_inputs;     // outo values
    std::vector<Eigen::VectorXd> candidate_inputs;       // outct_tilde values
    std::vector<Eigen::VectorXd> cell_state_tanh_inputs; // ct values (input to final tanh)
};
