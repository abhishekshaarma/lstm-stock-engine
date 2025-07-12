#include "lstm.h"
#include <iostream>
#include <vector>
#include <iomanip>

int main() {
    std::cout << "=== Simple LSTM Test ===" << std::endl;
    
    // Test with 1D inputs first (keep it simple)
    int n_neurons = 5;
    int input_size = 1;
    
    std::cout << "Creating LSTM: " << n_neurons << " neurons, " << input_size << "D inputs" << std::endl;
    
    // Create LSTM
    lstm network(n_neurons, input_size);
    
    // Create simple test sequence: [1], [2], [3]
    std::vector<Eigen::VectorXd> test_sequence;
    
    for(int i = 1; i <= 3; i++) {
        Eigen::VectorXd input(1);  // 1D input
        input(0) = i;
        test_sequence.push_back(input);
        std::cout << "Input " << i << ": [" << input(0) << "]" << std::endl;
    }
    
    // Run LSTM
    std::cout << "\nRunning LSTM forward pass..." << std::endl;
    LSTMOutput result = network.forward(test_sequence);
    
    // Get final hidden state
    Eigen::VectorXd final_hidden = result.H.back();
    std::cout << "Final hidden state: [";
    for(int i = 0; i < final_hidden.size(); i++) {
        std::cout << std::fixed << std::setprecision(3) << final_hidden(i);
        if(i < final_hidden.size()-1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Test with 3D inputs
    std::cout << "\n=== Testing 3D Inputs ===" << std::endl;
    
    int input_size_3d = 3;
    lstm network_3d(n_neurons, input_size_3d);
    
    std::vector<Eigen::VectorXd> test_sequence_3d;
    
    for(int i = 1; i <= 2; i++) {
        Eigen::VectorXd input(3);  // 3D input
        input(0) = i;           // price
        input(1) = i * 10;      // volume  
        input(2) = i * 0.5;     // some other feature
        test_sequence_3d.push_back(input);
        
        std::cout << "Input " << i << ": [" << input(0) << ", " 
                  << input(1) << ", " << input(2) << "]" << std::endl;
    }
    
    LSTMOutput result_3d = network_3d.forward(test_sequence_3d);
    Eigen::VectorXd final_hidden_3d = result_3d.H.back();
    
    std::cout << "Final hidden state (3D): [";
    for(int i = 0; i < final_hidden_3d.size(); i++) {
        std::cout << std::fixed << std::setprecision(3) << final_hidden_3d(i);
        if(i < final_hidden_3d.size()-1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Test dense layer prediction
    std::cout << "\n=== Testing Dense Layer ===" << std::endl;
    
    // Simple dense layer for stock price prediction
    Eigen::MatrixXd W = Eigen::MatrixXd::Random(1, n_neurons) * 0.1;  // 5 -> 1
    Eigen::VectorXd b = Eigen::VectorXd::Random(1) * 0.1;
    
    // Make prediction
    Eigen::VectorXd prediction = W * final_hidden + b;
    std::cout << "Stock price prediction: $" << std::fixed << std::setprecision(2) 
              << prediction(0) << std::endl;
    
    // Test with 3D input prediction
    Eigen::VectorXd prediction_3d = W * final_hidden_3d + b;
    std::cout << "Stock price prediction (3D input): $" << std::fixed << std::setprecision(2) 
              << prediction_3d(0) << std::endl;
    
    std::cout << "\n=== Test Complete! ===" << std::endl;
    
    return 0;
}
