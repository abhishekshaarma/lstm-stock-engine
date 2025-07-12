#include "lstm.h"
#include "StockCSVParser.h"  // Include the CSV parser
#include <iostream>
#include <iomanip>

int main()
{
    std::cout << "=== Stock LSTM with Real Data ===" << std::endl;
    
    // Load stock data from CSV
    StockCSVParser parser;
    if (!parser.loadFromFile("./data/GOOG.csv")) {  // Change to your CSV filename
        std::cerr << "Failed to load CSV file" << std::endl;
        return 1;
    }
    
    // Print sample data
    parser.printSample(3);
    
    // Configuration
    int sequence_length = 5;    // Look at 5 days to predict next day
    int feature_count = 4;      // OHLC (Open, High, Low, Close)
    int n_neurons = 10;         // LSTM neurons
    bool normalize = true;      // Normalize data to 0-1 range
    
    std::cout << "\n=== LSTM Configuration ===" << std::endl;
    std::cout << "Sequence length: " << sequence_length << " days" << std::endl;
    std::cout << "Features: " << feature_count << " (OHLC)" << std::endl;
    std::cout << "LSTM neurons: " << n_neurons << std::endl;
    std::cout << "Data normalization: " << (normalize ? "ON" : "OFF") << std::endl;
    
    // Create sequences for LSTM
    auto sequences = parser.createSequences(sequence_length, feature_count, normalize);
    auto targets = parser.getTargets(sequence_length, normalize);
    
    if (sequences.empty()) {
        std::cerr << "No sequences created!" << std::endl;
        return 1;
    }
    
    // Get price range for denormalization
    auto price_range = parser.getPriceRange();  // You'll need to add this method to StockCSVParser
    double min_price = price_range.first;
    double max_price = price_range.second;
    
    // Create LSTM network
    lstm network(n_neurons, feature_count);
    
    // Create dense layer for prediction (LSTM output -> stock price)
    Eigen::MatrixXd dense_W = Eigen::MatrixXd::Random(1, n_neurons) * 0.1;
    Eigen::VectorXd dense_b = Eigen::VectorXd::Random(1) * 0.1;
    
    std::cout << "\n=== Testing LSTM with Real Stock Data ===" << std::endl;
    
    // Test with first few sequences
    int test_count = std::min(5, (int)sequences.size());
    
    for (int i = 0; i < test_count; i++) {
        std::cout << "\n--- Sequence " << (i+1) << " ---" << std::endl;
        
        // Show input sequence
        std::cout << "Input sequence (last 3 days shown):" << std::endl;
        for (int day = std::max(0, sequence_length-3); day < sequence_length; day++) {
            const auto& input = sequences[i][day];
            std::cout << "  Day " << (day+1) << ": [";
            for (int f = 0; f < feature_count; f++) {
                std::cout << std::fixed << std::setprecision(3) << input(f);
                if (f < feature_count-1) std::cout << ", ";
            }
            std::cout << "]";
            if (!normalize) {
                std::cout << " (O:" << input(0) << " H:" << input(1) 
                          << " L:" << input(2) << " C:" << input(3) << ")";
            }
            std::cout << std::endl;
        }
        
        // Run LSTM
        LSTMOutput result = network.forward(sequences[i]);
        Eigen::VectorXd hidden = result.H.back();
        
        // Make prediction with dense layer
        Eigen::VectorXd prediction_vec = dense_W * hidden + dense_b;
        double prediction = prediction_vec(0);
        
        std::cout << "LSTM hidden state: [";
        for (int j = 0; j < std::min(3, (int)hidden.size()); j++) {
            std::cout << std::fixed << std::setprecision(3) << hidden(j);
            if (j < std::min(3, (int)hidden.size())-1) std::cout << ", ";
        }
        std::cout << "...]" << std::endl;
        
        std::cout << "Predicted next close: " << std::fixed << std::setprecision(3) << prediction;
        if (i < targets.size()) {
            std::cout << ", Actual: " << std::fixed << std::setprecision(3) << targets[i];
            double error = std::abs(prediction - targets[i]);
            std::cout << ", Error: " << std::fixed << std::setprecision(3) << error;
        }
        std::cout << std::endl;
        
        // Convert normalized values back to real prices
        if (normalize) {
            double real_predicted_price = prediction * (max_price - min_price) + min_price;
            std::cout << "Real predicted price: $" << std::fixed << std::setprecision(2) << real_predicted_price;
            
            if (i < targets.size()) {
                double real_actual_price = targets[i] * (max_price - min_price) + min_price;
                std::cout << ", Real actual price: $" << std::fixed << std::setprecision(2) << real_actual_price;
                double real_error = std::abs(real_predicted_price - real_actual_price);
                std::cout << ", Real error: $" << std::fixed << std::setprecision(2) << real_error;
            }
            std::cout << std::endl;
        }
    }
    
    // Test different feature configurations
    std::cout << "\n=== Testing Different Feature Sets ===" << std::endl;
    
    // Test with just closing price (1 feature)
    auto simple_sequences = parser.createSequences(sequence_length, 1, normalize);
    if (!simple_sequences.empty()) {
        lstm simple_network(n_neurons, 1);
        LSTMOutput simple_result = simple_network.forward(simple_sequences[0]);
        
        std::cout << "Single feature (close price only) LSTM output: [";
        auto simple_hidden = simple_result.H.back();
        for (int i = 0; i < std::min(3, (int)simple_hidden.size()); i++) {
            std::cout << std::fixed << std::setprecision(3) << simple_hidden(i);
            if (i < std::min(3, (int)simple_hidden.size())-1) std::cout << ", ";
        }
        std::cout << "...]" << std::endl;
    }
    
    // Test with OHLC + Volume (5 features)
    auto rich_sequences = parser.createSequences(sequence_length, 5, normalize);
    if (!rich_sequences.empty()) {
        lstm rich_network(n_neurons, 5);
        LSTMOutput rich_result = rich_network.forward(rich_sequences[0]);
        
        std::cout << "Rich features (OHLC+Volume) LSTM output: [";
        auto rich_hidden = rich_result.H.back();
        for (int i = 0; i < std::min(3, (int)rich_hidden.size()); i++) {
            std::cout << std::fixed << std::setprecision(3) << rich_hidden(i);
            if (i < std::min(3, (int)rich_hidden.size())-1) std::cout << ", ";
        }
        std::cout << "...]" << std::endl;
    }
    
    std::cout << "\n=== Real Stock Data Test Complete! ===" << std::endl;
    std::cout << "Next steps: Implement training to improve predictions!" << std::endl;
    
    return 0;
}
