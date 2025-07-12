#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <iomanip>
struct StockData
{
    std::string symbol;
    std::string date;
    double close;
    double high;
    double low;
    double open;
    long volume;
    double adjClose;
    double adjHigh;
    double adjLow;
    double adjOpen;
    long adjVolume;
    double divCash;
    double splitFactor;
};

class StockCSVParser
{
  private:
    std::vector<StockData> data;
    std::map<std::string, int> header_map;
    
    std::vector<std::string> split(const std::string& str, char delimiter) {
        std::vector<std::string> tokens;
        std::stringstream ss(str);
        std::string token;
        
        while (std::getline(ss, token, delimiter))
        {
            tokens.push_back(token);
        }
        return tokens;
    }
    
    double parseDouble(const std::string& str)
    {
        try
        {
            return std::stod(str);
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error parsing double: " << str << std::endl;
            return 0.0;
        }
    }
    
    long parseLong(const std::string& str)
    {
        try
        {
            return std::stol(str);
        } catch (const std::exception& e)
          {
              std::cerr << "Error parsing long: " << str << std::endl;
              return 0;
          }
    }

  public:
   std::pair<double, double> getPriceRange() const {
    if (data.empty()) return {0.0, 0.0};
    
    double min_price = data[0].close;
    double max_price = data[0].close;
    
    for (const auto& row : data) {
        min_price = std::min(min_price, row.close);
        max_price = std::max(max_price, row.close);
    }
    
    return {min_price, max_price};
}
    bool loadFromFile(const std::string& filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return false;
        }
        
        std::string line;
        bool first_line = true;
        
        while (std::getline(file, line))
        {
            if (line.empty()) continue;
            
            std::vector<std::string> tokens = split(line, ',');
            
            if (first_line)
            {
                // Parse header
                for (size_t i = 0; i < tokens.size(); i++)
                {
                    header_map[tokens[i]] = i;
                }
                first_line = false;
                
                std::cout << "CSV Headers found: ";
                for (const auto& token : tokens)
                {
                    std::cout << token << " ";
                }
                std::cout << std::endl;
                continue;
            }
            
            if (tokens.size() >= 14)
            {  // Ensure we have all required columns
                StockData row;
                row.symbol = tokens[0];
                row.date = tokens[1];
                row.close = parseDouble(tokens[2]);
                row.high = parseDouble(tokens[3]);
                row.low = parseDouble(tokens[4]);
                row.open = parseDouble(tokens[5]);
                row.volume = parseLong(tokens[6]);
                row.adjClose = parseDouble(tokens[7]);
                row.adjHigh = parseDouble(tokens[8]);
                row.adjLow = parseDouble(tokens[9]);
                row.adjOpen = parseDouble(tokens[10]);
                row.adjVolume = parseLong(tokens[11]);
                row.divCash = parseDouble(tokens[12]);
                row.splitFactor = parseDouble(tokens[13]);
                
                data.push_back(row);
            }
        }
        
        file.close();
        std::cout << "Loaded " << data.size() << " rows of stock data" << std::endl;
        return true;
    }
    
    // Create sequences for LSTM training/testing
    std::vector<std::vector<Eigen::VectorXd>> createSequences(
        int sequence_length, 
        int feature_count = 4,  // Default: open, high, low, close
        bool normalize = true
        )
    {
        if (data.size() < sequence_length)
        {
            std::cerr << "Not enough data for sequences of length " << sequence_length << std::endl;
            return {};
        }
        
        std::vector<std::vector<Eigen::VectorXd>> sequences;
        
        // Calculate normalization factors if needed
        double min_price = 1e9, max_price = -1e9;
        double min_volume = 1e9, max_volume = -1e9;
        
        if (normalize)
        {
            for (const auto& row : data)
            {
                min_price = std::min({min_price, row.open, row.high, row.low, row.close});
                max_price = std::max({max_price, row.open, row.high, row.low, row.close});
                min_volume = std::min(min_volume, (double)row.volume);
                max_volume = std::max(max_volume, (double)row.volume);
            }
            std::cout << "Price range: " << min_price << " to " << max_price << std::endl;
            std::cout << "Volume range: " << min_volume << " to " << max_volume << std::endl;
        }
        
        // Create sequences
        for (size_t i = 0; i <= data.size() - sequence_length; i++)
        {
            std::vector<Eigen::VectorXd> sequence;
            
            for (int j = 0; j < sequence_length; j++)
            {
                Eigen::VectorXd features(feature_count);
                const StockData& row = data[i + j];
                
                if (feature_count == 1)
                {
                    // Just closing price
                    features(0) = normalize ? (row.close - min_price) / (max_price - min_price) : row.close;
                } else if (feature_count == 4)
                {
                    // OHLC (Open, High, Low, Close)
                    if (normalize) {
                        features(0) = (row.open - min_price) / (max_price - min_price);
                        features(1) = (row.high - min_price) / (max_price - min_price);
                        features(2) = (row.low - min_price) / (max_price - min_price);
                        features(3) = (row.close - min_price) / (max_price - min_price);
                    }
                    else
                    {
                        features(0) = row.open;
                        features(1) = row.high;
                        features(2) = row.low;
                        features(3) = row.close;
                    }
                }
                else if (feature_count == 5)
                {
                    // OHLC + Volume
                    if (normalize) {
                        features(0) = (row.open - min_price) / (max_price - min_price);
                        features(1) = (row.high - min_price) / (max_price - min_price);
                        features(2) = (row.low - min_price) / (max_price - min_price);
                        features(3) = (row.close - min_price) / (max_price - min_price);
                        features(4) = (row.volume - min_volume) / (max_volume - min_volume);
                    }
                    else
                    {
                        features(0) = row.open;
                        features(1) = row.high;
                        features(2) = row.low;
                        features(3) = row.close;
                        features(4) = row.volume;
                    }
                }
                
                sequence.push_back(features);
            }
            
            sequences.push_back(sequence);
        }
        
        std::cout << "Created " << sequences.size() << " sequences of length " << sequence_length << std::endl;
        return sequences;
    }
    
    // Get target values (next day's closing price)
    std::vector<double> getTargets(int sequence_length, bool normalize = true)
    {
        std::vector<double> targets;
        
        if (normalize)
        {
            double min_price = 1e9, max_price = -1e9;
            for (const auto& row : data) {
                min_price = std::min(min_price, row.close);
                max_price = std::max(max_price, row.close);
            }
            
            for (size_t i = sequence_length; i < data.size(); i++) {
                double normalized_target = (data[i].close - min_price) / (max_price - min_price);
                targets.push_back(normalized_target);
            }
        }
        else
        {
            for (size_t i = sequence_length; i < data.size(); i++)
            {
                targets.push_back(data[i].close);
            }
        }
        
        return targets;
    }
    
    // Print some sample data
    void printSample(int count = 5)
    {
        std::cout << "\n=== Sample Stock Data ===" << std::endl;
        for (int i = 0; i < std::min(count, (int)data.size()); i++)
        {
            const auto& row = data[i];
            std::cout << "Date: " << row.date 
                      << ", Close: $" << std::fixed << std::setprecision(2) << row.close
                      << ", Volume: " << row.volume << std::endl;
        }
    }
    
    size_t size() const { return data.size(); }
    bool empty() const { return data.empty(); }
};
