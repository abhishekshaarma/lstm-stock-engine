#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <iomanip>
#include <algorithm>
#include <random>
#include <numeric>
#include <set>

struct StockData
{
    std::string date;
    double open;
    double high;
    double low;
    double close;
    long volume;
    std::string name;
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
            std::cerr << "Error parsing double: '" << str << "' - " << e.what() << std::endl;
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
            std::cerr << "Error parsing long: '" << str << "' - " << e.what() << std::endl;
            return 0;
        }
    }

public:
    void filterReasonablePrices(double min_price = 5.0, double max_price = 500.0) {
        std::cout << "\n=== Filtering to Reasonable Price Range ===" << std::endl;
        size_t original_size = data.size();
    
        auto it = std::remove_if(data.begin(), data.end(), [min_price, max_price](const StockData& row) {
            return row.open < min_price || row.high < min_price || 
                                                      row.low < min_price || row.close < min_price ||
                                                                                         row.open > max_price || row.high > max_price || 
                                                      row.low > max_price || row.close > max_price ||
                row.volume <= 0;
        });
    
        data.erase(it, data.end());
    
        std::cout << "Kept " << data.size() << " entries (removed " 
                  << (original_size - data.size()) << " outliers)" << std::endl;
    
        auto price_range = getPriceRange();
        std::cout << "New price range: $" << std::fixed << std::setprecision(2) 
                  << price_range.first << " to $" << price_range.second << std::endl;
    }
    void cleanBadData() {
        std::cout << "\n=== Cleaning Bad Data ===" << std::endl;
        size_t original_size = data.size();
    
        // Remove entries with zero/very low prices or invalid OHLC relationships
        auto it = std::remove_if(data.begin(), data.end(), [](const StockData& row) {
            // Remove if any price is too low (likely parsing error)
            if (row.open < 1.0 || row.high < 1.0 || row.low < 1.0 || row.close < 1.0) {
                return true;
            }
        
            // Remove if volume is zero (bad data)
            if (row.volume <= 0) {
                return true;
            }
        
            // Remove if OHLC relationships are invalid
            if (row.high < row.low || row.close > row.high || row.close < row.low || 
                row.open > row.high || row.open < row.low) {
                return true;
            }
        
            // Remove if prices are unreasonably high (likely parsing error)
            if (row.open > 2000 || row.high > 2000 || row.low > 2000 || row.close > 2000) {
                return true;
            }
        
            return false;
        });
    
        data.erase(it, data.end());
    
        size_t cleaned_size = data.size();
        size_t removed = original_size - cleaned_size;
    
        std::cout << "Removed " << removed << " bad entries (" 
                  << std::fixed << std::setprecision(1) 
                  << (removed * 100.0 / original_size) << "% of data)" << std::endl;
    
        // Show new price range
        if (!data.empty()) {
            auto price_range = getPriceRange();
            std::cout << "Cleaned price range: $" << std::fixed << std::setprecision(2) 
                      << price_range.first << " to $" << price_range.second << std::endl;
        }
    }
    void filterByPriceRange(double min_price = 5.0, double max_price = 500.0)
    {
        std::cout << "\n=== Filtering by Price Range [$" << min_price 
                  << " - $" << max_price << "] ===" << std::endl;
    
        size_t original_size = data.size();
    
        auto it = std::remove_if(data.begin(), data.end(), [min_price, max_price](const StockData& row) {
            return row.close < min_price || row.close > max_price;
        });
    
        data.erase(it, data.end());
    
        size_t filtered_size = data.size();
        std::cout << "Kept " << filtered_size << " entries out of " << original_size 
                  << " (" << std::fixed << std::setprecision(1) 
                  << (filtered_size * 100.0 / original_size) << "%)" << std::endl;
    
        // Show new range
        auto price_range = getPriceRange();
        std::cout << "Filtered price range: $" << std::fixed << std::setprecision(2) 
                  << price_range.first << " to $" << price_range.second << std::endl;
    }
    std::pair<double, double> getPriceRange() const
    {
        if (data.empty()) return {0.0, 0.0};
        
        double min_price = data[0].close;
        double max_price = data[0].close;
        
        for (const auto& row : data)
        {
            min_price = std::min({min_price, row.open, row.high, row.low, row.close});
            max_price = std::max({max_price, row.open, row.high, row.low, row.close});
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
        int line_count = 0;
        
        while (std::getline(file, line))
        {
            line_count++;
            if (line.empty()) continue;
            
            std::vector<std::string> tokens = split(line, ',');
            
            if (first_line)
            {
                // Parse header: date,open,high,low,close,volume,Name
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
                
                // Verify expected headers
                std::vector<std::string> expected = {"date", "open", "high", "low", "close", "volume", "Name"};
                bool headers_ok = true;
                for (const auto& exp : expected) {
                    if (header_map.find(exp) == header_map.end())
                    {
                        std::cerr << "Warning: Expected header '" << exp << "' not found!" << std::endl;
                        headers_ok = false;
                    }
                }
                if (headers_ok)
                {
                    std::cout << "✓ All expected headers found" << std::endl;
                }
                continue;
            }
            
            // Debug first few data lines
            if (data.size() < 3) {
                std::cout << "\nDebug - Line " << line_count << ": " << line << std::endl;
                std::cout << "Tokens (" << tokens.size() << "): ";
                for (size_t i = 0; i < tokens.size(); i++)
                {
                    std::cout << "[" << i << "]='" << tokens[i] << "' ";
                }
                std::cout << std::endl;
            }
            
            if (tokens.size() >= 7)
            {  
                StockData row;
                
                // Use header mapping for robust parsing
                row.date = tokens[header_map["date"]];
                row.open = parseDouble(tokens[header_map["open"]]);
                row.high = parseDouble(tokens[header_map["high"]]);
                row.low = parseDouble(tokens[header_map["low"]]);
                row.close = parseDouble(tokens[header_map["close"]]);
                row.volume = parseLong(tokens[header_map["volume"]]);
                row.name = tokens[header_map["Name"]];
                
                // Debug parsed values for first few rows
                if (data.size() < 3)
                {
                    std::cout << "Parsed - Date: '" << row.date 
                              << "', Name: '" << row.name
                              << "', Open: " << row.open 
                              << ", High: " << row.high 
                              << ", Low: " << row.low 
                              << ", Close: " << row.close 
                              << ", Volume: " << row.volume << std::endl;
                }
                
                data.push_back(row);
            }
            else
            {
                std::cerr << "Warning: Line " << line_count << " has only " << tokens.size() 
                          << " columns (expected 7)" << std::endl;
            }
        }
        
        file.close();
        std::cout << "Loaded " << data.size() << " rows of stock data" << std::endl;
        
        // Show data summary
        if (!data.empty())
        {
            auto price_range = getPriceRange();
            std::cout << "Price range: $" << std::fixed << std::setprecision(2) 
                      << price_range.first << " to $" << price_range.second << std::endl;
            
            // Show date range
            std::cout << "Date range: " << data.front().date << " to " << data.back().date << std::endl;
            
            // Show unique symbols
            std::set<std::string> symbols;
            for (const auto& row : data)
            {
                symbols.insert(row.name);
            }
            std::cout << "Stocks: " << symbols.size() << " symbols (";
            int count = 0;
            for (const auto& sym : symbols)
            {
                std::cout << sym;
                if (++count < symbols.size()) std::cout << ", ";
                if (count >= 5) { std::cout << "..."; break; }
            }
            std::cout << ")" << std::endl;
        }
        
        return true;
    }
    
    // Create sequences with random sampling to avoid timeline issues
    std::vector<std::vector<Eigen::VectorXd>> createSequences(
        int sequence_length, 
        int feature_count = 4,  // Default: open, high, low, close
        bool normalize = true,
        bool random_split = true  // New parameter for random sampling
        )
    {
        if ((int)data.size() < sequence_length)
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
            std::cout << "Normalization - Price range: $" << std::fixed << std::setprecision(2) 
                      << min_price << " to $" << max_price << std::endl;
            std::cout << "Normalization - Volume range: " << min_volume << " to " << max_volume << std::endl;
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
                } 
                else if (feature_count == 4)
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
        
        // Debug: Show first sequence
        if (!sequences.empty()) {
            std::cout << "\nFirst sequence:" << std::endl;
            for (int day = 0; day < std::min(3, sequence_length); day++) {
                const auto& features = sequences[0][day];
                std::cout << "  Day " << day << ": [";
                for (int f = 0; f < feature_count; f++) {
                    std::cout << std::fixed << std::setprecision(4) << features(f);
                    if (f < feature_count-1) std::cout << ", ";
                }
                std::cout << "]";
                
                // Show corresponding raw data
                const auto& raw = data[day];
                std::cout << " (Raw: O=$" << std::setprecision(2) << raw.open 
                          << " H=$" << raw.high << " L=$" << raw.low << " C=$" << raw.close 
                          << ", " << raw.name << ")";
                std::cout << std::endl;
            }
        }
        
        return sequences;
    }
    
    // Random train/test split to avoid timeline issues
    std::pair<std::vector<std::vector<Eigen::VectorXd>>, std::vector<std::vector<Eigen::VectorXd>>>
    createRandomSplit(const std::vector<std::vector<Eigen::VectorXd>>& sequences, double train_ratio = 0.8)
    {
        // Create indices for random sampling
        std::vector<size_t> indices(sequences.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        // Shuffle indices
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        // Split
        size_t split_point = (size_t)(sequences.size() * train_ratio);
        std::vector<std::vector<Eigen::VectorXd>> train_sequences, test_sequences;
        
        for (size_t i = 0; i < split_point; i++)
        {
            train_sequences.push_back(sequences[indices[i]]);
        }
        
        for (size_t i = split_point; i < sequences.size(); i++)
        {
            test_sequences.push_back(sequences[indices[i]]);
        }
        
        std::cout << "Random split: " << train_sequences.size() << " training, " 
                  << test_sequences.size() << " testing" << std::endl;
        
        return {train_sequences, test_sequences};
    }
    
    // Get target values (next day's closing price) 
    std::vector<double> getTargets(int sequence_length, bool normalize = true)
    {
        std::vector<double> targets;
        
        if (normalize)
        {
            auto price_range = getPriceRange();
            double min_price = price_range.first;
            double max_price = price_range.second;
            
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
        
        // Debug: Show first few targets
        std::cout << "First few targets:" << std::endl;
        for (int i = 0; i < std::min(3, (int)targets.size()); i++)
        {
            double raw_price = data[sequence_length + i].close;
            std::cout << "  Target " << i << ": " << std::fixed << std::setprecision(4) 
                      << targets[i] << " (Raw: $" << std::setprecision(2) << raw_price 
                      << ", Date: " << data[sequence_length + i].date << ")" << std::endl;
        }
        
        return targets;
    }
    
    // Random target split to match sequence split
    std::pair<std::vector<double>, std::vector<double>>
    createRandomTargetSplit(const std::vector<double>& targets, double train_ratio = 0.8)
    {
        // Use same random indices as sequences
        std::vector<size_t> indices(targets.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        size_t split_point = (size_t)(targets.size() * train_ratio);
        std::vector<double> train_targets, test_targets;
        
        for (size_t i = 0; i < split_point; i++)
        {
            train_targets.push_back(targets[indices[i]]);
        }
        
        for (size_t i = split_point; i < targets.size(); i++)
        {
            test_targets.push_back(targets[indices[i]]);
        }
        
        return {train_targets, test_targets};
    }
    
    // Print some sample data
    void printSample(int count = 5)
    {
        std::cout << "\n=== Sample Stock Data ===" << std::endl;
        for (int i = 0; i < std::min(count, (int)data.size()); i++)
        {
            const auto& row = data[i];
            std::cout << "Date: " << row.date 
                      << ", Symbol: " << row.name
                      << ", Open: $" << std::fixed << std::setprecision(2) << row.open
                      << ", High: $" << row.high
                      << ", Low: $" << row.low
                      << ", Close: $" << row.close
                      << ", Volume: " << row.volume << std::endl;
        }
    }
    
    size_t size() const { return data.size(); }
    bool empty() const { return data.empty(); }
};
