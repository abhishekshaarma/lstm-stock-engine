#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <thread>
#include <mutex>
#include <omp.h>
#include <Eigen/Dense>

class DataProcessor {
private:
    std::vector<std::vector<double>> raw_data;
    std::vector<std::vector<Eigen::VectorXd>> sequences;
    std::vector<double> targets;
    std::mutex data_mutex;
    
    // Normalization parameters
    double min_val, max_val;
    bool is_normalized;
    
    // Thread-safe random number generator
    std::mt19937 gen;
    std::mutex gen_mutex;

public:
    DataProcessor();
    
    // Parallel CSV loading
    bool loadCSVParallel(const std::string& filename, int num_threads = 0);
    
    // Parallel sequence creation
    std::vector<std::vector<Eigen::VectorXd>> createSequencesParallel(
        int sequence_length, 
        int num_features, 
        bool normalize = true,
        int num_threads = 0
    );
    
    // Parallel target extraction
    std::vector<double> extractTargetsParallel(
        int sequence_length,
        bool normalize = true,
        int num_threads = 0
    );
    
    // Parallel data splitting
    std::pair<std::vector<std::vector<Eigen::VectorXd>>, std::vector<std::vector<Eigen::VectorXd>>> 
    splitDataParallel(
        const std::vector<std::vector<Eigen::VectorXd>>& data,
        double train_ratio = 0.8,
        int num_threads = 0
    );
    
    // Parallel normalization
    void normalizeDataParallel(int num_threads = 0);
    void denormalizeDataParallel(int num_threads = 0);
    
    // Parallel data augmentation
    std::vector<std::vector<Eigen::VectorXd>> augmentDataParallel(
        const std::vector<std::vector<Eigen::VectorXd>>& sequences,
        double noise_factor = 0.01,
        int num_threads = 0
    );
    
    // Utility functions
    std::pair<double, double> getDataRange() const;
    size_t getDataSize() const;
    bool isDataNormalized() const { return is_normalized; }
    
    // Thread-safe random number generation
    double getRandomDouble(double min, double max);
    int getRandomInt(int min, int max);
};
