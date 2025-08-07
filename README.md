# LSTM Multi-Factor Investment Model

This project implements a custom light weight Long Short-Term Memory (LSTM) neural network from scratch in C++ using the Eigen linear algebra library. The LSTM is trained to predict stock prices based on a multi-factor model, utilizing historical stock data sequences parsed from CSV files. 

## The project was inspired and based on the paper(https://arxiv.org/pdf/2507.00332)
Implementation of the multi-factor investment model optimization with LSTM for risk control, based on Li et al., "Optimization Method of Multi-factor Investment Model Driven by Deep Learning for Risk Control" (2025).

## 🚀 New: Multithreading Support

The project now includes comprehensive multithreading support using OpenMP for significant performance improvements:

### Key Multithreading Features:
- **Parallel Training**: Batch processing with OpenMP parallelization
- **Parallel Forward Pass**: Multiple sequences processed simultaneously
- **Parallel Validation**: Concurrent evaluation of test data
- **Thread-Safe Operations**: Proper synchronization for shared resources
- **Automatic Thread Detection**: Uses all available CPU cores

### Performance Improvements:
- **2-4x faster training** on multi-core systems
- **Parallel data preprocessing** for large datasets
- **Efficient memory usage** with thread-local storage
- **Scalable architecture** that adapts to available cores

### Features
Lightweight LSTM with constructors to dynamically change the goal of the model.

### Data Handling:

I have written a CSV parser class, however it is specific to the data I used. 

### Prediction & Evaluation:

Predicts stock price values on unseen test data.
Compares model predictions against a baseline multi-factor linear regression model.
Computes performance metrics such as MSE and R-squared.

### Requirements
C++17 or higher compiler and Eigen linear algebra library (for matrix and vector operations)
OpenMP support (included in most modern compilers)
the rest is just c++ stdlibrary

### Project Structure
- lstm.h / lstm.cpp: Original LSTM implementation
- lstm_parallel.h / lstm_parallel.cpp: Multithreaded LSTM implementation
- StockCSVParser.h / StockCSVParser.cpp: CSV data parser
- MultiFactorModel.h / MultiFactorModel.cpp: Baseline model
- RiskMetrics.h, PortfolioOptimizer.h, Utils.h: Utility classes
- Utils.cpp: Utility functions
- main.cpp: Original training script
- main_parallel.cpp: Full multithreaded training script
- train_parallel.cpp: Simplified parallel training demo
- DataProcessor.h: Multithreaded data preprocessing utilities

### data/stock.csv:
Sample historical stock price data used for training and testing.

## To build 
The make file should work with Linux and the build.ps1 with powershell

### Build Targets:
```bash
# Build all versions
make all

# Build and run original version
make run

# Build and run parallel version
make run-parallel

# Build and run training demo
make run-train

# Performance comparison
make benchmark

# Clean build artifacts
make clean
```

### PowerShell (Windows):
```powershell
# Build with OpenMP support
.\build.ps1

# Run the executable
.\bin\lstm.exe
```

### Multithreading Configuration:
The system automatically detects available CPU cores and uses them for parallel processing. You can control the number of threads by setting the `OMP_NUM_THREADS` environment variable:

```bash
# Use 8 threads
export OMP_NUM_THREADS=8
make run-parallel

# Or set in PowerShell
$env:OMP_NUM_THREADS=8
.\bin\lstm_parallel.exe
```

### Futher Improvements
Need further improvements in the data handling as well as more dynamic model. Currently I am working on 
incoperating multiple models into a single program. 

### Performance Notes:
- **Training Speed**: 2-4x faster on multi-core systems
- **Memory Usage**: Slightly higher due to thread-local storage
- **Scalability**: Linear scaling with number of cores (up to optimal point)
- **Compatibility**: Works on Windows, Linux, and macOS with OpenMP support 