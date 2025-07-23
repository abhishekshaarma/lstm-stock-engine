# LSTM Multi-Factor Investment Model

This project implements a custom light weight Long Short-Term Memory (LSTM) neural network from scratch in C++ using the Eigen linear algebra library. The LSTM is trained to predict stock prices based on a multi-factor model, utilizing historical stock data sequences parsed from CSV files. 

## The project was inspired and based on the paper(https://arxiv.org/pdf/2507.00332)
Implementation of the multi-factor investment model optimization with LSTM for risk control, based on Li et al., "Optimization Method of Multi-factor Investment Model Driven by Deep Learning for Risk Control" (2025).



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
the rest is just c++ stdlibrary

### Project Structure
->lstm.h / lstm.cpp:
->StockCSVParser.h / StockCSVParser.cpp:
->MultiFactorModel.h / MultiFactorModel.cpp:
->RiskMetrics.h, PortfolioOptimizer.h, Utils.h:
->Utils.cpp
->main.cpp:

### data/stock.csv:
Sample historical stock price data used for training and testing.

## To build 
The make file should work with Linux and the build.ps1 with powershell

### Futher Improvements
Need further improvements in the data handling as well as more dynamic model. Currently I am working on 
incoperating multiple models into a single program. 