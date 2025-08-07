CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 -fopenmp
INCLUDES = -Iinclude -Iinclude/eigen-3.4.0

# Directories
OBJDIR = obj
BINDIR = bin

# Files
SOURCES = lstm.cpp main.cpp Utils.cpp
PARALLEL_SOURCES = lstm_parallel.cpp main_parallel.cpp Utils.cpp
TRAIN_SOURCES = lstm.cpp train_parallel.cpp Utils.cpp
PERF_SOURCES = lstm.cpp performance_test.cpp Utils.cpp

OBJECTS = $(SOURCES:%.cpp=$(OBJDIR)/%.o)
PARALLEL_OBJECTS = $(PARALLEL_SOURCES:%.cpp=$(OBJDIR)/%.o)
TRAIN_OBJECTS = $(TRAIN_SOURCES:%.cpp=$(OBJDIR)/%.o)
PERF_OBJECTS = $(PERF_SOURCES:%.cpp=$(OBJDIR)/%.o)

TARGET = $(BINDIR)/lstm
PARALLEL_TARGET = $(BINDIR)/lstm_parallel
TRAIN_TARGET = $(BINDIR)/train_parallel
PERF_TARGET = $(BINDIR)/performance_test

# Create directories
$(shell mkdir -p $(OBJDIR) $(BINDIR))

# Build targets
all: $(TARGET) $(PARALLEL_TARGET) $(TRAIN_TARGET) $(PERF_TARGET)

# Original LSTM
$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ -fopenmp

# Parallel LSTM
$(PARALLEL_TARGET): $(PARALLEL_OBJECTS)
	$(CXX) $(PARALLEL_OBJECTS) -o $@ -fopenmp

# Parallel training demo
$(TRAIN_TARGET): $(TRAIN_OBJECTS)
	$(CXX) $(TRAIN_OBJECTS) -o $@ -fopenmp

# Performance test
$(PERF_TARGET): $(PERF_OBJECTS)
	$(CXX) $(PERF_OBJECTS) -o $@ -fopenmp

# Compile rules
$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Utilities
clean:
	rm -rf $(OBJDIR) $(BINDIR)

run: $(TARGET)
	./$(TARGET)

run-parallel: $(PARALLEL_TARGET)
	./$(PARALLEL_TARGET)

run-train: $(TRAIN_TARGET)
	./$(TRAIN_TARGET)

run-perf: $(PERF_TARGET)
	./$(PERF_TARGET)

install-eigen:
	sudo dnf install eigen3-devel

# Performance comparison
benchmark: $(TARGET) $(TRAIN_TARGET)
	@echo "Running original LSTM..."
	@time ./$(TARGET)
	@echo "Running parallel LSTM..."
	@time ./$(TRAIN_TARGET)

# Quick performance test
perf-test: $(PERF_TARGET)
	@echo "Running performance comparison..."
	./$(PERF_TARGET)

.PHONY: all clean run run-parallel run-train run-perf install-eigen benchmark perf-test
