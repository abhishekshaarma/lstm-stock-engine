# Makefile for LSTM C++ implementation with Eigen

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
DEBUG_FLAGS = -g -DDEBUG -O0

# Eigen settings - adjust path as needed
EIGEN_PATH = ./eigen-3.4.0
# Alternative common paths:
# EIGEN_PATH = /usr/include/eigen3
# EIGEN_PATH = /usr/local/include/eigen3
# EIGEN_PATH = /opt/homebrew/include/eigen3  # macOS with Homebrew

INCLUDES = -I$(EIGEN_PATH)

# Source and target
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# Create directories if they don't exist
$(shell mkdir -p $(OBJDIR) $(BINDIR))

# Source files
SOURCES = lstm.cpp main.cpp
OBJECTS = $(SOURCES:%.cpp=$(OBJDIR)/%.o)
TARGET = $(BINDIR)/lstm

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@

# Compile source files
$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Debug build
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: $(TARGET)

# Clean build artifacts
clean:
	rm -rf $(OBJDIR)/*.o $(TARGET)
	rm -rf $(OBJDIR) $(BINDIR)

# Clean and rebuild
rebuild: clean all

# Run the program
run: $(TARGET)
	./$(TARGET)

# Install Eigen (Ubuntu/Debian)
install-eigen-ubuntu:
	sudo apt-get update
	sudo apt-get install libeigen3-dev

# Install Eigen (macOS with Homebrew)
install-eigen-macos:
	brew install eigen

# Install Eigen (CentOS/RHEL/Fedora)
install-eigen-fedora:
	sudo dnf install eigen3-devel

# Check if Eigen is properly installed
check-eigen:
	@echo "Checking for Eigen installation..."
	@if [ -d "$(EIGEN_PATH)" ]; then \
		echo "✓ Eigen found at $(EIGEN_PATH)"; \
		ls $(EIGEN_PATH)/Eigen/ | head -5; \
	else \
		echo "✗ Eigen not found at $(EIGEN_PATH)"; \
		echo "Please install Eigen or update EIGEN_PATH in Makefile"; \
		echo "Common locations to check:"; \
		echo "  /usr/include/eigen3"; \
		echo "  /usr/local/include/eigen3"; \
		echo "  /opt/homebrew/include/eigen3"; \
	fi

# Test compilation without running
test-compile: $(TARGET)
	@echo "✓ Compilation successful"

# Create a simple test
test: $(TARGET)
	@echo "Running LSTM test..."
	./$(TARGET)
	@echo "✓ Test completed"

# Help target
help:
	@echo "Available targets:"
	@echo "  all            - Build the LSTM executable (default)"
	@echo "  debug          - Build with debug flags"
	@echo "  clean          - Remove build artifacts"
	@echo "  rebuild        - Clean and build"
	@echo "  run            - Build and run the program"
	@echo "  test           - Build and run test"
	@echo "  test-compile   - Test compilation only"
	@echo "  check-eigen    - Check if Eigen is installed"
	@echo "  install-eigen-ubuntu - Install Eigen on Ubuntu/Debian"
	@echo "  install-eigen-macos   - Install Eigen on macOS"
	@echo "  install-eigen-fedora  - Install Eigen on Fedora/CentOS"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  CXX        = $(CXX)"
	@echo "  CXXFLAGS   = $(CXXFLAGS)"
	@echo "  EIGEN_PATH = $(EIGEN_PATH)"

# Declare phony targets
.PHONY: all debug clean rebuild run test test-compile check-eigen help
.PHONY: install-eigen-ubuntu install-eigen-macos install-eigen-fedora
