CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2
INCLUDES = -Iinclude -Iinclude/eigen-3.4.0

# Directories
OBJDIR = obj
BINDIR = bin

# Files
SOURCES = lstm.cpp main.cpp Utils.cpp
OBJECTS = $(SOURCES:%.cpp=$(OBJDIR)/%.o)
TARGET = $(BINDIR)/lstm

# Create directories
$(shell mkdir -p $(OBJDIR) $(BINDIR))

# Build
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@

$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Utilities
clean:
	rm -rf $(OBJDIR) $(BINDIR)

run: $(TARGET)
	./$(TARGET)

install-eigen:
	sudo dnf install eigen3-devel

.PHONY: all clean run install-eigen
