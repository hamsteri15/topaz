# Compiler and flags
CXX := g++
CXXFLAGS := -Iinclude -Itest -std=c++14 -Wall -Wextra

# Source files and output binary
SOURCES := test/catch_main.cpp test/test.cpp
OUTPUT := run_tests

# Default target
all: $(OUTPUT)

# Link the source files into a single executable
$(OUTPUT): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(OUTPUT)

# Run the tests
test: $(OUTPUT)
	./$(OUTPUT)

# Clean up build files
clean:
	rm -f $(OUTPUT)

