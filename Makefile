# Compiler and flags
CXX := g++
CXXFLAGS := -Iinclude -Itest -std=c++14 -Wall -Wextra

# Source files and output binaries
TEST_SOURCES := test/catch_main.cpp test/test.cpp
BENCHMARK_SOURCES := test/catch_main.cpp test/benchmark.cpp
TEST_OBJECTS := $(TEST_SOURCES:.cpp=.o)
BENCHMARK_OBJECTS := $(BENCHMARK_SOURCES:.cpp=.o)
TEST_OUTPUT := run_tests.bin
BENCHMARK_OUTPUT := run_benchmark.bin

# Dependency files
DEPS := $(TEST_OBJECTS:.o=.d) $(BENCHMARK_OBJECTS:.o=.d)

# Default target to build both test and benchmark binaries
all: $(TEST_OUTPUT) $(BENCHMARK_OUTPUT)

# Rule to create the test binary
$(TEST_OUTPUT): $(TEST_OBJECTS)
	$(CXX) $(CXXFLAGS) $(TEST_OBJECTS) -o $(TEST_OUTPUT)

# Rule to create the benchmark binary
$(BENCHMARK_OUTPUT): $(BENCHMARK_OBJECTS)
	$(CXX) $(CXXFLAGS) $(BENCHMARK_OBJECTS) -o $(BENCHMARK_OUTPUT)

# Force recompilation of test.o every time by adding a phony dependency
.PHONY: force_rebuild
test/test.o: test/test.cpp force_rebuild
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile catch_main.o and benchmark.o only if they or their dependencies change
test/catch_main.o: test/catch_main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test/benchmark.o: test/benchmark.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Include dependency files for header tracking
-include $(DEPS)

# Run the tests
test: $(TEST_OUTPUT)
	./$(TEST_OUTPUT)

# Run the benchmarks
benchmark: $(BENCHMARK_OUTPUT)
	./$(BENCHMARK_OUTPUT)

# Clean up build files
clean:
	rm -f $(TEST_OUTPUT) $(BENCHMARK_OUTPUT) $(TEST_OBJECTS) $(BENCHMARK_OBJECTS) $(DEPS)

