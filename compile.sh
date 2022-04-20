

#tests
g++ -Isrc/ -Itest/ test/test.cpp -o test_cpu.bin
nvcc -Isrc/ -Itest/ -x cu test/test.cpp -o test_gpu.bin

#benchmarks
g++ -Isrc/ -Itest/ test/benchmark.cpp -o benchmark_cpu.bin
nvcc -Isrc/ -Itest/ -x cu test/benchmark.cpp -o benchmark_gpu.bin
