

#tests
g++ -Iinclude/ -Itest/ test/test.cpp -o test_cpu.bin
nvcc -Iinclude/ -Itest/ -x cu test/test.cpp -o test_gpu.bin

#benchmarks
#g++ -Iinclude/ -Itest/ test/benchmark.cpp -o benchmark_cpu.bin
#nvcc -Iinclude/ -Itest/ -x cu test/benchmark.cpp -o benchmark_gpu.bin
