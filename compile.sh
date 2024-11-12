

#tests
g++ -pthread -std=c++14 -Iinclude/ -Itest/ test/test.cpp -o test_cpu.bin
#nvcc -expt-relaxed-constexpr -std=c++14 -Iinclude/ -Itest/ -x cu test/test.cpp -o test_gpu.bin
#nvc++ -std=c++14 -Iinclude/ -Itest/ -x cu test/test.cpp -o test_gpu.bin



#benchmarks
#g++ -Iinclude/ -Itest/ test/benchmark.cpp -o benchmark_cpu.bin
#nvcc -Iinclude/ -Itest/ -x cu test/benchmark.cpp -o benchmark_gpu.bin
#nvc++ -Iinclude/ -Itest/ -x cu test/benchmark.cpp -o benchmark_gpu.bin
