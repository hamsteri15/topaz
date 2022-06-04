

#tests
#g++ -std=c++17 -Iinclude/ -Itest/ test/test.cpp -o test_cpu.bin
#nvcc -std=c++14 -Iinclude/ -Itest/ -x cu test/test.cpp -o test_gpu.bin


nvcc -std=c++14 --default-stream per-thread -Iinclude/ -Itest/ -x cu profile/profile_stream.cu -o profile_streams.bin
#nvcc -std=c++14 -O3 -Iinclude/ -Itest/ -Iprofile/ -x cu profile/profile_stream.cu -o profile_streams.bin

#benchmarks
#g++ -Iinclude/ -Itest/ test/benchmark.cpp -o benchmark_cpu.bin
#nvcc -Iinclude/ -Itest/ -x cu test/benchmark.cpp -o benchmark_gpu.bin
