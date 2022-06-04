#define CATCH_CONFIG_ENABLE_BENCHMARKING
#define CATCH_CONFIG_MAIN // This tells the catch header to generate a main
#include "catch.hpp"
#include "all.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/execution_policy.h>
#include <thrust/async/copy.h>
#include <thrust/async/reduce.h>
#include <thrust/async/transform.h>
#include <thrust/transform.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
template <typename T>
struct uninitialized_allocator : thrust::device_malloc_allocator<T> {
    __host__ __device__ void construct(T* p) {}
};

template<class T>
using vector_t = thrust::device_vector<T>;

template<class T>
using allocator = uninitialized_allocator<T>; //thrust::device_malloc_allocator<T>;

template<class T>
using NVec_t = topaz::NumericArray<T, allocator<T>>;

template<class T>
using pinned_allocator = thrust::cuda::experimental::pinned_allocator<T>;

template<class T>
using host_vector = thrust::host_vector<T, pinned_allocator<T>>;

template <class Policy, class Range1_t, class Range2_t>
void copy(Policy p, const Range1_t& src, Range2_t& dst) {
    thrust::copy(p, src.begin(), src.end(), dst.begin());

}

std::vector<cudaStream_t> create_streams(size_t count) {

    std::vector<cudaStream_t> streams(count);
    for (auto& stream : streams){
        cudaStreamCreate(&stream);
    }
    return streams;
}

void destroy_streams(std::vector<cudaStream_t>& streams){

    for (auto& stream : streams){
        cudaStreamDestroy(stream);
    }
}

void sync_streams(std::vector<cudaStream_t>& streams) {
    for (auto& stream : streams){
        cudaStreamSynchronize(stream);
    }
}

struct Streams {

    Streams(size_t count)
        : current_(0)
        , streams_(create_streams(count)) {}

    ~Streams() { destroy_streams(streams_); }

    cudaStream_t get_next() {

        if (current_ == streams_.size()){
            current_ = 0;
        }

        current_++;
        return streams_[current_ - 1];

    }



    void sync_all() { sync_streams(streams_); }

private:
    size_t current_;

    std::vector<cudaStream_t> streams_;
};

template<class Vector_t>
auto arithmetic1(const Vector_t& v1, const Vector_t& v2, const Vector_t& v3){
    using T = typename Vector_t::value_type;
    return v1 * v2 + T(43) / v1 * v3 - v1 - T(32);
}

using element_t = double;

struct Data {

    Data(size_t n_elements_, size_t n_kernels_)
        : n_elements(n_elements_)
        , n_kernels(n_kernels_)
        , results_device(n_kernels, NVec_t<element_t>(n_elements))
        , results_host(n_kernels, host_vector<element_t>(n_elements))
        , v1(n_elements, 12.0)
        , v2(n_elements, 13.0)
        , v3(n_elements, 14.0)
        {}

    size_t n_elements;
    size_t n_kernels;

    std::vector<NVec_t<element_t>>              results_device;
    std::vector<host_vector<element_t>> results_host;
    NVec_t<element_t> v1;
    NVec_t<element_t> v2;
    NVec_t<element_t> v3;
};

struct NoOp{
    CUDA_HOSTDEV double operator()(const double& d) {return d;}
};


template<class T1, class T2>
__global__
void copy_custom(T1 in, T2 out, size_t size)
{
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < size) out[i] = in[i];
}



auto sequential(Data& data){

    for (size_t i = 0; i < data.n_kernels; ++i){
        auto kernel = arithmetic1(data.v1, data.v2, data.v3);
        topaz::copy(kernel, data.results_device[i]);
    }


    for (size_t i = 0; i < data.n_kernels; ++i){
        topaz::copy(data.results_device[i], data.results_host[i]);
    }


    cudaDeviceSynchronize();

    return data.results_host;
}



auto streamed(Data& data, Streams& streams){



    int N = data.n_elements;
    int threads = 256; //1024;
    int blocks = (N + threads + 1) / threads;

    for (size_t i = 0; i < data.n_kernels; ++i){

        auto stream = streams.get_next();

        auto kernel = arithmetic1(data.v1, data.v2, data.v3);
        //auto policy = thrust::cuda::par.on(stream);
        //thrust::transform(policy, kernel.begin(), kernel.end(), data.results_device[i].begin(), NoOp{});
        copy_custom<<<blocks, threads, 0, stream>>>(kernel.begin(), data.results_device[i].begin(), data.results_device[i].size());

        cudaMemcpyAsync(
            data.results_host[i].data(),
            thrust::raw_pointer_cast(data.results_device[i].data()),
            sizeof(element_t) * data.n_elements,
            cudaMemcpyDeviceToHost,
            stream
        );
    }

    /*
    for (size_t i = 0; i < data.n_kernels; ++i){

        cudaMemcpyAsync(
            data.results_host[i].data(),
            thrust::raw_pointer_cast(data.results_device[i].data()),
            sizeof(element_t) * data.n_elements,
            cudaMemcpyDeviceToHost,
            streams.get_next()
        );
        //copy(data.results_device[i], data.results_host[i]);
    }
    */
    streams.sync_all();

    //cudaDeviceSynchronize();
    return data.results_host;

}




auto async(Data& data){


    std::vector<thrust::device_event> events;

    for (size_t i = 0; i < data.n_kernels; ++i){

        auto kernel = arithmetic1(data.v1, data.v2, data.v3);

        events.push_back(thrust::async::transform(
            kernel.begin(), kernel.end(), data.results_device[i].begin(), NoOp{}
        ));

    }


    Streams streams(8);

    for (size_t i = 0; i < data.n_kernels; ++i){
        events[i].wait();
        cudaMemcpyAsync(
            data.results_host[i].data(),
            thrust::raw_pointer_cast(data.results_device[i].data()),
            sizeof(element_t) * data.n_elements,
            cudaMemcpyDeviceToHost,
            streams.get_next()
        );

    }

    streams.sync_all();

    //cudaDeviceSynchronize();

    return data.results_host;

}


auto createData() {
    size_t n_elements = 1E5;
    size_t n_kernels = 150;
    Data data(n_elements, n_kernels);
    return data;
}


TEST_CASE("Sequential"){

    auto data = createData();

    BENCHMARK("Sequential"){
        return sequential(data);
    };
}

TEST_CASE("Streamed"){

    auto data = createData();
    Streams streams(2);
    BENCHMARK("Streamed"){
        return streamed(data, streams);
    };

}
TEST_CASE("Async"){

    auto data = createData();

    BENCHMARK("Async"){
        return async(data);
    };


}
