#define CATCH_CONFIG_ENABLE_BENCHMARKING
#define CATCH_CONFIG_MAIN // This tells the catch header to generate a main
#include "catch.hpp"

#include "all.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/execution_policy.h>
#include <thrust/async/copy.h>
#include <thrust/async/transform.h>

template<class T>
using vector_t = thrust::device_vector<T>;

template<class T>
using NVec_t = topaz::NumericArray<T, thrust::device_malloc_allocator<T>>;

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

template<class Vector_t>
auto arithmetic1(const Vector_t& v1, const Vector_t& v2, const Vector_t& v3){
    using T = typename Vector_t::value_type;
    return v1 * v2 + T(43) / v1 * v3 - v1 - T(32);
}

struct NoOp{

    CUDA_HOSTDEV double operator()(const double& d) {return d;}

};
TEST_CASE("asd"){


    size_t n_elements = 2000;
    size_t n_kernels = 400;

    using element_t = double;
    std::vector<NVec_t<element_t>> results(n_kernels, NVec_t<element_t>(n_elements, 0));
    NVec_t<element_t> v1(n_elements, 1);
    NVec_t<element_t> v2(n_elements, 2);
    NVec_t<element_t> v3(n_elements, 3);

    std::vector<thrust::host_vector<element_t>> results_host(n_kernels, std::vector<element_t>(n_elements));

    BENCHMARK("Sequential kernel evaluation"){

        for (size_t i = 0; i < n_kernels; ++i){
            auto kernel = arithmetic1(v1, v2, v3);
            topaz::copy(kernel, results[i]);
        }
        for (size_t i = 0; i < n_kernels; ++i){
            topaz::copy(results[i], results_host[i]);
        }
        return results_host;
    };

    BENCHMARK("Streamed kernel evaluation"){

        size_t n_streams = n_kernels;
        auto streams = create_streams(n_streams);


        for (size_t i = 0; i < n_kernels; ++i){
            auto kernel = arithmetic1(v1, v2, v3);
            auto policy = thrust::cuda::par.on(streams[i]);
            copy(policy, kernel, results[i]);
        }

        for (size_t i = 0; i < n_kernels; ++i){
            cudaStreamSynchronize(streams[i]);
            topaz::copy(results[i], results_host[i]);
        }

        destroy_streams(streams);
        return results_host;
    };

    BENCHMARK("async copy"){


        std::vector<thrust::device_event> events;

        for (size_t i = 0; i < n_kernels; ++i){

            auto kernel = arithmetic1(v1, v2, v3);
            events.push_back(thrust::async::transform(
                kernel.begin(), kernel.end(), results[i].begin(), NoOp{}
            ));
            //events.push_back(event);
        }

        for (size_t i = 0; i < n_kernels; ++i){
            /*auto e = thrust::async::copy
            (
                thrust::host.after(events[i]),
                results[i].begin(), results[i].end(), results_host[i].begin()
            );
            e.wait();*/
            events[i].wait();
            topaz::copy(results[i], results_host[i]);
        }


        return results_host;
    };



}