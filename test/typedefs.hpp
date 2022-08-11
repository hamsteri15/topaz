#pragma once

#include "all.hpp"

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/host_vector.h>

template<class T>
using vector_t = thrust::device_vector<T>;

template<class T>
using NVec_t = topaz::NumericArray<T, thrust::device_malloc_allocator<T>>;


#else
#include <vector>
template<class T>
using vector_t = std::vector<T>;

template<class T>
using NVec_t = topaz::NumericArray<T, std::allocator<T>>;


#endif