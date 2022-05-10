#pragma once


#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif
#include "begin_end.hpp"
#include "traits.hpp"
#include "range.hpp"
#include "chunked_range.hpp"
#include "zip_range.hpp"
#include "zip.hpp"
#include "transform.hpp"
#include "numeric_array.hpp"
#include "numeric_soa.hpp"
#include "arithmetic_ops.hpp"