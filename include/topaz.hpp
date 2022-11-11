#pragma once


#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif
#include "bits/begin_end.hpp"
#include "bits/traits.hpp"
#include "bits/range.hpp"
#include "bits/chunked_range.hpp"
#include "bits/zip_range.hpp"
#include "bits/zip.hpp"
#include "bits/transform.hpp"
#include "bits/numeric_array.hpp"
#include "bits/numeric_soa.hpp"
#include "bits/arithmetic_ops.hpp"
#include "bits/parallel_force_evaluate.hpp"

#ifdef __CUDACC__
#include "bits/device_host_copy.hpp"
#endif
