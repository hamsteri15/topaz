#pragma once

#include "range.hpp"

#ifdef __CUDACC__
#include <thrust/copy.h>
#else
#include <algorithm>
#endif

namespace topaz {

template <class Range1_t, class Range2_t>
void copy(const Range1_t& src, Range2_t& dst) {


#ifdef __CUDACC__
    thrust::copy(src.begin(), src.end(), dst.begin());
#else
    //TODO: ifdef for c++17 parallel exec
    //std::copy(std::execution::par{}, src.begin(), src.end(), dst.begin());
    std::copy(src.begin(), src.end(), dst.begin());
#endif


}


} // namespace topaz