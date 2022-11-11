#pragma once

#include "range.hpp"

#ifdef __CUDACC__
#include <thrust/async/copy.h>
#include <thrust/copy.h>
#else
#include <algorithm>
#include <future>
#endif

namespace topaz {

#ifdef __CUDACC__

template <class Range1_t, class Range2_t>
void copy(const Range1_t& src, Range2_t& dst) {
    thrust::copy(src.begin(), src.end(), dst.begin());
}

template <class Range1_t, class Range2_t>
auto async_copy(const Range1_t& src, Range2_t& dst) {
    return thrust::async::copy(src.begin(), src.end(), dst.begin());
}

template <class Event, class Range1_t, class Range2_t>
auto async_copy(Event& e, const Range1_t& src, Range2_t& dst) {
    return thrust::async::copy(
        thrust::device.after(e), src.begin(), src.end(), dst.begin());
}

#else

template <class Range1_t, class Range2_t>
void copy(const Range1_t& src, Range2_t& dst) {
    std::copy(src.begin(), src.end(), dst.begin());
}

template <class Range1_t, class Range2_t>
auto async_copy(const Range1_t& src, Range2_t& dst) {

    using iterator1 = decltype(src.begin());
    using iterator2 = decltype(dst.begin());

    return std::async(std::launch::async,
                      std::copy<iterator1, iterator2>,
                      src.begin(),
                      src.end(),
                      dst.begin());
}

template <class Event, class Range1_t, class Range2_t>
auto async_copy(Event&& e, const Range1_t& src, Range2_t& dst) {

    //Note! The previous event is forced to be evaluated here
    //since standard does not support smart futures yet
    e.wait();
    return async_copy(src, dst);

}

#endif

} // namespace topaz