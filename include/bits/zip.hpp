#pragma once

#include "traits.hpp"
#include "zip_range.hpp"

namespace topaz {

template <class Range_t, std::enable_if_t<IsRange_v<Range_t>, bool> = true>
inline CUDA_HOSTDEV auto zip(Range_t& rng) {
    using iterator    = decltype(std::begin(rng));
    using result_type = ZipRange<Tuple<iterator>>;
    return result_type(adl_make_tuple(rng));
}

template <class Range_t, std::enable_if_t<IsRange_v<Range_t>, bool> = true>
inline CUDA_HOSTDEV auto zip(const Range_t& rng) {
    using iterator    = decltype(std::begin(rng));
    using result_type = ZipRange<Tuple<iterator>>;
    return result_type(adl_make_tuple(rng));
}

template <class T1, class T2>
inline CUDA_HOSTDEV auto zip(T1& rng1, T2& rng2) {
    return make_zip_range(rng1, rng2);
}

template <class T1, class T2>
inline CUDA_HOSTDEV auto zip(T1& rng1, const T2& rng2) {
    return make_zip_range(rng1, rng2);
}

template <class T1, class T2>
inline CUDA_HOSTDEV auto zip(const T1& rng1, T2& rng2) {
    return make_zip_range(rng1, rng2);
}

template <class T1, class T2>
inline CUDA_HOSTDEV auto zip(const T1& rng1, const T2& rng2) {
    return make_zip_range(rng1, rng2);
}
} // namespace topaz