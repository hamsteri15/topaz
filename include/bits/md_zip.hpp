#pragma once

#include "md_zip_range.hpp"
#include "traits.hpp"

namespace topaz {

/*
template <class MdMdRange_t>
inline CUDA_HOSTDEV auto md_zip(MdRange_t& rng) {
    using iterator = decltype((*rng.begin()).begin());
    using result_type = MdZipRange<Tuple<iterator>>;
    return result_type(adl_make_tuple(rng));
}

template <class MdRange_t>
inline CUDA_HOSTDEV auto md_zip(const MdRange_t& rng) {
    using iterator    = decltype(std::begin(rng));
    using result_type = ZipRange<Tuple<iterator>>;
    return result_type(adl_make_tuple(rng));
}
*/

template <class MdRange1_t, class MdRange2_t>
inline CUDA_HOSTDEV auto md_zip(MdRange1_t& rng1, MdRange2_t& rng2) {
    return make_md_zip_range(rng1, rng2);
}

template <class MdRange1_t, class MdRange2_t>
inline CUDA_HOSTDEV auto md_zip(MdRange1_t& rng1, const MdRange2_t& rng2) {
    return make_md_zip_range(rng1, rng2);
}

template <class MdRange1_t, class MdRange2_t>
inline CUDA_HOSTDEV auto md_zip(const MdRange1_t& rng1, MdRange2_t& rng2) {
    return make_md_zip_range(rng1, rng2);
}

template <class MdRange1_t, class MdRange2_t>
inline CUDA_HOSTDEV auto md_zip(const MdRange1_t& rng1,
                                const MdRange2_t& rng2) {
    return make_md_zip_range(rng1, rng2);
}
} // namespace topaz