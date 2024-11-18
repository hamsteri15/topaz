#pragma once

#include "traits.hpp"
#include "transform_mdrange.hpp"
#include "zip_mdrange.hpp"
#include "transform.hpp"

namespace topaz {

template <class Range_t, typename UnaryOp>
inline CUDA_HOSTDEV auto md_transform(Range_t& rng, UnaryOp f) {
    return make_md_transform_range(rng, f);
}

template <class Range_t, typename UnaryOp>
inline CUDA_HOSTDEV auto md_transform(const Range_t& rng, UnaryOp f) {
    return make_md_transform_range(rng, f);
}

/*
template <typename Range1_t, typename Range2_t, typename BinaryOp>
inline CUDA_HOSTDEV auto transform(Range1_t& rng1, Range2_t& rng2, BinaryOp f) {
    return transform(zip(rng1, rng2), ApplyBinaryOp<BinaryOp>(f));
}

template <typename Range1_t, typename Range2_t, typename BinaryOp>
inline CUDA_HOSTDEV auto
transform(Range1_t& rng1, const Range2_t& rng2, BinaryOp f) {
    return transform(zip(rng1, rng2), ApplyBinaryOp<BinaryOp>(f));
}

template <typename Range1_t, typename Range2_t, typename BinaryOp>
inline CUDA_HOSTDEV auto
transform(const Range1_t& rng1, Range2_t& rng2, BinaryOp f) {
    return transform(zip(rng1, rng2), ApplyBinaryOp<BinaryOp>(f));
}
*/

template <typename Range1_t, typename Range2_t, typename BinaryOp>
inline CUDA_HOSTDEV auto
md_transform(const Range1_t& rng1, const Range2_t& rng2, BinaryOp f) {
    return md_transform
    (
        make_md_zip_range(rng1, rng2), ApplyBinaryOp<BinaryOp>(f)
    );
}






} // namespace topaz