#pragma once

#include "traits.hpp"
#include "md_transform_range.hpp"
#include "md_zip.hpp"

namespace topaz {

template <class MdRange_t, typename UnaryOp>
inline CUDA_HOSTDEV auto md_transform(MdRange_t& rng, UnaryOp f) {
    return make_md_transform_range(rng, f);
}

template <class MdRange_t, typename UnaryOp>
inline CUDA_HOSTDEV auto md_transform(const MdRange_t& rng, UnaryOp f) {
    return make_md_transform_range(rng, f);
}


template <typename MdRange1_t, typename MdRange2_t, typename BinaryOp>
inline CUDA_HOSTDEV auto md_transform(MdRange1_t& rng1, MdRange2_t& rng2, BinaryOp f) {
    return md_transform(md_zip(rng1, rng2), ApplyBinaryOp<BinaryOp>(f));
}

template <typename MdRange1_t, typename MdRange2_t, typename BinaryOp>
inline CUDA_HOSTDEV auto
md_transform(MdRange1_t& rng1, const MdRange2_t& rng2, BinaryOp f) {
    return md_transform(md_zip(rng1, rng2), ApplyBinaryOp<BinaryOp>(f));
}

template <typename MdRange1_t, typename MdRange2_t, typename BinaryOp>
inline CUDA_HOSTDEV auto
md_transform(const MdRange1_t& rng1, MdRange2_t& rng2, BinaryOp f) {
    return md_transform(md_zip(rng1, rng2), ApplyBinaryOp<BinaryOp>(f));
}

template <typename MdRange1_t, typename MdRange2_t, typename BinaryOp>
inline CUDA_HOSTDEV auto
md_transform(const MdRange1_t& rng1, const MdRange2_t& rng2, BinaryOp f) {
    return md_transform(md_zip(rng1, rng2), ApplyBinaryOp<BinaryOp>(f));
}






} // namespace topaz