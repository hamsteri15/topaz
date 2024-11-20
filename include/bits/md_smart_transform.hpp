#pragma once

#include "md_traits.hpp"
#include "md_range.hpp"
#include "md_constant_range.hpp"
namespace topaz{



template <class T1, class T2>
inline CUDA_HOSTDEV auto md_determine_size(const T1&, const T2& rhs)
    -> std::enable_if_t<IsScalar_v<T1>, small_array<typename T2::difference_type>> {
    return md_size(rhs);
}
template <class T1, class T2>
inline CUDA_HOSTDEV auto md_determine_size(const T1& lhs, const T2&)
    -> std::enable_if_t<IsScalar_v<T2>, small_array<typename T1::difference_type>> {
    return md_size(lhs);
}
template <class T1,
          class T2,
          typename = std::enable_if_t<BothMdRangesOrMdNumericArrays_v<T1, T2>>>
inline CUDA_HOSTDEV auto md_determine_size(const T1& lhs, const T2&) {
    return md_size(lhs);
}



template <class T1, class T2>
inline CUDA_HOSTDEV auto md_determine_range_count(const T1&, const T2& rhs)
    -> std::enable_if_t<IsScalar_v<T1>, size_t> {
    return range_count(rhs);
}
template <class T1, class T2>
inline CUDA_HOSTDEV auto md_determine_range_count(const T1& lhs, const T2&)
    -> std::enable_if_t<IsScalar_v<T2>, size_t> {
    return range_count(lhs);
}
template <class T1,
          class T2,
          typename = std::enable_if_t<BothMdRangesOrMdNumericArrays_v<T1, T2>>>
inline CUDA_HOSTDEV auto md_determine_range_count(const T1& lhs, const T2&) {
    return range_count(lhs);
}






template <class MdRange_t,
          class Size,
          typename = std::enable_if_t<!IsScalar_v<MdRange_t>>>
inline CUDA_HOSTDEV auto md_rangify(MdRange_t& rng, const small_array<Size>& sizes, size_t count) {
    (void) sizes;
    (void) count;
    return make_md_range(rng);
    //return take(rng, n);
}

template <class MdRange_t,
          class Size,
          typename = std::enable_if_t<!IsScalar_v<MdRange_t>>>
inline CUDA_HOSTDEV auto md_rangify(const MdRange_t& rng, const small_array<Size>& sizes, size_t count) {
    (void) sizes;
    (void) count;
    return make_md_range(rng);
    //return take(rng, n);
}

template <class Scalar,
          class Size,
          std::enable_if_t<IsScalar_v<Scalar>, bool> = true>
inline CUDA_HOSTDEV auto md_rangify(const Scalar& s, const small_array<Size>& sizes, size_t count) {
    return make_constant_range<Scalar, Size>(s, count, sizes);
}


/*

template <class T1, class T2, class BinaryOp>
inline CUDA_HOSTDEV auto
md_smart_transform(const T1& lhs, const T2& rhs, BinaryOp f) {
    
    const auto size = md_determine_size(lhs, rhs);
    const auto count = md_determine_range_count(lhs, rhs);
    return md_transform(md_rangify(lhs, size, count), md_rangify(rhs, size, count), f);
}

*/

template <class T1,
          class T2,
          class BinaryOp,
          std::enable_if_t<AtleastOneIsMdRange_v<T1, T2>, bool> = true>
inline CUDA_HOSTDEV auto
smart_transform(const T1& lhs, const T2& rhs, BinaryOp f) {

    if constexpr (BothAreMdRanges_v<T1, T2>)
    {
        return transform(lhs, rhs, f); 
    }
    else if constexpr (IsMdRange_v<T1>)
    {
        auto count = range_count(lhs);
        auto sizes = md_size(lhs);
        return transform(lhs, make_constant_range(rhs, count, sizes), f);
    }
    else {
        auto count = range_count(rhs);
        auto sizes = md_size(rhs);
        return transform(make_constant_range(lhs, count, sizes), rhs, f); 
    }
    


}





}