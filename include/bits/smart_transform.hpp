#pragma once

#include "constant_range.hpp"
#include "range.hpp"
#include "traits.hpp"
namespace topaz {

template <class T1,
          class T2,
          class BinaryOp,
          std::enable_if_t<AtleastOneIsRange_v<T1, T2>, bool> = true>
inline CUDA_HOSTDEV auto
smart_transform(const T1& lhs, const T2& rhs, BinaryOp f) {

    if constexpr (BothRangesOrNumericArrays_v<T1, T2>) {
        return transform(lhs, rhs, f);
    } else if constexpr (IsRangeOrNumericArray_v<T1>) {
        return transform(lhs, make_constant_range(rhs, adl_size(lhs)), f);
    } else {
        return transform(make_constant_range(lhs, adl_size(rhs)), rhs, f);
    }
}

} // namespace topaz