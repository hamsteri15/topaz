#pragma once

#include <cmath>
//#include <math.h>
#include "range.hpp"
#include "smart_transform.hpp"
#include "traits.hpp"

namespace topaz {

struct Plus {
    template <class T>
    inline CUDA_HOSTDEV auto operator()(const T& lhs, const T& rhs) const
        -> decltype(lhs + rhs) {
        return lhs + rhs;
    }
};

struct Minus {
    template <class T>
    inline CUDA_HOSTDEV auto operator()(const T& lhs, const T& rhs) const
        -> decltype(lhs - rhs) {
        return lhs - rhs;
    }
};

struct Multiplies {
    template <class T>
    inline CUDA_HOSTDEV auto operator()(const T& lhs, const T& rhs) const
        -> decltype(lhs * rhs) {
        return lhs * rhs;
    }
};

struct Divides {
    template <class T>
    inline CUDA_HOSTDEV auto operator()(const T& lhs, const T& rhs) const
        -> decltype(lhs / rhs) {
        return lhs / rhs;
    }
};

inline CUDA_HOSTDEV float  adl_sqrt(float s) { return sqrtf(s); }
inline CUDA_HOSTDEV double adl_sqrt(double s) { return sqrt(s); }
struct Sqrt {

    template <class T>
    inline CUDA_HOSTDEV auto operator()(const T& t) const
        -> decltype(adl_sqrt(t)) {
        return adl_sqrt(t);
    }
};

template <class T1,
          class T2,
          typename = std::enable_if_t<SupportsBinaryExpression_v<T1, T2>>>
inline CUDA_HOSTDEV auto operator+(const T1& lhs, const T2& rhs) {

    return smart_transform(lhs, rhs, Plus{});
}

template <class T1,
          class T2,
          typename = std::enable_if_t<SupportsBinaryExpression_v<T1, T2>>>
inline CUDA_HOSTDEV auto operator-(const T1& lhs, const T2& rhs) {

    return smart_transform(lhs, rhs, Minus{});
}

template <class T1,
          class T2,
          typename = std::enable_if_t<SupportsBinaryExpression_v<T1, T2>>>
inline CUDA_HOSTDEV auto operator*(const T1& lhs, const T2& rhs) {

    return smart_transform(lhs, rhs, Multiplies{});
}

template <class T1,
          class T2,
          typename = std::enable_if_t<SupportsBinaryExpression_v<T1, T2>>>
inline CUDA_HOSTDEV auto operator/(const T1& lhs, const T2& rhs) {

    return smart_transform(lhs, rhs, Divides{});
}

template <class T, typename = std::enable_if_t<IsRangeOrNumericArray_v<T>>>
inline CUDA_HOSTDEV auto sqr(const T& t) {
    return t * t;
}

template <class T, typename = std::enable_if_t<IsRangeOrNumericArray_v<T>>>
inline CUDA_HOSTDEV auto sqrt(const T& t) {
    return transform(t, Sqrt{});
}

} // namespace topaz