#pragma once

#include "range.hpp"
#include "traits.hpp"
#include "smart_transform.hpp"

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

} // namespace topaz