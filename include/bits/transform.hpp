#pragma once

#include "zip.hpp"

namespace topaz {

namespace detail {

template <typename BinaryOp>
struct ApplyBinaryOp {

    BinaryOp op;

    inline CUDA_HOSTDEV ApplyBinaryOp() = default;

    inline CUDA_HOSTDEV ApplyBinaryOp(BinaryOp f)
        : op(f) {}

    template <typename Tuple>
    inline CUDA_HOSTDEV auto operator()(const Tuple& t) const {

        return op(get<0>(t), get<1>(t));
    }
};

struct Transform {

    template <class Range_t, class UnaryOp>
    inline CUDA_HOSTDEV auto operator()(Range_t& rng, UnaryOp f) const {
        return make_transform_range(rng, f);
    }

    template <class Range_t, class UnaryOp>
    inline CUDA_HOSTDEV auto operator()(const Range_t& rng, UnaryOp f) const {
        return make_transform_range(rng, f);
    }

    template <class Range1_t, class Range2_t, class BinaryOp>
    inline CUDA_HOSTDEV auto
    operator()(Range1_t& rng1, Range2_t& rng2, BinaryOp f) const {
        return make_transform_range(zip(rng1, rng2),
                                    ApplyBinaryOp<BinaryOp>(f));
    }

    template <class Range1_t, class Range2_t, class BinaryOp>
    inline CUDA_HOSTDEV auto
    operator()(const Range1_t& rng1, Range2_t& rng2, BinaryOp f) const {
        return make_transform_range(zip(rng1, rng2),
                                    ApplyBinaryOp<BinaryOp>(f));
    }

    template <class Range1_t, class Range2_t, class BinaryOp>
    inline CUDA_HOSTDEV auto
    operator()(Range1_t& rng1, const Range2_t& rng2, BinaryOp f) const {
        return make_transform_range(zip(rng1, rng2),
                                    ApplyBinaryOp<BinaryOp>(f));
    }

    template <class Range1_t, class Range2_t, class BinaryOp>
    inline CUDA_HOSTDEV auto
    operator()(const Range1_t& rng1, const Range2_t& rng2, BinaryOp f) const {
        return make_transform_range(zip(rng1, rng2),
                                    ApplyBinaryOp<BinaryOp>(f));
    }
};

} // namespace detail

//This is the niebloid we expose in topaz to allow for customization of make_transform_range and zip
//inline constexpr detail::Transform transform{};
inline constexpr const auto transform = detail::Transform{};

/*
template <class Range_t, typename UnaryOp>
inline CUDA_HOSTDEV auto transform(Range_t& rng, UnaryOp f) {
    return make_transform_range(rng, f);
}

template <class Range_t, typename UnaryOp>
inline CUDA_HOSTDEV auto transform(const Range_t& rng, UnaryOp f) {
    return make_transform_range(rng, f);
}

template <typename BinaryOp>
struct ApplyBinaryOp {

    BinaryOp op;

    inline CUDA_HOSTDEV ApplyBinaryOp() = default;

    inline CUDA_HOSTDEV ApplyBinaryOp(BinaryOp f)
        : op(f) {}

    template <typename Tuple>
    inline CUDA_HOSTDEV auto operator()(const Tuple& t) const {

        return op(get<0>(t), get<1>(t));
    }
};

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

template <typename Range1_t, typename Range2_t, typename BinaryOp>
inline CUDA_HOSTDEV auto
transform(const Range1_t& rng1, const Range2_t& rng2, BinaryOp f) {
    return transform(zip(rng1, rng2), ApplyBinaryOp<BinaryOp>(f));
}
*/

} // namespace topaz