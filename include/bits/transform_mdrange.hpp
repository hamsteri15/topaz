#pragma once

#include "begin_end.hpp"
#include "small_array.hpp"
#include "transform_range.hpp"

namespace topaz {

template <typename Function, class MdRange_t>
inline CUDA_HOSTDEV auto make_md_transform_ranges(const MdRange_t& rng,
                                                  Function         f) {

    using org_iterator = typename MdRange_t::iterator;
    using new_iterator = detail::transform_iterator<Function, org_iterator>;
    using trange       = Range<new_iterator>;
    small_array<trange> ret{};
    for (size_t i = 0; i < range_count(rng); ++i) {
        ret[i] = make_transform_range(rng.m_ranges[i], f);
    }
    return ret;
}

template <typename Function, class MdRange_t>
inline CUDA_HOSTDEV auto make_md_transform_ranges(MdRange_t& rng, Function f) {

    using org_iterator = typename MdRange_t::iterator;
    using new_iterator = detail::transform_iterator<Function, org_iterator>;
    using trange       = Range<new_iterator>;
    small_array<trange> ret{};

    for (size_t i = 0; i < range_count(rng); ++i) {
        ret[i] = make_transform_range(rng.m_ranges[i], f);
    }
    return ret;
}

template <typename UnaryFunction, typename Iterator>
struct MdTransformRange
    : public MdRange<detail::transform_iterator<UnaryFunction, Iterator>> {

    using parent = MdRange<detail::transform_iterator<UnaryFunction, Iterator>>;
    using iterator = typename parent::iterator;

    MdTransformRange() = default;

    template <class MdRange_t>
    inline CUDA_HOSTDEV MdTransformRange(MdRange_t& rng, UnaryFunction f)
        : parent(range_count(rng), make_md_transform_ranges(rng, f)) {}

    template <class MdRange_t>
    inline CUDA_HOSTDEV MdTransformRange(const MdRange_t& rng, UnaryFunction f)
        : parent(range_count(rng), make_md_transform_ranges(rng, f)) {}
};

template <typename Function, class MdRange_t>
inline CUDA_HOSTDEV auto make_md_transform_range(MdRange_t& rng, Function f) {

    auto md = make_md_range(rng);
    using iterator = typename decltype(md)::iterator;
    return MdTransformRange<Function, iterator>{md, f};
}

template <typename Function, class MdRange_t>
inline CUDA_HOSTDEV auto make_md_transform_range(const MdRange_t& rng,
                                                 Function         f) {
    auto md = make_md_range(rng);
    using iterator = typename decltype(md)::iterator;
    return MdTransformRange<Function, iterator>{md, f};
}

} // namespace topaz