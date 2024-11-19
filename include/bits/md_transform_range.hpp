#pragma once

#include "begin_end.hpp"
#include "md_transform_range.hpp"
#include "small_array.hpp"

namespace topaz {

template <typename UnaryFunction, typename Iterator>
struct MdTransformRange
    : public MdRange<detail::transform_iterator<UnaryFunction, Iterator>> {

    using iterator = detail::transform_iterator<UnaryFunction, Iterator>;
    using parent   = MdRange<iterator>;

    inline CUDA_HOSTDEV
    MdTransformRange(const small_array<Iterator>& begin_iters,
                     const small_array<Iterator>& end_iters,
                     UnaryFunction                f,
                     size_t                       count)
        : parent(count, combine(begin_iters, end_iters, f, count)) {}

private:
    static inline small_array<Range<iterator>>
    combine(const small_array<Iterator>& begin_iters,
            const small_array<Iterator>& end_iters,
            UnaryFunction                f,
            size_t                       count) {

        small_array<Range<iterator>> ret{};
        for (size_t i = 0; i < count; ++i) {
            ret[i] = make_transform_range(begin_iters[i], end_iters[i], f);
        }

        return ret;
    }
};

template <typename Function, class MdRange_t>
inline CUDA_HOSTDEV auto make_md_transform_range(MdRange_t& rng, Function f) {

    using iterator = typename decltype(make_md_range(rng))::iterator;

    return MdTransformRange<Function, iterator>{
        md_begin(rng), md_end(rng), f, range_count(rng)};
}

template <typename Function, class MdRange_t>
inline CUDA_HOSTDEV auto make_md_transform_range(const MdRange_t& rng,
                                                 Function         f) {
    using iterator = typename decltype(make_md_range(rng))::iterator;

    return MdTransformRange<Function, iterator>{
        md_begin(rng), md_end(rng), f, range_count(rng)};
}

} // namespace topaz