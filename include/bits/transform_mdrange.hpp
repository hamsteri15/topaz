#pragma once

#include "begin_end.hpp"
#include "small_array.hpp"
#include "transform_range.hpp"

namespace topaz {

template <typename UnaryFunction, typename Iterator>
struct MdTransformRange
    : public MdRange<detail::transform_iterator<UnaryFunction, Iterator>> {



    using parent = MdRange<detail::transform_iterator<UnaryFunction, Iterator>>;

    /*

    inline CUDA_HOSTDEV
    MdTransformRange(Iterator first, Iterator last, UnaryFunction f)
        : parent(detail::make_transform_iterator(first, f),
                 detail::make_transform_iterator(last, f)) {}

    template <class MdRange_t>
    inline CUDA_HOSTDEV TransformRange(MdRange_t& rng, UnaryFunction f)
        : TransformRange(adl_begin(rng), adl_end(rng), f) {}

    template <class Range_t>
    inline CUDA_HOSTDEV TransformRange(const Range_t& rng, UnaryFunction f)
        : TransformRange(adl_begin(rng), adl_end(rng), f) {}
    */
};

template <typename Function, class MdRange_t>
inline CUDA_HOSTDEV auto make_md_transform_range(MdRange_t& rng, Function f) {
    using iterator = typename MdRnage_t::iterator;
    return MdTransformRange<Function, iterator>(rng, f);
}

template <typename Function, class Range_t>
inline CUDA_HOSTDEV auto make_md_transform_range(const MdRange_t& rng,
                                                 Function         f) {
    using iterator = typename MdRnage_t::iterator;
    return MdTransformRange<Function, iterator>(rng, f);
}

} // namespace topaz