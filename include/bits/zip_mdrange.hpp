#pragma once

#include "mdrange.hpp"

namespace topaz {

template <typename IteratorTuple>
struct MdZipRange : public MdRange<detail::zip_iterator<IteratorTuple>>{

    using parent = MdRange<detail::zip_iterator<IteratorTuple>>;

    MdZipRange(small_array<IteratorTuple> firsts,
               small_array<IteratorTuple> lasts,
               size_t                     count)
        : parent(count, combine(firsts, lasts, count)) {}

private:
    static inline auto combine(small_array<IteratorTuple> firsts,
                               small_array<IteratorTuple> lasts,
                               size_t                     count) {

        small_array<Range<detail::zip_iterator<IteratorTuple>>> ret;

        for (size_t i = 0; i < count; ++i) {
            ret[i] = Range<detail::zip_iterator<IteratorTuple>>(firsts[i], lasts[i]);
        }
        return ret;
    }
};

template <typename MdRange1_t, typename MdRange2_t>
inline CUDA_HOSTDEV auto make_md_zip_range(MdRange1_t& rng1, MdRange2_t& rng2) {

    auto md1 = make_md_range(rng1);
    auto md2 = make_md_range(rng2);

    using iter1   = typename decltype(md1)::iterator;
    using iter2   = typename decltype(md2)::iterator;
    using tuple_t = Tuple<iter1, iter2>;

    auto firsts1 = md_begin(md1);
    auto firsts2 = md_begin(md2);

    auto lasts1 = md_end(md1);
    auto lasts2 = md_end(md2);

    auto count = range_count(rng1);

    small_array<tuple_t> firsts;
    small_array<tuple_t> lasts;

    for (size_t i = 0; i < count; ++i) {
        firsts[i] = adl_make_tuple(firsts1[i], firsts2[i]);
        lasts[i]  = adl_make_tuple(lasts1[i], lasts2[i]);
    }

    return MdZipRange<tuple_t>(firsts, lasts, count);
}

} // namespace topaz