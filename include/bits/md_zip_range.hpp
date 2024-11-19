#pragma once

#include "md_range.hpp"

namespace topaz {

template <typename IteratorTuple>
struct MdZipRange : public MdRange<detail::zip_iterator<IteratorTuple>> {

    using iterator = typename detail::zip_iterator<IteratorTuple>;
    using parent   = MdRange<iterator>;

    MdZipRange(const small_array<IteratorTuple>& firsts,
               const small_array<IteratorTuple>& lasts,
               size_t                            count)
        : parent(count, combine(firsts, lasts, count)) {}

private:
    static inline auto combine(const small_array<IteratorTuple>& firsts,
                               const small_array<IteratorTuple>& lasts,
                               size_t                            count) {

        small_array<Range<iterator>> ret{};

        for (size_t i = 0; i < count; ++i) {
            ret[i] = Range<iterator>(firsts[i], lasts[i]);
        }
        return ret;
    }
};

template <typename MdRange1_t, typename MdRange2_t>
inline CUDA_HOSTDEV auto make_md_zip_range(MdRange1_t& rng1, MdRange2_t& rng2) {

    auto firsts1 = md_begin(rng1);
    auto firsts2 = md_begin(rng2);

    auto lasts1 = md_end(rng1);
    auto lasts2 = md_end(rng2);

    using iter1 = typename decltype(firsts1)::value_type;
    using iter2 = typename decltype(firsts2)::value_type;

    using tuple_t = Tuple<iter1, iter2>;

    auto count = range_count(rng1);

    small_array<tuple_t> firsts{};
    small_array<tuple_t> lasts{};

    for (size_t i = 0; i < count; ++i) {
        firsts[i] = adl_make_tuple(firsts1[i], firsts2[i]);
        lasts[i]  = adl_make_tuple(lasts1[i], lasts2[i]);
    }

    return MdZipRange<tuple_t>(firsts, lasts, count);
}

template <typename MdRange1_t, typename MdRange2_t>
inline CUDA_HOSTDEV auto make_md_zip_range(MdRange1_t&       rng1,
                                           const MdRange2_t& rng2) {

    auto firsts1 = md_begin(rng1);
    auto firsts2 = md_begin(rng2);

    auto lasts1 = md_end(rng1);
    auto lasts2 = md_end(rng2);

    using iter1 = typename decltype(firsts1)::value_type;
    using iter2 = typename decltype(firsts2)::value_type;

    using tuple_t = Tuple<iter1, iter2>;

    auto count = range_count(rng1);

    small_array<tuple_t> firsts{};
    small_array<tuple_t> lasts{};

    for (size_t i = 0; i < count; ++i) {
        firsts[i] = adl_make_tuple(firsts1[i], firsts2[i]);
        lasts[i]  = adl_make_tuple(lasts1[i], lasts2[i]);
    }

    return MdZipRange<tuple_t>(firsts, lasts, count);
}

template <typename MdRange1_t, typename MdRange2_t>
inline CUDA_HOSTDEV auto make_md_zip_range(const MdRange1_t& rng1,
                                           MdRange2_t&       rng2) {

    auto firsts1 = md_begin(rng1);
    auto firsts2 = md_begin(rng2);

    auto lasts1 = md_end(rng1);
    auto lasts2 = md_end(rng2);

    using iter1 = typename decltype(firsts1)::value_type;
    using iter2 = typename decltype(firsts2)::value_type;

    using tuple_t = Tuple<iter1, iter2>;

    auto count = range_count(rng1);

    small_array<tuple_t> firsts{};
    small_array<tuple_t> lasts{};

    for (size_t i = 0; i < count; ++i) {
        firsts[i] = adl_make_tuple(firsts1[i], firsts2[i]);
        lasts[i]  = adl_make_tuple(lasts1[i], lasts2[i]);
    }

    return MdZipRange<tuple_t>(firsts, lasts, count);
}

template <typename MdRange1_t, typename MdRange2_t>
inline CUDA_HOSTDEV auto make_md_zip_range(const MdRange1_t& rng1,
                                           const MdRange2_t& rng2) {

    auto firsts1 = md_begin(rng1);
    auto firsts2 = md_begin(rng2);

    auto lasts1 = md_end(rng1);
    auto lasts2 = md_end(rng2);

    using iter1 = typename decltype(firsts1)::value_type;
    using iter2 = typename decltype(firsts2)::value_type;

    using tuple_t = Tuple<iter1, iter2>;

    auto count = range_count(rng1);

    small_array<tuple_t> firsts{};
    small_array<tuple_t> lasts{};

    for (size_t i = 0; i < count; ++i) {
        firsts[i] = adl_make_tuple(firsts1[i], firsts2[i]);
        lasts[i]  = adl_make_tuple(lasts1[i], lasts2[i]);
    }

    return MdZipRange<tuple_t>(firsts, lasts, count);
}

} // namespace topaz