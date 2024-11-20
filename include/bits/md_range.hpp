#pragma once

#include "begin_end.hpp"
#include "md_traits.hpp"
#include "small_array.hpp"
#include "range.hpp"

namespace topaz {

template <class Iterator>
struct MdRange {

    using iterator   = Iterator;
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    using reference  = typename std::iterator_traits<Iterator>::reference;
    using difference_type =
        typename std::iterator_traits<Iterator>::difference_type;
    inline CUDA_HOSTDEV MdRange(size_t                              count,
                                const small_array<Range<iterator>>& ranges)
        : m_count(count)
        , m_ranges(ranges) {}

    inline CUDA_HOSTDEV size_t size() const { return m_count; }

    Range<iterator>&       operator[](size_t i) { return m_ranges[i]; }
    const Range<iterator>& operator[](size_t i) const { return m_ranges[i]; }

    size_t                       m_count;
    small_array<Range<iterator>> m_ranges;
};

//These are the functions that are required to satisfy the IsMdRange type

template <typename Iterator>
CUDA_HOSTDEV size_t range_count(const MdRange<Iterator>& rng) {
    return rng.size();
}

template <typename Iterator>
CUDA_HOSTDEV auto md_begin(const MdRange<Iterator>& t) {

    small_array<Iterator> ret{};
    for (size_t i = 0; i < range_count(t); ++i) { ret[i] = adl_begin(t[i]); }
    return ret;
}

template <typename Iterator>
CUDA_HOSTDEV auto md_begin(MdRange<Iterator>& t) {

    small_array<Iterator> ret{};
    for (size_t i = 0; i < range_count(t); ++i) { ret[i] = adl_begin(t[i]); }
    return ret;
}

template <typename Iterator>
CUDA_HOSTDEV auto md_end(const MdRange<Iterator>& t) {

    small_array<Iterator> ret{};
    for (size_t i = 0; i < range_count(t); ++i) { ret[i] = adl_end(t[i]); }
    return ret;
}

template <typename Iterator>
CUDA_HOSTDEV auto md_end(MdRange<Iterator>& t) {

    small_array<Iterator> ret{};
    for (size_t i = 0; i < range_count(t); ++i) { ret[i] = adl_end(t[i]); }
    return ret;
}

/////////////////////



template <typename T, std::enable_if_t<IsMdRange_v<T>, bool> = true>
CUDA_HOSTDEV auto md_size(const T& t) {

    auto beg       = md_begin(t);
    auto end       = md_end(t);
    using iterator = typename decltype(beg)::value_type;
    using integer_type =
        typename std::iterator_traits<iterator>::difference_type;

    small_array<integer_type> ret{};
    for (size_t i = 0; i < range_count(t); ++i) {
        ret[i] = std::distance(beg[i], end[i]);
    }
    return ret;
}

/*
template <typename T, std::enable_if_t<IsMdRange_v<T>, bool> = true>
CUDA_HOSTDEV auto make_range(const T& t) {

    auto count = range_count(t);

    auto beg       = md_begin(t);
    auto end       = md_end(t);
    using iterator = typename decltype(beg)::value_type;

    small_array<Range<iterator>> ranges{};
    for (size_t i = 0; i < count; ++i) {
        ranges[i] = make_range(beg[i], end[i]);
    }
    return MdRange<iterator>(count, ranges);
}
*/

template <typename T, std::enable_if_t<IsMdRange_v<T>, bool> = true>
CUDA_HOSTDEV auto make_md_range(const T& t) {

    auto count = range_count(t);

    auto beg       = md_begin(t);
    auto end       = md_end(t);
    using iterator = typename decltype(beg)::value_type;

    small_array<Range<iterator>> ranges{};
    for (size_t i = 0; i < count; ++i) {
        ranges[i] = make_range(beg[i], end[i]);
    }
    return MdRange<iterator>(count, ranges);
}

template <typename T, std::enable_if_t<IsMdRange_v<T>, bool> = true>
CUDA_HOSTDEV auto make_md_range(T& t) {

    auto count = range_count(t);

    auto beg       = md_begin(t);
    auto end       = md_end(t);
    using iterator = typename decltype(beg)::value_type;

    small_array<Range<iterator>> ranges{};
    for (size_t i = 0; i < count; ++i) {
        ranges[i] = make_range(beg[i], end[i]);
    }
    return MdRange<iterator>(count, ranges);
}


} // namespace topaz