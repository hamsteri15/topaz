#pragma once

#include "begin_end.hpp"
#include "small_array.hpp"

namespace topaz {

template <class Iterator>
struct MdRange {

    using iterator   = Iterator;
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    using reference  = typename std::iterator_traits<Iterator>::reference;
    using difference_type = typename std::iterator_traits<Iterator>::difference_type;

    inline CUDA_HOSTDEV MdRange(size_t                       count,
                                const small_array<Range<iterator>>& ranges)
        : m_count(count)
        , m_ranges(ranges) {}

    auto CUDA_HOSTDEV begin() { return m_ranges.begin(); }
    auto CUDA_HOSTDEV begin() const { return m_ranges.begin(); }

    auto CUDA_HOSTDEV end() { return m_ranges.begin() + m_count; }
    auto CUDA_HOSTDEV end() const { return m_ranges.begin() + m_count; }

    inline CUDA_HOSTDEV size_t size() const { return m_count; }

    Range<iterator>&       operator[](size_t i) { return m_ranges[i]; }
    const Range<iterator>& operator[](size_t i) const { return m_ranges[i]; }

    size_t                       m_count;
    small_array<Range<iterator>> m_ranges;
};

template <typename T>
CUDA_HOSTDEV size_t range_count(const T& rng) {
    return rng.size();
}

template <typename T>
CUDA_HOSTDEV auto make_md_range(const T& t) {

    auto count = range_count(t);
    using iterator = decltype((*t.begin()).begin());
    small_array<Range<iterator>> ranges;
    for (size_t i = 0; i < count; ++i) { ranges[i] = make_range(t[i]); }
    return MdRange<iterator>(count, ranges);
}

template <typename T>
CUDA_HOSTDEV auto make_md_range(T& t) {

    auto count = range_count(t);
    using iterator = decltype((*t.begin()).begin());
    small_array<Range<iterator>> ranges;
    for (size_t i = 0; i < count; ++i) { ranges[i] = make_range(t[i]); }
    return MdRange<iterator>(count, ranges);
}

template <typename MdRange_t>
CUDA_HOSTDEV auto md_begin(const MdRange_t& t) {

    using iterator = decltype((*t.begin()).begin());
    small_array<iterator> ret{};
    for (size_t i = 0; i < range_count(t); ++i) {
        ret[i] = adl_begin(t[i]);
    }
    return ret;
}

template <typename MdRange_t>
CUDA_HOSTDEV auto md_begin(MdRange_t& t) {

    using iterator = decltype((*t.begin()).begin());
    small_array<iterator> ret{};
    for (size_t i = 0; i < range_count(t); ++i) {
        ret[i] = adl_begin(t[i]);
    }
    return ret;
}

template <typename MdRange_t>
CUDA_HOSTDEV auto md_end(const MdRange_t& t) {

    using iterator = decltype((*t.begin()).begin());
    small_array<iterator> ret{};
    for (size_t i = 0; i < range_count(t); ++i) {
        ret[i] = adl_end(t[i]);
    }
    return ret;
}

template <typename MdRange_t>
CUDA_HOSTDEV auto md_end(MdRange_t& t) {

    using iterator = decltype((*t.begin()).begin());
    small_array<iterator> ret{};
    for (size_t i = 0; i < range_count(t); ++i) {
        ret[i] = adl_end(t[i]);
    }
    return ret;
}

template <typename MdRange_t>
CUDA_HOSTDEV auto md_size(const MdRange_t& t) {

    using iterator = decltype((*t.begin()).begin());
    using integer_type = typename std::iterator_traits<iterator>::difference_type;
    small_array<integer_type> ret{};
    for (size_t i = 0; i < range_count(t); ++i) {
        ret[i] = adl_size(t[i]);
    }
    return ret;
}



} // namespace topaz