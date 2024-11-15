#pragma once

#include "begin_end.hpp"
#include "small_array.hpp"

namespace topaz {

template <class Iterator>
struct MdRange {

    using iterator   = Iterator;
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    using reference  = typename std::iterator_traits<Iterator>::reference;

    inline CUDA_HOSTDEV MdRange(size_t                              count,
                                const small_array<Range<iterator>>& ranges)
        : m_count(count)
        , m_ranges(ranges) {}

    // using difference_type = typename
    // std::iterator_traits<Iterator>::difference_type;

    /*
    inline CUDA_HOSTDEV auto begin() const { m_begin.begin(); }
    inline CUDA_HOSTDEV auto begin() { return m_begin.begin(); }

    inline CUDA_HOSTDEV auto end() const { return begin() + m_count; }
    inline CUDA_HOSTDEV auto end() { return begin() + m_count; }

    */

    inline CUDA_HOSTDEV size_t size() const { return m_count; }

    size_t                       m_count;
    small_array<Range<iterator>> m_ranges;




};

/*
template<typename T>
struct MdRangeIterator{

    using iterator =


};
*/

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

/*
template <typename Iterator>
CUDA_HOSTDEV small_array<Iterator> md_begin(const MdRange<Iterator>& rng) {

    small_array<Iterator> begins{};
    for (size_t i = 0; i < range_count(rng); ++i)
    {
        begins[i] = rng.m_ranges[i];
    }

}

template <typename Iterator>
CUDA_HOSTDEV small_array<Iterator> md_ends(const MdRange<Iterator>& rng) {

    small_array<Iterator> begins{};
    for (size_t i = 0; i < range_count(rng); ++i)
    {
        begins[i] = rng.m_ranges[i];
    }

}
*/

} // namespace topaz