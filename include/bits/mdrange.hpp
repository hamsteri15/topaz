#pragma once

#include "begin_end.hpp"
#include "small_array.hpp"

namespace topaz {


template<class MdArray>
struct DeduceInnerIterator{

};




template <class Iterator>
struct MdRange {

    using iterator   = Iterator;
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    using reference  = typename std::iterator_traits<Iterator>::reference;

    inline CUDA_HOSTDEV MdRange(size_t                       count,
                                small_array<Range<iterator>> ranges)
        : m_count(count)
        , m_ranges(ranges) {}


    auto begin() {return m_ranges.begin();}
    auto begin() const {return m_ranges.begin();}

    auto end() {return m_ranges.begin() + m_count;}
    auto end() const {return m_ranges.begin() + m_count;}

    inline CUDA_HOSTDEV size_t size() const { return m_count; }

    Range<iterator>& operator[](size_t i) {return m_ranges[i];}
    const Range<iterator>& operator[](size_t i) const {return m_ranges[i];}


    size_t                       m_count;
    small_array<Range<iterator>> m_ranges;
};


template<class T>
struct DeduceInnerIterator<std::vector<std::vector<T>>>{

    using iterator = typename std::vector<T>::iterator;

};

template<class T>
struct DeduceInnerIterator<const std::vector<std::vector<T>>>{

    using iterator = typename std::vector<T>::const_iterator;

};

template<class Iterator>
struct DeduceInnerIterator<MdRange<Iterator>>{

    using iterator = Iterator;

};

template<class Iterator>
struct DeduceInnerIterator<const MdRange<Iterator>>{

    using iterator = const Iterator;

};




template <typename T>
CUDA_HOSTDEV size_t range_count(const T& rng) {
    return rng.size();
}

template <typename T>
CUDA_HOSTDEV auto make_md_range(const T& t) {

    auto count = range_count(t);

    using iterator = decltype((*t.begin()).begin());

    //using iterator = typename DeduceInnerIterator<T>::iterator;
    small_array<Range<iterator>> ranges;

    for (size_t i = 0; i < count; ++i) { ranges[i] = make_range(t[i]); }

    return MdRange<iterator>(count, ranges);
}

template <typename T>
CUDA_HOSTDEV auto make_md_range(T& t) {

    auto count = range_count(t);

    //using iterator = typename DeduceInnerIterator<T>::iterator;

    using iterator = decltype((*t.begin()).begin());

    small_array<Range<iterator>> ranges;

    for (size_t i = 0; i < count; ++i) { ranges[i] = make_range(t[i]); }

    return MdRange<iterator>(count, ranges);
}

template <typename Iterator>
CUDA_HOSTDEV small_array<Iterator> md_begin(const MdRange<Iterator>& rng) {

    small_array<Iterator> begins{};
    for (size_t i = 0; i < range_count(rng); ++i) {
        begins[i] = adl_begin(rng.m_ranges[i]);
    }
    return begins;
}

template <typename Iterator>
CUDA_HOSTDEV small_array<Iterator> md_begin(MdRange<Iterator>& rng) {

    small_array<Iterator> begins{};
    for (size_t i = 0; i < range_count(rng); ++i) {
        begins[i] = adl_begin(rng.m_ranges[i]);
    }
    return begins;
}

template <typename Iterator>
CUDA_HOSTDEV small_array<Iterator> md_end(const MdRange<Iterator>& rng) {

    small_array<Iterator> ends{};
    for (size_t i = 0; i < range_count(rng); ++i) {
        ends[i] = adl_end(rng.m_ranges[i]);
    }
    return ends;
}

template <typename Iterator>
CUDA_HOSTDEV small_array<Iterator> md_end(MdRange<Iterator>& rng) {

    small_array<Iterator> ends{};
    for (size_t i = 0; i < range_count(rng); ++i) {
        ends[i] = adl_end(rng.m_ranges[i]);
    }
    return ends;
}

} // namespace topaz