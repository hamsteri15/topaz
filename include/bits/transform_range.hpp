#pragma once

#include "range.hpp"
#include <iterator>
#include <type_traits>

#ifdef __NVIDIA_COMPILER__
#include <thrust/iterator/transform_iterator.h>
#endif
namespace topaz {

namespace detail {

#ifdef __NVIDIA_COMPILER__

template <class Func, class Iter>
using transform_iterator = thrust::transform_iterator<Func, Iter>;

template <class Func, class Iter>
inline CUDA_HOSTDEV auto make_transform_iterator(Iter it, Func f) {
    return thrust::make_transform_iterator(it, f);
}

#else

template <class Func, class Iter>
class transform_iterator {

public:
    Iter m_it;
    Func m_func;

    inline CUDA_HOSTDEV transform_iterator(Iter it, Func func)
        : m_it(it)
        , m_func(func) {}

    using difference_type =
        typename std::iterator_traits<Iter>::difference_type;
    using reference  = typename std::result_of<Func(
        typename std::iterator_traits<Iter>::reference)>::type;
    using pointer    = void;
    using value_type = reference;
    // using iterator_category = std::input_iterator_tag;
    using iterator_category = std::random_access_iterator_tag;

    using my_type = transform_iterator<Func, Iter>;


    inline CUDA_HOSTDEV reference dereference() const {
        return m_func(*m_it);
    }


    inline CUDA_HOSTDEV bool operator==(const my_type& rhs) const {
        return m_it == rhs.m_it;
    }
    inline CUDA_HOSTDEV bool operator!=(const my_type& rhs) const {
        return m_it != rhs.m_it;
    }

    inline CUDA_HOSTDEV bool operator<(const my_type& rhs) const {
        return m_it < rhs.m_it;
    }
    inline CUDA_HOSTDEV bool operator<=(const my_type& rhs) const {
        return m_it <= rhs.m_it;
    }
    inline CUDA_HOSTDEV bool operator>(const my_type& rhs) const {
        return m_it > rhs.m_it;
    }
    inline CUDA_HOSTDEV bool operator>=(const my_type& rhs) const {
        return m_it >= rhs.m_it;
    }

    inline CUDA_HOSTDEV difference_type operator-(const my_type& rhs) const {
        return m_it - rhs.m_it;
    }

    inline CUDA_HOSTDEV difference_type operator+(const my_type& rhs) const {
        return m_it + rhs.m_it;
    }

    inline CUDA_HOSTDEV my_type operator+(difference_type i) const {
        return my_type(m_it + i, m_func);
    }
    inline CUDA_HOSTDEV my_type operator-(difference_type i) const {
        return my_type(m_it - i, m_func);
    }

    inline CUDA_HOSTDEV my_type& operator+=(difference_type i) {
        m_it = m_it + i;
        return *this;
    }
    inline CUDA_HOSTDEV my_type& operator-=(difference_type i) {
        m_it = m_it - i;
        return *this;
    }


    inline CUDA_HOSTDEV auto operator[](difference_type i) const {
        return m_func(m_it[i]);
    }


    // auto& operator[](difference_type i) { return m_func(m_it[i]); }

    inline CUDA_HOSTDEV auto operator*() const { return dereference(); }

    inline CUDA_HOSTDEV my_type& operator++() {
        ++m_it;
        return *this;
    }
    inline CUDA_HOSTDEV my_type& operator--() {
        --m_it;
        return *this;
    }
    inline CUDA_HOSTDEV my_type& operator++(int) {
        auto prev = *this;
        ++m_it;
        return prev;
    }
};

template <class Func, class Iter>
inline CUDA_HOSTDEV transform_iterator<Func, Iter>
                    make_transform_iterator(Iter it, Func f) {
    return transform_iterator<Func, Iter>(it, f);
}

#endif
} // namespace detail

template <typename UnaryFunction, typename Iterator>
struct TransformRange
    : public Range<detail::transform_iterator<UnaryFunction, Iterator>> {

    using parent = Range<detail::transform_iterator<UnaryFunction, Iterator>>;

    inline CUDA_HOSTDEV
    TransformRange(Iterator first, Iterator last, UnaryFunction f)
        : parent(detail::make_transform_iterator(first, f),
                 detail::make_transform_iterator(last, f)) {}

    template <class Range_t>
    inline CUDA_HOSTDEV TransformRange(Range_t& rng, UnaryFunction f)
        : TransformRange(adl_begin(rng), adl_end(rng), f) {}

    template <class Range_t>
    inline CUDA_HOSTDEV TransformRange(const Range_t& rng, UnaryFunction f)
        : TransformRange(adl_begin(rng), adl_end(rng), f) {}
};

template <typename Function, typename Iterator>
inline CUDA_HOSTDEV auto
make_transform_range(Iterator first, Iterator last, Function f) {
    return TransformRange<Function, Iterator>(first, last, f);
}

template <typename Function, class Range_t>
inline CUDA_HOSTDEV auto make_transform_range(Range_t& rng, Function f) {
    using iterator = decltype(std::begin(rng));
    return TransformRange<Function, iterator>(rng, f);
}

template <typename Function, class Range_t>
inline CUDA_HOSTDEV auto make_transform_range(const Range_t& rng, Function f) {
    using iterator = decltype(std::begin(rng));
    return TransformRange<Function, iterator>(rng, f);
}

} // namespace topaz