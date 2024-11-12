#pragma once

#include "transform_range.hpp"

#include <iterator>
#include <tuple>
#include <utility>


namespace topaz {

namespace detail {

template <typename T, size_t... I>
inline CUDA_HOSTDEV constexpr auto
deref_tuple_elements(const T& t, std::index_sequence<I...>) {
    // Note! This functions always returns a tuple of values and makes the
    // zip_iterator
    //  not usable as when dereferenced on the lhs of = .
    // TODO: Check std::tie, std::forward_as_tuple and std::get and use the
    // appropriate
    //  combination based on what the dereference operator of the underlying
    //  iterator returns. Also std::reference_wrapper may be useful.
    auto f = [](auto it) { return *it; };

    return std::make_tuple(f(std::get<I>(t))...);
}

template <typename DiffType, typename T, size_t... I>
inline CUDA_HOSTDEV constexpr auto
increment_by(DiffType inc, T& t, std::index_sequence<I...>) {

    auto f = [=](auto& it) { return it + inc; };
    return std::make_tuple(f(std::get<I>(t))...);

    //return std::make_tuple(f(std::get<I>(t))...);
}


} // namespace detail




template <typename T>
inline CUDA_HOSTDEV constexpr auto deref_tuple_elements(const T& t) {
    return detail::deref_tuple_elements<T>(
        t, std::make_index_sequence<std::tuple_size<T>::value>{});
}

//TODO: This function should not return anything
template <typename DiffType, typename T>
constexpr auto increment_by(DiffType inc, T& t) {

    //In c++17
    //std::apply([=](auto&... args) { ((args += inc), ...); }, t);
    return
    detail::increment_by<DiffType, T>
    (
        inc, t, std::make_index_sequence<std::tuple_size<T>::value>{}
    );

}

template <class Iterator>
struct iterator_reference
{
    typedef typename std::iterator_traits<Iterator>::reference type;
};

template <class Iterator>
struct iterator_value_type
{
    typedef typename std::iterator_traits<Iterator>::value_type type;
};

template <class T>
struct DeduceReference {};

template <class... T>
struct DeduceReference<std::tuple<T...>> {
    using type = std::tuple<typename iterator_reference<T>::type...>;
};

template <class T>
struct DeduceValue {};

template <class... T>
struct DeduceValue<std::tuple<T...>> {
    using type = std::tuple<typename iterator_value_type<T>::type...>;
};


template <class IteratorTuple>
class zip_iterator {

public:
    // TODO: use iterator_traits
    // using difference_type =
    //    decltype(std::distance(std::get<0>(m_tuple), std::get<0>(m_tuple)));
    // typename std::iterator_traits<first_type>::difference_type;

    //using reference = typename DeduceReference<IteratorTuple>::type;
    using reference = typename DeduceValue<IteratorTuple>::type; //TODO: This is wrong
    using value_type        = typename DeduceValue<IteratorTuple>::type;
    using difference_type   = std::ptrdiff_t;
    using pointer           = void;
    using iterator_category = std::random_access_iterator_tag;

    IteratorTuple m_tuple;


    using my_type = zip_iterator<IteratorTuple>;

    const IteratorTuple& get_tuple() const { return m_tuple; }

public:
    zip_iterator(IteratorTuple tuple)
        : m_tuple(tuple) {}

    inline CUDA_HOSTDEV reference dereference() const { return deref_tuple_elements(get_tuple()); }

    inline CUDA_HOSTDEV void advance(difference_type i) {
        // increment_by(i, get_tuple());
        m_tuple = increment_by(i, m_tuple);
    }

    inline CUDA_HOSTDEV void increment() { this->advance(1); }

    inline CUDA_HOSTDEV void decrement() { this->advance(-1); }

    inline CUDA_HOSTDEV bool operator==(const my_type& rhs) const {
        return std::get<0>(m_tuple) == std::get<0>(rhs.m_tuple);
    }
    inline CUDA_HOSTDEV bool operator!=(const my_type& rhs) const {
        return std::get<0>(m_tuple) != std::get<0>(rhs.m_tuple);
    }

    inline CUDA_HOSTDEV bool operator<(const my_type& rhs) const {
        return std::get<0>(m_tuple) < std::get<0>(rhs.m_tuple);
    }
    inline CUDA_HOSTDEV bool operator<=(const my_type& rhs) const {
        return std::get<0>(m_tuple) <= std::get<0>(rhs.m_tuple);
    }
    inline CUDA_HOSTDEV bool operator>(const my_type& rhs) const {
        return std::get<0>(m_tuple) > std::get<0>(rhs.m_tuple);
    }
    inline CUDA_HOSTDEV bool operator>=(const my_type& rhs) const {
        return std::get<0>(m_tuple) >= std::get<0>(rhs.m_tuple);
    }

    inline CUDA_HOSTDEV difference_type operator-(const my_type& rhs) const {
        return std::get<0>(m_tuple) - std::get<0>(rhs.m_tuple);
    }

    inline CUDA_HOSTDEV difference_type operator+(const my_type& rhs) const {
        return std::get<0>(m_tuple) + std::get<0>(rhs.m_tuple);
    }

    inline CUDA_HOSTDEV my_type& operator+=(difference_type i) {
        this->advance(i);
        return *this;
    }
    inline CUDA_HOSTDEV my_type& operator-=(difference_type i) {
        this->advance(-i);
        return *this;
    }

    inline CUDA_HOSTDEV my_type operator+(difference_type i) const {
        auto copy = zip_iterator(get_tuple());
        copy.advance(i);
        return copy;
    }

    inline CUDA_HOSTDEV my_type operator-(difference_type i) const {
        auto copy = zip_iterator(get_tuple());
        copy.advance(-i);
        return copy;
    }

    inline CUDA_HOSTDEV reference operator[](difference_type i) const {
        return (my_type(get_tuple()) + i).dereference();
    }

    inline CUDA_HOSTDEV auto operator*() const { return dereference(); }

    inline CUDA_HOSTDEV my_type& operator++() {
        this->increment();
        return *this;
    }
    inline CUDA_HOSTDEV my_type& operator--() {
        this->decrement();
        return *this;
    }

    /*
    inline CUDA_HOSTDEV my_type& operator++(int) {
        auto prev = *this;
        ++(*this);
        return prev;
    }
    */
};

template <class IteratorTuple>
inline CUDA_HOSTDEV zip_iterator<IteratorTuple>
                    make_zip_iterator(IteratorTuple tuple) {
    return zip_iterator<IteratorTuple>(tuple);
}

} // namespace topaz
