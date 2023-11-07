#pragma once

#include <iterator>

#ifdef __NVIDIA_COMPILER__
#include <thrust/iterator/constant_iterator.h>
#endif

namespace topaz {

#ifdef __NVIDIA_COMPILER__

template <class T>
using constant_iterator = thrust::constant_iterator<T>;

#else

template <class T>
class constant_iterator {
public:
    using iter = constant_iterator<T>;

    using value_type = T;
    using reference  = T; //No idea why this has to be a T instead of T&

    using difference_type   = std::ptrdiff_t;
    using pointer           = T*;
    using iterator_category = std::random_access_iterator_tag;

    inline CUDA_HOSTDEV constant_iterator(const T&        value,
                                          difference_type index = 0)
        : m_value(value)
        , m_index(index) {}

private:
    inline CUDA_HOSTDEV const T& dereference() const { return m_value; }
    inline CUDA_HOSTDEV T&       dereference() { return m_value; }

    inline CUDA_HOSTDEV bool equal(const constant_iterator<T>& other) const {
        return m_value == other.m_value && m_index == other.m_index;
    }

    inline CUDA_HOSTDEV void increment() { m_index++; }

    inline CUDA_HOSTDEV void decrement() { m_index--; }

    inline CUDA_HOSTDEV void advance(difference_type n) { m_index += n; }

    inline CUDA_HOSTDEV difference_type
    distance_to(const constant_iterator<T>& other) const {
        return static_cast<difference_type>(other.m_index - m_index);
    }

public:
    inline CUDA_HOSTDEV const T&  operator*() const { return dereference(); }
    inline CUDA_HOSTDEV T&        operator*() { return dereference(); }
    inline CUDA_HOSTDEV pointer   operator->() const { return &m_value; }
    inline CUDA_HOSTDEV T& operator[](difference_type) {
        return dereference();
    }

    inline CUDA_HOSTDEV const T& operator[](difference_type) const {
        return dereference();
    }

    inline CUDA_HOSTDEV iter& operator++() {
        increment();
        return *this;
    }
    // inline CUDA_HOSTDEV iter            operator++(int) {} //TODO:
    inline CUDA_HOSTDEV iter& operator--() {
        decrement();
        return *this;
    }
    // inline CUDA_HOSTDEV iter            operator--(int) {} //TODO:
    inline CUDA_HOSTDEV iter& operator+=(difference_type i) {
        advance(i);
        return *this;
    }
    inline CUDA_HOSTDEV iter& operator-=(difference_type i) {
        advance(-i);
        return *this;
    }

    inline CUDA_HOSTDEV iter operator+(difference_type i) const {
        return iter(m_value, m_index + i);
    }
    inline CUDA_HOSTDEV iter operator-(difference_type i) const {
        return iter(m_value, m_index - i);
    }
    inline CUDA_HOSTDEV difference_type operator-(const iter& rhs) const {
        return this->m_index - rhs.m_index;
    }
    inline CUDA_HOSTDEV bool operator==(const iter& rhs) const {
        return this->m_index == rhs.m_index; /*this->equal(rhs);*/
    }
    inline CUDA_HOSTDEV bool operator!=(const iter& rhs) const {
        return !(*this == rhs);
    }

    inline CUDA_HOSTDEV bool operator<(const iter& rhs) const {
        return this->m_index < rhs.m_index;
    }
    inline CUDA_HOSTDEV bool operator<=(const iter& rhs) const {
        return this->m_index <= rhs.m_index;
    }
    inline CUDA_HOSTDEV bool operator>(const iter& rhs) const {
        return this->m_index > rhs.m_index;
    }
    inline CUDA_HOSTDEV bool operator>=(const iter& rhs) const {
        return this->m_index >= rhs.m_index;
    }

private:
    T               m_value;
    difference_type m_index;
};


#endif

} // namespace topaz