#pragma once

#include "range.hpp"
#include "traits.hpp"
#include "constant_range.hpp"
#include "transform.hpp"

#ifdef __CUDACC__
#include <thrust/detail/vector_base.h>
#else
#include <vector>
#endif


namespace topaz {


#ifdef __CUDACC__
template<class T, class Allocator>
using vector_base_type = thrust::detail::vector_base<T, Allocator>;
#else
template<class T, class Allocator>
using vector_base_type = std::vector<T, Allocator>; //TODO: This is very bad, make some own base type
#endif


template <class T, class Allocator>
struct NumericArray : public vector_base_type<T, Allocator>{

private:
    using parent = vector_base_type<T, Allocator>;

public:
    using size_type  = typename parent::size_type;
    using value_type = typename parent::value_type;

    static constexpr bool is_numeric_vector = true;

    inline NumericArray() = default;

    inline explicit NumericArray(size_type n, const value_type& value)
        : parent(n, value) {}

    template <class Iterator,
              typename = std::enable_if_t<IsIterator_v<Iterator>>>
    inline explicit NumericArray(Iterator begin, Iterator end)
        : parent(begin, end) {}

    inline explicit NumericArray(std::initializer_list<T> l) : parent(l.size()){
        this->assign(l.begin(), l.end());
    }

    //TODO: this should maybe be marked explicit as well
    template <class Range_t,
              typename = std::enable_if_t<IsRangeOrNumericVector_v<Range_t>>>
    inline NumericArray(const Range_t& rng)
        : parent(adl_begin(rng), adl_end(rng)) {}


    template <class Range_t,
              typename = std::enable_if_t<IsRangeOrNumericVector_v<Range_t>>>
    inline NumericArray& operator=(const Range_t& rng) {
        this->assign(adl_begin(rng), adl_end(rng));
        return *this;
    }
};


template <class T1, class T2>
inline CUDA_HOSTDEV auto determine_size(const T1&, const T2& rhs)
    -> std::enable_if_t<IsScalar_v<T1>, typename T2::difference_type> {
    return adl_size(rhs);
}
template <class T1, class T2>
inline CUDA_HOSTDEV auto determine_size(const T1& lhs, const T2&)
    -> std::enable_if_t<IsScalar_v<T2>, typename T1::difference_type> {
    return adl_size(lhs);
}


template <class T1, class T2,
          typename = std::enable_if_t<BothRangesOrNumericVectors_v<T1,T2>>>
inline CUDA_HOSTDEV auto determine_size(const T1& lhs, const T2&) {
    return adl_size(lhs);
}



template <class Range_t,
          class Size,
          typename = std::enable_if_t<!IsScalar_v<Range_t>>>
inline CUDA_HOSTDEV auto rangify(Range_t& rng, Size n) {
    return take(rng, n);
}


template <class Range_t,
          class Size,
          typename = std::enable_if_t<!IsScalar_v<Range_t>>>
inline CUDA_HOSTDEV auto rangify(const Range_t& rng, Size n) {
    return take(rng, n);
}

template <class Scalar,
          class Size,
          std::enable_if_t<IsScalar_v<Scalar>, bool> = true>
inline CUDA_HOSTDEV auto rangify(const Scalar& s, Size n) {
    return make_constant_range<Scalar, Size>(s, n);
}





template<class T1, class T2, class BinaryOp>
inline CUDA_HOSTDEV auto smart_transform(const T1& lhs, const T2& rhs, BinaryOp f){
    auto size = determine_size(lhs, rhs);
    return transform(rangify(lhs, size), rangify(rhs, size), f);
}



template<class Lhs, class Rhs, class Enable = void>
struct ReturnType{
    using type = void;
};
template<class Lhs, class Rhs>
struct ReturnType<Lhs, Rhs, std::enable_if_t<IsScalar_v<Lhs> && !IsScalar_v<Rhs>>>{
    using type = Lhs;
};

template<class Lhs, class Rhs>
struct ReturnType<Lhs, Rhs, std::enable_if_t<!IsScalar_v<Lhs> && IsScalar_v<Rhs>>>{
    using type = Rhs;
};

template<class Lhs, class Rhs>
struct ReturnType<Lhs, Rhs, std::enable_if_t<!IsScalar_v<Lhs> && !IsScalar_v<Rhs>>>{
    using type = typename Lhs::value_type;
};



template<class T>
struct Plus{
    inline CUDA_HOSTDEV
    T operator()( const T& lhs, const T& rhs ) const {return lhs + rhs;}
};

template<class T>
struct Minus{
    inline CUDA_HOSTDEV
    T operator()( const T& lhs, const T& rhs ) const {return lhs - rhs;}
};

template<class T>
struct Multiplies{
    inline CUDA_HOSTDEV
    T operator()( const T& lhs, const T& rhs ) const {return lhs * rhs;}
};

template<class T>
struct Divides{
    inline CUDA_HOSTDEV
    T operator()( const T& lhs, const T& rhs ) const {return lhs / rhs;}
};



template <class T1,
          class T2,
          typename = std::enable_if_t<SupportsBinaryExpression_v<T1, T2>>>
inline CUDA_HOSTDEV auto operator+(const T1& lhs, const T2& rhs) {

    using value_type = typename ReturnType<T1, T2>::type;
    return smart_transform(lhs, rhs, Plus<value_type>{});
}

template <class T1,
          class T2,
          typename = std::enable_if_t<SupportsBinaryExpression_v<T1, T2>>>
inline CUDA_HOSTDEV auto operator-(const T1& lhs, const T2& rhs) {

    using value_type = typename ReturnType<T1, T2>::type;
    return smart_transform(lhs, rhs, Minus<value_type>{});
}


template <class T1,
          class T2,
          typename = std::enable_if_t<SupportsBinaryExpression_v<T1, T2>>>
inline CUDA_HOSTDEV auto operator*(const T1& lhs, const T2& rhs) {

    using value_type = typename ReturnType<T1, T2>::type;
    return smart_transform(lhs, rhs, Multiplies<value_type>{});
}


template <class T1,
          class T2,
          typename = std::enable_if_t<SupportsBinaryExpression_v<T1, T2>>>
inline CUDA_HOSTDEV auto operator/(const T1& lhs, const T2& rhs) {

    using value_type = typename ReturnType<T1, T2>::type;
    return smart_transform(lhs, rhs, Divides<value_type>{});
}

} // namespace topaz