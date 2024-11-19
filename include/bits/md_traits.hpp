#pragma once


#include <type_traits>
#include <iterator>

namespace topaz {



///////////////////////////////////////////////////////////////////////////////////
template<typename T, typename = void>
struct IsMdNumericVector : std::false_type {};

/*
template<typename T>
struct IsMdNumericVector<T, std::enable_if_t< T::is_md_numeric_vector>>
: public std::true_type {};
*/


template< typename T >
constexpr bool IsMdNumericVector_v = IsMdNumericVector<T>::value;

///////////////////////////////////////////////////////////////////////////////////


template<typename T, typename = void>
struct IsMdRange : std::false_type {};

/*
template<typename T>
struct IsMdRange<T, std::enable_if_t< T::is_md_range >>
: public std::true_type {};
*/



template< typename T >
constexpr bool IsMdRange_v = IsMdRange<T>::value;


///////////////////////////////////////////////////////////////////////////////////



template<typename T, typename = void>
struct IsMdRangeOrMdNumericArray : std::false_type {};



template<typename T>
struct IsMdRangeOrMdNumericArray<T, std::enable_if_t< (IsMdNumericVector_v<T>)||(IsMdRange_v<T>) >>
: public std::true_type {};




template< typename T >
constexpr bool IsMdRangeOrMdNumericArray_v = IsMdRangeOrMdNumericArray<T>::value;


///////////////////////////////////////////////////////////////////////////////////

template<typename T1, typename T2, typename = void>
struct BothMdRangesOrMdNumericArrays : public std::false_type {};

template<typename T1, typename T2>
struct BothMdRangesOrMdNumericArrays< T1, T2, std::enable_if_t< IsMdRangeOrMdNumericArray_v<T1> && IsMdRangeOrMdNumericArray_v<T2> > >
   : public std::true_type {};

template< typename T1, typename T2 >
constexpr bool BothMdRangesOrMdNumericArrays_v = BothMdRangesOrMdNumericArrays<T1,T2>::value;

///////////////////////////////////////////////////////////////////////////////////


template< typename T1, typename T2, typename = void >
struct SupportsMdBinaryExpression
   : public std::false_type {};



//Both are either expressions or fields
template< typename T1, typename T2 >
struct SupportsMdBinaryExpression< T1, T2, std::enable_if_t< BothMdRangesOrMdNumericArrays_v<T1, T2> > >
   : public std::true_type {};



//LHS is a scalar value
template< typename T1, typename T2 >
struct SupportsMdBinaryExpression< T1, T2, std::enable_if_t< IsScalar_v<T1> && IsMdRangeOrMdNumericArray_v<T2> > >
   : public std::true_type {};



//RHS is a scalar value
template< typename T1, typename T2 >
struct SupportsMdBinaryExpression< T1, T2, std::enable_if_t< IsMdRangeOrMdNumericArray_v<T1> && IsScalar_v<T2> > >
   : public std::true_type {};




//shorthand for getting the value
template< typename T1, typename T2 >
constexpr bool SupportsMdBinaryExpression_v = SupportsMdBinaryExpression<T1,T2>::value;
///////////////////////////////////////////////////////////////////////////////////


















} // namespace topaz
