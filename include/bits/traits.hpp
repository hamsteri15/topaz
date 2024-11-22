#pragma once


#include <type_traits>
#include <iterator>

namespace topaz {


template<class T, typename = void>
struct IsMathematical : public std::false_type{}; 


template<typename T>
struct IsMathematical<T, std::void_t<
    decltype(std::declval<T>() + std::declval<T>()),
    decltype(std::declval<T>() - std::declval<T>()),
    decltype(std::declval<T>() * std::declval<T>()),
    decltype(std::declval<T>() / std::declval<T>())
>> : public std::true_type{};

template<typename T>
static constexpr bool IsMathematical_v = IsMathematical<T>::value;


///////////////////////////////////////////////////////////////////////////////////


template <typename T1, typename T2, typename = void>
struct IsValueTypeOf : std::false_type {};

template <typename T1, typename T2>
struct IsValueTypeOf<T1, T2, std::void_t<typename T1::value_type>> 
    : std::is_same<typename T1::value_type, T2> {};

template <typename T1, typename T2>
inline constexpr bool IsValueTypeOf_v = IsValueTypeOf<T1, T2>::value;

///////////////////////////////////////////////////////////////////////////////////


// Primary template: default to false
template <typename T1, typename T2, typename = void>
struct ValueTypeIs : std::false_type {};

// Specialization: Check if T1 has a value_type that matches T2
template <typename T1, typename T2>
struct ValueTypeIs<T1, T2, std::void_t<typename T1::value_type>> 
    : std::is_same<typename T1::value_type, T2> {};

// Helper variable template for convenience
template <typename T1, typename T2>
inline constexpr bool ValueTypeIs_v = ValueTypeIs<T1, T2>::value;

///////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct IsScalar : std::is_arithmetic<T> {};

template<typename T>
static constexpr bool IsScalar_v = IsScalar<T>::value;

///////////////////////////////////////////////////////////////////////////////////





template <typename T, typename = void>
struct IsIterator : public std::false_type {};


template <typename T>
struct IsIterator<T, std::void_t<
    typename std::iterator_traits<T>::iterator_category  // Check if iterator_traits is valid
>> : std::true_type {};

template<typename T>
constexpr bool IsIterator_v = IsIterator<T>::value;

///////////////////////////////////////////////////////////////////////////////////


template <typename, typename = void>
struct IsMathematicalIterator : std::false_type {};

template <typename T>
struct IsMathematicalIterator<T, std::void_t<
    typename std::iterator_traits<T>::value_type  // Extract value type
>> : IsMathematical<typename std::iterator_traits<T>::value_type> {};


template<typename T>
constexpr bool IsMathematicalIterator_v = IsMathematicalIterator<T>::value;



///////////////////////////////////////////////////////////////////////////////////



template<typename T, typename = void>
struct IsNumericVector : std::false_type {};


template<typename T>
struct IsNumericVector<T, std::enable_if_t< T::is_numeric_vector >>
: public std::true_type {};



template< typename T >
constexpr bool IsNumericVector_v = IsNumericVector<T>::value;

///////////////////////////////////////////////////////////////////////////////////


template<typename T, typename = void>
struct IsRange : std::false_type {};

/*
template<typename T>
struct IsRange<T, std::enable_if_t< T::is_range >>
: public std::true_type {};
*/

template <typename T>
struct IsRange<T, std::void_t<
    decltype(std::declval<T>().begin()), // Check if begin() exists
    decltype(std::declval<T>().end()),   // Check if end() exists
    typename std::iterator_traits<decltype(std::declval<T>().begin())>::value_type, // Ensure value_type is valid
    typename std::iterator_traits<decltype(std::declval<T>().end())>::value_type
>> : std::bool_constant<
    IsIterator_v<decltype(std::declval<T>().begin())> &&  // begin() is an iterator
    IsIterator_v<decltype(std::declval<T>().end())> &&    // end() is an iterator
    !IsRange<typename std::iterator_traits<decltype(std::declval<T>().begin())>::value_type>::value // Recursive: value_type is not a range
> {};



template< typename T >
constexpr bool IsRange_v = IsRange<T>::value;


///////////////////////////////////////////////////////////////////////////////////


template <typename, typename = void>
struct IsMathematicalRange : std::false_type {};

template <typename T>
struct IsMathematicalRange<T, std::void_t<
    decltype(std::declval<T>().begin()),  // Check if begin() is valid
    decltype(std::declval<T>().end())    // Check if end() is valid
>> : std::conditional_t<
    IsMathematicalIterator_v<decltype(std::declval<T>().begin())> &&  // Check if begin() returns a MathematicalIterator
    IsMathematicalIterator_v<decltype(std::declval<T>().end())>,      // Check if end() returns a MathematicalIterator
    std::true_type,
    std::false_type
> {};

template< typename T >
constexpr bool IsMathematicalRange_v = IsMathematicalRange<T>::value;

///////////////////////////////////////////////////////////////////////////////////


template<typename T, typename = void>
struct IsRangeOrNumericArray : std::false_type {};



template<typename T>
struct IsRangeOrNumericArray<T, std::enable_if_t< (IsNumericVector_v<T>)||(IsRange_v<T>) >>
: public std::true_type {};




template< typename T >
constexpr bool IsRangeOrNumericArray_v = IsRangeOrNumericArray<T>::value;


///////////////////////////////////////////////////////////////////////////////////

template<typename T1, typename T2, typename = void>
struct BothRangesOrNumericArrays : public std::false_type {};

template<typename T1, typename T2>
struct BothRangesOrNumericArrays< T1, T2, std::enable_if_t< IsRangeOrNumericArray_v<T1> && IsRangeOrNumericArray_v<T2> > >
   : public std::true_type {};

template< typename T1, typename T2 >
constexpr bool BothRangesOrNumericArrays_v = BothRangesOrNumericArrays<T1,T2>::value;

///////////////////////////////////////////////////////////////////////////////////


template< typename T1, typename T2, typename = void >
struct SupportsBinaryExpression
   : public std::false_type {};



//Both are either expressions or fields
template< typename T1, typename T2 >
struct SupportsBinaryExpression< T1, T2, std::enable_if_t< BothRangesOrNumericArrays_v<T1, T2> > >
   : public std::true_type {};



//LHS is a scalar value
template< typename T1, typename T2 >
struct SupportsBinaryExpression< T1, T2, std::enable_if_t< IsScalar_v<T1> && IsRangeOrNumericArray_v<T2> > >
   : public std::true_type {};



//RHS is a scalar value
template< typename T1, typename T2 >
struct SupportsBinaryExpression< T1, T2, std::enable_if_t< IsRangeOrNumericArray_v<T1> && IsScalar_v<T2> > >
   : public std::true_type {};




//shorthand for getting the value
template< typename T1, typename T2 >
constexpr bool SupportsBinaryExpression_v = SupportsBinaryExpression<T1,T2>::value;
///////////////////////////////////////////////////////////////////////////////////


template< typename T1, typename T2, typename = void >
struct AtleastOneIsRange
   : public std::false_type {};


template<typename T1, typename T2>
struct AtleastOneIsRange<T1, T2, std::enable_if_t<IsRange_v<T1>||IsRange_v<T2>>> : public std::true_type{};

template< typename T1, typename T2 >
constexpr bool AtleastOneIsRange_v = AtleastOneIsRange<T1,T2>::value;

///////////////////////////////////////////////////////////////////////////////////


template <typename T1, typename T2>
struct BothAreRanges : std::bool_constant<IsRange_v<T1> && IsRange_v<T2>> {};


template <typename T1, typename T2>
inline constexpr bool BothAreRanges_v = BothAreRanges<T1, T2>::value;

///////////////////////////////////////////////////////////////////////////////////



} // namespace topaz
