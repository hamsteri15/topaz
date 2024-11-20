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
template <typename T>
struct IsRange<T, std::void_t<
    decltype(std::declval<T>().begin()),  // Check begin() is valid
    decltype(std::declval<T>().end()),    // Check end() is valid
    typename std::iterator_traits<decltype(std::declval<T>().begin())>::value_type, // Ensure iterator_traits works
    typename std::iterator_traits<decltype(std::declval<T>().end())>::value_type
>> : std::bool_constant<
    !std::is_base_of_v<
        std::input_iterator_tag,
        typename std::iterator_traits<typename std::iterator_traits<decltype(std::declval<T>().begin())>::value_type>::iterator_category // Check value type is not an iterator
    >
> {};
*()

*/

/*
template <typename T>
struct IsRange<T, std::void_t<
    decltype(std::declval<T>().begin()),  // Check if begin() exists
    decltype(std::declval<T>().end()),    // Check if end() exists
    typename std::iterator_traits<decltype(std::declval<T>().begin())>::value_type, // Ensure begin()'s value_type is valid
    typename std::iterator_traits<decltype(std::declval<T>().end())>::value_type    // Ensure end()'s value_type is valid
>> : std::bool_constant<
    IsIterator_v<decltype(std::declval<T>().begin())> &&  // Check if begin() is an iterator
    IsIterator_v<decltype(std::declval<T>().end())> &&    // Check if end() is an iterator
    !IsIterator_v<typename std::iterator_traits<decltype(std::declval<T>().begin())>::value_type> // Check if value_type is NOT an iterator
> {};
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

/*
template<typename T>
struct IsRange<T, std::void_t<
    decltype(std::declval<T>().begin()),
    decltype(std::declval<T>().end())
>> : public std::true_type{};
*/


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
