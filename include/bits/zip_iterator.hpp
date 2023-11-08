#pragma once

//#include "invoke.hpp"
#include <iterator>
#include <tuple>
#include <utility>

namespace topaz{

namespace detail{

template <typename T, typename F, std::size_t... Is>
constexpr auto map_tuple_elements(T&& tup, F& f, std::index_sequence<Is...>) {
    return std::make_tuple(f(std::get<Is>(std::forward<T>(tup)))...);
}

template<typename T, size_t... I>
constexpr auto deref_tuple_elements(T t ,std::index_sequence<I...>)
{ 
    //return std::make_tuple(*std::get<0>(t), *std::get<1>(t));
    return std::tie(*std::get<I>(t)...) ;
}



template<typename DiffType, typename T, size_t... I>
constexpr void increment(DiffType inc, T t ,  std::index_sequence<I...>)
{ 
    //(std::get<I>(t)++)...;
    //{(std::get<I>(t) + inc);...};
}



template<typename DiffType, typename ...T, size_t... I>
constexpr auto increment_and_deref(DiffType inc, std::tuple<T...> t ,  std::index_sequence<I...>)
{ return std::tie(*(std::get<I>(t) + inc)...) ;}

/*
template<typename ...T, size_t... I>
constexpr auto deref_tuple_elements(const std::tuple<T...>& t ,  std::index_sequence<I...>)
{ return std::tie(*std::get<I>(t)...) ;}
*/
} // namespace detail

template <typename T,
          typename F,
          std::size_t TupSize = std::tuple_size<std::decay_t<T>>::value>
constexpr auto map_tuple_elements(T&& tup, F f) {
    return detail::map_tuple_elements(
        std::forward<T>(tup), f, std::make_index_sequence<TupSize>{});
}



template<typename T>
constexpr auto deref_tuple_elements(T t ){
	return detail::deref_tuple_elements<T>(t, std::make_index_sequence<std::tuple_size_v<T>>{});
}



template<typename DiffType, typename T>
constexpr auto increment(DiffType inc, T t ){
	return detail::increment<DiffType, T>(inc, t, std::make_index_sequence<std::tuple_size_v<T>>{});
}


template<typename DiffType, typename ...T>
constexpr auto increment_and_deref(DiffType inc, std::tuple<T...> t ){
	return detail::increment_and_deref<DiffType, T...>(inc, t, std::make_index_sequence<sizeof...(T)>{});
}


/*
template<typename ...T>
constexpr auto deref_tuple_elements( const std::tuple<T...>& t ){
	return detail::deref_tuple_elements<T...>(t, std::make_index_sequence<sizeof...(T)>{});
}
*/





template <class Iterator>
struct iterator_reference
{
    typedef typename std::iterator_traits<Iterator>::reference type;
};






template <class DifferenceType>
struct Increment {

    Increment(DifferenceType inc)
        : m_inc(inc) {}

    template <class It>
    void operator()(It it) const {
        it += m_inc;
    }

    DifferenceType m_inc;
};


template<class T>
struct DeduceReference{

};

template<class ...T>
struct DeduceReference<std::tuple<T...>>
{
    using type = std::tuple<typename iterator_reference<T>::type...>;
};


template <class IteratorTuple>
class zip_iterator {

public:

    // TODO: use iterator_traits
    //using difference_type =
    //    decltype(std::distance(std::get<0>(m_tuple), std::get<0>(m_tuple)));
    // typename std::iterator_traits<first_type>::difference_type;
    

    using reference  = typename DeduceReference<IteratorTuple>::type; 
    using value_type = reference;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using iterator_category = std::random_access_iterator_tag;


    IteratorTuple m_tuple;

    //using first_type =
    //    decltype(std::get<0>(m_tuple)); // std::tuple_element<0, IteratorTuple>;
    using my_type = zip_iterator<IteratorTuple>;

    const IteratorTuple& get_tuple() const {
        return m_tuple;
    }

public:
    
    zip_iterator(IteratorTuple tuple)
        : m_tuple(tuple) {}


    /*
    reference dereference() {
        return deref_tuple_elements(m_tuple);
    }
    */
    reference dereference() const {

        //return std::apply([](auto && ... args){ 
        //        return *args...; }, m_tuple);


        return deref_tuple_elements(get_tuple());
    }
    

    
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

    inline CUDA_HOSTDEV my_type operator+(difference_type i) const {
        
        auto copy = m_tuple;
        //std::apply([=](auto iter){iter+=i;}, copy);
        return my_type(copy);
    }

    inline CUDA_HOSTDEV my_type operator-(difference_type i) const {
        auto copy = m_tuple;
        std::apply([=](auto iter){iter-=i;}, copy);
        return my_type(copy);
    }

    inline CUDA_HOSTDEV my_type& operator+=(difference_type i) {
        //invoke_hpp::apply(m_tuple, Increment<difference_type>(i));
        //std::apply([=](auto iter){iter+=i;}, m_tuple);
        return *this;
        


    }
    inline CUDA_HOSTDEV my_type& operator-=(difference_type i) {
        //invoke_hpp::apply(m_tuple, Increment<difference_type>(-i));
        std::apply([=](auto iter){iter-=i;}, m_tuple);
        return *this;
    }

    
    inline CUDA_HOSTDEV auto operator[](difference_type i) const {
        return increment_and_deref(i, m_tuple);
    }
    
    
    // auto& operator[](difference_type i) { return m_func(m_it[i]); }

    inline CUDA_HOSTDEV auto operator*() const { return dereference(); }

    inline CUDA_HOSTDEV my_type& operator++() {
        
        std::apply([](auto && ... args){ ((args += 1), ...); }, m_tuple);
        return *this;

    }
    inline CUDA_HOSTDEV my_type& operator--() {
        std::apply([](auto && ... args){ ((args -= 1), ...); }, m_tuple);
        return *this;
    }
    /*
    inline CUDA_HOSTDEV my_type& operator++(int) {
        auto prev = *this;
        ++m_it;
        return prev;
    }

    */
};

template <class IteratorTuple>
inline CUDA_HOSTDEV zip_iterator<IteratorTuple> make_zip_iterator(IteratorTuple tuple) {
    return zip_iterator<IteratorTuple>(tuple);
}

} // namespace Zip
