#pragma once

//#include "invoke.hpp"
#include <iterator>
#include <tuple>
#include <utility>
#include "temp.hpp"

namespace topaz{

namespace detail{

template <typename T, typename F, std::size_t... Is>
constexpr auto map_tuple_elements(T&& tup, F& f, std::index_sequence<Is...>) {
    return std::make_tuple(f(std::get<Is>(std::forward<T>(tup)))...);
}

template<int N, typename Tu> using NthTypeOf =
        typename std::tuple_element<N, Tu>::type;

template<int N, typename Tu> using NthValueOf =
    typename NthTypeOf<N, Tu>::value_type;

template<int N, typename Tu> using Good =
    std::remove_reference<NthTypeOf<N, Tu>>;





template<typename T, size_t... I>
constexpr auto deref_tuple_elements(const T& t ,std::index_sequence<I...>)
{
    //return std::make_tuple(*std::get<0>(t), *std::get<1>(t));

    //return std::tie(*std::get<I>(t)...);

    auto f = [](auto it) {

        using deref_type = decltype(*it);
        //using value_type = typename decltype(it)::value_type;

        //using return_type = std::remove_reference_t<deref_type>;
        //using return_type = std::reference_wrapper<std::remove_reference_t<value_type>>;
        //using return_type = std::remove_reference<typename decltype(*it)::value_type>;
        //using return_type = deref_type;
        //return static_cast<return_type>(*it);

        return std::forward<deref_type>(*it);
        //return *it;
    };
    //return std::make_tuple(f(std::get<I>(t))...);
    return std::forward_as_tuple(f(std::get<I>(t))...);
    //return std::make_tuple(f(std::get<I>(std::forward<T>(t)))...);
    //return std::tie(f(std::get<I>(t))...);

}


} // namespace detail

template <typename T,
          typename F,
          std::size_t TupSize = std::tuple_size<std::decay_t<T>>::value>
constexpr auto map_tuple_elements(T&& tup, F f) {
    return detail::map_tuple_elements(
        std::forward<T>(tup), f, std::make_index_sequence<TupSize>{});
}

/*
template<typename DiffType, typename T, size_t... I>
constexpr void increment(DiffType inc, T t ,  std::index_sequence<I...>)
{
    auto f = [=](auto it){
        it += inc;
    };

    map_tuple_elements(t, f);
    //(std::get<I>(t)++)...;
    //{(std::get<I>(t) + inc);...};
}
*/


template<typename T>
constexpr auto deref_tuple_elements(const T t ){
	return detail::deref_tuple_elements<T>(
        t, std::make_index_sequence<std::tuple_size_v<T>>{});
}



template<typename DiffType, typename T>
constexpr auto increment_by(DiffType inc, T& t ){

    /*
    auto f = [=](auto it){
        it += inc;
    };
    */
    std::get<0>(t) += inc;
    std::get<1>(t) += inc;
    //std::apply([=](auto& ... args){ ((args += inc), ...); }, t);


    //map_tuple_elements(t, f);

    //return detail::increment<DiffType, T>(inc, t, std::make_index_sequence<std::tuple_size_v<T>>{});
}















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
    //using reference = typename tuple_of_references<IteratorTuple>::type;
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
    auto dereference() const {
        /*
        auto op = [](auto && ... args){

            return std::tie(*(args)...);
        };
        return std::apply(op, m_tuple);
        */

        /*
        typedef converter<reference> gen;
        return gen::call(boost::fusion::transform(
          get_tuple(),
          dereference_iterator()));
        */
        return deref_tuple_elements(get_tuple());
    }


    void advance(difference_type i) {
        //increment_by(i, get_tuple());
        increment_by(i, m_tuple);
    }

    void increment() {
        this->advance(1);
    }

    void decrement() {
        this->advance(-1);
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
        //return copy += i;
    }

    inline CUDA_HOSTDEV my_type operator-(difference_type i) const {
        auto copy = zip_iterator(get_tuple());
        copy.advance(-i);
        //std::apply([=](auto iter){iter-=i;}, copy);
        //return copy -= i;
        return copy;
    }




    inline CUDA_HOSTDEV auto operator[](difference_type i) const {
        //return increment_and_deref(i, m_tuple);
        return dereference();
    }


    // auto& operator[](difference_type i) { return m_func(m_it[i]); }

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
inline CUDA_HOSTDEV zip_iterator<IteratorTuple> make_zip_iterator(IteratorTuple tuple) {
    return zip_iterator<IteratorTuple>(tuple);
}

} // namespace Zip
