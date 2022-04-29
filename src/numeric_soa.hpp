#pragma once

#include "numeric_array.hpp"
#include <array>

namespace topaz {


template <size_t N, class T, class Allocator>
struct NumericSoa {

private:
    using array_type = NumericArray<T, Allocator>;

public:
    using size_type = typename array_type::size_type;

    using iterator = typename array_type::iterator;
    using value_type = typename std::iterator_traits<iterator>::value_type;
    using reference  = typename std::iterator_traits<iterator>::reference;
    using difference_type = typename std::iterator_traits<iterator>::difference_type;

    static constexpr bool is_numeric_vector = true;

    NumericSoa() = default;

    inline explicit NumericSoa(size_type n)
        : m_data(n * N, T(0)) {}

    /*
    template<class... Ts>
    inline explicit NumericSoa(size_type n, Ts... ts) :
    NumericSoa(n)
    {
        //TODO: get chunks and set one by one, _dont_ use the zip iterator
    }

    */

    inline auto begin() { return m_data.begin(); }
    inline auto begin() const { return m_data.begin(); }

    inline auto zipped_begin() {
        return zip_begins(std::make_index_sequence<N>{});
    }
    inline auto zipped_begin() const {
        return zip_begins(std::make_index_sequence<N>{});
    }

    inline auto end() { return m_data.end(); }
    inline auto end() const { return m_data.end(); }

    inline auto zipped_end() { return zip_ends(std::make_index_sequence<N>{}); }
    inline auto zipped_end() const {
        return zip_ends(std::make_index_sequence<N>{});
    }

    template <size_t I>
    auto get_chunk() {
        static_assert(I < N, "Index out of bounds");
        return slice(m_data, I * chunk_size(), (I + size_t(1)) * chunk_size());
    }

    template <size_t I>
    auto get_chunk() const {
        static_assert(I < N, "Index out of bounds");
        return slice(m_data, I * chunk_size(), (I + size_t(1)) * chunk_size());
    }

    inline size_type chunk_size() const { return m_data.size() / N; }
    inline size_type chunk_size() { return m_data.size() / N; }

private:
    template <size_t... Is>
    inline auto zip_begins(std::index_sequence<Is...>) {
        return detail::make_zip_iterator(
            adl_make_tuple(get_chunk<Is>().begin()...));
    }
    template <size_t... Is>
    inline auto zip_begins(std::index_sequence<Is...>) const {
        return detail::make_zip_iterator(
            adl_make_tuple(get_chunk<Is>().begin()...));
    }
    template <size_t... Is>
    inline auto zip_ends(std::index_sequence<Is...>) {
        return detail::make_zip_iterator(
            adl_make_tuple(get_chunk<Is>().end()...));
    }
    template <size_t... Is>
    inline auto zip_ends(std::index_sequence<Is...>) const {
        return detail::make_zip_iterator(
            adl_make_tuple(get_chunk<Is>().end()...));
    }

private:
    array_type m_data;
};

} // namespace topaz