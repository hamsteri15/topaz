#pragma once

#include "numeric_array.hpp"
#include "copy.hpp"
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

    inline explicit NumericSoa(const array_type& other) : m_data(other) {}

    template<class Range_t>
    inline NumericSoa(const std::array<Range_t, N>& ranges)
    : NumericSoa(ranges.front().size())
    {
        for (size_t i = 0; i < N; ++i){
            set_chunk(i, ranges[i]);
        }
    }

    inline size_type chunk_size() const { return m_data.size() / N; }

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


    auto get_chunk(size_t I) {
        //assert(I < N, "Index out of bounds");
        return slice(m_data, I * chunk_size(), (I + size_type(1)) * chunk_size());
    }

    auto get_chunk(size_t I) const {
        //assert(I < N, "Index out of bounds");
        return slice(m_data, I * chunk_size(), (I + size_type(1)) * chunk_size());
    }

    auto get_all_chunks() const {
        return get_all_chunks_helper(std::make_index_sequence<N>{});
    }

    auto get_all_chunks() {
        return get_all_chunks_helper(std::make_index_sequence<N>{});
    }


    template<class Range_t>
    void set_chunk(size_t i, const Range_t& rng){
        if (size_type(adl_size(rng)) != chunk_size()){
            throw std::runtime_error("Size mismatch error");
        }
        auto dest = get_chunk(i);
        copy(rng, dest);

    }


private:


    template<size_t... Is>
    auto get_all_chunks_helper(std::index_sequence<Is...>){
        return std::make_tuple(get_chunk<Is>()...);
    }
    
    template<size_t... Is>
    auto get_all_chunks_helper(std::index_sequence<Is...>) const{
        return std::make_tuple(get_chunk<Is>()...);
    }

    template <size_t I>
    auto get_chunk() {
        static_assert(I < N, "Index out of bounds");
        return slice(m_data, I * chunk_size(), (I + size_type(1)) * chunk_size());
    }

    template <size_t I>
    auto get_chunk() const {
        static_assert(I < N, "Index out of bounds");
        return slice(m_data, I * chunk_size(), (I + size_type(1)) * chunk_size());
    }


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