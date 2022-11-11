#pragma once

#include "chunked_range.hpp"
#include "copy.hpp"
#include "numeric_array.hpp"
#include <array>

namespace topaz {

template <size_t N, class T, class Allocator>
struct NumericSoa {

private:
    using array_type = NumericArray<T, Allocator>;

public:
    using size_type = typename array_type::size_type;

    using iterator   = typename array_type::iterator;
    using value_type = typename std::iterator_traits<iterator>::value_type;
    using reference  = typename std::iterator_traits<iterator>::reference;
    using difference_type =
        typename std::iterator_traits<iterator>::difference_type;

    static constexpr bool   is_numeric_vector = true;
    static constexpr size_t n_chunks = N; // TODO: does not belong here

    NumericSoa() = default;

    inline explicit NumericSoa(size_type n)
        : m_data(n * N, T(0)) {}

    inline explicit NumericSoa(const array_type& other)
        : m_data(other) {}

    template <class Range_t>
    inline NumericSoa(const std::array<Range_t, N>& ranges)
        : NumericSoa(ranges.front().size()) {
        for (size_t i = 0; i < N; ++i) { set_chunk(i, ranges[i]); }
    }

    inline auto begin() { return m_data.begin(); }
    inline auto begin() const { return m_data.begin(); }

    inline auto zipped_begin() { return zip_begins<N>(*this); }
    inline auto zipped_begin() const { return zip_begins<N>(*this); }

    inline auto end() { return m_data.end(); }
    inline auto end() const { return m_data.end(); }

    inline auto zipped_end() { return zip_ends<N>(*this); }
    inline auto zipped_end() const { return zip_ends<N>(*this); }

    template <class Range_t>
    void set_chunk(size_t i, const Range_t& rng) {
        if (size_type(adl_size(rng)) != chunk_size(*this)) {
            throw std::runtime_error("Size mismatch error");
        }
        auto dest = get_chunk(i, *this);
        copy(rng, dest);
    }

private:
    array_type m_data;
};

} // namespace topaz