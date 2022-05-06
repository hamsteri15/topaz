#pragma once

#include "range.hpp"
#include "runtime_assert.hpp"

namespace topaz {

template <size_t N, typename Iterator>
class ChunkedRange : public Range<Iterator> {
private:
    using parent = Range<Iterator>;

public:
    using iterator       = typename parent::iterator;
    using value_type     = typename parent::value_type;
    using reference      = typename parent::reference;
    using differece_type = typename parent::difference_type;

    inline CUDA_HOSTDEV ChunkedRange(iterator first, iterator last)
        : Range<iterator>(first, last) {
        runtime_assert((size_t(last - first)) % N == 0,
                       "Range not divisible by the chunk count.");
    }
};

template <size_t N, typename Iterator>
CUDA_HOSTDEV auto chunk_size(const ChunkedRange<N, Iterator>& rng) {
    return size(rng) / N;
}

template <size_t I, size_t N, typename Iterator>
CUDA_HOSTDEV auto get_chunk(ChunkedRange<N, Iterator>& rng) {
    static_assert(I < N, "Chunk index out of bounds.");
    return slice(rng, I * chunk_size(rng), (I + size_t(1)) * chunk_size(rng));
}

template <size_t I, size_t N, typename Iterator>
CUDA_HOSTDEV auto get_chunk(const ChunkedRange<N, Iterator>& rng) {
    static_assert(I < N, "Chunk index out of bounds.");
    return slice(rng, I * chunk_size(rng), (I + size_t(1)) * chunk_size(rng));
}

template <size_t N, typename Iterator>
CUDA_HOSTDEV auto make_chunked_range(Iterator first, Iterator last) {
    return ChunkedRange<N, Iterator>(first, last);
}

template <size_t N, class Range_t>
CUDA_HOSTDEV auto make_chunked_range(Range_t& rng) {
    using iterator = decltype(std::begin(rng));
    return make_chunked_range<N, iterator>(adl_begin(rng), adl_end(rng));
}

template <size_t N, class Range_t>
CUDA_HOSTDEV auto make_chunked_range(const Range_t& rng) {
    using iterator = decltype(std::begin(rng));
    return make_chunked_range<N, iterator>(adl_begin(rng), adl_end(rng));
}

namespace detail {

template <size_t N, class Iterator, size_t... Is>
CUDA_HOSTDEV auto
get_chunks_impl(std::index_sequence<Is...>,
                const ChunkedRange<N, Iterator>& rng) {

    return adl_make_tuple(get_chunk<Is>(rng)...);
}
} // namespace detail

template <size_t N, class Range_t>
CUDA_HOSTDEV auto get_chunks(Range_t& rng) {

    using iterator = decltype(std::begin(rng));
    return detail::get_chunks_impl(std::make_index_sequence<N>{},
                                   make_chunked_range<N>(rng));
}

template <size_t N, class Range_t>
CUDA_HOSTDEV auto get_chunks(const Range_t& rng) {

    using iterator = decltype(std::begin(rng));
    return detail::get_chunks_impl(std::make_index_sequence<N>{},
                                   make_chunked_range<N>(rng));
}
