#pragma once

#include "range.hpp"
#include "runtime_assert.hpp"
#include "tuple.hpp"
#include "zip_range.hpp"

namespace topaz {

template <size_t N, typename Iterator>
class ChunkedRange : public Range<Iterator> {
private:
    using parent = Range<Iterator>;

public:
    static constexpr size_t n_chunks = N;

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

template <class ChunkedRange_t>
static constexpr CUDA_HOSTDEV auto chunk_count(const ChunkedRange_t&) {
    return ChunkedRange_t::n_chunks;
}

template <class ChunkedRange_t>
CUDA_HOSTDEV auto chunk_size(const ChunkedRange_t& rng) {
    return adl_size(rng) / chunk_count(rng);
}

template <size_t I, class ChunkedRange_t>
CUDA_HOSTDEV auto get_chunk(ChunkedRange_t& rng) {
    // static_assert(I < chunk_count(rng), "Chunk index out of bounds.");
    // TODO: should be a static assert
    runtime_assert(I < chunk_count(rng), "Chunk index out of bounds.");
    return slice(rng, I * chunk_size(rng), (I + size_t(1)) * chunk_size(rng));
}

template <size_t I, class ChunkedRange_t>
CUDA_HOSTDEV auto get_chunk(const ChunkedRange_t& rng) {
    // static_assert(I < chunk_count(rng), "Chunk index out of bounds.");
    // TODO: should be a static assert
    runtime_assert(I < chunk_count(rng), "Chunk index out of bounds.");
    return slice(rng, I * chunk_size(rng), (I + size_t(1)) * chunk_size(rng));
}

template <class ChunkedRange_t>
CUDA_HOSTDEV auto get_chunk(size_t I, ChunkedRange_t& rng) {
    // static_assert(I < chunk_count(rng), "Chunk index out of bounds.");
    // TODO: should be a static assert
    runtime_assert(I < chunk_count(rng), "Chunk index out of bounds.");
    return slice(rng, I * chunk_size(rng), (I + size_t(1)) * chunk_size(rng));
}

template <class ChunkedRange_t>
CUDA_HOSTDEV auto get_chunk(size_t I, const ChunkedRange_t& rng) {
    // static_assert(I < chunk_count(rng), "Chunk index out of bounds.");
    // TODO: should be a static assert
    runtime_assert(I < chunk_count(rng), "Chunk index out of bounds.");
    return slice(rng, I * chunk_size(rng), (I + size_t(1)) * chunk_size(rng));
}

namespace detail {

template <size_t N, class Iterator, size_t... Is>
CUDA_HOSTDEV auto get_chunks_impl(std::index_sequence<Is...>,
                                  const ChunkedRange<N, Iterator>& rng) {
    return adl_make_tuple(get_chunk<Is>(rng)...);
}

template <class ChunkTuple, class NaryOp, size_t... Is>
CUDA_HOSTDEV auto chunked_reduce_impl(std::index_sequence<Is...>,
                                      const ChunkTuple& tpl,
                                      NaryOp            op) {
    return op(get<Is>(tpl)...);
}

template <size_t N, class Iterator, size_t... Is>
CUDA_HOSTDEV auto zip_begins_impl(std::index_sequence<Is...>,
                                  ChunkedRange<N, Iterator>& rng) {
    return detail::make_zip_iterator(
        adl_make_tuple(get_chunk<Is>(rng).begin()...));
}

template <size_t N, class Iterator, size_t... Is>
CUDA_HOSTDEV auto zip_begins_impl(std::index_sequence<Is...>,
                                  const ChunkedRange<N, Iterator>& rng) {
    return detail::make_zip_iterator(
        adl_make_tuple(get_chunk<Is>(rng).begin()...));
}

template <size_t N, class Iterator, size_t... Is>
CUDA_HOSTDEV auto zip_ends_impl(std::index_sequence<Is...>,
                                ChunkedRange<N, Iterator>& rng) {
    return detail::make_zip_iterator(
        adl_make_tuple(get_chunk<Is>(rng).end()...));
}

template <size_t N, class Iterator, size_t... Is>
CUDA_HOSTDEV auto zip_ends_impl(std::index_sequence<Is...>,
                                const ChunkedRange<N, Iterator>& rng) {
    return detail::make_zip_iterator(
        adl_make_tuple(get_chunk<Is>(rng).end()...));
}

} // namespace detail

template <size_t N, class Range_t>
CUDA_HOSTDEV auto zip_begins(Range_t& rng) {

    return detail::zip_begins_impl(std::make_index_sequence<N>{},
                                   make_chunked_range<N>(rng));
}

template <size_t N, class Range_t>
CUDA_HOSTDEV auto zip_begins(const Range_t& rng) {

    return detail::zip_begins_impl(std::make_index_sequence<N>{},
                                   make_chunked_range<N>(rng));
}

template <size_t N, class Range_t>
CUDA_HOSTDEV auto zip_ends(Range_t& rng) {

    return detail::zip_ends_impl(std::make_index_sequence<N>{},
                                 make_chunked_range<N>(rng));
}

template <size_t N, class Range_t>
CUDA_HOSTDEV auto zip_ends(const Range_t& rng) {

    return detail::zip_ends_impl(std::make_index_sequence<N>{},
                                 make_chunked_range<N>(rng));
}

template <size_t N, class Range_t>
CUDA_HOSTDEV auto get_chunks(Range_t& rng) {

    return detail::get_chunks_impl(std::make_index_sequence<N>{},
                                   make_chunked_range<N>(rng));
}

template <size_t N, class Range_t>
CUDA_HOSTDEV auto get_chunks(const Range_t& rng) {

    return detail::get_chunks_impl(std::make_index_sequence<N>{},
                                   make_chunked_range<N>(rng));
}

template <size_t N, class Range_t, class NaryOp>
CUDA_HOSTDEV auto chunked_reduce(const Range_t& rng, NaryOp op) {

    return detail::chunked_reduce_impl(
        std::make_index_sequence<N>{}, get_chunks<N>(rng), op);
}

} // namespace topaz