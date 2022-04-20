#pragma once

#include "range.hpp"
#include <thrust/iterator/zip_iterator.h>
#include <boost/iterator/zip_iterator.hpp>

namespace topaz {

namespace detail{


    #ifdef __CUDACC__

        template<class... Types>
        using Tuple = thrust::tuple<Types...>;

        template< class... Types >
        inline CUDA_HOSTDEV
        auto make_tuple( Types&&... args ) {
            return thrust::make_tuple(std::forward<Types>(args)...);
        }

        template<class T>
        using zip_iterator = thrust::zip_iterator<T>;

    #else

        template<class... Types>
        using Tuple = boost::tuple<Types...>;

        template< class... Types >
        auto make_tuple( Types&&... args ) {
            return boost::make_tuple(std::forward<Types>(args)...);
        }

        template<class T>
        using zip_iterator = boost::zip_iterator<T>;


    #endif
}


template <typename IteratorTuple>
struct ZipRange : public Range<detail::zip_iterator<IteratorTuple>> {

    using parent = Range<detail::zip_iterator<IteratorTuple>>;

    inline CUDA_HOSTDEV ZipRange(IteratorTuple first, IteratorTuple last)
        : parent(first, last) {}
};

template <typename Range1_t, typename Range2_t>
inline CUDA_HOSTDEV auto make_zip_range(Range1_t& rng1, Range2_t& rng2) {

    using iter1   = decltype(adl_begin(rng1));
    using iter2   = decltype(adl_begin(rng2));
    using tuple_t = detail::Tuple<iter1, iter2>;

    return ZipRange<tuple_t>(detail::make_tuple(adl_begin(rng1), adl_begin(rng2)),
                              detail::make_tuple(adl_end(rng1), adl_end(rng2)));
}

template <typename Range1_t, typename Range2_t>
inline CUDA_HOSTDEV auto make_zip_range(Range1_t& rng1, const Range2_t& rng2) {

    using iter1   = decltype(adl_begin(rng1));
    using iter2   = decltype(adl_begin(rng2));
    using tuple_t = detail::Tuple<iter1, iter2>;

    return ZipRange<tuple_t>(detail::make_tuple(adl_begin(rng1), adl_begin(rng2)),
                              detail::make_tuple(adl_end(rng1), adl_end(rng2)));
}

template <typename Range1_t, typename Range2_t>
inline CUDA_HOSTDEV auto make_zip_range(const Range1_t& rng1, Range2_t& rng2) {

    using iter1   = decltype(adl_begin(rng1));
    using iter2   = decltype(adl_begin(rng2));
    using tuple_t = detail::Tuple<iter1, iter2>;

    return ZipRange<tuple_t>(detail::make_tuple(adl_begin(rng1), adl_begin(rng2)),
                              detail::make_tuple(adl_end(rng1), adl_end(rng2)));
}

template <typename Range1_t, typename Range2_t>
inline CUDA_HOSTDEV auto make_zip_range(const Range1_t& rng1,
                                        const Range2_t& rng2) {

    using iter1   = decltype(adl_begin(rng1));
    using iter2   = decltype(adl_begin(rng2));
    using tuple_t = detail::Tuple<iter1, iter2>;

    return ZipRange<tuple_t>(detail::make_tuple(adl_begin(rng1), adl_begin(rng2)),
                              detail::make_tuple(adl_end(rng1), adl_end(rng2)));
}

} // namespace topaz