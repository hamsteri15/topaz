#pragma once

namespace topaz {
/*
template <class Range_t, std::enable_if_t<IsRange_v<Range_t>, bool> = true>
inline CUDA_HOSTDEV auto zip(Range_t& rng) {
    using iterator    = decltype(std::begin(rng));
    using result_type = ZipRange<Tuple<iterator>>;
    return result_type(adl_make_tuple(rng));
}

template <class Range_t, std::enable_if_t<IsRange_v<Range_t>, bool> = true>
inline CUDA_HOSTDEV auto zip(const Range_t& rng) {
    using iterator    = decltype(std::begin(rng));
    using result_type = ZipRange<Tuple<iterator>>;
    return result_type(adl_make_tuple(rng));
}
*/

namespace detail {

struct Zip {

    template <class Range1_t, class Range2_t>
    inline CUDA_HOSTDEV auto operator()(Range1_t& rng1, Range2_t& rng2) const {
        return make_zip_range(rng1, rng2);
    }

    template <class Range1_t, class Range2_t>
    inline CUDA_HOSTDEV auto operator()(const Range1_t& rng1,
                                        Range2_t&       rng2) const {
        return make_zip_range(rng1, rng2);
    }

    template <class Range1_t, class Range2_t>
    inline CUDA_HOSTDEV auto operator()(Range1_t&       rng1,
                                        const Range2_t& rng2) const {
        return make_zip_range(rng1, rng2);
    }

    template <class Range1_t, class Range2_t>
    inline CUDA_HOSTDEV auto operator()(const Range1_t& rng1,
                                        const Range2_t& rng2) const {
        return make_zip_range(rng1, rng2);
    }
};

} // namespace detail

// This is the niebloid we expose in topaz to allow for customization of
// make_zip_range
inline constexpr detail::Zip zip{};

/*
template <class T1, class T2>
inline CUDA_HOSTDEV auto zip(T1& rng1, T2& rng2) {
    return make_zip_range(rng1, rng2);
}

template <class T1, class T2>
inline CUDA_HOSTDEV auto zip(T1& rng1, const T2& rng2) {
    return make_zip_range(rng1, rng2);
}

template <class T1, class T2>
inline CUDA_HOSTDEV auto zip(const T1& rng1, T2& rng2) {
    return make_zip_range(rng1, rng2);
}

template <class T1, class T2>
inline CUDA_HOSTDEV auto zip(const T1& rng1, const T2& rng2) {
    return make_zip_range(rng1, rng2);
}
*/
} // namespace topaz