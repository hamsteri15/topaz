#pragma once

#include "begin_end.hpp"
#include "small_array.hpp"

namespace topaz {

template <class Iterator>
struct MdRange {

    using iterator   = Iterator;
    using value_type = typename std::iterator_traits<Iterator>::value_type;
    using reference  = typename std::iterator_traits<Iterator>::reference;
    // using difference_type = typename
    // std::iterator_traits<Iterator>::difference_type;

    inline CUDA_HOSTDEV small_array<iterator> begin() const {return m_begin;}
    inline CUDA_HOSTDEV small_array<iterator> begin() {return m_begin;}

    inline CUDA_HOSTDEV small_array<iterator> end() const {return m_end;}
    inline CUDA_HOSTDEV small_array<iterator> end() {return m_end;}




    small_array<iterator> m_begin, m_end;
};

template <typename Iterator>
CUDA_HOSTDEV size_t range_count(const MdRange<Iterator>& rng) {



}


template <typename Iterator>
CUDA_HOSTDEV auto size(const MdRange<Iterator>& rng) {






}

} // namespace topaz