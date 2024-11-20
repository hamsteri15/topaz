#pragma once

#include "constant_iterator.hpp"
#include "md_range.hpp"

namespace topaz {

template <typename Value>
struct MdConstantRange : public MdRange<constant_iterator<Value>> {

private:
    using iterator        = constant_iterator<Value>;
    using parent          = MdRange<constant_iterator<Value>>;
    using value_type      = typename parent::value_type;
    using difference_type = typename parent::difference_type;

public:
    inline CUDA_HOSTDEV MdConstantRange(
        value_type c, size_t count, const small_array<difference_type>& sizes)
        : parent(count, make(c, count, sizes)) {}

private:
    inline CUDA_HOSTDEV small_array<Range<iterator>> make(
        value_type c, size_t count, const small_array<difference_type>& sizes) {

        small_array<Range<iterator>> ret{};
        for (size_t i = 0; i < count; ++i) {
            ret[i] = make_constant_range(c, sizes[i]);
        }
        return ret;
    }
};


template <typename Value, typename Size>
inline CUDA_HOSTDEV auto
make_constant_range(Value c, size_t count, const small_array<Size>& sizes) {

    return MdConstantRange<Value>(c, count, sizes);
}


} // namespace topaz