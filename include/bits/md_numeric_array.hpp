#pragma once

#include "md_range.hpp"
#include "md_traits.hpp"
#include "numeric_array.hpp"

namespace topaz {

template <class T, class Allocator>
struct MdNumericArray {

    static constexpr bool is_md_numeric_vector = true;

    using iterator        = typename NumericArray<T, Allocator>::iterator;
    using value_type      = T;
    using size_type       = typename NumericArray<T, Allocator>::size_type;
    using difference_type = std::ptrdiff_t;

    inline MdNumericArray() = default;

    inline MdNumericArray(
        const NumericArray<T, Allocator>&              internal,
        const std::vector<NumericArray<T, Allocator>>& boundary)
        : m_internal(internal)
        , m_boundary(boundary) {}

    auto& internalField() { return m_internal; }

    const auto& internalField() const { return m_internal; }

    auto& boundaryField() { return m_boundary; }

    const auto& boundaryField() const { return m_boundary; }

private:
    NumericArray<T, Allocator>              m_internal;
    std::vector<NumericArray<T, Allocator>> m_boundary;
};

template <class T, class Allocator>
size_t range_count(const MdNumericArray<T, Allocator>& a) {
    return 1 + a.boundaryField().size();
}

template <class T, class Allocator>
auto make_md_range(const MdNumericArray<T, Allocator>& a) {

    using iterator                     = decltype(a.internalField().begin());
    auto                         count = range_count(a);
    small_array<Range<iterator>> ranges{};
    ranges[0] = make_range(a.internalField());

    for (size_t i = 0; i < a.boundaryField().size(); ++i) {
        ranges[i + 1] = make_range(a.boundaryField()[i]);
    }
    return MdRange<iterator>(count, ranges);
}

template <class T, class Allocator>
auto make_md_range(MdNumericArray<T, Allocator>& a) {

    using iterator                     = decltype(a.internalField().begin());
    auto                         count = range_count(a);
    small_array<Range<iterator>> ranges{};
    ranges[0] = make_range(a.internalField());

    for (size_t i = 0; i < a.boundaryField().size(); ++i) {
        ranges[i + 1] = make_range(a.boundaryField()[i]);
    }
    return MdRange<iterator>(count, ranges);
}

template <class T, class Allocator>
auto md_begin(const MdNumericArray<T, Allocator>& a) {

    using iterator = decltype(a.internalField().begin());
    small_array<iterator> ret{};

    ret[0] = a.internalField().begin();

    for (size_t i = 0; i < a.boundaryField().size(); ++i) {
        ret[i + 1] = a.boundaryField()[i].begin();
    }

    return ret;
}

template <class T, class Allocator>
auto md_begin(MdNumericArray<T, Allocator>& a) {

    using iterator = decltype(a.internalField().begin());
    small_array<iterator> ret{};

    ret[0] = a.internalField().begin();

    for (size_t i = 0; i < a.boundaryField().size(); ++i) {

        ret[i + 1] = a.boundaryField()[i].begin();
    }

    return ret;
}

template <class T, class Allocator>
auto md_end(const MdNumericArray<T, Allocator>& a) {

    using iterator = decltype(a.internalField().begin());
    small_array<iterator> ret{};

    ret[0] = a.internalField().end();

    for (size_t i = 0; i < a.boundaryField().size(); ++i) {

        ret[i + 1] = a.boundaryField()[i].end();
    }

    return ret;
}

template <class T, class Allocator>
auto md_end(MdNumericArray<T, Allocator>& a) {

    using iterator              = decltype(a.internalField().begin());
    auto                  count = range_count(a);
    small_array<iterator> ret{};

    ret[0] = a.internalField().end();

    for (size_t i = 0; i < a.boundaryField().size(); ++i) {
        ret[i + 1] = a.boundaryField()[i].end();
    }

    return ret;
}

template <class T, class Allocator>
CUDA_HOSTDEV auto md_size(const MdNumericArray<T, Allocator>& a) {

    using iterator = decltype(a.internalField().begin());

    using integer_type =
        typename std::iterator_traits<iterator>::difference_type;
    small_array<integer_type> ret{};
    ret[0] = a.internalField().size();

    for (size_t i = 0; i < a.boundaryField().size(); ++i) {
        ret[i + 1] = a.boundaryField()[i].size();
    }
    return ret;
}

template <class T, class Allocator>
struct SupportsBinaryExpression<MdNumericArray<T, Allocator>,
                                MdNumericArray<T, Allocator>> : std::true_type {
};


} // namespace topaz
