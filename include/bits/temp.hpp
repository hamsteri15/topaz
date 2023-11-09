#pragma once

#include <type_traits>
#include <utility>
#include <iterator>

#include <boost/mpl/at.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/placeholders.hpp>

#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/fusion/algorithm/transformation/transform.hpp>
#include <boost/fusion/sequence/convert.hpp>
#include <boost/fusion/sequence/intrinsic/at_c.hpp>
#include <boost/fusion/sequence/comparison/equal_to.hpp>
#include <boost/fusion/support/tag_of_fwd.hpp>

namespace topaz{




template <class Iterator>
struct iterator_reference
{
    typedef typename std::iterator_traits<Iterator>::reference type;
};

struct dereference_iterator
{
    template<typename>
    struct result;

    template<typename This, typename Iterator>
    struct result<This(Iterator)>
    {
    using iterator = typename
        std::remove_cv<typename std::remove_reference<Iterator>::type>::type;

    using type = typename iterator_reference<iterator>::type;
    };

    template<typename Iterator>
    typename result<dereference_iterator(Iterator)>::type
    operator()(Iterator const& it) const
    { return *it; }
};

// Metafunction to obtain the type of the tuple whose element types
    // are the reference types of an iterator tuple.
    //
template<typename IteratorTuple>
struct tuple_of_references
    : boost::mpl::transform<
        IteratorTuple,
        iterator_reference<boost::mpl::_1>
        >
{
};

template <typename reference>
struct converter
{
    template <typename Seq>
    static reference call(Seq seq)
    {
        typedef typename boost::fusion::traits::tag_of<reference>::type tag;
        return boost::fusion::convert<tag>(seq);
    }
};


}