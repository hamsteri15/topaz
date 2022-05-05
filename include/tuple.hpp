#pragma once

#ifdef __CUDACC__
#include <thrust/tuple.h>
#else
#include "boost/tuple/tuple.hpp"
#endif
namespace topaz {


    #ifdef __CUDACC__

        template<class... Types>
        using Tuple = thrust::tuple<Types...>;

        using thrust::get;

        template< class... Types >
        inline constexpr CUDA_HOSTDEV
        auto adl_make_tuple( Types&&... args ) {
            return thrust::make_tuple(std::forward<Types>(args)...);
        }

    #else

        template<class... Types>
        using Tuple = boost::tuple<Types...>;

        using boost::get;

        template< class... Types >
        inline constexpr auto adl_make_tuple( Types&&... args ) {
            return boost::make_tuple(std::forward<Types>(args)...);
        }

    #endif


}


