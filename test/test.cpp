#define CATCH_CONFIG_ENABLE_BENCHMARKING
#define CATCH_CONFIG_MAIN // This tells the catch header to generate a main
#include "catch.hpp"

#include "all.hpp"

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#include <thrust/device_malloc_allocator.h>
template<class T>
using vector_t = thrust::device_vector<T>;

template<class T>
using NVec_t = topaz::NumericArray<T, thrust::device_malloc_allocator<T>>;

template<size_t N, class T>
using NSoa_t = topaz::NumericSoa<N, T, thrust::device_malloc_allocator<T>>;

#else
#include <vector>
template<class T>
using vector_t = std::vector<T>;

template<class T>
using NVec_t = topaz::NumericArray<T, std::allocator<T>>;

template<size_t N, class T>
using NSoa_t = topaz::NumericSoa<N, T, std::allocator<T>>;

#endif

TEST_CASE("Tuple"){

    using namespace topaz;

    auto tpl = adl_make_tuple(int(1), double(4), float(5));
    CHECK(get<0>(tpl) == int(1));

    static_assert(tuple_size<decltype(tpl)>::value == size_t(3));

    auto s_tpl = to_std_tuple(tpl);
    CHECK(std::get<0>(s_tpl) == int(1));

}


TEST_CASE("Range"){


    using namespace topaz;


    SECTION("make_range"){
        vector_t<int> v = std::vector<int>{1,2,3};
        auto rng = make_range(v.begin(), v.end());

        CHECK(*rng.begin() == 1);
        CHECK(rng.size() == 3);
        CHECK(!rng.empty());
        CHECK(rng[1] == 2);

    }

    SECTION("make_zip_range"){

        vector_t<int> v1 = std::vector<int>{1,2,3};
        vector_t<double> v2 = std::vector<double>{4.0,5.0,6.0};
        const vector_t<int> v3 = std::vector<int>{7,8,9};

        auto z1 = make_zip_range(v1, v3);
        auto z2 = make_zip_range(v2, v3);
        auto z3 = make_zip_range(v3, v3);

        /*
        CHECK(std::get<0>(z1[0]) == 1);
        CHECK(std::get<0>(z2[1]) == 5.0);
        CHECK(std::get<1>(z3[0]) == 7);
        */
    }


    SECTION("make_transform_iterator"){

        std::vector<int> v = {1,2,3};
        auto op = [](int i) {return i + 1;};

        auto tr = detail::make_transform_iterator(v.begin(), op);


        CHECK(tr[1] == 3);

    }

    SECTION("make_constant_range"){


        auto r = make_constant_range(4, 4);
        CHECK(r[0] == 4);

    }


    SECTION("zip"){

        SECTION("unary"){

        }

        SECTION("binary"){
            /*
            const vector_t<int> v1 = std::vector<int>{1,2,3};
            vector_t<int> v2 = std::vector<int>{4,5,6};

            auto z1 = zip(v1, v2);

            auto tuple1 = z1[0];
            CHECK(std::get<0>(tuple1) == 1);
            CHECK(std::get<1>(tuple1) == 4);

            auto z2 = zip(v2, v1);
            auto tuple2 = z2[1];
            CHECK(std::get<0>(tuple2) == 5);
            CHECK(std::get<1>(tuple2) == 2);
            */
        }


    }


    SECTION("transform()"){

        SECTION("unary"){
            auto op = [](int i) {return i+1;};
            vector_t<int> v1 = std::vector<int>{1,2,3};

            auto s1 = transform(v1, op);
            CHECK(std::vector<int>(s1.begin(), s1.end()) == std::vector<int>{2,3,4});

            auto s2 = transform(v1, op);
            auto s3 = transform(s2, op);

            CHECK(std::vector<int>(s3.begin(), s3.end()) == std::vector<int>{3, 4, 5});


        }


        SECTION("binary"){

            SECTION("test 1"){
                const vector_t<int> v1 = std::vector<int>{1,1,1};
                auto s1 = transform(v1, v1, Plus{});
                CHECK(std::vector<int>(s1.begin(), s1.end()) == std::vector<int>{2,2,2});
            }


            SECTION("test 2"){
                const vector_t<int> v1 = std::vector<int>{1,2,3};
                const vector_t<int> v2 = std::vector<int>{4,5,6};
                auto s1 = transform(v1, v2, Plus{});
                CHECK(std::vector<int>(s1.begin(), s1.end()) == std::vector<int>{5,7,9});
            }

            SECTION("test 3"){
                const vector_t<int> v1 = std::vector<int>{1,2,3};
                const vector_t<int> v2 = std::vector<int>{4,5,6};
                auto s1 = transform(v1, v2, Plus{}); //{5,7,9}
                auto s2 = transform(s1, v2, Plus{}); //{9, 12, 15}
                CHECK(std::vector<int>(s2.begin(), s2.end()) == std::vector<int>{9,12,15});
            }


        }


    }



}

struct PlusTriplet{

    template<class T>
    CUDA_HOSTDEV auto operator()(const T& a, const T& b, const T& c) ->decltype(a + b + c)
    {
        return a+b+c;
    }

};

TEST_CASE("ChunkedRange"){

    using namespace topaz;

    SECTION("make_chunked_range"){

        vector_t<int> v = std::vector<int>{1,2,3,4};

        auto r1 = make_chunked_range<2>(v);
        //CHECK(r1[0] == 1);
        CHECK(chunk_size(r1) == 2);

        //REQUIRE_THROWS(make_chunked_range<3>(v));

    }

    SECTION("get_chunk"){
        vector_t<int> v = std::vector<int>{1,2,3,4};
        auto r1 = make_chunked_range<2>(v);
        auto c1 = get_chunk<0>(r1);
        auto c2 = get_chunk<1>(r1);
        CHECK(size(c1) == 2);
        CHECK(size(c2) == 2);
        CHECK(c1[0] == 1);
        CHECK(c1[1] == 2);
        CHECK(c2[0] == 3);
        CHECK(c2[1] == 4);
    }


    SECTION("get_chunks"){
       const vector_t<int> v = std::vector<int>{1,2,3,4};

        auto tpl = get_chunks<2>(v);
        auto c1 = get<0>(tpl);
        auto c2 = get<1>(tpl);
        CHECK(size(c1) == 2);
        CHECK(size(c2) == 2);
        CHECK(c1[0] == 1);
        CHECK(c1[1] == 2);
        CHECK(c2[0] == 3);
        CHECK(c2[1] == 4);

    }


    SECTION("zip_begins"){


       const vector_t<int> v = std::vector<int>{1,2,3,4};

        auto it = zip_begins<2>(v);

        auto tpl1 = *it++;
        CHECK(get<0>(tpl1) == 1);
        CHECK(get<1>(tpl1) == 3);


    }



    SECTION("chunked_reduce"){

        SECTION("test1"){

            const vector_t<int> v = std::vector<int>{1,2,3,4};

            auto temp = chunked_reduce<2>(v, Plus{});

            CHECK(adl_size(temp) == 2);
            CHECK(temp[0] == 1 + 3);
            CHECK(temp[1] == 2 + 4);

        }

        SECTION("test2"){

            const vector_t<int> v = std::vector<int>{1,2,3,4,5,6};

            auto temp = chunked_reduce<3>(v, PlusTriplet{});

            CHECK(adl_size(temp) == 2);
            CHECK(temp[0] == 1 + 3 + 5);
            CHECK(temp[1] == 2 + 4 + 6);

        }



    }





}

TEST_CASE("NumericArray"){


    using namespace topaz;


    SECTION("Constructors"){

        NVec_t<int> v(5, 2);
        CHECK(v.size() == 5);
        CHECK(*v.begin() == 2);

        NVec_t<int> v2(v);
        CHECK(v2.size() == 5);
        CHECK(*v2.begin() == 2);

        NVec_t<int> v3(v.begin(), v.end());
        CHECK(v3.size() == 5);
        CHECK(*v3.begin() == 2);

        CHECK(v3[0] == 2);
        CHECK(v3[3] == 2);

    }

    SECTION("Assignment"){
        NVec_t<int> v1(3, 2);
        NVec_t<int> v2(3, 3);
        NVec_t<int> v3(3, 0);

        v1 = v2;
        CHECK(std::vector<int>(v1.begin(), v1.end()) == std::vector<int>{3,3,3});

        v3 = v1 + v2 + v1;

        CHECK(std::vector<int>(v3.begin(), v3.end()) == std::vector<int>{9,9,9});

    }


    SECTION("transform"){

        SECTION("Unary"){

            NVec_t<int> v1(10, 1);

            auto op = [](int i) {return i + 1;};
            auto s = transform(v1, op);
            CHECK(s[0] == 2);
            CHECK(s[1] == 2);

        }


        SECTION("Binary"){
            NVec_t<int> v1(10, 1);
            NVec_t<int> v2(10, 3);

            auto s = transform(v1, v2, Plus{});
            auto ss = transform(s, v2, Plus{});
            CHECK(s[0] == 4);
            CHECK(ss[0] == 7);
        }

    }

    SECTION("determine_size"){

        const NVec_t<int> v1(3, 1);
        int t = 43;


        static_assert(IsScalar_v<int>, "Is scalar");
        static_assert(IsScalar_v<double>, "Is scalar");

        CHECK(determine_size(v1, v1) == 3);
        CHECK(determine_size(v1, t) == 3);
        CHECK(determine_size(t, v1) == 3);

        auto tr = transform(v1, v1, Plus{});

        static_assert(IsRange_v<decltype(tr)>, "Not range");

        CHECK(determine_size(tr, t) == 3);
        CHECK(determine_size(tr, tr) == 3);
        CHECK(determine_size(t, tr) == 3);

    }


    SECTION("smart_transform()"){

        const NVec_t<int> v1{1,1,1};
        const NVec_t<int> v2{2,2,2};
        int t = 2;

        auto r1 = smart_transform(v1, t, Plus{});
        CHECK(std::vector<int>{r1.begin(), r1.end()} == std::vector<int>{3,3,3});

        auto r2 = smart_transform(v1, v2, Plus{});
        CHECK(std::vector<int>{r2.begin(), r2.end()} == std::vector<int>{3,3,3});

        auto r3 = smart_transform(t, v1, Plus{});
        CHECK(std::vector<int>{r3.begin(), r3.end()} == std::vector<int>{3,3,3});

        auto r4 = smart_transform(r3, t, Plus{});
        CHECK(std::vector<int>{r4.begin(), r4.end()} == std::vector<int>{5,5,5});

        auto r5 = smart_transform(r3, r3, Plus{});
        CHECK(std::vector<int>{r5.begin(), r5.end()} == std::vector<int>{6,6,6});


    }

    SECTION("Arithmetic"){

        const NVec_t<int> v1{1,2,3};
        const NVec_t<int> v2{4,5,6};

        auto v3 = 12 * v1 * v2 / v1 / 3 - v1 * v2 + 43 * v1 - 2;


        std::vector<int> correct = {
            12 * 1 * 4 / 1 / 3 - 1 * 4 + 43 * 1 - 2,
            12 * 2 * 5 / 2 / 3 - 2 * 5 + 43 * 2 - 2,
            12 * 3 * 6 / 3 / 3 - 3 * 6 + 43 * 3 - 2
        };

        CHECK(std::vector<int>{v3.begin(), v3.end()} == correct);


        auto t1 = v1 + v2;
        auto t2 = t1 + v1;
        CHECK(std::vector<int>{t2.begin(), t2.end()} == std::vector<int>{6, 9, 12});


    }

}

TEST_CASE("NumericSoa"){


    using namespace topaz;

    SECTION("Constructors"){

        REQUIRE_NOTHROW(NSoa_t<3, int>());
        REQUIRE_NOTHROW(NSoa_t<3, int>(50));

        NVec_t<int> a(10, 1);
        NVec_t<int> b(10, 2);
        NVec_t<int> c(7, 3);

        REQUIRE_NOTHROW(NSoa_t<2, int> (std::array<NVec_t<int>, 2>{a,b}));
        REQUIRE_THROWS(NSoa_t<2, int> (std::array<NVec_t<int>, 2>{a,c}));

        NSoa_t<2, int> soa(std::array<NVec_t<int>, 2>{a,b});


        for (auto it = soa.zipped_begin(); it != soa.zipped_end(); ++it){
            REQUIRE(get<0>(*it) == 1);
            REQUIRE(get<1>(*it) == 2);
        }



    }

    SECTION("begin/end"){

        NSoa_t<3, int> soa(10);
        auto begin = soa.begin();
        auto end = soa.end();
        REQUIRE_NOTHROW(++begin);
        REQUIRE_NOTHROW(--end);


        const NSoa_t<3, int> soa2(10);
        auto z_begin1 = soa2.zipped_begin();
        auto z_begin2 = soa.zipped_begin();


        CHECK(get<0>(*z_begin1) == 0);
        CHECK(get<0>(*z_begin2) == 0);

        auto z_end1 = soa2.zipped_end();
        auto z_end2 = soa.zipped_end();

        --z_end1;
        --z_end2;

        CHECK(get<0>(*z_end1) == 0);
        CHECK(get<0>(*z_end2) == 0);

        for (auto it = soa.zipped_begin(); it != soa.zipped_end(); ++it){
            *it = adl_make_tuple(1,2,3);
        }


        //CHECK(get<0>(t1) == 0);



    }

    SECTION("Access"){
        NSoa_t<3, int> soa(10);
        for (auto it = soa.zipped_begin(); it != soa.zipped_end(); ++it){
            *it = adl_make_tuple(1,2,3);
        }
        auto tpl = get_chunks<3>(soa);
        auto x = get<0>(tpl);
        auto y = get<1>(tpl);
        auto z = get<2>(tpl);
        CHECK(x[1] == 1);
        CHECK(y[1] == 2);
        CHECK(z[1] == 3);

    }



    SECTION("transform"){
        /*
        NSoa_t<3, int> soa(10);


        auto op = [](Tuple<int, int, int> t) {
            return adl_make_tuple(1,2,3);
        };

        auto tra_rng = make_transform_range(soa, op);

        CHECK(get<0>(tra_rng[1]) == 1);
        CHECK(get<1>(tra_rng[1]) == 2);
        CHECK(get<2>(tra_rng[1]) == 3);
        */

    }

    SECTION("arithmetic"){
        NSoa_t<3, int> s1(10);
        NSoa_t<3, int> s2(10);

        auto s3 = s1 + s2;

    }

}
