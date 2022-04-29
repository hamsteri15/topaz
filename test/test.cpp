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

#else
#include <vector>
template<class T>
using vector_t = std::vector<T>;

template<class T>
using NVec_t = topaz::NumericArray<T, std::allocator<T>>;
#endif

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
                auto s1 = transform(v1, v1, Plus<int>{});
                CHECK(std::vector<int>(s1.begin(), s1.end()) == std::vector<int>{2,2,2});
            }


            SECTION("test 2"){
                const vector_t<int> v1 = std::vector<int>{1,2,3};
                const vector_t<int> v2 = std::vector<int>{4,5,6};
                auto s1 = transform(v1, v2, Plus<int>{});
                CHECK(std::vector<int>(s1.begin(), s1.end()) == std::vector<int>{5,7,9});
            }

            SECTION("test 3"){
                const vector_t<int> v1 = std::vector<int>{1,2,3};
                const vector_t<int> v2 = std::vector<int>{4,5,6};
                auto s1 = transform(v1, v2, Plus<int>{}); //{5,7,9}
                auto s2 = transform(s1, v2, Plus<int>{}); //{9, 12, 15}
                CHECK(std::vector<int>(s2.begin(), s2.end()) == std::vector<int>{9,12,15});
            }


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

            auto s = transform(v1, v2, Plus<int>{});
            auto ss = transform(s, v2, Plus<int>{});
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

        auto tr = transform(v1, v1, Plus<int>{});

        static_assert(IsRange_v<decltype(tr)>, "Not range");

        CHECK(determine_size(tr, t) == 3);
        CHECK(determine_size(tr, tr) == 3);
        CHECK(determine_size(t, tr) == 3);

    }


    SECTION("smart_transform()"){

        const NVec_t<int> v1{1,1,1};
        const NVec_t<int> v2{2,2,2};
        int t = 2;

        auto r1 = smart_transform(v1, t, Plus<int>{});
        CHECK(std::vector<int>{r1.begin(), r1.end()} == std::vector<int>{3,3,3});

        auto r2 = smart_transform(v1, v2, Plus<int>{});
        CHECK(std::vector<int>{r2.begin(), r2.end()} == std::vector<int>{3,3,3});

        auto r3 = smart_transform(t, v1, Plus<int>{});
        CHECK(std::vector<int>{r3.begin(), r3.end()} == std::vector<int>{3,3,3});

        auto r4 = smart_transform(r3, t, Plus<int>{});
        CHECK(std::vector<int>{r4.begin(), r4.end()} == std::vector<int>{5,5,5});

        auto r5 = smart_transform(r3, r3, Plus<int>{});
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
