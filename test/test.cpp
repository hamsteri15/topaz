#define CATCH_CONFIG_ENABLE_BENCHMARKING
#define CATCH_CONFIG_MAIN // This tells the catch header to generate a main
#include "catch.hpp"

#include "all.hpp"

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/host_vector.h>
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

    SECTION("async_copy()"){

        vector_t<int> v1 = std::vector<int>{1,2,3};
        vector_t<int> v2 = std::vector<int>{1,1,1};
        vector_t<int> v3 = std::vector<int>{0,0,0};

        auto event1 = async_copy(v1, v2);
        auto event2 = async_copy(event1, v1, v3);

        event2.wait();

        CHECK(std::vector<int>(v2.begin(), v2.end()) == std::vector<int>{1,2,3});
        CHECK(std::vector<int>(v3.begin(), v3.end()) == std::vector<int>{1,2,3});


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

        SECTION("plus, minus, divides, multiplies"){

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
        SECTION("min/max"){
            const NVec_t<int> v1{1,8,3};
            const NVec_t<int> v2{4,5,6};
            auto m1 = topaz::max(v1, v2);
            CHECK(std::vector<int>{m1.begin(), m1.end()} == std::vector<int>{4, 8, 6});

            auto m2 = topaz::max(v1, 3);
            CHECK(std::vector<int>{m2.begin(), m2.end()} == std::vector<int>{3, 8, 3});

            auto m3 = topaz::max(3, v1);
            CHECK(std::vector<int>{m3.begin(), m3.end()} == std::vector<int>{3, 8, 3});

            auto m4 = topaz::min(v1, v2);
            CHECK(std::vector<int>{m4.begin(), m4.end()} == std::vector<int>{1, 5, 3});

            auto m5 = topaz::min(v1, 3);
            CHECK(std::vector<int>{m5.begin(), m5.end()} == std::vector<int>{1, 3, 3});

            auto m6 = topaz::min(3, v1);
            CHECK(std::vector<int>{m6.begin(), m6.end()} == std::vector<int>{1, 3, 3});

        }

        SECTION("sqr"){

            const NVec_t<int> v1{1,2,3};
            auto t1 = sqr(v1);
            CHECK(std::vector<int>{t1.begin(), t1.end()} == std::vector<int>{1, 4, 9});

        }
        SECTION("sqrt float"){

            const NVec_t<float> v1{1,2,3};
            auto t1 = sqrt(v1);
            CHECK(std::vector<float>{t1.begin(), t1.end()}
                    == std::vector<float>{adl_sqrt(float(1)), adl_sqrt(float(2)), adl_sqrt(float(3))});

        }
        SECTION("sqrt float"){

            const NVec_t<double> v1{1,2,3};
            auto t1 = sqrt(v1);
            CHECK(std::vector<double>{t1.begin(), t1.end()}
                    == std::vector<double>{adl_sqrt(double(1)), adl_sqrt(double(2)), adl_sqrt(double(3))});

        }
        SECTION("pow float"){

            const NVec_t<float> v1{1,2,3};
            auto t1 = pow(v1, float(2.0));
            CHECK(std::vector<float>{t1.begin(), t1.end()}
                    == std::vector<float>{pow(float(1), float(2)), pow(float(2),float(2)), pow(float(3),float(2))});

        }
        SECTION("pow double"){

            const NVec_t<double> v1{1,2,3};
            auto t1 = pow(v1, 2.0);
            CHECK(std::vector<double>{t1.begin(), t1.end()}
                    == std::vector<double>{pow(1.0, 2.0), pow(2.0, 2.0), pow(3.0, 2.0)});

        }
        SECTION("pow2"){

            const NVec_t<double> v1{1,2,3};
            auto t1 = pow2(v1);
            CHECK(std::vector<double>{t1.begin(), t1.end()}
                    == std::vector<double>{pow(1.0, 2.0), pow(2.0, 2.0), pow(3.0, 2.0)});

        }
        SECTION("pow3"){

            const NVec_t<double> v1{1,2,3};
            auto t1 = pow3(v1);
            CHECK(std::vector<double>{t1.begin(), t1.end()}
                    == std::vector<double>{pow(1.0, 3.0), pow(2.0, 3.0), pow(3.0, 3.0)});

        }
        SECTION("exp float"){

            const NVec_t<float> v1{1,2,3};
            auto t1 = exp(v1);
            CHECK(std::vector<float>{t1.begin(), t1.end()}
                    == std::vector<float>{exp(float(1)), exp(float(2)), exp(float(3))});

        }
        SECTION("exp double"){

            const NVec_t<double> v1{1,2,3};
            auto t1 = exp(v1);
            CHECK(std::vector<double>{t1.begin(), t1.end()}
                    == std::vector<double>{exp(1.0), exp(2.0), exp(3.0)});

        }
        SECTION("log float"){

            const NVec_t<float> v1{1,2,3};
            auto t1 = log(v1);
            CHECK(std::vector<float>{t1.begin(), t1.end()}
                    == std::vector<float>{log(float(1)), log(float(2)), log(float(3))});

        }
        SECTION("log double"){

            const NVec_t<double> v1{1,2,3};
            auto t1 = log(v1);
            CHECK(std::vector<double>{t1.begin(), t1.end()}
                    == std::vector<double>{log(1.0), log(2.0), log(3.0)});

        }
        SECTION("erf float"){

            const NVec_t<float> v1{1,2,3};
            auto t1 = erf(v1);
            CHECK(std::vector<float>{t1.begin(), t1.end()}
                    == std::vector<float>{erff(float(1)), erff(float(2)), erff(float(3))});

        }
        SECTION("erf double"){

            const NVec_t<double> v1{1,2,3};
            auto t1 = erf(v1);
            CHECK(std::vector<double>{t1.begin(), t1.end()}
                    == std::vector<double>{erf(1.0), erf(2.0), erf(3.0)});

        }

    }

}


#ifdef __CUDACC__
TEST_CASE("Cuda only"){

    SECTION("parallel_force_evaluate"){

        const NVec_t<int> v1{1,2,3};
        const NVec_t<int> v2{4,5,6};
        NVec_t<int> result{0,0,0};
        auto kernel = v1 + v2;


            cudaStream_t s;
            cudaStreamCreate(&s);
            topaz::parallel_force_evaluate(
                thrust::cuda::par.on(s), kernel, result
            );
            cudaStreamSynchronize(s);
            cudaStreamDestroy(s);



        CHECK(std::vector<int>{result.begin(), result.end()}
            ==std::vector<int>{5,7,9});


    }


    SECTION("memcopies"){

        SECTION("serial"){
            thrust::host_vector<int> v1 = std::vector<int>{1,2,3};
            thrust::device_vector<int> v2 = std::vector<int>{0,0,0};
            thrust::host_vector<int> v3 = std::vector<int>{0,0,0};

            topaz::host_to_device(v1, v2);

            topaz::device_to_host(v2, v3);

            CHECK(std::vector<int>{v3.begin(), v3.end()} ==
                std::vector<int>{1, 2, 3});

        }

        SECTION("async"){
            thrust::host_vector<int> v1 = std::vector<int>{1,2,3};
            thrust::device_vector<int> v2 = std::vector<int>{0,0,0};
            thrust::host_vector<int> v3 = std::vector<int>{0,0,0};

            cudaStream_t s;
            cudaStreamCreate(&s);

            topaz::async_host_to_device(v1, v2, s);

            topaz::async_device_to_host(v2, v3, s);

            cudaStreamSynchronize(s);
            cudaStreamDestroy(s);

            CHECK(std::vector<int>{v3.begin(), v3.end()} ==
                std::vector<int>{1, 2, 3});

        }
    }
}
#endif

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


struct Vec3{

    Vec3() = default;

    Vec3(std::initializer_list<double> l) {
        std::copy(l.begin(), l.end(), data_);
    }

    double data_[3];
    //std::array<double, 3> data_;

};

std::ostream& operator<<(std::ostream& os, const Vec3& v){
    os << "{ ";
    os << v.data_[0] << " ";
    os << v.data_[1] << " ";
    os << v.data_[2] << " ";
    os << "}";
    return os;
}

bool operator==(const Vec3& lhs, const Vec3& rhs){
    for (size_t i = 0; i < 3; ++i){
        if (lhs.data_[i] != rhs.data_[i]){
            return false;
        }
    }
    return true;
}

CUDA_HOSTDEV
auto operator+(const Vec3& lhs, const Vec3& rhs){
    Vec3 ret;
    for (size_t i = 0; i < 3; ++i){
        ret.data_[i] = lhs.data_[i] + rhs.data_[i];
    }
    return ret;    
}


TEST_CASE("Custom type Numeric Array"){

    SECTION("Arithmetic"){
        NVec_t<Vec3> v1(3, Vec3{1,2,3});
        NVec_t<Vec3> v2(3, Vec3{4,5,6});

        NVec_t<Vec3> v3 = v1 + v2;

        CHECK(v3[0] == Vec3{5, 7, 9});
        CHECK(v3[1] == Vec3{5, 7, 9});

        //Vec3 v1 = {1.0, 2.0, 3.0};
        //Vec
    }



}
