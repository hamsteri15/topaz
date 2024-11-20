//#define CATCH_CONFIG_ENABLE_BENCHMARKING
//#define CATCH_CONFIG_MAIN // This tells the catch header to generate a main
#include "catch.hpp"

#include "topaz.hpp"
#include <iostream>

#ifdef __NVIDIA_COMPILER__
#include <thrust/device_vector.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
template<class T>
using vector_t = thrust::device_vector<T>;

template<class T>
using NVec_t = topaz::NumericArray<T, thrust::device_malloc_allocator<T>>;

namespace alglib = thrust;


template<class T>
using MDArray_t = topaz::MdNumericArray<T, thrust::device_malloc_allocator<T>>;

#else
#include <vector>
template<class T>
using vector_t = std::vector<T>;

template<class T>
using NVec_t = topaz::NumericArray<T, std::allocator<T>>;

namespace alglib = std;

template<class T>
using MDArray_t = topaz::MdNumericArray<T, std::allocator<T>>;


#endif

TEST_CASE("Tuple"){

    using namespace topaz;

    auto tpl = adl_make_tuple(int(1), double(4), float(5));
    CHECK(get<0>(tpl) == int(1));

    static_assert(tuple_size<decltype(tpl)>::value == size_t(3));

    auto s_tpl = to_std_tuple(tpl);
    CHECK(std::get<0>(s_tpl) == int(1));

}

TEST_CASE("zip_iterator"){

    SECTION("Comparison"){

        using namespace topaz;
        std::vector<int> v1 = {1,2,3,4};
        std::vector<int> v2 = {1,2,3,4};
        const std::vector<double> v3 = {1.0,2.0,3.0,4.0};

        auto begins = std::make_tuple(v1.begin(), v2.begin(), v3.begin());
        //auto ends = std::make_tuple(v1.end(), v2.end(), v3.end());

        auto iter = make_zip_iterator(begins);
        auto copy = iter;
        CHECK(iter == iter);
        CHECK((iter + 1) != iter);
        CHECK((iter + 1) > iter);
        CHECK((iter + 1) >= (iter + 1));
        CHECK(iter < (iter+1));
        CHECK((iter + 1) <= (iter + 1));




        ++iter;
        CHECK(iter != copy);

        --iter;
        CHECK(iter == copy);

        iter += 1;;
        CHECK(*iter == std::make_tuple(2, 2, 2.0));

    }

    /*
    //TODO: make sure these work
    SECTION("Dereference 1")
    {
        using namespace topaz;
        std::vector<int> v1 = {1,2,3,4};
        std::vector<int> v2 = {1,2,3,4};

        auto begins = std::make_tuple(v1.begin(), v2.begin());


        auto iter = make_zip_iterator(begins);
        std::tuple<int&, int&> vals = *iter;
        //auto vals = iter.dereference();
        std::get<1>(vals) = 5;
        CHECK(v2 == std::vector<int>{5, 2, 3, 4});
    }




    SECTION("Dereference 2")
    {
        using namespace topaz;
        std::vector<int> v1 = {1,2,3,4};
        std::vector<int> v2 = {1,2,3,4};
        std::vector<double> v3 = {1.0,2.0,3.0,4.0};

        auto begins = std::make_tuple(v1.begin(), v2.begin(), v3.begin());


        auto iter = make_zip_iterator(begins);
        std::tuple<int&, int&, double&> vals = *iter;
        //auto vals = iter.dereference();
        std::get<1>(vals) = 5;
        CHECK(v2 == std::vector<int>{5, 2, 3, 4});
    }




    SECTION("Dereference 3")
    {
        using namespace topaz;
        std::vector<int> v1 = {1,2,3,4};
        const std::vector<int> v2 = {4,3,2,1};
        std::vector<double> v3 = {1.0,2.0,3.0,4.0};

        auto begins = std::make_tuple(v1.begin(), v2.begin(), v3.begin());
        auto ends = std::make_tuple(v1.end(), v2.end(), v3.end());


        auto iter = make_zip_iterator(begins);
        std::tuple<int&, const int&, double&> vals = *iter;

        CHECK(std::get<0>(vals) == 1);
        CHECK(std::get<1>(vals) ==  4);
        CHECK(std::get<2>(vals) == 1.0);

        ++iter;

        std::tuple<int&, const int&, double&> vals2 = *iter;

        CHECK(std::get<0>(vals2) == 2);
        CHECK(std::get<1>(vals2) ==  3);
        CHECK(std::get<2>(vals2) == 2.0);

    }

    */


}





TEST_CASE("constant_iterator"){

    using namespace topaz;

    auto rng1 = make_constant_range(int(10), 4);

    #ifdef __NVIDIA_COMPILER__
    //It appears the thrust::constant_iterator can not be used for sorting
    //thrust::sort(thrust::host, rng1.begin(), rng1.end());
    #else

    std::sort(rng1.begin(), rng1.end());

    #endif
    CHECK(std::vector<int>(rng1.begin(), rng1.end()) == std::vector<int>{10, 10, 10, 10});

    CHECK(std::distance(rng1.begin(), rng1.begin() + 2) == 2);


}

TEST_CASE("transform_iterator"){

    using namespace topaz;


    auto op = [] CUDA_HOSTDEV (int i) {return i + 1;};

    vector_t<int> v = std::vector<int>{1,2,3};
    auto r = make_transform_range(v.begin(), v.begin() + 1, op);
    vector_t<int> v2(r.begin(), r.end());
    CHECK(v2[0] == 2);
    CHECK(v2.size() == 1);

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


        CHECK(get<0>(z1[0]) == 1);
        CHECK(get<0>(z2[1]) == 5.0);
        CHECK(get<1>(z3[0]) == 7);

    }


    SECTION("make_transform_iterator"){

        std::vector<int> v = {1,2,3};
        auto op = [](int i) {return i + 1;};

        auto tr = detail::make_transform_iterator(v.begin(), op);
        CHECK(*tr == 2);

    }


    SECTION("make_constant_range"){


        auto r = make_constant_range(4, 4);
        CHECK(r[0] == 4);

    }


    SECTION("zip"){

        SECTION("unary"){

        }

        SECTION("binary"){

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
                vector_t<int> v1 = std::vector<int>{1,2,3};
                vector_t<int> v2 = std::vector<int>{4,5,6};
                auto s1 = transform(v1, v2, Plus{}); //{5,7,9}
                auto s2 = transform(s1, v2, Plus{}); //{9, 12, 15}

                int i  = *s1.begin();
                int i2 = *s2.begin();

                CHECK(i == 5);
                CHECK(i2 == 9);
                //CHECK(std::vector<int>(s2.begin(), s2.end()) == std::vector<int>{9,12,15});
            }


            SECTION("test 4"){
                const vector_t<int> v1 = std::vector<int>{1,2,3};
                const vector_t<int> v2 = std::vector<int>{4,5,6};
                auto s1 = transform(v1, v2, Plus{}); //{5,7,9}
                auto s2 = transform(s1, v2, Plus{}); //{9, 12, 15}
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

            auto s = transform(v1, v2, Plus{});
            auto ss = transform(s, v2, Plus{});
            NVec_t<int> r1(s);
            NVec_t<int> r2(ss);
            CHECK(r1[0] == 4);
            CHECK(r2[0] == 7);
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



#ifdef __NVIDIA_COMPILER__
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


struct Tester{

    CUDA_HOSTDEV
    void operator()(int& e ) const {e += 1;}
};

struct PlusOne{

    CUDA_HOSTDEV
    int operator()(int e ) const {return e + 1;}
};

struct BinaryTester{

    template<class Tuple>
    CUDA_HOSTDEV int operator()(const Tuple& tpl) const {return topaz::get<0>(tpl) + topaz::get<1>(tpl);}
};

namespace topaz{

template<class T>
size_t range_count(const std::vector<NVec_t<T>>& a){return a.size();}

template<class T>
auto md_begin(const std::vector<NVec_t<T>>& a){

    using iterator = typename NVec_t<T>::const_iterator;
    small_array<iterator> ret{};
    for (size_t i = 0; i < range_count(a); ++i)
    {
        ret[i] = adl_begin(a[i]);
    }
    return ret;
}

template<class T>
auto md_begin(std::vector<NVec_t<T>>& a){

    using iterator = typename NVec_t<T>::iterator;
    small_array<iterator> ret{};
    for (size_t i = 0; i < range_count(a); ++i)
    {
        ret[i] = adl_begin(a[i]);
    }
    return ret;
}

template<class T>
auto md_end(const std::vector<NVec_t<T>>& a){

    using iterator = typename NVec_t<T>::const_iterator;
    small_array<iterator> ret{};
    for (size_t i = 0; i < range_count(a); ++i)
    {
        ret[i] = adl_end(a[i]);
    }
    return ret;
}

template<class T>
auto md_end(std::vector<NVec_t<T>>& a){

    using iterator = typename NVec_t<T>::iterator;
    small_array<iterator> ret{};
    for (size_t i = 0; i < range_count(a); ++i)
    {
        ret[i] = adl_end(a[i]);
    }
    return ret;
}


}

TEST_CASE("Test MdRange"){

    using namespace topaz;

    using Array = std::vector<NVec_t<int>>;

    SECTION("make_md_range"){
        Array a1 = {NVec_t<int>{1,3,4}, NVec_t<int>{1,2}};

        REQUIRE_NOTHROW(make_md_range(a1));

        REQUIRE_NOTHROW(make_md_range(make_md_range(a1)));
    }

    SECTION("range_count"){
        Array a1 = {NVec_t<int>{1,3,4}, NVec_t<int>{1,2}};

        Array a2 = {NVec_t<int>{1,3,4}, NVec_t<int>{}, NVec_t<int>{1,2}};

        CHECK(range_count(a1) == 2);
        CHECK(range_count(a2) == 3);

        CHECK(range_count(make_md_range(a1)) == 2);
        CHECK(range_count(make_md_range(a2)) == 3);
    }

    SECTION("md_size()"){
        Array a1 = {NVec_t<int>{1,3,4}, NVec_t<int>{1,2}};
        Array a2 = {NVec_t<int>{}, NVec_t<int>{}};

        small_array<std::ptrdiff_t> c1{};
        c1[0] = 3;
        c1[1] = 2;

        CHECK(md_size(a1) == c1);
        CHECK(md_size(a2) == small_array<std::ptrdiff_t>{});

    }


}

TEST_CASE("Test MdConstantRange"){

    using namespace topaz;

    using Array = std::vector<NVec_t<int>>;

    SECTION("make_md_constant_range"){

        auto sizes = [](){
            small_array<std::ptrdiff_t> ret{};
            ret[0] = 3;
            ret[1] = 5;
            return ret;
        };

        auto rng = make_md_constant_range(int(4), 2, sizes());

        CHECK(rng[0][0] == 4);
        CHECK(rng[1][0] == 4);
        CHECK(rng[1][4] == 4);
        

        CHECK(range_count(rng) == 2);

        CHECK(md_size(rng) == sizes());

    }

        


}




TEST_CASE("Test MdTransformRange"){
    using namespace topaz;

    using Array = std::vector<NVec_t<int>>;

    /*
    SECTION("Test 1 "){
        Array a1 = {NVec_t<int>{1,3,4}, NVec_t<int>{1,2}};
        auto arr = make_md_transform_ranges(make_md_range(a1), Tester{});
        auto& t1 = arr[0];
        for (size_t i = 0; i < t1.size(); ++i){
            t1[i];
        }
        CHECK(std::vector<int>(a1[0].begin(), a1[0].end()) == std::vector<int>{2,4,5});
        CHECK(std::vector<int>(a1[1].begin(), a1[1].end()) == std::vector<int>{1, 2});
    }
    */
    

    SECTION("Test 2 "){
        Array a1 = {NVec_t<int>{1,3,4}, NVec_t<int>{1,2}};

        auto rng = make_md_transform_range(a1, Tester{});


        //auto tr = make_md_transform_range(a1, Tester{});
    }

    SECTION("md_transform()")
    {
           const Array a1 = {NVec_t<int>{1,2,3}, NVec_t<int>{4,5}};

        SECTION("unary"){
            
            
            auto op = PlusOne{};

            auto s1 = md_transform(a1, op);

            CHECK(std::vector<int>(s1[0].begin(), s1[0].end()) == std::vector<int>{2,3,4});
            CHECK(std::vector<int>(s1[1].begin(), s1[1].end()) == std::vector<int>{5,6});

            
            auto s2 = md_transform(a1, op);
            auto s3 = md_transform(s2, op);

            CHECK(std::vector<int>(s3[0].begin(), s3[0].end()) == std::vector<int>{3,4,5});
            CHECK(std::vector<int>(s3[1].begin(), s3[1].end()) == std::vector<int>{6,7});
            

        }
        
        SECTION("binary"){

            SECTION("test 1"){
                auto s = md_transform(a1, a1, Plus{});

                CHECK(std::vector<int>(s[0].begin(), s[0].end()) == std::vector<int>{2,4,6});
                CHECK(std::vector<int>(s[1].begin(), s[1].end()) == std::vector<int>{8,10});
            }

            SECTION("test 2"){
                Array a2 = {NVec_t<int>{1,2}, NVec_t<int>{}};
                Array a3 = {NVec_t<int>{4,5}, NVec_t<int>{}};
                auto s = md_transform(a2, a3, Plus{});

                CHECK(std::vector<int>(s[0].begin(), s[0].end()) == std::vector<int>{5,7});
                CHECK(std::vector<int>(s[1].begin(), s[1].end()) == std::vector<int>{});
            }


            SECTION("test 3"){
                const Array a2 = {NVec_t<int>{1,2}, NVec_t<int>{}};
                const Array a3 = {NVec_t<int>{4,5}, NVec_t<int>{}};

                auto s1 = md_transform(a2, a3, Plus{}); //{5,7}
                auto s2 = md_transform(s1, a2, Plus{}); //{6,9}
                CHECK(std::vector<int>(s2[0].begin(), s2[0].end()) == std::vector<int>{6,9});
                CHECK(std::vector<int>(s2[1].begin(), s2[1].end()) == std::vector<int>{});
            }

        }


        SECTION("md_transform1"){

            Array a1 = {NVec_t<int>{1,3,4}, NVec_t<int>{1,2}};
            Array a2 = {NVec_t<int>{1,3,4}, NVec_t<int>{1,2}};


            auto zip = make_md_zip_range(a1, a2);


            auto rng2 = md_transform(zip, BinaryTester{});


            Array a3(a1);
            a3[0] = NVec_t<int>(rng2[0].begin(), rng2[0].end());
            a3[1] = NVec_t<int>(rng2[1].begin(), rng2[1].end());
            CHECK(a3[0] == NVec_t<int>{2, 6, 8});
            CHECK(a3[1] == NVec_t<int>{2, 4});

            //auto rng2 = md_transform(a1, a2, std::plus<int>{});

        }
        
        SECTION("md_transform2"){

            Array a1 = {NVec_t<int>{1,3,4}, NVec_t<int>{1,2}};
            Array a2 = {NVec_t<int>{1,3,4}, NVec_t<int>{1,2}};


            auto rng2 = md_transform(a1, a2, std::plus<int>{});

            Array a3(a1);
            a3[0] = NVec_t<int>(rng2[0].begin(), rng2[0].end());
            a3[1] = NVec_t<int>(rng2[1].begin(), rng2[1].end());
            CHECK(a3[0] == NVec_t<int>{2, 6, 8});
            CHECK(a3[1] == NVec_t<int>{2, 4});

        }
        
    }

}

TEST_CASE("md_smart_transform"){

    using namespace topaz;
    using Array = std::vector<NVec_t<int>>;

    //using Array = std::vector<NVec_t<int>>;

    Array a1 = {NVec_t<int>{1,3,4}, NVec_t<int>{1,2}};
    //Array a2 = {NVec_t<int>{1,3,4}, NVec_t<int>{1,2}};

    SECTION("md_determine_size"){
        
        CHECK(md_determine_size(a1, int(4)) == md_size(a1));
        CHECK(md_determine_size(int(4), a1) == md_size(a1));
        //CHECK(md_determine_size(a1, a1) == md_size(a1));

    }

    SECTION("md_determine_range_count"){
        
        CHECK(md_determine_range_count(a1, int(4)) == 2);
        CHECK(md_determine_range_count(int(4), a1) == 2);
        //CHECK(md_determine_size(a1, a1) == md_size(a1));

    }

    SECTION("Test 1"){


        //auto asd = smart_transform(a1, a1, std::plus<int>{});

        //auto s1 = smart_transform(a1, int(3), std::plus<int>{});

        //CHECK(std::vector<int>(s1[0].begin(), s1[0].end()) == std::vector<int>{4, 6, 7});
        //CHECK(std::vector<int>(s1[1].begin(), s1[1].end()) == std::vector<int>{4, 5});

        //auto s2 = smart_transform(int(3), a1, std::plus<int>{});

        //CHECK(std::vector<int>(s2[0].begin(), s2[0].end()) == std::vector<int>{4, 6, 7});
        //CHECK(std::vector<int>(s2[1].begin(), s2[1].end()) == std::vector<int>{4, 5});


        //auto t1 = md_smart_transform(a1, int(3), std::plus<int>{});

    }




}


TEST_CASE("Test MdZipRange"){
    using namespace topaz;
    using Array = std::vector<NVec_t<int>>;
    


    SECTION("make_md_zip_range"){

        Array a1 = {NVec_t<int>{1,3,4}, NVec_t<int>{1,2}};

        auto z = make_md_zip_range(a1, a1);
    }
}

TEST_CASE("Test MdNumericArray"){

    using namespace topaz;

    using MDVec_t = MDArray_t<int>;

    MDVec_t a1(NVec_t<int>{3,3}, {NVec_t<int>{3,3,3}, NVec_t<int>{}});
    MDVec_t a2(NVec_t<int>{3,3}, {NVec_t<int>{3,3,3}, NVec_t<int>{}});

    CHECK(range_count(a1) == 3);

    //auto rng = a1 + a2;
    //CHECK(std::vector<int>(rng[0].begin(), rng[0].end()) == std::vector<int>{6,6});

    /*
    auto rng = md_smart_transform(a1, a2, std::plus<int>{});


    CHECK(SupportsBinaryExpression_v<MDVec_t, MDVec_t> == false);
    CHECK(IsNumericVector_v<MDVec_t> == false);
    CHECK(IsRange_v<MDVec_t> == false);
    CHECK(IsScalar_v<MDVec_t> == false);
    CHECK(IsRangeOrNumericArray_v<MDVec_t> == false);
    CHECK(BothRangesOrNumericArrays_v<MDVec_t, MDVec_t> == false);

    */
    //CHECK(SupportsBinaryExpression_v<MDVec_t, MDVec_t> == false);

    //auto r3 = a1 + a2;


}

