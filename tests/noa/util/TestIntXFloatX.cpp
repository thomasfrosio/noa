#include <catch2/catch.hpp>

#include "noa/util/IntX.h"
#include "noa/util/FloatX.h"

using namespace ::Noa;

#define REQUIRE_FOR_ALL_TYPES(type_trait, type)               \
REQUIRE((type_trait<type>));                                  \
REQUIRE((type_trait<std::add_const_t<type>>));                \
REQUIRE((type_trait<std::add_volatile_t<type>>));             \
REQUIRE((type_trait<std::add_cv_t<type>>));                   \
REQUIRE((type_trait<std::add_lvalue_reference_t<type>>));     \
REQUIRE((type_trait<std::add_rvalue_reference_t<type>>))

#define REQUIRE_FALSE_FOR_ALL_TYPES(type_trait, type)               \
REQUIRE_FALSE((type_trait<type>));                                  \
REQUIRE_FALSE((type_trait<std::add_const_t<type>>));                \
REQUIRE_FALSE((type_trait<std::add_volatile_t<type>>));             \
REQUIRE_FALSE((type_trait<std::add_cv_t<type>>));                   \
REQUIRE_FALSE((type_trait<std::add_lvalue_reference_t<type>>));     \
REQUIRE_FALSE((type_trait<std::add_rvalue_reference_t<type>>))

#define REQUIRE_FOR_ALL_TYPES_INT(type_traits)              \
REQUIRE_FOR_ALL_TYPES(type_traits, ::Noa::Int2<TestType>);  \
REQUIRE_FOR_ALL_TYPES(type_traits, ::Noa::Int3<TestType>);  \
REQUIRE_FOR_ALL_TYPES(type_traits, ::Noa::Int4<TestType>)

#define REQUIRE_FALSE_FOR_ALL_TYPES_INT(type_traits)                \
REQUIRE_FALSE_FOR_ALL_TYPES(type_traits, ::Noa::Int2<TestType>);    \
REQUIRE_FALSE_FOR_ALL_TYPES(type_traits, ::Noa::Int3<TestType>);    \
REQUIRE_FALSE_FOR_ALL_TYPES(type_traits, ::Noa::Int4<TestType>)

#define REQUIRE_FOR_ALL_TYPES_FLOAT(type_traits)                \
REQUIRE_FOR_ALL_TYPES(type_traits, ::Noa::Float2<TestType>);    \
REQUIRE_FOR_ALL_TYPES(type_traits, ::Noa::Float3<TestType>);    \
REQUIRE_FOR_ALL_TYPES(type_traits, ::Noa::Float4<TestType>)

#define REQUIRE_FALSE_FOR_ALL_TYPES_FLOAT(type_traits)              \
REQUIRE_FALSE_FOR_ALL_TYPES(type_traits, ::Noa::Float2<TestType>);  \
REQUIRE_FALSE_FOR_ALL_TYPES(type_traits, ::Noa::Float3<TestType>);  \
REQUIRE_FALSE_FOR_ALL_TYPES(type_traits, ::Noa::Float4<TestType>)

#define F(x) static_cast<TestType>(x)


TEMPLATE_TEST_CASE("Traits: vectors", "[noa][traits]",
                   uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t,
                   float, double) {
    using namespace ::Noa::Traits;

    if constexpr (std::is_same_v<TestType, float> ||
                  std::is_same_v<TestType, double>) {
        REQUIRE_FOR_ALL_TYPES_FLOAT(is_floatX_v);

        REQUIRE_FALSE(is_float3_v<Float2<TestType>>);
        REQUIRE_FALSE(is_float4_v<Float2<TestType>>);
        REQUIRE_FALSE(is_float2_v<Float3<TestType>>);
        REQUIRE_FALSE(is_float4_v<Float3<TestType>>);
        REQUIRE_FALSE(is_float2_v<Float4<TestType>>);
        REQUIRE_FALSE(is_float3_v<Float4<TestType>>);

        REQUIRE_FALSE_FOR_ALL_TYPES_FLOAT(is_intX_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_FLOAT(is_int2_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_FLOAT(is_int3_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_FLOAT(is_int4_v);

    } else {
        REQUIRE_FOR_ALL_TYPES_INT(is_intX_v);

        REQUIRE_FALSE(is_int3_v<Int2<TestType>>);
        REQUIRE_FALSE(is_int4_v<Int2<TestType>>);
        REQUIRE_FALSE(is_int2_v<Int3<TestType>>);
        REQUIRE_FALSE(is_int4_v<Int3<TestType>>);
        REQUIRE_FALSE(is_int2_v<Int4<TestType>>);
        REQUIRE_FALSE(is_int3_v<Int4<TestType>>);

        REQUIRE_FALSE_FOR_ALL_TYPES_INT(is_floatX_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_INT(is_float2_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_INT(is_float3_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_INT(is_float4_v);
    }
}


TEMPLATE_TEST_CASE("Vectors: Int2", "[noa][vectors]", int32_t, int64_t, uint32_t, uint64_t) {
    using Int = Int2<TestType>;

    //@CLION-formatter:off
    Int test{};
    REQUIRE((test == Int{TestType{0}}));
    test = TestType{2};
    test += TestType{1}; REQUIRE(test == TestType{3});
    test -= TestType{2}; REQUIRE(test == TestType{1});
    test *= TestType{3}; REQUIRE(test == TestType{3});
    test /= TestType{2}; REQUIRE(test == TestType{1});

    test = TestType{30};
    REQUIRE((test + TestType{10} == TestType{40}));
    REQUIRE((test - TestType{5} == TestType{25}));
    REQUIRE((test * TestType{3} == TestType{90}));
    REQUIRE((test / TestType{2} == TestType{15}));

    test = TestType{30};
    REQUIRE((TestType{40} == TestType{10} + test));
    REQUIRE((TestType{15} == TestType{45} - test));
    REQUIRE((TestType{90} == TestType{3} * test));
    REQUIRE((TestType{2} == TestType{60} / test));

    test = {4, 10};
    REQUIRE_FALSE(test > TestType{5});
    REQUIRE(test < TestType{11});
    REQUIRE_FALSE(test >= TestType{7});
    REQUIRE_FALSE(test <= TestType{9});
    REQUIRE(test != TestType{4});

    REQUIRE_FALSE(TestType{5} < test);
    REQUIRE(TestType{11} > test);
    REQUIRE_FALSE(TestType{7} <= test);
    REQUIRE_FALSE(TestType{9} >= test);
    REQUIRE(TestType{4} != test);

    test = Int2<int>{0, 2};
    test += Int(35, 20); REQUIRE((test == Int(35, 22)));
    test -= Int(22, 18); REQUIRE((test == Int(13, 4)));
    test *= Int(2, 6);   REQUIRE((test == Int(26, 24)));
    test /= Int(2, 9);   REQUIRE((test == Int(13, 2)));

    //@CLION-formatter:on
    test.x = 20;
    test.y = 50;
    REQUIRE((test + Int(10, 12) == Int(30, 62)));
    REQUIRE((test - Int(3, 2) == Int(17, 48)));
    REQUIRE((test * Int(1, 5) == Int(20, 250)));
    REQUIRE((test / Int(3, 2) == Int(6, 25)));

    test = {2, 4};
    REQUIRE(test > Int{1, 2});
    REQUIRE(test < Int{4, 5});
    REQUIRE(test >= Int{2, 4});
    REQUIRE(test <= Int{10, 4});
    REQUIRE(test != Int{4, 4});

    // Min & Max
    REQUIRE((Math::min(Int{3, 4}, Int{5, 2}) == Int{3, 2}));
    REQUIRE((Math::max(Int{3, 4}, Int{5, 2}) == Int{5, 4}));
    REQUIRE((Math::min(Int{3, 6}, TestType{5}) == Int{3, 5}));
    REQUIRE((Math::max(Int{9, 0}, TestType{2}) == Int{9, 2}));

    test = Float2<float>{3.4f, 90.6f};
    REQUIRE(test == Int(3, 90));
    Int2<int> test1(test);
    REQUIRE((test1 == static_cast<Int2<int>>(test)));

    test.x = 23;
    test.y = 52;
    REQUIRE(test.size() == 2);
    REQUIRE(Math::sum(test) == 75);
    REQUIRE(Math::prod(test) == 1196);
    REQUIRE(Math::elements(test) == 1196);
    REQUIRE(Math::elementsFFT(test) == 624);
    REQUIRE((test.toString() == std::string{"(23, 52)"}));

    std::array<TestType, 2> test2 = test.toArray();
    REQUIRE(test2[0] == test.x);
    REQUIRE(test2[1] == test.y);
}


TEMPLATE_TEST_CASE("Vectors: Int3", "[noa][vectors]", int32_t, int64_t, uint32_t, uint64_t) {
    using Int = Int3<TestType>;

    //@CLION-formatter:off
    Int test{};
    REQUIRE((test == Int{TestType{0}}));
    test = TestType{2};
    test += TestType{1}; REQUIRE(test == TestType{3});
    test -= TestType{2}; REQUIRE(test == TestType{1});
    test *= TestType{3}; REQUIRE(test == TestType{3});
    test /= TestType{2}; REQUIRE(test == TestType{1});

    test = TestType{30};
    REQUIRE((test + TestType{10} == TestType{40}));
    REQUIRE((test - TestType{5} == TestType{25}));
    REQUIRE((test * TestType{3} == TestType{90}));
    REQUIRE((test / TestType{2} == TestType{15}));

    test = TestType{30};
    REQUIRE((TestType{40} == TestType{10} + test));
    REQUIRE((TestType{15} == TestType{45} - test));
    REQUIRE((TestType{90} == TestType{3} * test));
    REQUIRE((TestType{2} == TestType{60} / test));

    test = {4, 10, 7};
    REQUIRE_FALSE(test > TestType{9});
    REQUIRE(test < TestType{11});
    REQUIRE_FALSE(test >= TestType{7});
    REQUIRE_FALSE(test <= TestType{9});
    REQUIRE(test != TestType{4});

    REQUIRE_FALSE(TestType{9} > test);
    REQUIRE(TestType{11} > test);
    REQUIRE_FALSE(TestType{7} >= test);
    REQUIRE_FALSE(TestType{9} <= test);
    REQUIRE(TestType{4} != test);

    test = Int3<int>{0, 2, 130};
    test += Int(35, 20, 4);     REQUIRE((test == Int(35, 22, 134)));
    test -= Int(22, 18, 93);    REQUIRE((test == Int(13, 4, 41)));
    test *= Int(2, 6, 2);       REQUIRE((test == Int(26, 24, 82)));
    test /= Int(2, 9, 1);       REQUIRE((test == Int(13, 2, 82)));

    //@CLION-formatter:on
    test.x = 20;
    test.y = 50;
    test.z = 10;
    REQUIRE((test + Int(10, 12, 5) == Int(30, 62, 15)));
    REQUIRE((test - Int(3, 2, 7) == Int(17, 48, 3)));
    REQUIRE((test * Int(1, 5, 3) == Int(20, 250, 30)));
    REQUIRE((test / Int(3, 2, 2) == Int(6, 25, 5)));

    test = {2, 4, 4};
    REQUIRE(test > Int{1, 2, 3});
    REQUIRE(test < Int{4, 5, 6});
    REQUIRE(test >= Int{2, 4, 3});
    REQUIRE(test <= Int{10, 4, 6});
    REQUIRE(test != Int{4, 4, 4});

    // Min & Max
    REQUIRE((Math::min(Int{3, 4, 8}, Int{5, 2, 10}) == Int{3, 2, 8}));
    REQUIRE((Math::max(Int{3, 4, 1000}, Int{5, 2, 30}) == Int{5, 4, 1000}));
    REQUIRE((Math::min(Int{3, 6, 4}, TestType{5}) == Int{3, 5, 4}));
    REQUIRE((Math::max(Int{9, 0, 1}, TestType{2}) == Int{9, 2, 2}));

    test = Float3<float>{3.4f, 90.6f, 5.f};
    REQUIRE(test == Int(3, 90, 5));
    Int3<long> test1(test);
    REQUIRE((test1 == static_cast<Int3<long>>(test)));

    test.x = 23;
    test.y = 52;
    test.z = 128;
    REQUIRE(test.size() == 3);
    REQUIRE(Math::sum(test) == 203);
    REQUIRE(Math::prod(test) == 153088);
    REQUIRE(Math::elements(test) == 153088);
    REQUIRE(Math::elementsFFT(test) == 79872);
    REQUIRE(Math::slice(test) == Int(23, 52, 1));
    REQUIRE(Math::elementsSlice(test) == 1196);

    REQUIRE((test.toString() == std::string{"(23, 52, 128)"}));

    std::array<TestType, 3> test2 = test.toArray();
    REQUIRE(test2[0] == test.x);
    REQUIRE(test2[1] == test.y);
    REQUIRE(test2[2] == test.z);
}


TEMPLATE_TEST_CASE("Vectors: Int4", "[noa][vectors]", int32_t, int64_t, uint32_t, uint64_t) {

    using Int = Int4<TestType>;

    //@CLION-formatter:off
    Int test{};
    REQUIRE((test == Int{TestType{0}}));
    test = TestType{2};
    test += TestType{1}; REQUIRE(test == TestType{3});
    test -= TestType{2}; REQUIRE(test == TestType{1});
    test *= TestType{3}; REQUIRE(test == TestType{3});
    test /= TestType{2}; REQUIRE(test == TestType{1});

    test = TestType{30};
    REQUIRE((test + TestType{10} == TestType{40}));
    REQUIRE((test - TestType{5} == TestType{25}));
    REQUIRE((test * TestType{3} == TestType{90}));
    REQUIRE((test / TestType{2} == TestType{15}));

    test = TestType{30};
    REQUIRE((TestType{40} == TestType{10} + test));
    REQUIRE((TestType{15} == TestType{45} - test));
    REQUIRE((TestType{90} == TestType{3} * test));
    REQUIRE((TestType{2} == TestType{60} / test));

    test = {15, 130, 70, 2};
    REQUIRE_FALSE(test > TestType{9});
    REQUIRE(test < TestType{131});
    REQUIRE_FALSE(test >= TestType{7});
    REQUIRE_FALSE(test <= TestType{50});
    REQUIRE(test != TestType{15});

    REQUIRE_FALSE(TestType{9} < test);
    REQUIRE(TestType{131} > test);
    REQUIRE_FALSE(TestType{7} >= test);
    REQUIRE_FALSE(TestType{50} <= test);
    REQUIRE(TestType{15} != test);

    test = Int4<long>{130, 5, 130, 120};
    test += Int(35, 20, 4, 20); REQUIRE(test == Int(165, 25, 134, 140));
    test -= Int(22, 1, 93, 2);  REQUIRE(test == Int(143, 24, 41, 138));
    test *= Int(2, 6, 2, 9);    REQUIRE(test == Int(286, 144, 82, 1242));
    test /= Int(2, 9, 1, 4);    REQUIRE(test == Int(143, 16, 82, 310));

    //@CLION-formatter:on
    test.x = 20;
    test.y = 50;
    test.z = 10;
    test.w = 3;
    REQUIRE((test + Int(10, 12, 5, 30) == Int(30, 62, 15, 33)));
    REQUIRE((test - Int(3, 2, 7, 1) == Int(17, 48, 3, 2)));
    REQUIRE((test * Int(1, 5, 3, 103) == Int(20, 250, 30, 309)));
    REQUIRE((test / Int(3, 2, 2, 10) == Int(6, 25, 5, 0)));

    test = {2, 4, 4, 13405};
    REQUIRE_FALSE(test > Int{5, 5, 5, 100});
    REQUIRE_FALSE(test < Int{5, 5, 6, 100});
    REQUIRE(test >= Int{2, 4, 3, 4});
    REQUIRE_FALSE(test <= Int{10, 4, 6, 2});
    REQUIRE_FALSE(test != Int{2, 4, 4, 13405});

    // Min & Max
    REQUIRE((Math::min(Int{3, 4, 8, 1230}, Int{5, 2, 10, 312}) == Int{3, 2, 8, 312}));
    REQUIRE((Math::max(Int{3, 4, 1000, 2}, Int{5, 2, 30, 1}) == Int{5, 4, 1000, 2}));
    REQUIRE((Math::min(Int{3, 6, 4, 74}, TestType{5}) == Int{3, 5, 4, 5}));
    REQUIRE((Math::max(Int{9, 0, 1, 4}, TestType{2}) == Int{9, 2, 2, 4}));

    test = Float4<double>{3.4, 90.6, 5., 12.99};
    REQUIRE(test == Int(3, 90, 5, 12));
    Int4<int> test1(test);
    REQUIRE((test1 == static_cast<Int4<int>>(test)));

    test.x = 23;
    test.y = 52;
    test.z = 128;
    test.w = 4;
    REQUIRE(test.size() == 4);
    REQUIRE(Math::sum(test) == 207);
    REQUIRE(Math::prod(test) == 612352);
    REQUIRE(Math::elements(test) == 612352);
    REQUIRE(Math::elementsFFT(test) == 319488);
    REQUIRE(Math::slice(test) == Int(23, 52, 1, 1));
    REQUIRE(Math::elementsSlice(test) == 1196);

    REQUIRE((test.toString() == std::string{"(23, 52, 128, 4)"}));

    std::array<TestType, 4> test2 = test.toArray();
    REQUIRE(test2[0] == test.x);
    REQUIRE(test2[1] == test.y);
    REQUIRE(test2[2] == test.z);
    REQUIRE(test2[3] == test.w);
}


TEMPLATE_TEST_CASE("Vectors: Float2", "[noa][vectors]", float, double) {
    using Float = Float2<TestType>;

    //@CLION-formatter:off
    Float test{};
    REQUIRE((test == Float{TestType{0}}));
    test = TestType{2};  REQUIRE(test == TestType{2});
    test += F(1.34);     REQUIRE(Math::isEqual(test, F(3.34)));
    test -= F(23.134);   REQUIRE(Math::isEqual(test, F(-19.794)));
    test *= F(-2.45);    REQUIRE(Math::isEqual(test, F(48.4953)));
    test /= F(567.234);  REQUIRE(Math::isEqual(test, F(0.085494)));

    test = F(3.30);
    auto tmp = test + F(3.234534);   REQUIRE((Math::isEqual(tmp, F(6.534534))));
    tmp = test - F(-234.2);          REQUIRE((Math::isEqual(tmp, F(237.5))));
    tmp = test * F(3);               REQUIRE((Math::isEqual(tmp, F(9.90))));
    tmp = test / F(0.001);           REQUIRE((Math::isEqual(tmp, F(3299.999f), F(1e-3))));


    test = {4, 10};
    REQUIRE(Math::isEqual(test, Float{4, 10}));
    REQUIRE_FALSE(test > TestType{5});
    REQUIRE(test < TestType{11});
    REQUIRE_FALSE(test >= TestType{7});
    REQUIRE_FALSE(test <= TestType{9});
    REQUIRE(test != TestType{4});

    test = Float2<float>{0, 2};
    test += Float(35, 20);                  REQUIRE((Math::isEqual(test, Float(35, 22))));
    test -= Float(F(-0.12), F(23.2123));    REQUIRE((Math::isEqual(test, Float(F(35.12), F(-1.2123)))));
    test *= Float(F(0), F(10));             REQUIRE((Math::isEqual(test, Float(0, F(-12.123)), F(1e-5))));
    test /= Float(2, 9);                    REQUIRE((Math::isEqual(test, Float(0, F(-1.347)))));

    test.x = 20;
    test.y = 50;
    tmp = test + Float(10, 12);                     REQUIRE((Math::isEqual(tmp, Float(30, 62))));
    tmp = test - Float(F(10.32), F(-112.001));      REQUIRE((Math::isEqual(tmp, Float(F(9.68), F(162.001)))));
    tmp = test * Float(F(2.5), F(3.234));           REQUIRE((Math::isEqual(tmp, Float(50, F(161.7)))));
    tmp = test / Float(F(10), F(-12));              REQUIRE((Math::isEqual(tmp, Float(F(2), F(-4.166667)))));

    test = {2, 4};
    REQUIRE(test > Float{1, 2});
    REQUIRE(test < Float{4, 5});
    REQUIRE(test >= Float{2, 4});
    REQUIRE(test <= Float{10, 4});
    REQUIRE(test != Float{4, 4});
    REQUIRE(Math::isEqual(test, Float{2, 4}));

    // Min & Max
    REQUIRE((Math::min(Float{3, 4}, Float{5, 2}) == Float{3, 2}));
    REQUIRE((Math::max(Float{3, 4}, Float{5, 2}) == Float{5, 4}));
    REQUIRE((Math::min(Float{3, 6}, TestType{5}) == Float{3, 5}));
    REQUIRE((Math::max(Float{9, 0}, TestType{2}) == Float{9, 2}));

    test = Int2<long>{3, 90};
    REQUIRE(Math::isEqual(test, Float(3, 90)));
    Float2<double> test1(test);
    REQUIRE((test1 == static_cast<Float2<double>>(test)));

    test.x = F(23.23);
    test.y = F(-12.252);
    REQUIRE(test.size() == 2);
    REQUIRE_THAT(Math::sum(test), Catch::WithinAbs(10.978, 1e-6));
    REQUIRE_THAT(Math::prod(test), Catch::WithinAbs(static_cast<double>(test.x * test.y), 1e-6));
    tmp = Math::ceil(test); REQUIRE((Math::isEqual(tmp, Float(F(24), F(-12)), F(0))));
    tmp = Math::floor(test); REQUIRE((Math::isEqual(tmp, Float(F(23), F(-13)), F(0))));

    auto lengthSq = static_cast<double>(test.x * test.x + test.y * test.y);
    REQUIRE_THAT(Math::lengthSq(test), Catch::WithinAbs(lengthSq, 1e-6));
    REQUIRE_THAT(Math::length(test), Catch::WithinAbs(std::sqrt(lengthSq), 1e-6));
    tmp = Math::normalize(test); REQUIRE_THAT(Math::length(tmp), Catch::WithinAbs(1, 1e-6));
    REQUIRE_THAT(Math::dot(test, Float(F(-12.23), F(-21.23))), Catch::WithinAbs(-23.992940, 1e-4));

    //@CLION-formatter:on
    REQUIRE((test.toString() == std::string{"(23.23, -12.252)"}));

    std::array<TestType, 2> test2 = test.toArray();
    REQUIRE(test2[0] == test.x);
    REQUIRE(test2[1] == test.y);
}


TEMPLATE_TEST_CASE("Vectors: Float3", "[noa][vectors]", float, double) {
    using Float = Float3<TestType>;

    //@CLION-formatter:off
    Float test{};
    REQUIRE((test == Float{TestType{0}}));
    test = TestType{2};     REQUIRE(test == TestType{2});
    test += F(1.34);     REQUIRE(Math::isEqual(test, F(3.34)));
    test -= F(23.134);   REQUIRE(Math::isEqual(test, F(-19.794)));
    test *= F(-2.45);    REQUIRE(Math::isEqual(test, F(48.4953)));
    test /= F(567.234);  REQUIRE(Math::isEqual(test, F(0.085494)));

    test = F(3.30);
    auto tmp = test + F(3.234534);   REQUIRE((Math::isEqual(tmp, F(6.534534))));
    tmp = test - F(-234.2);          REQUIRE((Math::isEqual(tmp, F(237.5))));
    tmp = test * F(3);               REQUIRE((Math::isEqual(tmp, F(9.90))));
    tmp = test / F(0.001);           REQUIRE((Math::isEqual(tmp, F(3299.999f), F(1e-3))));


    test = {4, 10, 4};
    REQUIRE(Math::isEqual(test, Float{4, 10, 4}));
    REQUIRE_FALSE(test > TestType{5});
    REQUIRE(test < TestType{11});
    REQUIRE_FALSE(test >= TestType{7});
    REQUIRE_FALSE(test <= TestType{9});
    REQUIRE(test != TestType{4});

    test = Float3<float>{0, 2, 123};
    test += Float(35, 20, -12);                     REQUIRE((Math::isEqual(test, Float(35, 22, 111))));
    test -= Float(F(-0.12), F(23.2123), F(0.23));   REQUIRE((Math::isEqual(test, Float(F(35.12), F(-1.2123), F(110.77)))));
    test *= Float(0, 10, F(-3.2));                  REQUIRE((Math::isEqual(test, Float(0, F(-12.123), F(-354.464)), F(1e-5))));
    test /= Float(2, 9, 2);                         REQUIRE((Math::isEqual(test, Float(0, F(-1.347), F(-177.232)))));

    test.x = 20;
    test.y = 50;
    test.z = 33;
    tmp = test + Float(10, 12, -1232);                      REQUIRE((Math::isEqual(tmp, Float(30, 62, -1199))));
    tmp = test - Float(F(10.32), F(-112.001), F(0.5541));   REQUIRE((Math::isEqual(tmp, Float(F(9.68), F(162.001), F(32.4459)))));
    tmp = test * Float(F(2.5), F(3.234), F(58.12));         REQUIRE((Math::isEqual(tmp, Float(50, F(161.7), F(1917.959999)))));
    tmp = test / Float(F(10), F(-12), F(-2.3));             REQUIRE((Math::isEqual(tmp, Float(F(2), F(-4.166667), F(-14.3478261)))));

    test = {2, 4, -1};
    REQUIRE(test > Float{1, 2, -3});
    REQUIRE(test < Float{4, 5, 0});
    REQUIRE(test >= Float{2, 4, -1});
    REQUIRE(test <= Float{10, 4, 3});
    REQUIRE(test != Float{4, 4, -1});
    REQUIRE(Math::isEqual(test, Float{2, 4, -1}));

    // Min & Max
    REQUIRE((Math::min(Float{3, 4, -34}, Float{5, 2, -12}) == Float{3, 2, -34}));
    REQUIRE((Math::max(Float{3, 4, -3}, Float{5, 2, 23}) == Float{5, 4, 23}));
    REQUIRE((Math::min(Float{3, 6, 32}, TestType{5}) == Float{3, 5, 5}));
    REQUIRE((Math::max(Float{9, 0, -99}, TestType{2}) == Float{9, 2, 2}));

    test = Int3<long>{3, 90, -123};
    REQUIRE(Math::isEqual(test, Float(3, 90, -123)));
    Float3<double> test1(test);
    REQUIRE((test1 == static_cast<Float3<double>>(test)));

    test.x = F(23.23);
    test.y = F(-12.252);
    test.z = F(95.12);
    REQUIRE(test.size() == 3);
    REQUIRE_THAT(Math::sum(test), Catch::WithinAbs(static_cast<double>(test.x + test.y + test.z), 1e-6));
    REQUIRE_THAT(Math::prod(test), Catch::WithinAbs(static_cast<double>(test.x * test.y * test.z), 1e-6));
    tmp = Math::ceil(test); REQUIRE((Math::isEqual(tmp, Float(F(24), F(-12), F(96)), F(0))));
    tmp = Math::floor(test); REQUIRE((Math::isEqual(tmp, Float(F(23), F(-13), F(95)), F(0))));

    auto lengthSq = static_cast<double>(test.x * test.x + test.y * test.y + test.z * test.z);
    REQUIRE_THAT(Math::lengthSq(test), Catch::WithinAbs(lengthSq, 1e-6));
    REQUIRE_THAT(Math::length(test), Catch::WithinAbs(std::sqrt(lengthSq), 1e-6));
    tmp = Math::normalize(test); REQUIRE_THAT(Math::length(tmp), Catch::WithinAbs(1, 1e-6));

    auto tmp1 = Float(F(-12.23), F(-21.23), F(123.22));
    REQUIRE_THAT(Math::dot(test, tmp1), Catch::WithinAbs(11696.69346, 1e-3));

    //@CLION-formatter:on

    Float t1(2, 3, 4);
    Float t2(5, 6, 7);
    Float t3(Math::cross(t1, t2));
    REQUIRE((Math::isEqual(t3, Float(-3, 6, -3), F(0))));

    REQUIRE((test.toString() == std::string{"(23.23, -12.252, 95.12)"}));

    std::array<TestType, 3> test2 = test.toArray();
    REQUIRE(test2[0] == test.x);
    REQUIRE(test2[1] == test.y);
    REQUIRE(test2[2] == test.z);
}


TEMPLATE_TEST_CASE("Vectors: Float4", "[noa][vectors]", float, double) {
    using Float = Float4<TestType>;

    //@CLION-formatter:off
    Float test{};
    REQUIRE((test == Float{TestType{0}}));
    test = TestType{2};     REQUIRE(test == TestType{2});
    test += F(1.34);     REQUIRE(Math::isEqual(test, F(3.34)));
    test -= F(23.134);   REQUIRE(Math::isEqual(test, F(-19.794)));
    test *= F(-2.45);    REQUIRE(Math::isEqual(test, F(48.4953)));
    test /= F(567.234);  REQUIRE(Math::isEqual(test, F(0.085494)));

    test = F(3.30);
    auto tmp = test + F(3.234534);   REQUIRE((Math::isEqual(tmp, F(6.534534))));
    tmp = test - F(-234.2);          REQUIRE((Math::isEqual(tmp, F(237.5))));
    tmp = test * F(3);               REQUIRE((Math::isEqual(tmp, F(9.90))));
    tmp = test / F(0.001);           REQUIRE((Math::isEqual(tmp, F(3299.999f), F(1e-3))));


    test = {4, 10, 4, 1};
    REQUIRE(Math::isEqual(test, Float{4, 10, 4, 1}));
    REQUIRE_FALSE(test > TestType{5});
    REQUIRE(test < TestType{11});
    REQUIRE_FALSE(test >= TestType{7});
    REQUIRE_FALSE(test <= TestType{9});
    REQUIRE(test != TestType{4});

    test = Float4<float>{0, 2, 123, 32};
    test += Float(35, 20, -12, 1);                      REQUIRE((Math::isEqual(test, Float(35, 22, 111, 33))));
    test -= Float(F(-0.12), F(23.2123), F(0.23), 2);    REQUIRE((Math::isEqual(test, Float(F(35.12), F(-1.2123), F(110.77), 31))));
    test *= Float(0, 10, F(-3.2), F(-0.324));           REQUIRE((Math::isEqual(test, Float(0, F(-12.123), F(-354.464), F(-10.044)), F(1e-5))));
    test /= Float(2, 9, 2, F(-0.5));                    REQUIRE((Math::isEqual(test, Float(0, F(-1.347), F(-177.232), F(20.088)))));

    test.x = 20;
    test.y = 50;
    test.z = 33;
    test.w = 5;
    tmp = test + Float(10, 12, -1232, F(2.3));                  REQUIRE((Math::isEqual(tmp, Float(30, 62, -1199, F(7.3)))));
    tmp = test - Float(F(10.32), F(-112.001), F(0.5541), 1);    REQUIRE((Math::isEqual(tmp, Float(F(9.68), F(162.001), F(32.4459), 4))));
    tmp = test * Float(F(2.5), F(3.234), F(58.12), F(8.81));    REQUIRE((Math::isEqual(tmp, Float(50, F(161.7), F(1917.959999), F(44.050)))));
    tmp = test / Float(F(10), F(-12), F(-2.3), F(0.01));        REQUIRE((Math::isEqual(tmp, Float(F(2), F(-4.166667), F(-14.3478261), 500))));

    test = {2, 4, -1, 12};
    REQUIRE(test > Float{1, 2, -3, 11});
    REQUIRE(test < Float{4, 5, 0, 13});
    REQUIRE(test >= Float{2, 4, -1, 12});
    REQUIRE(test <= Float{10, 4, 3, 12});
    REQUIRE(test != Float{4, 4, -1, 12});
    REQUIRE(Math::isEqual(test, Float{2, 4, -1, 12}));

    // Min & Max
    REQUIRE((Math::min(Float{3, 4, -34, F(2.34)}, Float{5, 2, -12, F(120.12)}) == Float{3, 2, -34, F(2.34)}));
    REQUIRE((Math::max(Float{3, 4, -3, F(-9.9)}, Float{5, 2, 23, F(-10)}) == Float{5, 4, 23, F(-9.9)}));
    REQUIRE((Math::min(Float{3, 6, 32, F(5.01)}, TestType{5}) == Float{3, 5, 5, 5}));
    REQUIRE((Math::max(Float{9, 0, -99, F(2.01)}, TestType{2}) == Float{9, 2, 2, F(2.01)}));

    // .data()
    test = Int4<long>{3, 90, -123, 12};
    REQUIRE(Math::isEqual(test, Float(3, 90, -123, 12)));
    Float4<double> test1(test);
    REQUIRE((test1 == static_cast<Float4<double>>(test)));

    test.x = F(23.23);
    test.y = F(-12.252);
    test.z = F(95.12);
    test.w = F(2.34);
    REQUIRE(test.size() == 4);
    REQUIRE_THAT(Math::sum(test), Catch::WithinAbs(static_cast<double>(test.x + test.y + test.z + test.w), 1e-6));
    REQUIRE_THAT(Math::prod(test), Catch::WithinAbs(static_cast<double>(test.x * test.y * test.z * test.w), 1e-6));
    tmp = Math::ceil(test); REQUIRE((Math::isEqual(tmp, Float(F(24), F(-12), F(96), F(3)), F(0))));
    tmp = Math::floor(test); REQUIRE((Math::isEqual(tmp, Float(F(23), F(-13), F(95), F(2)), F(0))));

    auto lengthSq = static_cast<double>(test.x * test.x + test.y * test.y + test.z * test.z + test.w * test.w);
    REQUIRE_THAT(Math::lengthSq(test), Catch::WithinAbs(lengthSq, 1e-6));
    REQUIRE_THAT(Math::length(test), Catch::WithinAbs(std::sqrt(lengthSq), 1e-6));
    tmp = Math::normalize(test); REQUIRE_THAT(Math::length(tmp), Catch::WithinAbs(1, 1e-6));

    //@CLION-formatter:on
    REQUIRE((test.toString() == std::string{"(23.23, -12.252, 95.12, 2.34)"}));

    std::array<TestType, 4> test2 = test.toArray();
    REQUIRE(test2[0] == test.x);
    REQUIRE(test2[1] == test.y);
    REQUIRE(test2[2] == test.z);
    REQUIRE(test2[3] == test.w);
}

#undef F