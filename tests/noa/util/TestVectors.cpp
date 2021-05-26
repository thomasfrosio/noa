#include <noa/util/BoolX.h>
#include <noa/util/IntX.h>
#include <noa/util/FloatX.h>

#include <catch2/catch.hpp>

using namespace ::Noa;

TEST_CASE("Vectors: typedefs", "[noa][vectors]") {
    static_assert(std::is_same_v<int2_t, Int2<int>>);
    static_assert(std::is_same_v<uint2_t, Int2<unsigned int>>);
    static_assert(std::is_same_v<long2_t, Int2<long long>>);
    static_assert(std::is_same_v<ulong2_t, Int2<unsigned long long>>);

    static_assert(std::is_same_v<int3_t, Int3<int>>);
    static_assert(std::is_same_v<uint3_t, Int3<unsigned int>>);
    static_assert(std::is_same_v<long3_t, Int3<long long>>);
    static_assert(std::is_same_v<ulong3_t, Int3<unsigned long long>>);

    static_assert(std::is_same_v<int4_t, Int4<int>>);
    static_assert(std::is_same_v<uint4_t, Int4<unsigned int>>);
    static_assert(std::is_same_v<long4_t, Int4<long long>>);
    static_assert(std::is_same_v<ulong4_t, Int4<unsigned long long>>);

    static_assert(std::is_same_v<float2_t, Float2<float>>);
    static_assert(std::is_same_v<double2_t, Float2<double>>);

    static_assert(std::is_same_v<float3_t, Float3<float>>);
    static_assert(std::is_same_v<double3_t, Float3<double>>);

    static_assert(std::is_same_v<float4_t, Float4<float>>);
    static_assert(std::is_same_v<double4_t, Float4<double>>);
}

TEMPLATE_TEST_CASE("Vectors: Int2", "[noa][vectors]",
                   int, long, long long,
                   unsigned int, unsigned long, unsigned long long) {
    using Int = Int2<TestType>;

    //@CLION-formatter:off
    Int test;
    REQUIRE(all(test == Int{TestType{0}}));

    test = 2;
    test += TestType{1}; REQUIRE(all(test == TestType{3}));
    test -= TestType{2}; REQUIRE(all(test == TestType{1}));
    test *= TestType{3}; REQUIRE(all(test == TestType{3}));
    test /= TestType{2}; REQUIRE(all(test == TestType{1}));

    test = 30;
    REQUIRE(all(test + TestType{10} == TestType{40}));
    REQUIRE(all(test - TestType{5} == TestType{25}));
    REQUIRE(all(test * TestType{3} == TestType{90}));
    REQUIRE(all(test / TestType{2} == TestType{15}));

    REQUIRE(all(TestType{40} == TestType{10} + test));
    REQUIRE(all(TestType{15} == TestType{45} - test));
    REQUIRE(all(TestType{90} == TestType{3} * test));
    REQUIRE(all(TestType{2} == TestType{60} / test));

    test = {4, 10};
    REQUIRE(Bool2(0, 1) == (test > TestType{5}));
    REQUIRE(Bool2(1, 1) == (test < TestType{11}));
    REQUIRE(Bool2(0, 1) == (test >= TestType{7}));
    REQUIRE(Bool2(1, 1) == (test <= TestType{10}));
    REQUIRE(any(test == TestType{4}));
    REQUIRE_FALSE(all(test == TestType{4}));

    REQUIRE((TestType{5} < test) == Bool2(0, 1));
    REQUIRE((TestType{11} > test) == Bool2(1, 1));
    REQUIRE((TestType{7} <= test) == Bool2(0, 1));
    REQUIRE((TestType{9} >= test) == Bool2(1, 0));
    REQUIRE(any(TestType{4} == test));
    REQUIRE_FALSE(all(TestType{4} == test));

    test = int2_t{0, 2};
    test += Int(35, 20); REQUIRE(all(test == Int(35, 22)));
    test -= Int(22, 18); REQUIRE(all(test == Int(13, 4)));
    test *= Int(2, 6);   REQUIRE(all(test == Int(26, 24)));
    test /= Int(2, 9);   REQUIRE(all(test == Int(13, 2)));
    //@CLION-formatter:on

    test.x = 20;
    test.y = 50;
    REQUIRE(all(test + Int(10, 12) == Int(30, 62)));
    REQUIRE(all(test - Int(3, 2) == Int(17, 48)));
    REQUIRE(all(test * Int(1, 5) == Int(20, 250)));
    REQUIRE(all(test / Int(3, 2) == Int(6, 25)));

    test = {2, 4};
    REQUIRE((test > Int{1, 2}) == Bool2(1, 1));
    REQUIRE((test < Int{4, 5}) == Bool2(1, 1));
    REQUIRE((test >= Int{2, 4}) == Bool2(1, 1));
    REQUIRE((test <= Int{1, 4}) == Bool2(0, 1));
    REQUIRE((test != Int{4, 4}) == Bool2(1, 0));

    // Min & Max
    REQUIRE(all(Math::min(Int{3, 4}, Int{5, 2}) == Int{3, 2}));
    REQUIRE(all(Math::max(Int{3, 4}, Int{5, 2}) == Int{5, 4}));
    REQUIRE(all(Math::min(Int{3, 6}, TestType{5}) == Int{3, 5}));
    REQUIRE(all(Math::max(Int{9, 0}, TestType{2}) == Int{9, 2}));

    test = float2_t{3.4f, 90.6f};
    REQUIRE(all(test == Int(3, 90)));
    int2_t test1(std::move(Int(test)));
    REQUIRE(all(test1 == static_cast<int2_t>(test)));

    test.x = 23;
    test.y = 52;
    const Int test2(23, 52);
    REQUIRE((test[0] == 23 && test[1] == 52));
    REQUIRE((test2[0] == 23 && test[1] == 52));
    REQUIRE(test.size() == 2);
    REQUIRE(test.elements() == 2);
    REQUIRE(Math::sum(test) == 75);
    REQUIRE(Math::prod(test) == 1196);
    REQUIRE(getElements(test) == 1196);
    REQUIRE(getElementsFFT(test) == 624);

    REQUIRE((String::format("{}", test) == "(23,52)"));

    std::array<TestType, 2> test3 = toArray(test);
    REQUIRE(test3[0] == test.x);
    REQUIRE(test3[1] == test.y);
}

TEMPLATE_TEST_CASE("Vectors: Int3", "[noa][vectors]",
                   int, long, long long,
                   unsigned int, unsigned long, unsigned long long) {
    using Int = Int3<TestType>;

    //@CLION-formatter:off
    Int test;
    REQUIRE(all(test == Int{TestType{0}}));

    test = 2;
    test += TestType{1}; REQUIRE(all(test == TestType{3}));
    test -= TestType{2}; REQUIRE(all(test == TestType{1}));
    test *= TestType{3}; REQUIRE(all(test == TestType{3}));
    test /= TestType{2}; REQUIRE(all(test == TestType{1}));

    test = 30;
    REQUIRE(all(test + TestType{10} == TestType{40}));
    REQUIRE(all(test - TestType{5} == TestType{25}));
    REQUIRE(all(test * TestType{3} == TestType{90}));
    REQUIRE(all(test / TestType{2} == TestType{15}));

    REQUIRE(all(TestType{40} == TestType{10} + test));
    REQUIRE(all(TestType{15} == TestType{45} - test));
    REQUIRE(all(TestType{90} == TestType{3} * test));
    REQUIRE(all(TestType{2} == TestType{60} / test));

    test = {4, 10, 7};
    REQUIRE(Bool3(0, 1, 0) == (test > TestType{9}));
    REQUIRE(Bool3(1, 1, 1) == (test < TestType{11}));
    REQUIRE(Bool3(0, 1, 1) == (test >= TestType{7}));
    REQUIRE(Bool3(1, 0, 1) == (test <= TestType{9}));
    REQUIRE(any(test == TestType{4}));
    REQUIRE_FALSE(all(test == TestType{4}));

    REQUIRE(Bool3(1, 0, 1) == (TestType{9} > test));
    REQUIRE(Bool3(1, 1, 1) == (TestType{11} > test));
    REQUIRE(Bool3(1, 0, 1) == (TestType{7} >= test));
    REQUIRE(Bool3(0, 1, 0) == (TestType{9} <= test));
    REQUIRE(any(TestType{4} == test));
    REQUIRE_FALSE(all(TestType{4} == test));

    test = int3_t{0, 2, 130};
    test += Int(35, 20, 4);     REQUIRE(all(test == Int(35, 22, 134)));
    test -= Int(22, 18, 93);    REQUIRE(all(test == Int(13, 4, 41)));
    test *= Int(2, 6, 2);       REQUIRE(all(test == Int(26, 24, 82)));
    test /= Int(2, 9, 1);       REQUIRE(all(test == Int(13, 2, 82)));
    //@CLION-formatter:on

    test.x = 20;
    test.y = 50;
    test.z = 10;
    REQUIRE(all(test + Int(10, 12, 5) == Int(30, 62, 15)));
    REQUIRE(all(test - Int(3, 2, 7) == Int(17, 48, 3)));
    REQUIRE(all(test * Int(1, 5, 3) == Int(20, 250, 30)));
    REQUIRE(all(test / Int(3, 2, 2) == Int(6, 25, 5)));

    test = {2, 4, 4};
    REQUIRE(Bool3(1, 1, 0) == (test > Int{1, 2, 4}));
    REQUIRE(Bool3(0, 1, 1) == (test < Int{2, 5, 6}));
    REQUIRE(Bool3(1, 1, 1) == (test >= Int{2, 4, 3}));
    REQUIRE(Bool3(0, 1, 1) == (test <= Int{1, 4, 6}));
    REQUIRE(Bool3(1, 0, 0) == (test != Int{4, 4, 4}));
    REQUIRE(Bool3(0, 0, 1) == (test == Int{4, 2, 4}));

    // Min & Max
    REQUIRE(all(Math::min(Int{3, 4, 8}, Int{5, 2, 10}) == Int{3, 2, 8}));
    REQUIRE(all(Math::max(Int{3, 4, 1000}, Int{5, 2, 30}) == Int{5, 4, 1000}));
    REQUIRE(all(Math::min(Int{3, 6, 4}, TestType{5}) == Int{3, 5, 4}));
    REQUIRE(all(Math::max(Int{9, 0, 1}, TestType{2}) == Int{9, 2, 2}));

    test = float3_t{3.4f, 90.6f, 5.f};
    REQUIRE(all(test == Int(3, 90, 5)));
    Int3<long> test1(test);
    REQUIRE(all(test1 == static_cast<Int3<long>>(test)));

    test.x = 23;
    test.y = 52;
    test.z = 128;
    const Int test2(test);
    REQUIRE((test[0] == 23 && test[1] == 52 && test[2] == 128));
    REQUIRE((test2[0] == 23 && test2[1] == 52 && test2[2] == 128));
    REQUIRE(test.size() == 3);
    REQUIRE(test.elements() == 3);
    REQUIRE(Math::sum(test) == 203);
    REQUIRE(Math::prod(test) == 153088);
    REQUIRE(getElements(test) == 153088);
    REQUIRE(getElementsFFT(test) == 79872);
    REQUIRE(all(getShapeSlice(test) == Int(23, 52, 1)));
    REQUIRE(getElementsSlice(test) == 1196);

    REQUIRE((String::format("{}", test) == std::string{"(23,52,128)"}));

    std::array<TestType, 3> test3 = toArray(test);
    REQUIRE(test3[0] == test.x);
    REQUIRE(test3[1] == test.y);
    REQUIRE(test3[2] == test.z);
}

TEMPLATE_TEST_CASE("Vectors: Int4", "[noa][vectors]",
                   int, long, long long,
                   unsigned int, unsigned long, unsigned long long) {
    using Int = Int4<TestType>;

    //@CLION-formatter:off
    Int test;
    REQUIRE(all(test == Int{TestType{0}}));
    test = 2;
    test += TestType{1}; REQUIRE(all(test == TestType{3}));
    test -= TestType{2}; REQUIRE(all(test == TestType{1}));
    test *= TestType{3}; REQUIRE(all(test == TestType{3}));
    test /= TestType{2}; REQUIRE(all(test == TestType{1}));

    test = 30;
    REQUIRE(all(test + TestType{10} == TestType{40}));
    REQUIRE(all(test - TestType{5} == TestType{25}));
    REQUIRE(all(test * TestType{3} == TestType{90}));
    REQUIRE(all(test / TestType{2} == TestType{15}));

    test = 30;
    REQUIRE(all(TestType{40} == TestType{10} + test));
    REQUIRE(all(TestType{15} == TestType{45} - test));
    REQUIRE(all(TestType{90} == TestType{3} * test));
    REQUIRE(all(TestType{2} == TestType{60} / test));

    test = {15, 130, 70, 2};
    REQUIRE(Bool4(1, 1, 1, 0) == (test > TestType{9}));
    REQUIRE(Bool4(1, 0, 1, 1) == (test < TestType{130}));
    REQUIRE(Bool4(1, 1, 1, 0) == (test >= TestType{7}));
    REQUIRE(Bool4(1, 0, 0, 1) == (test <= TestType{50}));
    REQUIRE(Bool4(0, 1, 1, 1) == (test != TestType{15}));
    REQUIRE(Bool4(1, 0, 0, 0) == (test == TestType{15}));

    REQUIRE(Bool4(1, 1, 1, 0) == (TestType{9} < test));
    REQUIRE(Bool4(1, 0, 1, 1) == (TestType{130} > test));
    REQUIRE(Bool4(1, 1, 1, 0) == (TestType{7} <= test));
    REQUIRE(Bool4(1, 0, 0, 1) == (TestType{50} >= test));
    REQUIRE(Bool4(0, 1, 1, 1) == (TestType{15} != test));
    REQUIRE(Bool4(1, 0, 0, 0) == (TestType{15} == test));

    test = int4_t(130, 5, 130, 120);
    test += Int(35, 20, 4, 20); REQUIRE(all(test == Int(165, 25, 134, 140)));
    test -= Int(22, 1, 93, 2);  REQUIRE(all(test == Int(143, 24, 41, 138)));
    test *= Int(2, 6, 2, 9);    REQUIRE(all(test == Int(286, 144, 82, 1242)));
    test /= Int(2, 9, 1, 4);    REQUIRE(all(test == Int(143, 16, 82, 310)));

    //@CLION-formatter:on
    test.x = 20;
    test.y = 50;
    test.z = 10;
    test.w = 3;
    const Int test2(test);
    REQUIRE((test[0] == 20 && test[1] == 50 && test[2] == 10 && test[3] == 3));
    REQUIRE((test2[0] == 20 && test2[1] == 50 && test2[2] == 10 && test2[3] == 3));
    REQUIRE(all(test + Int(10, 12, 5, 30) == Int(30, 62, 15, 33)));
    REQUIRE(all(test - Int(3, 2, 7, 1) == Int(17, 48, 3, 2)));
    REQUIRE(all(test * Int(1, 5, 3, 103) == Int(20, 250, 30, 309)));
    REQUIRE(all(test / Int(3, 2, 2, 10) == Int(6, 25, 5, 0)));

    test = {2, 4, 4, 13405};
    REQUIRE(Bool4(0, 0, 0, 1) == (test > Int{5, 5, 5, 100}));
    REQUIRE(Bool4(1, 1, 1, 0) == (test < Int{5, 5, 6, 100}));
    REQUIRE(Bool4(1, 1, 1, 1) == (test >= Int{2, 4, 3, 4}));
    REQUIRE(Bool4(1, 1, 0, 0) == (test <= Int{10, 4, 1, 2}));
    REQUIRE(Bool4(0, 0, 0, 1) == (test != Int{2, 4, 4, 13404}));
    REQUIRE(Bool4(0, 1, 0, 1) == (test == Int{3, 4, 3, 13405}));

    // Min & Max
    REQUIRE(all(Math::min(Int{3, 4, 8, 1230}, Int{5, 2, 10, 312}) == Int{3, 2, 8, 312}));
    REQUIRE(all(Math::max(Int{3, 4, 1000, 2}, Int{5, 2, 30, 1}) == Int{5, 4, 1000, 2}));
    REQUIRE(all(Math::min(Int{3, 6, 4, 74}, TestType{5}) == Int{3, 5, 4, 5}));
    REQUIRE(all(Math::max(Int{9, 0, 1, 4}, TestType{2}) == Int{9, 2, 2, 4}));

    test = Float4<double>{3.4, 90.6, 5., 12.99};
    REQUIRE(all(test == Int(3, 90, 5, 12)));
    Int4<int> test1(test);
    REQUIRE(all(test1 == static_cast<Int4<int>>(test)));

    test.x = 23;
    test.y = 52;
    test.z = 128;
    test.w = 4;
    REQUIRE(test.size() == 4);
    REQUIRE(Math::sum(test) == 207);
    REQUIRE(Math::prod(test) == 612352);
    REQUIRE(getElements(test) == 612352);
    REQUIRE(getElementsFFT(test) == 319488);
    REQUIRE(all(getShapeSlice(test) == Int(23, 52, 1, 1)));
    REQUIRE(getElementsSlice(test) == 1196);

    REQUIRE((String::format("{}", test) == std::string{"(23,52,128,4)"}));

    std::array<TestType, 4> test3 = toArray(test);
    REQUIRE(test3[0] == test.x);
    REQUIRE(test3[1] == test.y);
    REQUIRE(test3[2] == test.z);
    REQUIRE(test3[3] == test.w);
}

#define F(x) static_cast<TestType>(x)

TEMPLATE_TEST_CASE("Vectors: Float2", "[noa][vectors]", float, double) {
    using Float = Float2<TestType>;

    //@CLION-formatter:off
    Float test;
    REQUIRE(all(test == Float(0)));
    test = 2;            REQUIRE(all(test == TestType{2}));
    test += F(1.34);     REQUIRE(all(Math::isEqual(test, F(3.34))));
    test -= F(23.134);   REQUIRE(all(Math::isEqual(test, F(-19.794))));
    test *= F(-2.45);    REQUIRE(all(Math::isEqual(test, F(48.4953))));
    test /= F(567.234);  REQUIRE(all(Math::isEqual(test, F(0.085494))));

    test = 3.30;
    auto tmp = test + F(3.234534);   REQUIRE(all(Math::isEqual(tmp, F(6.534534))));
    tmp = test - F(-234.2);          REQUIRE(all(Math::isEqual(tmp, F(237.5))));
    tmp = test * F(3);               REQUIRE(all(Math::isEqual(tmp, F(9.90))));
    tmp = test / F(0.001);           REQUIRE(all(Math::isEqual(tmp, F(3299.999f), F(1e-3))));

    test = {4, 10};
    REQUIRE(all(Math::isEqual(test, Float{4, 10})));
    REQUIRE(Bool2(0, 1) == (test > TestType{5}));
    REQUIRE(Bool2(1, 1) == (test < TestType{11}));
    REQUIRE(Bool2(0, 1) == (test >= TestType{7}));
    REQUIRE(Bool2(1, 0) == (test <= TestType{9}));
    REQUIRE(Bool2(0, 1) == (test != TestType{4}));

    test = Float2<float>{0, 2};
    test += Float(35, 20);                  REQUIRE(all(Math::isEqual(test, Float(35, 22))));
    test -= Float(F(-0.12), F(23.2123));    REQUIRE(all(Math::isEqual(test, Float(35.12, -1.2123))));
    test *= Float(F(0), F(10));             REQUIRE(all(Math::isEqual(test, Float(0, -12.123), F(1e-5))));
    test /= Float(2, 9);                    REQUIRE(all(Math::isEqual(test, Float(0, -1.347))));

    test.x = 20;
    test.y = 50;
    tmp = test + Float(10, 12);                     REQUIRE(all(Math::isEqual(tmp, Float(30, 62))));
    tmp = test - Float(F(10.32), F(-112.001));      REQUIRE(all(Math::isEqual(tmp, Float(9.68, 162.001))));
    tmp = test * Float(F(2.5), F(3.234));           REQUIRE(all(Math::isEqual(tmp, Float(50, 161.7))));
    tmp = test / Float(F(10), F(-12));              REQUIRE(all(Math::isEqual(tmp, Float(2, -4.166667))));

    test = {2, 4};
    REQUIRE(Bool2(1, 1) == (test > Float{1, 2}));
    REQUIRE(Bool2(1, 0) == (test < Float{4, 4}));
    REQUIRE(Bool2(1, 1) == (test >= Float{2, 4}));
    REQUIRE(Bool2(0, 1) == (test <= Float{1, 4}));
    REQUIRE(Bool2(1, 0) == (test != Float{4, 4}));
    REQUIRE(Bool2(1, 1) == (test == Float{2, 4}));

    // Min & Max
    REQUIRE(all(Math::min(Float{3, 4}, Float{5, 2}) == Float{3, 2}));
    REQUIRE(all(Math::max(Float{3, 4}, Float{5, 2}) == Float{5, 4}));
    REQUIRE(all(Math::min(Float{3, 6}, TestType{5}) == Float{3, 5}));
    REQUIRE(all(Math::max(Float{9, 0}, TestType{2}) == Float{9, 2}));

    test = Int2<long>{3, 90};
    REQUIRE(all(Math::isEqual(test, Float(3, 90))));
    Float2<double> test1(test);
    REQUIRE(all(test1 == static_cast<Float2<double>>(test)));

    test.x = F(23.23);
    test.y = F(-12.252);
    const Float test2(test);
    REQUIRE(test.size() == 2);
    REQUIRE(test.elements() == 2);
    REQUIRE((test[0] == F(23.23) && test[1] == F(-12.252)));
    REQUIRE((test2[0] == F(23.23) && test2[1] == F(-12.252)));
    REQUIRE_THAT(Math::sum(test), Catch::WithinAbs(10.978, 1e-6));
    REQUIRE_THAT(Math::prod(test), Catch::WithinAbs(static_cast<double>(test.x * test.y), 1e-6));
    tmp = Math::ceil(test); REQUIRE(all(Math::isEqual(tmp, Float(24, -12), F(0))));
    tmp = Math::floor(test); REQUIRE(all(Math::isEqual(tmp, Float(23, -13), F(0))));

    auto lengthSq = static_cast<double>(test.x * test.x + test.y * test.y);
    REQUIRE_THAT(Math::lengthSq(test), Catch::WithinAbs(lengthSq, 1e-6));
    REQUIRE_THAT(Math::length(test), Catch::WithinAbs(std::sqrt(lengthSq), 1e-6));
    tmp = Math::normalize(test); REQUIRE_THAT(Math::length(tmp), Catch::WithinAbs(1, 1e-6));
    REQUIRE_THAT(Math::dot(test, Float(-12.23, -21.23)), Catch::WithinAbs(-23.992940, 1e-4));
    //@CLION-formatter:on

    REQUIRE((String::format("{}", test) == std::string{"(23.230,-12.252)"}));

    std::array<TestType, 2> test3 = toArray(test);
    REQUIRE(test3[0] == test.x);
    REQUIRE(test3[1] == test.y);
}

TEMPLATE_TEST_CASE("Vectors: Float3", "[noa][vectors]", float, double) {
    using Float = Float3<TestType>;

    //@CLION-formatter:off
    Float test;
    REQUIRE(all(test == Float{TestType{0}}));
    test = 2;            REQUIRE(all(test == TestType{2}));
    test += 1.34;     REQUIRE(all(Math::isEqual(test, F(3.34))));
    test -= 23.134;   REQUIRE(all(Math::isEqual(test, F(-19.794))));
    test *= -2.45;    REQUIRE(all(Math::isEqual(test, F(48.4953))));
    test /= 567.234;  REQUIRE(all(Math::isEqual(test, F(0.085494))));

    test = F(3.30);
    auto tmp = test + F(3.234534);   REQUIRE(all(Math::isEqual(tmp, F(6.534534))));
    tmp = test - F(-234.2);          REQUIRE(all(Math::isEqual(tmp, F(237.5))));
    tmp = test * F(3);               REQUIRE(all(Math::isEqual(tmp, F(9.90))));
    tmp = test / F(0.001);           REQUIRE(all(Math::isEqual(tmp, F(3299.999f), F(1e-3))));

    test = {4, 10, 4};
    REQUIRE(Bool3(0, 1, 1) == (Math::isEqual(test, Float{3.99, 10, 4})));
    REQUIRE(Bool3(0, 1, 0) == (test > TestType{5}));
    REQUIRE(Bool3(1, 1, 1) == (test < TestType{11}));
    REQUIRE(Bool3(0, 1, 0) == (test >= TestType{7}));
    REQUIRE(Bool3(1, 0, 1) == (test <= TestType{9}));
    REQUIRE(Bool3(0, 1, 0) == (test != TestType{4}));
    REQUIRE(Bool3(0, 1, 0) == (test == TestType{10}));

    test = Float3<float>{0, 2, 123};
    test += Float(35, 20, -12);             REQUIRE(all(Math::isEqual(test, Float(35, 22, 111))));
    test -= Float(-0.12, 23.2123, 0.23);    REQUIRE(all(Math::isEqual(test, Float(35.12, -1.2123, 110.77))));
    test *= Float(0, 10, -3.2);             REQUIRE(all(Math::isEqual(test, Float(0, -12.123, -354.464), F(1e-5))));
    test /= Float(2, 9, 2);                 REQUIRE(all(Math::isEqual(test, Float(0, -1.347, -177.232))));

    test.x = 20;
    test.y = 50;
    test.z = 33;
    tmp = test + Float(10, 12, -1232);              REQUIRE(all(Math::isEqual(tmp, Float(30, 62, -1199))));
    tmp = test - Float(10.32, -112.001, 0.5541);    REQUIRE(all(Math::isEqual(tmp, Float(9.68, 162.001, 32.4459))));
    tmp = test * Float(2.5, 3.234, 58.12);          REQUIRE(all(Math::isEqual(tmp, Float(50, 161.7, 1917.959999))));
    tmp = test / Float(10, -12, -2.3);              REQUIRE(all(Math::isEqual(tmp, Float(2, -4.166667, -14.3478261))));

    test = {2, 4, -1};
    REQUIRE(Bool3(1, 1, 1) == (test > Float{1, 2, -3}));
    REQUIRE(Bool3(1, 0, 1) == (test < Float{4, 4, 0}));
    REQUIRE(Bool3(1, 1, 0) == (test >= Float{2, 4, 0}));
    REQUIRE(Bool3(1, 1, 1) == (test <= Float{10, 4, 3}));
    REQUIRE(Bool3(1, 0, 0) == (test != Float{4, 4, -1}));
    REQUIRE(Bool3(1, 1, 0) == (Math::isEqual(test, Float{2, 4, 0.99})));

    // Min & Max
    REQUIRE(all(Math::min(Float{3, 4, -34}, Float{5, 2, -12}) == Float{3, 2, -34}));
    REQUIRE(all(Math::max(Float{3, 4, -3}, Float{5, 2, 23}) == Float{5, 4, 23}));
    REQUIRE(all(Math::min(Float{3, 6, 32}, TestType{5}) == Float{3, 5, 5}));
    REQUIRE(all(Math::max(Float{9, 0, -99}, TestType{2}) == Float{9, 2, 2}));

    test = Int3<long>{3, 90, -123};
    REQUIRE(all(Math::isEqual(test, Float(3, 90, -123))));
    Float3<double> test1(test);
    REQUIRE(all(test1 == static_cast<Float3<double>>(test)));

    test.x = F(23.23);
    test.y = F(-12.252);
    test.z = F(95.12);
    const Float test2(test);
    REQUIRE(test.size() == 3);
    REQUIRE(test.elements() == 3);
    REQUIRE((test[0] == F(23.23) && test[1] == F(-12.252) && test[2] == F(95.12)));
    REQUIRE((test2[0] == F(23.23) && test2[1] == F(-12.252) && test2[2] == F(95.12)));
    REQUIRE_THAT(Math::sum(test), Catch::WithinAbs(static_cast<double>(test.x + test.y + test.z), 1e-6));
    REQUIRE_THAT(Math::prod(test), Catch::WithinAbs(static_cast<double>(test.x * test.y * test.z), 1e-6));
    tmp = Math::ceil(test); REQUIRE(all(Math::isEqual(tmp, Float(24, -12, 96), F(0))));
    tmp = Math::floor(test); REQUIRE(all(Math::isEqual(tmp, Float(23, -13, 95), F(0))));

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
    REQUIRE(all(Math::isEqual(t3, Float(-3, 6, -3), F(0))));

    REQUIRE((String::format("{}", test) == std::string{"(23.230,-12.252,95.120)"}));

    std::array<TestType, 3> test3 = toArray(test);
    REQUIRE(test3[0] == test.x);
    REQUIRE(test3[1] == test.y);
    REQUIRE(test3[2] == test.z);
}

TEMPLATE_TEST_CASE("Vectors: Float4", "[noa][vectors]", float, double) {
    using Float = Float4<TestType>;

    //@CLION-formatter:off
    Float test;
    REQUIRE(all(test == Float{TestType{0}}));
    test = 2;            REQUIRE(all(test == TestType{2}));
    test += F(1.34);     REQUIRE(all(Math::isEqual(test, F(3.34))));
    test -= F(23.134);   REQUIRE(all(Math::isEqual(test, F(-19.794))));
    test *= F(-2.45);    REQUIRE(all(Math::isEqual(test, F(48.4953))));
    test /= F(567.234);  REQUIRE(all(Math::isEqual(test, F(0.085494))));

    test = 3.30;
    auto tmp = test + F(3.234534);   REQUIRE(all(Math::isEqual(tmp, F(6.534534))));
    tmp = test - F(-234.2);          REQUIRE(all(Math::isEqual(tmp, F(237.5))));
    tmp = test * F(3);               REQUIRE(all(Math::isEqual(tmp, F(9.90))));
    tmp = test / F(0.001);           REQUIRE(all(Math::isEqual(tmp, F(3299.999f), F(1e-3))));

    test = {4, 10, 4, 1};
    REQUIRE(Bool4(1, 1, 1, 0) == (Math::isEqual(test, Float{4, 10, 4, 1.001})));
    REQUIRE(Bool4(0, 1, 0, 0) == (test > TestType{5}));
    REQUIRE(Bool4(1, 0, 1, 1) == (test < TestType{10}));
    REQUIRE(Bool4(0, 1, 0, 0) == (test >= TestType{7}));
    REQUIRE(Bool4(1, 0, 1, 1) == (test <= TestType{9}));
    REQUIRE(Bool4(0, 1, 0, 1) == (test != TestType{4}));

    test = Float4<float>{0, 2, 123, 32};
    test += Float(35, 20, -12, 1);          REQUIRE(all(Math::isEqual(test, Float(35, 22, 111, 33))));
    test -= Float(-0.12, 23.2123, 0.23, 2); REQUIRE(all(Math::isEqual(test, Float(35.12, -1.2123, 110.77, 31))));
    test *= Float(0, 10, -3.2, -0.324);     REQUIRE(all(Math::isEqual(test, Float(0, -12.123, -354.464, -10.044), F(1e-5))));
    test /= Float(2, 9, 2, -0.5);           REQUIRE(all(Math::isEqual(test, Float(0, -1.347, -177.232, 20.088))));

    test.x = 20;
    test.y = 50;
    test.z = 33;
    test.w = 5;
    tmp = test + Float(10, 12, -1232, 2.3);         REQUIRE(all(Math::isEqual(tmp, Float(30, 62, -1199, 7.3))));
    tmp = test - Float(10.32, -112.001, 0.5541, 1); REQUIRE(all(Math::isEqual(tmp, Float(9.68, 162.001, 32.4459, 4))));
    tmp = test * Float(2.5, 3.234, 58.12, 8.81);    REQUIRE(all(Math::isEqual(tmp, Float(50, 161.7, 1917.959999, 44.050))));
    tmp = test / Float(10, -12, -2.3, 0.01);        REQUIRE(all(Math::isEqual(tmp, Float(2, -4.166667, -14.3478261, 500))));

    test = {2, 4, -1, 12};
    REQUIRE(Bool4(1, 1, 1, 1) == (test > Float{1, 2, -3, 11}));
    REQUIRE(Bool4(0, 1, 1, 1) == (test < Float{2, 5, 0, 13}));
    REQUIRE(Bool4(1, 1, 1, 1) == (test >= Float{2, 4, -1, 11}));
    REQUIRE(Bool4(1, 1, 0, 1) == (test <= Float{10, 4, -3, 12}));
    REQUIRE(Bool4(1, 0, 0, 1) == (test != Float{4, 4, -1, 11}));
    REQUIRE(Bool4(1, 1, 0, 1) == (Math::isEqual(test, Float{2, 4, -1.001, 12})));

    // Min & Max
    REQUIRE(all(Math::min(Float{3, 4, -34, 2.34}, Float{5, 2, -12, 120.12}) == Float{3, 2, -34, 2.34}));
    REQUIRE(all(Math::max(Float{3, 4, -3, -9.9}, Float{5, 2, 23, -10}) == Float{5, 4, 23, -9.9}));
    REQUIRE(all(Math::min(Float{3, 6, 32, 5.01}, TestType{5}) == Float{3, 5, 5, 5}));
    REQUIRE(all(Math::max(Float{9, 0, -99, 2.01}, TestType{2}) == Float{9, 2, 2, 2.01}));

    // .data()
    test = Int4<long>{3, 90, -123, 12};
    REQUIRE(all(Math::isEqual(test, Float(3, 90, -123, 12))));
    Float4<double> test1(test);
    REQUIRE(all(test1 == static_cast<Float4<double>>(test)));

    test.x = F(23.23);
    test.y = F(-12.252);
    test.z = F(95.12);
    test.w = F(2.34);
    const Float test2(test);
    REQUIRE(test.size() == 4);
    REQUIRE(test.elements() == 4);
    REQUIRE((test[0] == F(23.23) && test[1] == F(-12.252) && test[2] == F(95.12) && test[3] == F(2.34)));
    REQUIRE((test2[0] == F(23.23) && test2[1] == F(-12.252) && test2[2] == F(95.12) && test2[3] == F(2.34)));
    REQUIRE_THAT(Math::sum(test), Catch::WithinAbs(static_cast<double>(test.x + test.y + test.z + test.w), 1e-6));
    REQUIRE_THAT(Math::prod(test), Catch::WithinAbs(static_cast<double>(test.x * test.y * test.z * test.w), 1e-6));
    tmp = Math::ceil(test); REQUIRE(all(Math::isEqual(tmp, Float(24, -12, 96, 3), F(0))));
    tmp = Math::floor(test); REQUIRE(all(Math::isEqual(tmp, Float(23, -13, 95, 2), F(0))));

    auto lengthSq = static_cast<double>(test.x * test.x + test.y * test.y + test.z * test.z + test.w * test.w);
    REQUIRE_THAT(Math::lengthSq(test), Catch::WithinAbs(lengthSq, 1e-6));
    REQUIRE_THAT(Math::length(test), Catch::WithinAbs(std::sqrt(lengthSq), 1e-6));
    tmp = Math::normalize(test); REQUIRE_THAT(Math::length(tmp), Catch::WithinAbs(1, 1e-6));

    //@CLION-formatter:on
    REQUIRE((String::format("{}", test) == std::string{"(23.230,-12.252,95.120,2.340)"}));

    std::array<TestType, 4> test3 = toArray(test);
    REQUIRE(test3[0] == test.x);
    REQUIRE(test3[1] == test.y);
    REQUIRE(test3[2] == test.z);
    REQUIRE(test3[3] == test.w);
}
