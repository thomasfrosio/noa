#include <noa/common/Types.h>

#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("IntX, FloatX, typedefs", "[noa][common][types]") {
    static_assert(std::is_same_v<int2_t, Int2<int>>);
    static_assert(std::is_same_v<uint2_t, Int2<unsigned int>>);
    static_assert(std::is_same_v<long2_t, Int2<int64_t>>);
    static_assert(std::is_same_v<ulong2_t, Int2<uint64_t>>);

    static_assert(std::is_same_v<int3_t, Int3<int>>);
    static_assert(std::is_same_v<uint3_t, Int3<unsigned int>>);
    static_assert(std::is_same_v<long3_t, Int3<int64_t>>);
    static_assert(std::is_same_v<ulong3_t, Int3<uint64_t>>);

    static_assert(std::is_same_v<int4_t, Int4<int>>);
    static_assert(std::is_same_v<uint4_t, Int4<unsigned int>>);
    static_assert(std::is_same_v<long4_t, Int4<int64_t>>);
    static_assert(std::is_same_v<ulong4_t, Int4<uint64_t>>);

    static_assert(std::is_same_v<float2_t, Float2<float>>);
    static_assert(std::is_same_v<double2_t, Float2<double>>);

    static_assert(std::is_same_v<float3_t, Float3<float>>);
    static_assert(std::is_same_v<double3_t, Float3<double>>);

    static_assert(std::is_same_v<float4_t, Float4<float>>);
    static_assert(std::is_same_v<double4_t, Float4<double>>);
}

TEMPLATE_TEST_CASE("Int2", "[noa][common][types]",
                   int, long, long long,
                   unsigned int, unsigned long, unsigned long long) {
    using Int = Int2<TestType>;

    //@CLION-formatter:off
    Int test;
    REQUIRE(all(test == Int{0}));

    test = 2;
    test += 1; REQUIRE(all(test == 3));
    test -= 2; REQUIRE(all(test == 1));
    test *= 3; REQUIRE(all(test == 3));
    test /= 2; REQUIRE(all(test == 1));

    test = 30;
    REQUIRE(all(test + 10 == 40));
    REQUIRE(all(test - 5 == 25));
    REQUIRE(all(test * 3 == 90));
    REQUIRE(all(test / 2 == 15));

    REQUIRE(all(40 == 10 + test));
    REQUIRE(all(15 == 45 - test));
    REQUIRE(all(90 == 3 * test));
    REQUIRE(all(2 == 60 / test));

    test = {4, 10};
    REQUIRE(all(Bool2(0, 1) == (test > 5)));
    REQUIRE(all(Bool2(1, 1) == (test < 11)));
    REQUIRE(all(Bool2(0, 1) == (test >= 7)));
    REQUIRE(all(Bool2(1, 1) == (test <= 10)));
    REQUIRE(any(test == 4));
    REQUIRE_FALSE(all(test == 4));

    REQUIRE(all((5 < test) == Bool2(0, 1)));
    REQUIRE(all((11 > test) == Bool2(1, 1)));
    REQUIRE(all((7 <= test) == Bool2(0, 1)));
    REQUIRE(all((9 >= test) == Bool2(1, 0)));
    REQUIRE(any(4 == test));
    REQUIRE_FALSE(all(4 == test));

    test = Int{0, 2};
    test += Int(35, 20); REQUIRE(all(test == Int(35, 22)));
    test -= Int(22, 18); REQUIRE(all(test == Int(13, 4)));
    test *= Int(2, 6);   REQUIRE(all(test == Int(26, 24)));
    test /= Int(2, 9);   REQUIRE(all(test == Int(13, 2)));
    //@CLION-formatter:on

    test[0] = 20;
    test[1] = 50;
    REQUIRE(all(test + Int(10, 12) == Int(30, 62)));
    REQUIRE(all(test - Int(3, 2) == Int(17, 48)));
    REQUIRE(all(test * Int(1, 5) == Int(20, 250)));
    REQUIRE(all(test / Int(3, 2) == Int(6, 25)));

    test = {2, 4};
    REQUIRE(all((test > Int{1, 2}) == Bool2(1, 1)));
    REQUIRE(all((test < Int{4, 5}) == Bool2(1, 1)));
    REQUIRE(all((test >= Int{2, 4}) == Bool2(1, 1)));
    REQUIRE(all((test <= Int{1, 4}) == Bool2(0, 1)));
    REQUIRE(all((test != Int{4, 4}) == Bool2(1, 0)));

    // Min & Max
    REQUIRE(all(math::min(Int{3, 4}, Int{5, 2}) == Int{3, 2}));
    REQUIRE(all(math::max(Int{3, 4}, Int{5, 2}) == Int{5, 4}));
    REQUIRE(all(math::min(Int{3, 6}, TestType{5}) == Int{3, 5}));
    REQUIRE(all(math::max(Int{9, 0}, TestType{2}) == Int{9, 2}));

    test = Int{3.4f, 90.6f};
    REQUIRE(all(test == Int(3, 90)));
    int2_t test1(std::move(Int(test)));
    REQUIRE(all(test1 == static_cast<int2_t>(test)));

    test[0] = 23;
    test[1] = 52;
    const Int test2(23, 52);
    REQUIRE((test[0] == 23 && test[1] == 52));
    REQUIRE((test2[0] == 23 && test[1] == 52));
    REQUIRE(test.COUNT == 2);
    REQUIRE(test.COUNT == 2);
    REQUIRE(math::sum(test) == 75);
    REQUIRE(math::prod(test) == 1196);
    REQUIRE(test.elements() == 1196);
    REQUIRE(test.fft().elements() == 621);
    REQUIRE(all(test.stride() == Int{52, 1}));
    REQUIRE(all(test.fft().stride() == Int{27, 1}));
    REQUIRE(test.ndim() == 2);
    REQUIRE(Int{2, 1}.ndim() == 2);
    REQUIRE(Int{1, 10}.ndim() == 1);

    REQUIRE((string::format("{}", test) == "(23,52)"));

    std::array<TestType, 2> test3 = toArray(test);
    REQUIRE(test3[0] == test[0]);
    REQUIRE(test3[1] == test[1]);
}

TEMPLATE_TEST_CASE("Int3", "[noa][common][types]",
                   int, long, long long,
                   unsigned int, unsigned long, unsigned long long) {
    using Int = Int3<TestType>;

    //@CLION-formatter:off
    Int test;
    REQUIRE(all(test == Int{0}));

    test = 2;
    test += 1; REQUIRE(all(test == 3));
    test -= 2; REQUIRE(all(test == 1));
    test *= 3; REQUIRE(all(test == 3));
    test /= 2; REQUIRE(all(test == 1));

    test = 30;
    REQUIRE(all(test + 10 == 40));
    REQUIRE(all(test - 5 == 25));
    REQUIRE(all(test * 3 == 90));
    REQUIRE(all(test / 2 == 15));

    REQUIRE(all(40 == 10 + test));
    REQUIRE(all(15 == 45 - test));
    REQUIRE(all(90 == 3 * test));
    REQUIRE(all(2 == 60 / test));

    test = {4, 10, 7};
    REQUIRE(all(Bool3(0, 1, 0) == (test > 9)));
    REQUIRE(all(Bool3(1, 1, 1) == (test < 11)));
    REQUIRE(all(Bool3(0, 1, 1) == (test >= 7)));
    REQUIRE(all(Bool3(1, 0, 1) == (test <= 9)));
    REQUIRE(any(test == 4));
    REQUIRE_FALSE(all(test == 4));

    REQUIRE(all(Bool3(1, 0, 1) == (9 > test)));
    REQUIRE(all(Bool3(1, 1, 1) == (11 > test)));
    REQUIRE(all(Bool3(1, 0, 1) == (7 >= test)));
    REQUIRE(all(Bool3(0, 1, 0) == (9 <= test)));
    REQUIRE(any(4 == test));
    REQUIRE_FALSE(all(4 == test));

    test = Int{0, 2, 130};
    test += Int(35, 20, 4);     REQUIRE(all(test == Int(35, 22, 134)));
    test -= Int(22, 18, 93);    REQUIRE(all(test == Int(13, 4, 41)));
    test *= Int(2, 6, 2);       REQUIRE(all(test == Int(26, 24, 82)));
    test /= Int(2, 9, 1);       REQUIRE(all(test == Int(13, 2, 82)));
    //@CLION-formatter:on

    test[0] = 20;
    test[1] = 50;
    test[2] = 10;
    REQUIRE(all(test + Int(10, 12, 5) == Int(30, 62, 15)));
    REQUIRE(all(test - Int(3, 2, 7) == Int(17, 48, 3)));
    REQUIRE(all(test * Int(1, 5, 3) == Int(20, 250, 30)));
    REQUIRE(all(test / Int(3, 2, 2) == Int(6, 25, 5)));

    test = {2, 4, 4};
    REQUIRE(all(Bool3(1, 1, 0) == (test > Int{1, 2, 4})));
    REQUIRE(all(Bool3(0, 1, 1) == (test < Int{2, 5, 6})));
    REQUIRE(all(Bool3(1, 1, 1) == (test >= Int{2, 4, 3})));
    REQUIRE(all(Bool3(0, 1, 1) == (test <= Int{1, 4, 6})));
    REQUIRE(all(Bool3(1, 0, 0) == (test != Int{4, 4, 4})));
    REQUIRE(all(Bool3(0, 0, 1) == (test == Int{4, 2, 4})));

    // Min & Max
    REQUIRE(all(math::min(Int{3, 4, 8}, Int{5, 2, 10}) == Int{3, 2, 8}));
    REQUIRE(all(math::max(Int{3, 4, 1000}, Int{5, 2, 30}) == Int{5, 4, 1000}));
    REQUIRE(all(math::min(Int{3, 6, 4}, TestType{5}) == Int{3, 5, 4}));
    REQUIRE(all(math::max(Int{9, 0, 1}, TestType{2}) == Int{9, 2, 2}));

    test = Int{3.4f, 90.6f, 5.f};
    REQUIRE(all(test == Int(3, 90, 5)));
    Int3<long> test1(test);
    REQUIRE(all(test1 == static_cast<Int3<long>>(test)));

    test[0] = 23;
    test[1] = 52;
    test[2] = 128;
    const Int test2(test);
    REQUIRE((test[0] == 23 && test[1] == 52 && test[2] == 128));
    REQUIRE((test2[0] == 23 && test2[1] == 52 && test2[2] == 128));
    REQUIRE(test.COUNT == 3);
    REQUIRE(test.COUNT == 3);
    REQUIRE(math::sum(test) == 203);
    REQUIRE(math::prod(test) == 153088);
    REQUIRE(test.elements() == 153088);
    REQUIRE(test.fft().elements() == 77740);
    REQUIRE(all(test.stride() == Int{6656,128,1}));
    REQUIRE(all(test.fft().stride() == Int{3380, 65, 1}));
    REQUIRE(test.ndim() == 3);
    REQUIRE(Int{2, 2, 1}.ndim() == 3);
    REQUIRE(Int{1, 2, 1}.ndim() == 2);
    REQUIRE(Int{2, 1, 5}.ndim() == 3);

    REQUIRE((string::format("{}", test) == std::string{"(23,52,128)"}));

    std::array<TestType, 3> test3 = toArray(test);
    REQUIRE(test3[0] == test[0]);
    REQUIRE(test3[1] == test[1]);
    REQUIRE(test3[2] == test[2]);
}

TEMPLATE_TEST_CASE("Int4", "[noa][common][types]",
                   int, long, long long,
                   unsigned int, unsigned long, unsigned long long) {
    using Int = Int4<TestType>;

    //@CLION-formatter:off
    Int test;
    REQUIRE(all(test == Int{0}));
    test = 2;
    test += 1; REQUIRE(all(test == 3));
    test -= 2; REQUIRE(all(test == 1));
    test *= 3; REQUIRE(all(test == 3));
    test /= 2; REQUIRE(all(test == 1));

    test = 30;
    REQUIRE(all(test + 10 == 40));
    REQUIRE(all(test - 5 == 25));
    REQUIRE(all(test * 3 == 90));
    REQUIRE(all(test / 2 == 15));

    test = 30;
    REQUIRE(all(40 == 10 + test));
    REQUIRE(all(15 == 45 - test));
    REQUIRE(all(90 == 3 * test));
    REQUIRE(all(2 == 60 / test));

    test = {15, 130, 70, 2};
    REQUIRE(all(Bool4(1, 1, 1, 0) == (test > 9)));
    REQUIRE(all(Bool4(1, 0, 1, 1) == (test < 130)));
    REQUIRE(all(Bool4(1, 1, 1, 0) == (test >= 7)));
    REQUIRE(all(Bool4(1, 0, 0, 1) == (test <= 50)));
    REQUIRE(all(Bool4(0, 1, 1, 1) == (test != 15)));
    REQUIRE(all(Bool4(1, 0, 0, 0) == (test == 15)));

    REQUIRE(all(Bool4(1, 1, 1, 0) == (9 < test)));
    REQUIRE(all(Bool4(1, 0, 1, 1) == (130 > test)));
    REQUIRE(all(Bool4(1, 1, 1, 0) == (7 <= test)));
    REQUIRE(all(Bool4(1, 0, 0, 1) == (50 >= test)));
    REQUIRE(all(Bool4(0, 1, 1, 1) == (15 != test)));
    REQUIRE(all(Bool4(1, 0, 0, 0) == (15 == test)));

    test = Int(130, 5, 130, 120);
    test += Int(35, 20, 4, 20); REQUIRE(all(test == Int(165, 25, 134, 140)));
    test -= Int(22, 1, 93, 2);  REQUIRE(all(test == Int(143, 24, 41, 138)));
    test *= Int(2, 6, 2, 9);    REQUIRE(all(test == Int(286, 144, 82, 1242)));
    test /= Int(2, 9, 1, 4);    REQUIRE(all(test == Int(143, 16, 82, 310)));

    //@CLION-formatter:on
    test[0] = 20;
    test[1] = 50;
    test[2] = 10;
    test[3] = 3;
    const Int test2(test);
    REQUIRE((test[0] == 20 && test[1] == 50 && test[2] == 10 && test[3] == 3));
    REQUIRE((test2[0] == 20 && test2[1] == 50 && test2[2] == 10 && test2[3] == 3));
    REQUIRE(all(test + Int(10, 12, 5, 30) == Int(30, 62, 15, 33)));
    REQUIRE(all(test - Int(3, 2, 7, 1) == Int(17, 48, 3, 2)));
    REQUIRE(all(test * Int(1, 5, 3, 103) == Int(20, 250, 30, 309)));
    REQUIRE(all(test / Int(3, 2, 2, 10) == Int(6, 25, 5, 0)));

    test = {2, 4, 4, 13405};
    REQUIRE(all(Bool4(0, 0, 0, 1) == (test > Int{5, 5, 5, 100})));
    REQUIRE(all(Bool4(1, 1, 1, 0) == (test < Int{5, 5, 6, 100})));
    REQUIRE(all(Bool4(1, 1, 1, 1) == (test >= Int{2, 4, 3, 4})));
    REQUIRE(all(Bool4(1, 1, 0, 0) == (test <= Int{10, 4, 1, 2})));
    REQUIRE(all(Bool4(0, 0, 0, 1) == (test != Int{2, 4, 4, 13404})));
    REQUIRE(all(Bool4(0, 1, 0, 1) == (test == Int{3, 4, 3, 13405})));

    // Min & Max
    REQUIRE(all(math::min(Int{3, 4, 8, 1230}, Int{5, 2, 10, 312}) == Int{3, 2, 8, 312}));
    REQUIRE(all(math::max(Int{3, 4, 1000, 2}, Int{5, 2, 30, 1}) == Int{5, 4, 1000, 2}));
    REQUIRE(all(math::min(Int{3, 6, 4, 74}, TestType{5}) == Int{3, 5, 4, 5}));
    REQUIRE(all(math::max(Int{9, 0, 1, 4}, TestType{2}) == Int{9, 2, 2, 4}));

    test = Int{3.4, 90.6, 5., 12.99};
    REQUIRE(all(test == Int(3, 90, 5, 12)));
    Int4<int> test1(test);
    REQUIRE(all(test1 == static_cast<Int4<int>>(test)));

    test[0] = 4;
    test[1] = 52;
    test[2] = 128;
    test[3] = 58;
    REQUIRE(test.COUNT == 4);
    REQUIRE(math::sum(test) == 242);
    REQUIRE(math::prod(test) == 1544192);
    REQUIRE(test.elements() == 1544192);
    REQUIRE(test.fft().elements() == 798720);
    REQUIRE(all(test.stride() == Int{386048,7424,58,1}));
    REQUIRE(all(test.fft().stride() == Int{199680, 3840, 30, 1}));
    REQUIRE(test.ndim() == 4);
    REQUIRE(Int{2, 2, 1, 1}.ndim() == 4);
    REQUIRE(Int{1, 1, 2, 1}.ndim() == 2);
    REQUIRE(Int{1, 1, 1, 4}.ndim() == 1);

    REQUIRE((string::format("{}", test) == std::string{"(4,52,128,58)"}));

    std::array<TestType, 4> test3 = toArray(test);
    REQUIRE(test3[0] == test[0]);
    REQUIRE(test3[1] == test[1]);
    REQUIRE(test3[2] == test[2]);
    REQUIRE(test3[3] == test[3]);
}

#define F(x) static_cast<TestType>(x)

TEMPLATE_TEST_CASE("Float2", "[noa][common][types]", float, double) {
    using Float = Float2<TestType>;

    //@CLION-formatter:off
    Float test;
    REQUIRE(all(test == Float(0)));
    test = 2;            REQUIRE(all(test == TestType{2}));
    test += F(1.34);     REQUIRE(all(math::isEqual(test, F(3.34))));
    test -= F(23.134);   REQUIRE(all(math::isEqual(test, F(-19.794))));
    test *= F(-2.45);    REQUIRE(all(math::isEqual(test, F(48.4953))));
    test /= F(567.234);  REQUIRE(all(math::isEqual(test, F(0.085494))));

    test = static_cast<TestType>(3.30);
    auto tmp = test + F(3.234534);   REQUIRE(all(math::isEqual(tmp, F(6.534534))));
    tmp = test - F(-234.2);          REQUIRE(all(math::isEqual(tmp, F(237.5))));
    tmp = test * F(3);               REQUIRE(all(math::isEqual(tmp, F(9.90))));
    tmp = test / F(0.001);           REQUIRE(all(math::isEqual(tmp, F(3299.999f), F(1e-3))));

    test = {4, 10};
    REQUIRE(all(math::isEqual(test, Float{4, 10})));
    REQUIRE(all(Bool2(0, 1) == (test > 5)));
    REQUIRE(all(Bool2(1, 1) == (test < 11)));
    REQUIRE(all(Bool2(0, 1) == (test >= 7)));
    REQUIRE(all(Bool2(1, 0) == (test <= 9)));
    REQUIRE(all(Bool2(0, 1) == (test != 4)));

    test = Float{0, 2};
    test += Float(35, 20);                  REQUIRE(all(math::isEqual(test, Float(35, 22))));
    test -= Float(F(-0.12), F(23.2123));    REQUIRE(all(math::isEqual(test, Float(35.12, -1.2123))));
    test *= Float(F(0), F(10));             REQUIRE(all(math::isEqual(test, Float(0, -12.123), F(1e-5))));
    test /= Float(2, 9);                    REQUIRE(all(math::isEqual(test, Float(0, -1.347))));

    test[0] = 20;
    test[1] = 50;
    tmp = test + Float(10, 12);                     REQUIRE(all(math::isEqual(tmp, Float(30, 62))));
    tmp = test - Float(F(10.32), F(-112.001));      REQUIRE(all(math::isEqual(tmp, Float(9.68, 162.001))));
    tmp = test * Float(F(2.5), F(3.234));           REQUIRE(all(math::isEqual(tmp, Float(50, 161.7))));
    tmp = test / Float(F(10), F(-12));              REQUIRE(all(math::isEqual(tmp, Float(2, -4.166667))));

    test = {2, 4};
    REQUIRE(all(Bool2(1, 1) == (test > Float{1, 2})));
    REQUIRE(all(Bool2(1, 0) == (test < Float{4, 4})));
    REQUIRE(all(Bool2(1, 1) == (test >= Float{2, 4})));
    REQUIRE(all(Bool2(0, 1) == (test <= Float{1, 4})));
    REQUIRE(all(Bool2(1, 0) == (test != Float{4, 4})));
    REQUIRE(all(Bool2(1, 1) == (test == Float{2, 4})));

    // Min & Max
    REQUIRE(all(math::min(Float{3, 4}, Float{5, 2}) == Float{3, 2}));
    REQUIRE(all(math::max(Float{3, 4}, Float{5, 2}) == Float{5, 4}));
    REQUIRE(all(math::min(Float{3, 6}, TestType{5}) == Float{3, 5}));
    REQUIRE(all(math::max(Float{9, 0}, TestType{2}) == Float{9, 2}));

    test = Float{3, 90};
    REQUIRE(all(math::isEqual(test, Float(3, 90))));
    Float2<double> test1(test);
    REQUIRE(all(test1 == static_cast<Float2<double>>(test)));

    test[0] = F(23.23);
    test[1] = F(-12.252);
    const Float test2(test);
    REQUIRE(test.COUNT == 2);
    REQUIRE((test[0] == F(23.23) && test[1] == F(-12.252)));
    REQUIRE((test2[0] == F(23.23) && test2[1] == F(-12.252)));
    REQUIRE_THAT(math::sum(test), Catch::WithinAbs(10.978, 1e-6));
    REQUIRE_THAT(math::prod(test), Catch::WithinAbs(static_cast<double>(test[0] * test[1]), 1e-6));
    tmp = math::ceil(test); REQUIRE(all(math::isEqual(tmp, Float(24, -12), F(0))));
    tmp = math::floor(test); REQUIRE(all(math::isEqual(tmp, Float(23, -13), F(0))));

    auto lengthSq = static_cast<double>(test[0] * test[0] + test[1] * test[1]);
    REQUIRE_THAT(math::dot(test, test), Catch::WithinAbs(lengthSq, 1e-6));
    REQUIRE_THAT(math::length(test), Catch::WithinAbs(std::sqrt(lengthSq), 1e-6));
    tmp = math::normalize(test); REQUIRE_THAT(math::length(tmp), Catch::WithinAbs(1, 1e-6));
    REQUIRE_THAT(math::dot(test, Float(-12.23, -21.23)), Catch::WithinAbs(-23.992940, 1e-4));
    //@CLION-formatter:on

    REQUIRE((string::format("{:.5}", test) == std::string{"(23.230,-12.252)"}));

    std::array<TestType, 2> test3 = toArray(test);
    REQUIRE(test3[0] == test[0]);
    REQUIRE(test3[1] == test[1]);
}

TEMPLATE_TEST_CASE("Float3", "[noa][common][types]", float, double) {
    using Float = Float3<TestType>;

    //@CLION-formatter:off
    Float test;
    REQUIRE(all(test == Float{TestType{0}}));
    test = 2;            REQUIRE(all(test == TestType{2}));
    test += static_cast<TestType>(1.34);     REQUIRE(all(math::isEqual(test, F(3.34))));
    test -= static_cast<TestType>(23.134);   REQUIRE(all(math::isEqual(test, F(-19.794))));
    test *= static_cast<TestType>(-2.45);    REQUIRE(all(math::isEqual(test, F(48.4953))));
    test /= static_cast<TestType>(567.234);  REQUIRE(all(math::isEqual(test, F(0.085494))));

    test = F(3.30);
    auto tmp = test + F(3.234534);   REQUIRE(all(math::isEqual(tmp, F(6.534534))));
    tmp = test - F(-234.2);          REQUIRE(all(math::isEqual(tmp, F(237.5))));
    tmp = test * F(3);               REQUIRE(all(math::isEqual(tmp, F(9.90))));
    tmp = test / F(0.001);           REQUIRE(all(math::isEqual(tmp, F(3299.999f), F(1e-3))));

    test = {4, 10, 4};
    REQUIRE(all(Bool3(0, 1, 1) == (math::isEqual(test, Float{3.99, 10, 4}))));
    REQUIRE(all(Bool3(0, 1, 0) == (test > 5)));
    REQUIRE(all(Bool3(1, 1, 1) == (test < 11)));
    REQUIRE(all(Bool3(0, 1, 0) == (test >= 7)));
    REQUIRE(all(Bool3(1, 0, 1) == (test <= 9)));
    REQUIRE(all(Bool3(0, 1, 0) == (test != 4)));
    REQUIRE(all(Bool3(0, 1, 0) == (test == 10)));

    test = Float{0, 2, 123};
    test += Float(35, 20, -12);             REQUIRE(all(math::isEqual(test, Float(35, 22, 111))));
    test -= Float(-0.12, 23.2123, 0.23);    REQUIRE(all(math::isEqual(test, Float(35.12, -1.2123, 110.77))));
    test *= Float(0, 10, -3.2);             REQUIRE(all(math::isEqual(test, Float(0, -12.123, -354.464), F(1e-5))));
    test /= Float(2, 9, 2);                 REQUIRE(all(math::isEqual(test, Float(0, -1.347, -177.232))));

    test[0] = 20;
    test[1] = 50;
    test[2] = 33;
    tmp = test + Float(10, 12, -1232);              REQUIRE(all(math::isEqual(tmp, Float(30, 62, -1199))));
    tmp = test - Float(10.32, -112.001, 0.5541);    REQUIRE(all(math::isEqual(tmp, Float(9.68, 162.001, 32.4459))));
    tmp = test * Float(2.5, 3.234, 58.12);          REQUIRE(all(math::isEqual(tmp, Float(50, 161.7, 1917.959999))));
    tmp = test / Float(10, -12, -2.3);              REQUIRE(all(math::isEqual(tmp, Float(2, -4.166667, -14.3478261))));

    test = {2, 4, -1};
    REQUIRE(all(Bool3(1, 1, 1) == (test > Float{1, 2, -3})));
    REQUIRE(all(Bool3(1, 0, 1) == (test < Float{4, 4, 0})));
    REQUIRE(all(Bool3(1, 1, 0) == (test >= Float{2, 4, 0})));
    REQUIRE(all(Bool3(1, 1, 1) == (test <= Float{10, 4, 3})));
    REQUIRE(all(Bool3(1, 0, 0) == (test != Float{4, 4, -1})));
    REQUIRE(all(Bool3(1, 1, 0) == (math::isEqual(test, Float{2, 4, 0.99}))));

    // Min & Max
    REQUIRE(all(math::min(Float{3, 4, -34}, Float{5, 2, -12}) == Float{3, 2, -34}));
    REQUIRE(all(math::max(Float{3, 4, -3}, Float{5, 2, 23}) == Float{5, 4, 23}));
    REQUIRE(all(math::min(Float{3, 6, 32}, TestType{5}) == Float{3, 5, 5}));
    REQUIRE(all(math::max(Float{9, 0, -99}, TestType{2}) == Float{9, 2, 2}));

    test = Float{3, 90, -123};
    REQUIRE(all(math::isEqual(test, Float(3, 90, -123))));
    Float3<double> test1(test);
    REQUIRE(all(test1 == static_cast<Float3<double>>(test)));

    test[0] = F(23.23);
    test[1] = F(-12.252);
    test[2] = F(95.12);
    const Float test2(test);
    REQUIRE(test.COUNT == 3);
    REQUIRE((test[0] == F(23.23) && test[1] == F(-12.252) && test[2] == F(95.12)));
    REQUIRE((test2[0] == F(23.23) && test2[1] == F(-12.252) && test2[2] == F(95.12)));
    REQUIRE_THAT(math::sum(test), Catch::WithinAbs(static_cast<double>(test[0] + test[1] + test[2]), 1e-6));
    REQUIRE_THAT(math::prod(test), Catch::WithinAbs(static_cast<double>(test[0] * test[1] * test[2]), 1e-6));
    tmp = math::ceil(test); REQUIRE(all(math::isEqual(tmp, Float(24, -12, 96), F(0))));
    tmp = math::floor(test); REQUIRE(all(math::isEqual(tmp, Float(23, -13, 95), F(0))));

    auto lengthSq = static_cast<double>(test[0] * test[0] + test[1] * test[1] + test[2] * test[2]);
    REQUIRE_THAT(math::dot(test, test), Catch::WithinAbs(lengthSq, 1e-6));
    REQUIRE_THAT(math::length(test), Catch::WithinAbs(std::sqrt(lengthSq), 1e-6));
    tmp = math::normalize(test); REQUIRE_THAT(math::length(tmp), Catch::WithinAbs(1, 1e-6));

    auto tmp1 = Float(F(-12.23), F(-21.23), F(123.22));
    REQUIRE_THAT(math::dot(test, tmp1), Catch::WithinAbs(11696.69346, 1e-3));
    //@CLION-formatter:on

    Float t1(2, 3, 4);
    Float t2(5, 6, 7);
    Float t3(math::cross(t1, t2));
    REQUIRE(all(math::isEqual(t3, Float(-3, 6, -3), F(0))));

    REQUIRE((string::format("{:.5}", test) == std::string{"(23.230,-12.252,95.120)"}));

    std::array<TestType, 3> test3 = toArray(test);
    REQUIRE(test3[0] == test[0]);
    REQUIRE(test3[1] == test[1]);
    REQUIRE(test3[2] == test[2]);
}

TEMPLATE_TEST_CASE("Float4", "[noa][common][types]", float, double) {
    using Float = Float4<TestType>;

    //@CLION-formatter:off
    Float test;
    REQUIRE(all(test == Float{0}));
    test = 2;            REQUIRE(all(test == 2));
    test += F(1.34);     REQUIRE(all(math::isEqual(test, F(3.34))));
    test -= F(23.134);   REQUIRE(all(math::isEqual(test, F(-19.794))));
    test *= F(-2.45);    REQUIRE(all(math::isEqual(test, F(48.4953))));
    test /= F(567.234);  REQUIRE(all(math::isEqual(test, F(0.085494))));

    test = static_cast<TestType>(3.30);
    auto tmp = test + F(3.234534);   REQUIRE(all(math::isEqual(tmp, F(6.534534))));
    tmp = test - F(-234.2);          REQUIRE(all(math::isEqual(tmp, F(237.5))));
    tmp = test * F(3);               REQUIRE(all(math::isEqual(tmp, F(9.90))));
    tmp = test / F(0.001);           REQUIRE(all(math::isEqual(tmp, F(3299.999f), F(1e-3))));

    test = {4, 10, 4, 1};
    REQUIRE(all(Bool4(1, 1, 1, 0) == (math::isEqual(test, Float{4, 10, 4, 1.001}))));
    REQUIRE(all(Bool4(0, 1, 0, 0) == (test > 5)));
    REQUIRE(all(Bool4(1, 0, 1, 1) == (test < 10)));
    REQUIRE(all(Bool4(0, 1, 0, 0) == (test >= 7)));
    REQUIRE(all(Bool4(1, 0, 1, 1) == (test <= 9)));
    REQUIRE(all(Bool4(0, 1, 0, 1) == (test != 4)));

    test = Float{0, 2, 123, 32};
    test += Float(35, 20, -12, 1);          REQUIRE(all(math::isEqual(test, Float(35, 22, 111, 33))));
    test -= Float(-0.12, 23.2123, 0.23, 2); REQUIRE(all(math::isEqual(test, Float(35.12, -1.2123, 110.77, 31))));
    test *= Float(0, 10, -3.2, -0.324);     REQUIRE(all(math::isEqual(test, Float(0, -12.123, -354.464, -10.044), F(1e-5))));
    test /= Float(2, 9, 2, -0.5);           REQUIRE(all(math::isEqual(test, Float(0, -1.347, -177.232, 20.088))));

    test[0] = 20;
    test[1] = 50;
    test[2] = 33;
    test[3] = 5;
    tmp = test + Float(10, 12, -1232, 2.3);         REQUIRE(all(math::isEqual(tmp, Float(30, 62, -1199, 7.3))));
    tmp = test - Float(10.32, -112.001, 0.5541, 1); REQUIRE(all(math::isEqual(tmp, Float(9.68, 162.001, 32.4459, 4))));
    tmp = test * Float(2.5, 3.234, 58.12, 8.81);    REQUIRE(all(math::isEqual(tmp, Float(50, 161.7, 1917.959999, 44.050))));
    tmp = test / Float(10, -12, -2.3, 0.01);        REQUIRE(all(math::isEqual(tmp, Float(2, -4.166667, -14.3478261, 500))));

    test = {2, 4, -1, 12};
    REQUIRE(all(Bool4(1, 1, 1, 1) == (test > Float{1, 2, -3, 11})));
    REQUIRE(all(Bool4(0, 1, 1, 1) == (test < Float{2, 5, 0, 13})));
    REQUIRE(all(Bool4(1, 1, 1, 1) == (test >= Float{2, 4, -1, 11})));
    REQUIRE(all(Bool4(1, 1, 0, 1) == (test <= Float{10, 4, -3, 12})));
    REQUIRE(all(Bool4(1, 0, 0, 1) == (test != Float{4, 4, -1, 11})));
    REQUIRE(all(Bool4(1, 1, 0, 1) == (math::isEqual(test, Float{2, 4, -1.001, 12}))));

    // Min & Max
    REQUIRE(all(math::min(Float{3, 4, -34, 2.34}, Float{5, 2, -12, 120.12}) == Float{3, 2, -34, 2.34}));
    REQUIRE(all(math::max(Float{3, 4, -3, -9.9}, Float{5, 2, 23, -10}) == Float{5, 4, 23, -9.9}));
    REQUIRE(all(math::min(Float{3, 6, 32, 5.01}, TestType{5}) == Float{3, 5, 5, 5}));
    REQUIRE(all(math::max(Float{9, 0, -99, 2.01}, TestType{2}) == Float{9, 2, 2, 2.01}));

    // .data()
    test = Float{3, 90, -123, 12};
    REQUIRE(all(math::isEqual(test, Float(3, 90, -123, 12))));
    Float4<double> test1(test);
    REQUIRE(all(test1 == static_cast<Float4<double>>(test)));

    test[0] = F(23.23);
    test[1] = F(-12.252);
    test[2] = F(95.12);
    test[3] = F(2.34);
    const Float test2(test);
    REQUIRE(test.COUNT == 4);
    REQUIRE((test[0] == F(23.23) && test[1] == F(-12.252) && test[2] == F(95.12) && test[3] == F(2.34)));
    REQUIRE((test2[0] == F(23.23) && test2[1] == F(-12.252) && test2[2] == F(95.12) && test2[3] == F(2.34)));
    REQUIRE_THAT(math::sum(test), Catch::WithinAbs(static_cast<double>(test[0] + test[1] + test[2] + test[3]), 1e-6));
    REQUIRE_THAT(math::prod(test), Catch::WithinAbs(static_cast<double>(test[0] * test[1] * test[2] * test[3]), 1e-6));
    tmp = math::ceil(test); REQUIRE(all(math::isEqual(tmp, Float(24, -12, 96, 3), F(0))));
    tmp = math::floor(test); REQUIRE(all(math::isEqual(tmp, Float(23, -13, 95, 2), F(0))));

    auto lengthSq = static_cast<double>(test[0] * test[0] + test[1] * test[1] + test[2] * test[2] + test[3] * test[3]);
    REQUIRE_THAT(math::dot(test, test), Catch::WithinAbs(lengthSq, 1e-6));
    REQUIRE_THAT(math::length(test), Catch::WithinAbs(std::sqrt(lengthSq), 1e-6));
    tmp = math::normalize(test); REQUIRE_THAT(math::length(tmp), Catch::WithinAbs(1, 1e-6));

    //@CLION-formatter:on
    REQUIRE((string::format("{:.5}", test) == std::string{"(23.230,-12.252,95.120,2.3400)"}));

    std::array<TestType, 4> test3 = toArray(test);
    REQUIRE(test3[0] == test[0]);
    REQUIRE(test3[1] == test[1]);
    REQUIRE(test3[2] == test[2]);
    REQUIRE(test3[3] == test[3]);
}
