#include <noa/core/Types.hpp>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("core::Vec2<T>", "[noa][common][types]", i32, i64, u32, u64, f32, f64) {
    using Vec2Test = noa::Vec<TestType, 2>;

    SECTION("Initialization and assignment") {
        Vec2Test a;
        REQUIRE(noa::all(a == Vec2Test{0, 0}));
        REQUIRE(noa::all(a == Vec2Test{}));
        REQUIRE(noa::all(Vec2Test(1) == Vec2Test{1, 1}));
        const Vec2Test d(1, 2);
        const Vec2Test e = {1, 2};
        const Vec2Test f{1, 2};
        REQUIRE((noa::all(d == e) && noa::all(d == f)));

        a = Vec2Test{1};
        a = d;
        a = {1};
        REQUIRE(noa::all(a == Vec2Test{1, 1}));
        a = {1, 2};
        REQUIRE(noa::all(a == Vec2Test{1, 2}));

        a[0] = 23;
        a[1] = 52;
        REQUIRE((a[0] == 23 && a[1] == 52));

        const auto [a0, a1] = a;
        REQUIRE((a0 == 23 && a1 == 52));

        Vec2Test c{3, 4};
        REQUIRE(noa::all(Vec2Test(c.data()) == c));
    }

    SECTION("Cast") {
        const Vec2Test a{1};
        const Vec<float, 2> a_{1};
        const auto d = static_cast<Vec<float, 2>>(a);
        const auto c = noa::clamp_cast<Vec<float, 2>>(a);
        const auto b = noa::safe_cast<Vec<float, 2>>(a);
        const auto e = a.template as<float>();
        const auto f = a.template as_clamp<float>();
        const auto g = a.template as_safe<float>();
        REQUIRE((noa::all(a_ == b) &&
                 noa::all(a_ == c) &&
                 noa::all(a_ == d) &&
                 noa::all(a_ == e) &&
                 noa::all(a_ == f) &&
                 noa::all(a_ == g)));
    }

    SECTION("Arithmetics and comparisons") {
        auto randomizer = test::Randomizer<TestType>(1, 20);
        auto v0 = Vec2Test(randomizer.get(), randomizer.get());
        auto v1 = Vec2Test(randomizer.get(), randomizer.get());

        REQUIRE(noa::all(v0 + 1 == Vec2Test(v0[0] + 1, v0[1] + 1)));
        REQUIRE(noa::all(v0 - 2 == Vec2Test(v0[0] - 2, v0[1] - 2)));
        REQUIRE(noa::all(v0 * 3 == Vec2Test(v0[0] * 3, v0[1] * 3)));
        REQUIRE(noa::all(v0 / 2 == Vec2Test(v0[0] / 2, v0[1] / 2)));
        REQUIRE(noa::all(1 + v0  == Vec2Test(1 + v0[0], 1 + v0[1])));
        REQUIRE(noa::all(2 - v0  == Vec2Test(2 - v0[0], 2 - v0[1])));
        REQUIRE(noa::all(3 * v0  == Vec2Test(3 * v0[0], 3 * v0[1])));
        REQUIRE(noa::all(2 / v0  == Vec2Test(2 / v0[0], 2 / v0[1])));
        REQUIRE(noa::all(v0 + v1 == Vec2Test(v0[0] + v1[0], v0[1] + v1[1])));
        REQUIRE(noa::all(v0 - v1 == Vec2Test(v0[0] - v1[0], v0[1] - v1[1])));
        REQUIRE(noa::all(v0 * v1 == Vec2Test(v0[0] * v1[0], v0[1] * v1[1])));
        REQUIRE(noa::all(v0 / v1 == Vec2Test(v0[0] / v1[0], v0[1] / v1[1])));

        auto v2 = v0;
        v0 += 1; REQUIRE(noa::all(v0 == Vec2Test(v2[0] + 1, v2[1] + 1))); v2 = v0;
        v0 -= 2; REQUIRE(noa::all(v0 == Vec2Test(v2[0] - 2, v2[1] - 2))); v2 = v0;
        v0 *= 3; REQUIRE(noa::all(v0 == Vec2Test(v2[0] * 3, v2[1] * 3))); v2 = v0;
        v0 /= 2; REQUIRE(noa::all(v0 == Vec2Test(v2[0] / 2, v2[1] / 2))); v2 = v0;
        v0 += v1; REQUIRE(noa::all(v0 == Vec2Test(v2[0] + v1[0], v2[1] + v1[1]))); v2 = v0;
        v0 -= v1; REQUIRE(noa::all(v0 == Vec2Test(v2[0] - v1[0], v2[1] - v1[1]))); v2 = v0;
        v0 *= v1; REQUIRE(noa::all(v0 == Vec2Test(v2[0] * v1[0], v2[1] * v1[1]))); v2 = v0;
        v0 /= v1; REQUIRE(noa::all(v0 == Vec2Test(v2[0] / v1[0], v2[1] / v1[1])));

        v0 = {4, 10};
        REQUIRE(noa::all(Vec2<bool>(0, 1) == (v0 > 5)));
        REQUIRE(noa::all(Vec2<bool>(1, 1) == (v0 < 11)));
        REQUIRE(noa::all(Vec2<bool>(0, 1) == (v0 >= 7)));
        REQUIRE(noa::all(Vec2<bool>(1, 1) == (v0 <= 10)));
        REQUIRE(noa::any(v0 == 4));
        REQUIRE_FALSE(all(v0 == 4));

        REQUIRE(noa::all((5 < v0) == Vec2<bool>(0, 1)));
        REQUIRE(noa::all((11 > v0) == Vec2<bool>(1, 1)));
        REQUIRE(noa::all((7 <= v0) == Vec2<bool>(0, 1)));
        REQUIRE(noa::all((9 >= v0) == Vec2<bool>(1, 0)));
        REQUIRE(noa::any(4 == v0));
        REQUIRE_FALSE(noa::all(4 == v0));

        v0 = {2, 4};
        REQUIRE(noa::all((v0 > Vec2Test{1, 2}) == Vec2<bool>(1, 1)));
        REQUIRE(noa::all((v0 < Vec2Test{4, 5}) == Vec2<bool>(1, 1)));
        REQUIRE(noa::all((v0 >= Vec2Test{2, 4}) == Vec2<bool>(1, 1)));
        REQUIRE(noa::all((v0 <= Vec2Test{1, 4}) == Vec2<bool>(0, 1)));
        REQUIRE(noa::all((v0 != Vec2Test{4, 4}) == Vec2<bool>(1, 0)));

        // Min & Max
        REQUIRE(noa::all(math::min(Vec2Test{3, 4}, Vec2Test{5, 2}) == Vec2Test{3, 2}));
        REQUIRE(noa::all(math::max(Vec2Test{3, 4}, Vec2Test{5, 2}) == Vec2Test{5, 4}));
        REQUIRE(noa::all(math::min(Vec2Test{3, 6}, TestType{5}) == Vec2Test{3, 5}));
        REQUIRE(noa::all(math::max(Vec2Test{9, 0}, TestType{2}) == Vec2Test{9, 2}));

        if constexpr (std::is_floating_point_v<TestType>) {
            v0 = 2;
            REQUIRE(all(v0 == TestType{2}));
            v0 += static_cast<TestType>(1.34);
            REQUIRE(all(math::are_almost_equal(v0, static_cast<TestType>(3.34))));
            v0 -= static_cast<TestType>(23.134);
            REQUIRE(all(math::are_almost_equal(v0, static_cast<TestType>(-19.794))));
            v0 *= static_cast<TestType>(-2.45);
            REQUIRE(all(math::are_almost_equal(v0, static_cast<TestType>(48.4953))));
            v0 /= static_cast<TestType>(567.234);
            REQUIRE(all(math::are_almost_equal(v0, static_cast<TestType>(0.085494))));

            v0 = static_cast<TestType>(3.30);
            auto tmp = v0 + static_cast<TestType>(3.234534);
            REQUIRE(all(math::are_almost_equal(tmp, static_cast<TestType>(6.534534))));
            tmp = v0 - static_cast<TestType>(-234.2);
            REQUIRE(all(math::are_almost_equal(tmp, static_cast<TestType>(237.5))));
            tmp = v0 * static_cast<TestType>(3);
            REQUIRE(all(math::are_almost_equal(tmp, static_cast<TestType>(9.90))));
            tmp = v0 / static_cast<TestType>(0.001);
            REQUIRE(all(math::are_almost_equal(tmp, static_cast<TestType>(3299.999f), static_cast<TestType>(1e-3))));

            v0 = {0, 2};
            v0 += {35, 20};
            REQUIRE(all(math::are_almost_equal(v0, Vec2Test(35, 22))));
            v0 -= Vec2Test(-0.12, 23.2123);
            REQUIRE(all(math::are_almost_equal(v0, Vec2Test(35.12, -1.2123))));
            v0 *= Vec2Test(0, 10);
            REQUIRE(all(math::are_almost_equal(v0, Vec2Test(0, -12.123), static_cast<TestType>(1e-5))));
            v0 /= Vec2Test(2, 9);
            REQUIRE(all(math::are_almost_equal(v0, Vec2Test(0, -1.347))));

            v0[0] = 20;
            v0[1] = 50;
            tmp = v0 + Vec2Test(10, 12);
            REQUIRE(all(math::are_almost_equal(tmp, Vec2Test(30, 62))));
            tmp = v0 - Vec2Test(10.32, -112.001);
            REQUIRE(all(math::are_almost_equal(tmp, Vec2Test(9.68, 162.001))));
            tmp = v0 * Vec2Test(2.5, 3.234);
            REQUIRE(all(math::are_almost_equal(tmp, Vec2Test(50, 161.7))));
            tmp = v0 / Vec2Test(10, -12);
            REQUIRE(all(math::are_almost_equal(tmp, Vec2Test(2, -4.166667))));
        }
    }

    SECTION("Maths") {
        const Vec2Test a{123, 43};
        REQUIRE(math::sum(a) == 166);
        REQUIRE(math::product(a) == 5289);

        if constexpr (std::is_floating_point_v<TestType>) {
            const auto x = static_cast<TestType>(5.2);
            const auto y = static_cast<TestType>(12.3);
            auto b = Vec2Test(x, y);
            REQUIRE(all(math::are_almost_equal(math::cos(b), Vec2Test(math::cos(x), math::cos(y)))));
            REQUIRE(all(math::are_almost_equal(math::sin(b), Vec2Test(math::sin(x), math::sin(y)))));
            REQUIRE(all(math::are_almost_equal(math::tan(b), Vec2Test(math::tan(x), math::tan(y)))));
            REQUIRE(all(math::are_almost_equal(math::log(b), Vec2Test(math::log(x), math::log(y)))));
            REQUIRE(all(math::are_almost_equal(math::exp(b), Vec2Test(math::exp(x), math::exp(y)))));
            REQUIRE(all(math::are_almost_equal(math::round(b), Vec2Test(math::round(x), math::round(y)))));
            REQUIRE(all(math::are_almost_equal(math::ceil(b), Vec2Test(math::ceil(x), math::ceil(y)))));
            REQUIRE(all(math::are_almost_equal(math::abs(b), Vec2Test(math::abs(x), math::abs(y)))));
            REQUIRE(all(math::are_almost_equal(math::sqrt(b), Vec2Test(math::sqrt(x), math::sqrt(y)))));

            b = Vec2Test(23.23, -12.252);
            auto dot = static_cast<double>(b[0] * b[0] + b[1] * b[1]);
            REQUIRE_THAT(math::dot(b, b), Catch::WithinAbs(dot, 1e-6));
            REQUIRE_THAT(math::norm(b), Catch::WithinAbs(std::sqrt(dot), 1e-6));
            const auto b_normalized = math::normalize(b);
            REQUIRE_THAT(math::norm(b_normalized), Catch::WithinAbs(1, 1e-6));
            REQUIRE_THAT(math::dot(b, Vec2Test(-12.23, -21.23)), Catch::WithinAbs(-23.992940, 1e-4));
        }
    }

    SECTION("Other") {
        Vec2Test b{123, 43};
        REQUIRE(noa::all(noa::sort(b) == b.flip()));
        REQUIRE(noa::all(b.circular_shift(1) == b.flip()));
        REQUIRE(noa::all(b.circular_shift(-1) == b.flip()));
        REQUIRE(noa::all(b.circular_shift(2) == b));
        REQUIRE(b.pop_front()[0] == b[1]);
        REQUIRE(b.pop_back()[0] == b[0]);

        REQUIRE(noa::all(b.push_front(1) == Vec3<TestType>{1, 123, 43}));
        REQUIRE(noa::all(b.push_front(Vec2Test{1, 2}) == Vec4<TestType>(1, 2, 123, 43)));
        REQUIRE(noa::all(b.push_back(Vec2Test{1, 2}) == Vec4<TestType>(123, 43, 1, 2)));

        REQUIRE(noa::all(b.reorder({1, 0}) == b.flip()));
        REQUIRE(b.filter(0)[0] == b[0]);
        REQUIRE(b.filter(1)[0] == b[1]);

        const std::array<TestType, 2> b_array = {123, 43};
        REQUIRE(noa::string::format("{}", b) == noa::string::format("{}", b_array));
    }
}

TEMPLATE_TEST_CASE("core::Vec3<T>", "[noa][common][types]", i32, i64, u32, u64, f32, f64) {
    using Vec3Test = Vec3<TestType>;

    SECTION("Initialization and assignment") {
        Vec3Test a;
        REQUIRE(noa::all(a == Vec3Test{0, 0, 0}));
        REQUIRE(noa::all(a == Vec3Test{}));
        REQUIRE(noa::all(Vec3Test(1) == Vec3Test{1, 1, 1}));
        const Vec3Test d(1, 2, 3);
        const Vec3Test e = {1, 2, 3};
        const Vec3Test f{1, 2, 3};
        REQUIRE((noa::all(d == e) && noa::all(d == f)));

        a = Vec3Test{1};
        a = d;
        a = {1};
        REQUIRE(noa::all(a == Vec3Test{1, 1, 1}));
        a = {1, 2, 3};
        REQUIRE(noa::all(a == Vec3Test{1, 2, 3}));

        a[0] = 23;
        a[1] = 52;
        a[2] = 12;
        REQUIRE((a[0] == 23 && a[1] == 52 && a[2] == 12));

        const auto [a0, a1, a2] = a;
        REQUIRE((a0 == 23 && a1 == 52 && a2 == 12));

        Vec3Test c{3, 4, 5};
        REQUIRE(noa::all(Vec3Test(c.data()) == c));
    }

    SECTION("Cast") {
        const Vec3Test a{1};
        const Vec<float, 3> a_{1};
        const auto d = static_cast<Vec<float, 3>>(a);
        const auto c = noa::clamp_cast<Vec<float, 3>>(a);
        const auto b = noa::safe_cast<Vec<float, 3>>(a);
        const auto e = a.template as<float>();
        const auto f = a.template as_clamp<float>();
        const auto g = a.template as_safe<float>();
        REQUIRE((noa::all(a_ == b) &&
                 noa::all(a_ == c) &&
                 noa::all(a_ == d) &&
                 noa::all(a_ == e) &&
                 noa::all(a_ == f) &&
                 noa::all(a_ == g)));
    }

    SECTION("Arithmetics and comparisons") {
        auto randomizer = test::Randomizer<TestType>(1, 30);
        auto v0 = Vec3Test(randomizer.get(), randomizer.get(), randomizer.get());
        auto v1 = Vec3Test(randomizer.get(), randomizer.get(), randomizer.get());

        REQUIRE(noa::all(v0 + 1 == Vec3Test(v0[0] + 1, v0[1] + 1, v0[2] + 1)));
        REQUIRE(noa::all(v0 - 2 == Vec3Test(v0[0] - 2, v0[1] - 2, v0[2] - 2)));
        REQUIRE(noa::all(v0 * 3 == Vec3Test(v0[0] * 3, v0[1] * 3, v0[2] * 3)));
        REQUIRE(noa::all(v0 / 2 == Vec3Test(v0[0] / 2, v0[1] / 2, v0[2] / 2)));
        REQUIRE(noa::all(1 + v0  == Vec3Test(1 + v0[0], 1 + v0[1], 1 + v0[2])));
        REQUIRE(noa::all(2 - v0  == Vec3Test(2 - v0[0], 2 - v0[1], 2 - v0[2])));
        REQUIRE(noa::all(3 * v0  == Vec3Test(3 * v0[0], 3 * v0[1], 3 * v0[2])));
        REQUIRE(noa::all(2 / v0  == Vec3Test(2 / v0[0], 2 / v0[1], 2 / v0[2])));
        REQUIRE(noa::all(v0 + v1 == Vec3Test(v0[0] + v1[0], v0[1] + v1[1], v0[2] + v1[2])));
        REQUIRE(noa::all(v0 - v1 == Vec3Test(v0[0] - v1[0], v0[1] - v1[1], v0[2] - v1[2])));
        REQUIRE(noa::all(v0 * v1 == Vec3Test(v0[0] * v1[0], v0[1] * v1[1], v0[2] * v1[2])));
        REQUIRE(noa::all(v0 / v1 == Vec3Test(v0[0] / v1[0], v0[1] / v1[1], v0[2] / v1[2])));

        auto v2 = v0;
        v0 += 1; REQUIRE(noa::all(v0 == Vec3Test(v2[0] + 1, v2[1] + 1, v2[2] + 1))); v2 = v0;
        v0 -= 2; REQUIRE(noa::all(v0 == Vec3Test(v2[0] - 2, v2[1] - 2, v2[2] - 2))); v2 = v0;
        v0 *= 3; REQUIRE(noa::all(v0 == Vec3Test(v2[0] * 3, v2[1] * 3, v2[2] * 3))); v2 = v0;
        v0 /= 2; REQUIRE(noa::all(v0 == Vec3Test(v2[0] / 2, v2[1] / 2, v2[2] / 2))); v2 = v0;
        v0 += v1; REQUIRE(noa::all(v0 == Vec3Test(v2[0] + v1[0], v2[1] + v1[1], v2[2] + v1[2]))); v2 = v0;
        v0 -= v1; REQUIRE(noa::all(v0 == Vec3Test(v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]))); v2 = v0;
        v0 *= v1; REQUIRE(noa::all(v0 == Vec3Test(v2[0] * v1[0], v2[1] * v1[1], v2[2] * v1[2]))); v2 = v0;
        v0 /= v1; REQUIRE(noa::all(v0 == Vec3Test(v2[0] / v1[0], v2[1] / v1[1], v2[2] / v1[2])));

        v0 = {4, 10, 7};
        REQUIRE(all(Vec3<bool>(0, 1, 0) == (v0 > 9)));
        REQUIRE(all(Vec3<bool>(1, 1, 1) == (v0 < 11)));
        REQUIRE(all(Vec3<bool>(0, 1, 1) == (v0 >= 7)));
        REQUIRE(all(Vec3<bool>(1, 0, 1) == (v0 <= 9)));
        REQUIRE(any(v0 == 4));
        REQUIRE_FALSE(all(v0 == 4));

        REQUIRE(all(Vec3<bool>(1, 0, 1) == (9 > v0)));
        REQUIRE(all(Vec3<bool>(1, 1, 1) == (11 > v0)));
        REQUIRE(all(Vec3<bool>(1, 0, 1) == (7 >= v0)));
        REQUIRE(all(Vec3<bool>(0, 1, 0) == (9 <= v0)));
        REQUIRE(any(4 == v0));
        REQUIRE_FALSE(all(4 == v0));

        v0 = {2, 4, 4};
        REQUIRE(all(Vec3<bool>(1, 1, 0) == (v0 > Vec3Test{1, 2, 4})));
        REQUIRE(all(Vec3<bool>(0, 1, 1) == (v0 < Vec3Test{2, 5, 6})));
        REQUIRE(all(Vec3<bool>(1, 1, 1) == (v0 >= Vec3Test{2, 4, 3})));
        REQUIRE(all(Vec3<bool>(0, 1, 1) == (v0 <= Vec3Test{1, 4, 6})));
        REQUIRE(all(Vec3<bool>(1, 0, 0) == (v0 != Vec3Test{4, 4, 4})));
        REQUIRE(all(Vec3<bool>(0, 0, 1) == (v0 == Vec3Test{4, 2, 4})));

        // Min & Max
        REQUIRE(all(math::min(Vec3Test{3, 4, 8}, Vec3Test{5, 2, 10}) == Vec3Test{3, 2, 8}));
        REQUIRE(all(math::max(Vec3Test{3, 4, 1000}, Vec3Test{5, 2, 30}) == Vec3Test{5, 4, 1000}));
        REQUIRE(all(math::min(Vec3Test{3, 6, 4}, TestType{5}) == Vec3Test{3, 5, 4}));
        REQUIRE(all(math::max(Vec3Test{9, 0, 1}, TestType{2}) == Vec3Test{9, 2, 2}));

        if constexpr (std::is_floating_point_v<TestType>) {
            v0 = 2;
            REQUIRE(all(v0 == TestType{2}));
            v0 += static_cast<TestType>(1.34);
            REQUIRE(all(math::are_almost_equal(v0, static_cast<TestType>(3.34))));
            v0 -= static_cast<TestType>(23.134);
            REQUIRE(all(math::are_almost_equal(v0, static_cast<TestType>(-19.794))));
            v0 *= static_cast<TestType>(-2.45);
            REQUIRE(all(math::are_almost_equal(v0, static_cast<TestType>(48.4953))));
            v0 /= static_cast<TestType>(567.234);
            REQUIRE(all(math::are_almost_equal(v0, static_cast<TestType>(0.085494))));

            v0 = static_cast<TestType>(3.30);
            auto tmp = v0 + static_cast<TestType>(3.234534);
            REQUIRE(all(math::are_almost_equal(tmp, static_cast<TestType>(6.534534))));
            tmp = v0 - static_cast<TestType>(-234.2);
            REQUIRE(all(math::are_almost_equal(tmp, static_cast<TestType>(237.5))));
            tmp = v0 * static_cast<TestType>(3);
            REQUIRE(all(math::are_almost_equal(tmp, static_cast<TestType>(9.90))));
            tmp = v0 / static_cast<TestType>(0.001);
            REQUIRE(all(math::are_almost_equal(tmp, static_cast<TestType>(3299.999f), static_cast<TestType>(1e-3))));

            v0 = Vec3Test{0, 2, 123};
            v0 += Vec3Test(35, 20, -12);
            REQUIRE(all(math::are_almost_equal(v0, Vec3Test(35, 22, 111))));
            v0 -= Vec3Test(-0.12, 23.2123, 0.23);
            REQUIRE(all(math::are_almost_equal(v0, Vec3Test(35.12, -1.2123, 110.77))));
            v0 *= Vec3Test(0, 10, -3.2);
            REQUIRE(all(math::are_almost_equal(v0, Vec3Test(0, -12.123, -354.464), static_cast<TestType>(1e-5))));
            v0 /= Vec3Test(2, 9, 2);
            REQUIRE(all(math::are_almost_equal(v0, Vec3Test(0, -1.347, -177.232))));

            v0[0] = 20;
            v0[1] = 50;
            v0[2] = 33;
            tmp = v0 + Vec3Test(10, 12, -1232);
            REQUIRE(all(math::are_almost_equal(tmp, Vec3Test(30, 62, -1199))));
            tmp = v0 - Vec3Test(10.32, -112.001, 0.5541);
            REQUIRE(all(math::are_almost_equal(tmp, Vec3Test(9.68, 162.001, 32.4459))));
            tmp = v0 * Vec3Test(2.5, 3.234, 58.12);
            REQUIRE(all(math::are_almost_equal(tmp, Vec3Test(50, 161.7, 1917.959999))));
            tmp = v0 / Vec3Test(10, -12, -2.3);
            REQUIRE(all(math::are_almost_equal(tmp, Vec3Test(2, -4.166667, -14.3478261))));
        }

        SECTION("Maths") {
            const Vec3Test a{123, 43, 11};
            REQUIRE(math::sum(a) == 177);
            REQUIRE(math::product(a) == 58179);

            if constexpr (std::is_floating_point_v<TestType>) {
                const auto x = static_cast<TestType>(1.42);
                const auto y = static_cast<TestType>(3.14);
                const auto z = static_cast<TestType>(1.34);
                auto b = Vec3Test(x, y, z);
                REQUIRE(all(math::are_almost_equal(math::cos(b), Vec3Test(math::cos(x), math::cos(y), math::cos(z)))));
                REQUIRE(all(math::are_almost_equal(math::sin(b), Vec3Test(math::sin(x), math::sin(y), math::sin(z)))));
                REQUIRE(all(math::are_almost_equal(math::tan(b), Vec3Test(math::tan(x), math::tan(y), math::tan(z)))));
                REQUIRE(all(math::are_almost_equal(math::sinh(b), Vec3Test(math::sinh(x), math::sinh(y), math::sinh(z)))));
                REQUIRE(all(math::are_almost_equal(math::log(b), Vec3Test(math::log(x), math::log(y), math::log(z)))));
                REQUIRE(all(math::are_almost_equal(math::exp(b), Vec3Test(math::exp(x), math::exp(y), math::exp(z)))));
                REQUIRE(all(math::are_almost_equal(math::round(b), Vec3Test(math::round(x), math::round(y), math::round(z)))));
                REQUIRE(all(math::are_almost_equal(math::ceil(b), Vec3Test(math::ceil(x), math::ceil(y), math::ceil(z)))));
                REQUIRE(all(math::are_almost_equal(math::abs(b), Vec3Test(math::abs(x), math::abs(y), math::abs(z)))));
                REQUIRE(all(math::are_almost_equal(math::sqrt(b), Vec3Test(math::sqrt(x), math::sqrt(y), math::sqrt(z)))));

                b = Vec3Test(23.23, -12.252, 1.246);
                auto dot = static_cast<double>(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
                REQUIRE_THAT(math::dot(b, b), Catch::WithinAbs(dot, 1e-6));
                REQUIRE_THAT(math::norm(b), Catch::WithinAbs(std::sqrt(dot), 1e-6));
                const auto b_normalized = math::normalize(b);
                REQUIRE_THAT(math::norm(b_normalized), Catch::WithinAbs(1, 1e-6));
            }
        }

        SECTION("Other") {
            Vec3Test b{123, 43, 87};
            REQUIRE(noa::all(noa::sort(b) == Vec3Test{43, 87, 123}));
            REQUIRE(noa::all(b.circular_shift(1) == Vec3Test{87, 123, 43}));
            REQUIRE(noa::all(b.circular_shift(-1) == Vec3Test{43, 87, 123}));
            REQUIRE(noa::all(b.circular_shift(2) == Vec3Test{43, 87, 123}));
            REQUIRE(noa::all(b.circular_shift(3) == Vec3Test{123, 43, 87}));
            REQUIRE(b.pop_front()[0] == b[1]);
            REQUIRE(b.pop_back()[0] == b[0]);
            REQUIRE(b.template pop_front<2>()[0] == b[2]);

            REQUIRE(noa::all(b.push_front(1) == Vec4<TestType>{1, 123, 43, 87}));
            REQUIRE(noa::all(b.push_front(Vec2<TestType>{1, 2}) == Vec<TestType, 5>(1, 2, 123, 43, 87)));
            REQUIRE(noa::all(b.push_back(Vec2<TestType>{1, 2}) == Vec<TestType, 5>(123, 43, 87, 1, 2)));

            REQUIRE(noa::all(b.reorder({1, 2, 0}) == Vec3Test{43, 87, 123}));
            REQUIRE(b.filter(0)[0] == b[0]);
            REQUIRE(b.filter(1)[0] == b[1]);
            REQUIRE(b.filter(2)[0] == b[2]);
            REQUIRE(noa::all(b.filter(0, 2) == Vec2<TestType>{123, 87}));

            const std::array<TestType, 3> b_array = {123, 43, 87};
            REQUIRE(noa::string::format("{}", b) == noa::string::format("{}", b_array));
        }
    }
}

TEMPLATE_TEST_CASE("core::Vec1<T>", "[noa][common][types]", i32, i64, u32, u64, f32, f64) {
    using Vec1Test = Vec1<TestType>;

    SECTION("Initialization and assignment") {
        Vec1Test a;
        REQUIRE(noa::all(a == Vec1Test{0}));
        const Vec1Test d(3);
        const Vec1Test e{3};
        REQUIRE((noa::all(d == e)));

        a = Vec1Test{1};
        a = d;
        a = {1};
        REQUIRE(noa::all(a == Vec1Test{1}));
        a = 3;
        REQUIRE(noa::all(a == Vec1Test{3}));

        a[0] = 23;
        REQUIRE(a[0] == 23);

        const auto [a0] = a;
        REQUIRE(a0 == 23);

        Vec1Test c{5};
        REQUIRE(noa::all(Vec1Test(c.data()) == c));
    }

    SECTION("Cast") {
        const Vec1Test a{1};
        const Vec<float, 1> a_{1};
        const auto d = static_cast<Vec<float, 1>>(a);
        const auto c = noa::clamp_cast<Vec<float, 1>>(a);
        const auto b = noa::safe_cast<Vec<float, 1>>(a);
        const auto e = a.template as<float>();
        const auto f = a.template as_clamp<float>();
        const auto g = a.template as_safe<float>();
        REQUIRE((noa::all(a_ == b) &&
                 noa::all(a_ == c) &&
                 noa::all(a_ == d) &&
                 noa::all(a_ == e) &&
                 noa::all(a_ == f) &&
                 noa::all(a_ == g)));
    }

    SECTION("Arithmetics and comparisons") {
        auto randomizer = test::Randomizer<TestType>(1, 10);
        auto v0 = Vec1Test(randomizer.get());
        auto v1 = Vec1Test(randomizer.get());

        REQUIRE(noa::all(v0 + 1 == Vec1Test(v0[0] + 1)));
        REQUIRE(noa::all(v0 - 2 == Vec1Test(v0[0] - 2)));
        REQUIRE(noa::all(v0 * 3 == Vec1Test(v0[0] * 3)));
        REQUIRE(noa::all(v0 / 2 == Vec1Test(v0[0] / 2)));
        REQUIRE(noa::all(1 + v0  == Vec1Test(1 + v0[0])));
        REQUIRE(noa::all(2 - v0  == Vec1Test(2 - v0[0])));
        REQUIRE(noa::all(3 * v0  == Vec1Test(3 * v0[0])));
        REQUIRE(noa::all(2 / v0  == Vec1Test(2 / v0[0])));
        REQUIRE(noa::all(v0 + v1 == Vec1Test(v0[0] + v1[0])));
        REQUIRE(noa::all(v0 - v1 == Vec1Test(v0[0] - v1[0])));
        REQUIRE(noa::all(v0 * v1 == Vec1Test(v0[0] * v1[0])));
        REQUIRE(noa::all(v0 / v1 == Vec1Test(v0[0] / v1[0])));

        auto v2 = v0;
        v0 += 1; REQUIRE(noa::all(v0 == Vec1Test(v2[0] + 1))); v2 = v0;
        v0 -= 2; REQUIRE(noa::all(v0 == Vec1Test(v2[0] - 2))); v2 = v0;
        v0 *= 3; REQUIRE(noa::all(v0 == Vec1Test(v2[0] * 3))); v2 = v0;
        v0 /= 2; REQUIRE(noa::all(v0 == Vec1Test(v2[0] / 2))); v2 = v0;
        v0 += v1; REQUIRE(noa::all(v0 == Vec1Test(v2[0] + v1[0]))); v2 = v0;
        v0 -= v1; REQUIRE(noa::all(v0 == Vec1Test(v2[0] - v1[0]))); v2 = v0;
        v0 *= v1; REQUIRE(noa::all(v0 == Vec1Test(v2[0] * v1[0]))); v2 = v0;
        v0 /= v1; REQUIRE(noa::all(v0 == Vec1Test(v2[0] / v1[0])));

        v0 = 4;
        REQUIRE(noa::all(Vec1<bool>(0) == (v0 > 5)));
        REQUIRE(noa::all(Vec1<bool>(1) == (v0 < 11)));
        REQUIRE(noa::all(Vec1<bool>(0) == (v0 >= 7)));
        REQUIRE(noa::all(Vec1<bool>(1) == (v0 <= 10)));
        REQUIRE(noa::any(v0 == 4));
        REQUIRE(noa::all(v0 == 4));

        REQUIRE(noa::all((5 < v0) == Vec1<bool>(0)));
        REQUIRE(noa::all((11 > v0) == Vec1<bool>(1)));
        REQUIRE(noa::all((7 <= v0) == Vec1<bool>(0)));
        REQUIRE(noa::all((9 >= v0) == Vec1<bool>(1)));
        REQUIRE(noa::any(4 == v0));
        REQUIRE(noa::all(4 == v0));

        v0 = 2;
        REQUIRE(noa::all((v0 > Vec1Test{1}) == Vec1<bool>(1)));
        REQUIRE(noa::all((v0 < Vec1Test{4}) == Vec1<bool>(1)));
        REQUIRE(noa::all((v0 >= Vec1Test{2}) == Vec1<bool>(1)));
        REQUIRE(noa::all((v0 <= Vec1Test{1}) == Vec1<bool>(0)));
        REQUIRE(noa::all((v0 != Vec1Test{4}) == Vec1<bool>(1)));

        // Min & Max
        REQUIRE(noa::all(math::min(Vec1Test{3}, Vec1Test{5}) == Vec1Test{3}));
        REQUIRE(noa::all(math::max(Vec1Test{3}, Vec1Test{5}) == Vec1Test{5}));

        if constexpr (std::is_floating_point_v<TestType>) {
            v0 = 2;
            REQUIRE(all(v0 == TestType{2}));
            v0 += static_cast<TestType>(1.34);
            REQUIRE(all(math::are_almost_equal(v0, static_cast<TestType>(3.34))));
            v0 -= static_cast<TestType>(23.134);
            REQUIRE(all(math::are_almost_equal(v0, static_cast<TestType>(-19.794))));
            v0 *= static_cast<TestType>(-2.45);
            REQUIRE(all(math::are_almost_equal(v0, static_cast<TestType>(48.4953))));
            v0 /= static_cast<TestType>(567.234);
            REQUIRE(all(math::are_almost_equal(v0, static_cast<TestType>(0.085494))));

            v0 = static_cast<TestType>(3.30);
            auto tmp = v0 + static_cast<TestType>(3.234534);
            REQUIRE(all(math::are_almost_equal(tmp, static_cast<TestType>(6.534534))));
            tmp = v0 - static_cast<TestType>(-234.2);
            REQUIRE(all(math::are_almost_equal(tmp, static_cast<TestType>(237.5))));
            tmp = v0 * static_cast<TestType>(3);
            REQUIRE(all(math::are_almost_equal(tmp, static_cast<TestType>(9.90))));
            tmp = v0 / static_cast<TestType>(0.001);
            REQUIRE(all(math::are_almost_equal(tmp, static_cast<TestType>(3299.999f), static_cast<TestType>(1e-3))));

            v0 = {0};
            v0 += {35};
            REQUIRE(all(math::are_almost_equal(v0, Vec1Test(35))));
            v0 -= Vec1Test(-0.12);
            REQUIRE(all(math::are_almost_equal(v0, Vec1Test(35.12))));
            v0 *= Vec1Test(2);
            REQUIRE(all(math::are_almost_equal(v0, Vec1Test(70.24), static_cast<TestType>(1e-5))));
            v0 /= Vec1Test(9);
            REQUIRE(all(math::are_almost_equal(v0, Vec1Test(7.804444444))));

            v0[0] = 20;
            tmp = v0 + Vec1Test(10);
            REQUIRE(all(math::are_almost_equal(tmp, Vec1Test(30))));
            tmp = v0 - Vec1Test(10.32);
            REQUIRE(all(math::are_almost_equal(tmp, Vec1Test(9.68))));
            tmp = v0 * Vec1Test(2.5);
            REQUIRE(all(math::are_almost_equal(tmp, Vec1Test(50))));
            tmp = v0 / Vec1Test(10);
            REQUIRE(all(math::are_almost_equal(tmp, Vec1Test(2))));
        }
    }

    SECTION("Maths") {
        const Vec1Test a{123};
        REQUIRE(math::sum(a) == 123);
        REQUIRE(math::product(a) == 123);

        if constexpr (std::is_floating_point_v<TestType>) {
            const auto x = static_cast<TestType>(2.35);
            auto b = Vec1Test(x);
            REQUIRE(all(math::are_almost_equal(math::cos(b), Vec1Test(math::cos(x)))));
            REQUIRE(all(math::are_almost_equal(math::sin(b), Vec1Test(math::sin(x)))));
            REQUIRE(all(math::are_almost_equal(math::tan(b), Vec1Test(math::tan(x)))));
            REQUIRE(all(math::are_almost_equal(math::sinh(b), Vec1Test(math::sinh(x)))));
            REQUIRE(all(math::are_almost_equal(math::log(b), Vec1Test(math::log(x)))));
            REQUIRE(all(math::are_almost_equal(math::exp(b), Vec1Test(math::exp(x)))));
            REQUIRE(all(math::are_almost_equal(math::round(b), Vec1Test(math::round(x)))));
            REQUIRE(all(math::are_almost_equal(math::ceil(b), Vec1Test(math::ceil(x)))));
            REQUIRE(all(math::are_almost_equal(math::abs(b), Vec1Test(math::abs(x)))));
            REQUIRE(all(math::are_almost_equal(math::sqrt(b), Vec1Test(math::sqrt(x)))));

            b = Vec1Test(-12.252);
            auto dot = static_cast<double>(b[0] * b[0]);
            REQUIRE_THAT(math::dot(b, b), Catch::WithinAbs(dot, 1e-6));
            REQUIRE_THAT(math::norm(b), Catch::WithinAbs(std::sqrt(dot), 1e-6));
            const auto b_normalized = math::normalize(b);
            REQUIRE_THAT(math::norm(b_normalized), Catch::WithinAbs(1, 1e-6));
        }
    }

    SECTION("Other") {
        Vec1Test b{123};
        REQUIRE(noa::all(noa::sort(b) == b.flip()));
        REQUIRE(noa::all(b.circular_shift(1) == b.flip()));
        REQUIRE(noa::all(b.circular_shift(-1) == b.flip()));
        REQUIRE(noa::all(b.circular_shift(2) == b));
        // REQUIRE(b.pop_front()[0] == b[1]); // error
        // REQUIRE(b.pop_back()[0] == b[0]); // error

        REQUIRE(noa::all(b.push_front(1) == Vec2<TestType>{1, 123}));
        REQUIRE(noa::all(b.push_front(Vec3<TestType>{1, 2, 3}) == Vec4<TestType>(1, 2, 3, 123)));
        REQUIRE(noa::all(b.push_back(Vec2<TestType>{1, 2}) == Vec3<TestType>(123, 1, 2)));

        REQUIRE(noa::all(b.reorder(Vec1<int>{0}) == b.flip()));
        REQUIRE(b.filter(0)[0] == b[0]);

        const std::array<TestType, 1> b_array = {123};
        REQUIRE(noa::string::format("{}", b) == noa::string::format("{}", b_array));
        REQUIRE(noa::string::format("{::.2f}", Vec<c32, 2>{}) == "[(0.00,0.00), (0.00,0.00)]");
    }
}
