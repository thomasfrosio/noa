#include <noa/core/types/Vec.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;

static_assert(std::is_aggregate_v<Vec<f64, 4>>);
static_assert(alignof(Vec<f64, 4>) == 16);
static_assert(alignof(Vec<f32, 2>) == 8);
static_assert(alignof(Vec<f32, 6>) == 4);
static_assert(alignof(Vec<f64, 4, 32>) == 32);

TEMPLATE_TEST_CASE("core::Vec", "", i32, i64, u32, u64, f32, f64) {
    // Vec is an essential part of the library. It is similar to std::array, in that it is an aggregate storing
    // an array of a given type and static size. However, it only supports numeric types or a Vec, i.e. Vec<f32,2>,
    // Vec<Vec<f32,2>,2> are both allowed.
    static_assert(std::is_aggregate_v<Vec<Vec<Vec<f32, 1>, 2>, 3>>);

    // By default, Vec tries to overalign the underlying array for better efficiency, e.g. Vec<f32,2> as an alignment
    // of 8 bytes in supported platforms. This is handled by the third template parameter, which defaults to 0, meaning
    // to let Vec decide of the best alignment (which is at least alignof(T)).
    // However, we can set the alignment ourselves, which can be especially useful on the GPU.
    static_assert(alignof(Vec<f32, 3>) == alignof(f32));
    static_assert(alignof(Vec<f32, 3, 16>) == 16); // the alignment requires structure padding the Vec

    // Empty Vecs are allowed and used throughout the library.
    // Note that thanks to [[no_unique_address]], storing an empty Vec doesn't take space.
    static_assert(sizeof(Vec<f64, 0>) == 1 and alignof(Vec<f32, 0>) == 1);

    using vec2_t = Vec<TestType, 2>;
    using vec3_t = Vec<TestType, 3>;
    using vec4_t = Vec<TestType, 4>;
    using vec5_t = Vec<TestType, 5>;
    using vec6_t = Vec<TestType, 6>;

    SECTION("Initialization and assignment") {
        {
            // Vec is an aggregate, but has a bunch of useful factory functions.
            vec4_t a{};
            REQUIRE(noa::all(a == vec4_t{0, 0, 0, 0}));
            REQUIRE(noa::all(vec4_t{1} == vec4_t{1, 0, 0, 0}));
            REQUIRE(noa::all(vec4_t{1, 2, 3, 4} == vec4_t::from_values(1, 2, 3, 4)));
            REQUIRE(noa::all(vec4_t{2, 2, 2, 2} == vec4_t::filled_with(2)));
            REQUIRE(noa::all(vec4_t{1, 2, 3, 4} == vec4_t::from_vec(vec4_t{1, 2, 3, 4})));
            a = {4, 3, 2, 1};
            REQUIRE(noa::all(vec4_t{4, 3, 2, 1} == vec4_t::from_pointer(a.data())));
            REQUIRE(noa::all(vec4_t{0, 1, 2, 3} == vec4_t::arange()));
            REQUIRE(noa::all(vec4_t{2, 5, 8, 11} == vec4_t::arange(2, 3)));
        }

        {
            // Indexing and assignment are pretty much as expected.
            vec2_t a{};
            const vec2_t d{1, 2};
            const vec2_t e = {1, 2};
            REQUIRE(noa::all(d == e));

            a = vec2_t{1, 1};
            a = d;
            a = {1, 1};
            REQUIRE(noa::all(a == vec2_t{1, 1}));
            a = {1, 2};
            REQUIRE(noa::all(a == vec2_t{1, 2}));

            a[0] = 23;
            a[1] = 52;
            REQUIRE((a[0] == 23 and a[1] == 52));

            const auto [a0, a1] = a;
            REQUIRE((a0 == 23 and a1 == 52));
        }

        {
            // It's also a range
            vec3_t a{};
            for (i32 i{}; auto& e: a)
                e = static_cast<TestType>(i++);
            REQUIRE(noa::all(a == vec3_t{0, 1, 2}));
            static_assert(vec3_t::SIZE == 3);
        }
    }

    SECTION("Cast") {
        const vec5_t a{0, 1, 2, 3, 4};
        const Vec<f32, 5> a_{0, 1, 2, 3, 4};
        const auto d = static_cast<Vec<f32, 5>>(a);
        const auto c = noa::clamp_cast<Vec<f32, 5>>(a);
        const auto b = noa::safe_cast<Vec<f32, 5>>(a);
        const auto e = a.template as<f32>();
        const auto f = a.template as_clamp<f32>();
        const auto g = a.template as_safe<f32>();
        REQUIRE((noa::all(a_ == b) and
                 noa::all(a_ == c) and
                 noa::all(a_ == d) and
                 noa::all(a_ == e) and
                 noa::all(a_ == f) and
                 noa::all(a_ == g)));
    }

    SECTION("Arithmetics and comparisons") {
        auto randomizer = test::Randomizer<TestType>(1, 20);
        auto v0 = vec2_t{randomizer.get(), randomizer.get()};
        auto v1 = vec2_t{randomizer.get(), randomizer.get()};

        REQUIRE(noa::all(v0 + 1 == vec2_t{v0[0] + 1, v0[1] + 1}));
        REQUIRE(noa::all(v0 - 2 == vec2_t{v0[0] - 2, v0[1] - 2}));
        REQUIRE(noa::all(v0 * 3 == vec2_t{v0[0] * 3, v0[1] * 3}));
        REQUIRE(noa::all(v0 / 2 == vec2_t{v0[0] / 2, v0[1] / 2}));
        REQUIRE(noa::all(1 + v0  == vec2_t{1 + v0[0], 1 + v0[1]}));
        REQUIRE(noa::all(2 - v0  == vec2_t{2 - v0[0], 2 - v0[1]}));
        REQUIRE(noa::all(3 * v0  == vec2_t{3 * v0[0], 3 * v0[1]}));
        REQUIRE(noa::all(2 / v0  == vec2_t{2 / v0[0], 2 / v0[1]}));
        REQUIRE(noa::all(v0 + v1 == vec2_t{v0[0] + v1[0], v0[1] + v1[1]}));
        REQUIRE(noa::all(v0 - v1 == vec2_t{v0[0] - v1[0], v0[1] - v1[1]}));
        REQUIRE(noa::all(v0 * v1 == vec2_t{v0[0] * v1[0], v0[1] * v1[1]}));
        REQUIRE(noa::all(v0 / v1 == vec2_t{v0[0] / v1[0], v0[1] / v1[1]}));

        auto v2 = v0;
        v0 += 1; REQUIRE(noa::all(v0 == vec2_t{v2[0] + 1, v2[1] + 1})); v2 = v0;
        v0 -= 2; REQUIRE(noa::all(v0 == vec2_t{v2[0] - 2, v2[1] - 2})); v2 = v0;
        v0 *= 3; REQUIRE(noa::all(v0 == vec2_t{v2[0] * 3, v2[1] * 3})); v2 = v0;
        v0 /= 2; REQUIRE(noa::all(v0 == vec2_t{v2[0] / 2, v2[1] / 2})); v2 = v0;
        v0 += v1; REQUIRE(noa::all(v0 == vec2_t{v2[0] + v1[0], v2[1] + v1[1]})); v2 = v0;
        v0 -= v1; REQUIRE(noa::all(v0 == vec2_t{v2[0] - v1[0], v2[1] - v1[1]})); v2 = v0;
        v0 *= v1; REQUIRE(noa::all(v0 == vec2_t{v2[0] * v1[0], v2[1] * v1[1]})); v2 = v0;
        v0 /= v1; REQUIRE(noa::all(v0 == vec2_t{v2[0] / v1[0], v2[1] / v1[1]}));

        v0 = {4, 10};
        REQUIRE(noa::all(Vec2<bool>{0, 1} == (v0 > 5)));
        REQUIRE(noa::all(Vec2<bool>{1, 1} == (v0 < 11)));
        REQUIRE(noa::all(Vec2<bool>{0, 1} == (v0 >= 7)));
        REQUIRE(noa::all(Vec2<bool>{1, 1} == (v0 <= 10)));
        REQUIRE(noa::any(v0 == 4));
        REQUIRE_FALSE(all(v0 == 4));

        REQUIRE(noa::all((5 < v0) == Vec2<bool>{0, 1}));
        REQUIRE(noa::all((11 > v0) == Vec2<bool>{1, 1}));
        REQUIRE(noa::all((7 <= v0) == Vec2<bool>{0, 1}));
        REQUIRE(noa::all((9 >= v0) == Vec2<bool>{1, 0}));
        REQUIRE(noa::any(4 == v0));
        REQUIRE_FALSE(noa::all(4 == v0));

        v0 = {2, 4};
        REQUIRE(noa::all((v0 > vec2_t{1, 2}) == Vec2<bool>{1, 1}));
        REQUIRE(noa::all((v0 < vec2_t{4, 5}) == Vec2<bool>{1, 1}));
        REQUIRE(noa::all((v0 >= vec2_t{2, 4}) == Vec2<bool>{1, 1}));
        REQUIRE(noa::all((v0 <= vec2_t{1, 4}) == Vec2<bool>{0, 1}));
        REQUIRE(noa::all((v0 != vec2_t{4, 4}) == Vec2<bool>{1, 0}));

        // Min & Max
        REQUIRE(noa::all(noa::min(vec2_t{3, 4}, vec2_t{5, 2}) == vec2_t{3, 2}));
        REQUIRE(noa::all(noa::max(vec2_t{3, 4}, vec2_t{5, 2}) == vec2_t{5, 4}));
        REQUIRE(noa::all(noa::min(vec2_t{3, 6}, TestType{5}) == vec2_t{3, 5}));
        REQUIRE(noa::all(noa::max(vec2_t{9, 0}, TestType{2}) == vec2_t{9, 2}));

        if constexpr (std::is_floating_point_v<TestType>) {
            v0 = 2;
            REQUIRE(all(v0 == TestType{2}));
            v0 += static_cast<TestType>(1.34);
            REQUIRE(all(noa::allclose(v0, static_cast<TestType>(3.34))));
            v0 -= static_cast<TestType>(23.134);
            REQUIRE(all(noa::allclose(v0, static_cast<TestType>(-19.794))));
            v0 *= static_cast<TestType>(-2.45);
            REQUIRE(all(noa::allclose(v0, static_cast<TestType>(48.4953))));
            v0 /= static_cast<TestType>(567.234);
            REQUIRE(all(noa::allclose(v0, static_cast<TestType>(0.085494))));

            v0 = static_cast<TestType>(3.30);
            auto tmp = v0 + static_cast<TestType>(3.234534);
            REQUIRE(all(noa::allclose(tmp, static_cast<TestType>(6.534534))));
            tmp = v0 - static_cast<TestType>(-234.2);
            REQUIRE(all(noa::allclose(tmp, static_cast<TestType>(237.5))));
            tmp = v0 * static_cast<TestType>(3);
            REQUIRE(all(noa::allclose(tmp, static_cast<TestType>(9.90))));
            tmp = v0 / static_cast<TestType>(0.001);
            REQUIRE(all(noa::allclose(tmp, static_cast<TestType>(3299.999f), static_cast<TestType>(1e-3))));

            v0 = {0, 2};
            v0 += {35, 20};
            REQUIRE(all(noa::allclose(v0, vec2_t::from_values(35, 22))));
            v0 -= vec2_t::from_values(-0.12, 23.2123);
            REQUIRE(all(noa::allclose(v0, vec2_t::from_values(35.12, -1.2123))));
            v0 *= vec2_t::from_values(0, 10);
            REQUIRE(all(noa::allclose(v0, vec2_t::from_values(0, -12.123), static_cast<TestType>(1e-5))));
            v0 /= vec2_t::from_values(2, 9);
            REQUIRE(all(noa::allclose(v0, vec2_t::from_values(0, -1.347))));

            v0[0] = 20;
            v0[1] = 50;
            tmp = v0 + vec2_t::from_values(10, 12);
            REQUIRE(all(noa::allclose(tmp, vec2_t::from_values(30, 62))));
            tmp = v0 - vec2_t::from_values(10.32, -112.001);
            REQUIRE(all(noa::allclose(tmp, vec2_t::from_values(9.68, 162.001))));
            tmp = v0 * vec2_t::from_values(2.5, 3.234);
            REQUIRE(all(noa::allclose(tmp, vec2_t::from_values(50, 161.7))));
            tmp = v0 / vec2_t::from_values(10, -12);
            REQUIRE(all(noa::allclose(tmp, vec2_t::from_values(2, -4.166667))));
        }
    }

    SECTION("Maths") {
        const vec2_t a{123, 43};
        REQUIRE(noa::sum(a) == 166);
        REQUIRE(noa::product(a) == 5289);

        if constexpr (std::is_floating_point_v<TestType>) {
            const auto x = static_cast<TestType>(5.2);
            const auto y = static_cast<TestType>(12.3);
            auto b = vec2_t::from_values(x, y);
            REQUIRE(all(noa::allclose(noa::cos(b), vec2_t::from_values(noa::cos(x), noa::cos(y)))));
            REQUIRE(all(noa::allclose(noa::sin(b), vec2_t::from_values(noa::sin(x), noa::sin(y)))));
            REQUIRE(all(noa::allclose(noa::tan(b), vec2_t::from_values(noa::tan(x), noa::tan(y)))));
            REQUIRE(all(noa::allclose(noa::log(b), vec2_t::from_values(noa::log(x), noa::log(y)))));
            REQUIRE(all(noa::allclose(noa::exp(b), vec2_t::from_values(noa::exp(x), noa::exp(y)))));
            REQUIRE(all(noa::allclose(noa::round(b), vec2_t::from_values(noa::round(x), noa::round(y)))));
            REQUIRE(all(noa::allclose(noa::ceil(b), vec2_t::from_values(noa::ceil(x), noa::ceil(y)))));
            REQUIRE(all(noa::allclose(noa::abs(b), vec2_t::from_values(noa::abs(x), noa::abs(y)))));
            REQUIRE(all(noa::allclose(noa::sqrt(b), vec2_t::from_values(noa::sqrt(x), noa::sqrt(y)))));

            b = vec2_t::from_values(23.23, -12.252);
            auto dot = static_cast<double>(b[0] * b[0] + b[1] * b[1]);
            REQUIRE_THAT(noa::dot(b, b), Catch::Matchers::WithinAbs(dot, 1e-6));
            REQUIRE_THAT(noa::norm(b), Catch::Matchers::WithinAbs(std::sqrt(dot), 1e-6));
            const auto b_normalized = noa::normalize(b);
            REQUIRE_THAT(noa::norm(b_normalized), Catch::Matchers::WithinAbs(1, 1e-6));
            REQUIRE_THAT(noa::dot(b, vec2_t::from_values(-12.23, -21.23)), Catch::Matchers::WithinAbs(-23.992940, 1e-4));
        }
    }

    SECTION("Other") {
        vec4_t b{123, 43, 32, 12};
        REQUIRE(noa::all(noa::sort(b) == b.flip()));
        REQUIRE(noa::all(b.circular_shift(1) == b.filter(3, 0, 1, 2)));
        REQUIRE(noa::all(b.circular_shift(-1) == b.filter(1, 2, 3, 0)));
        REQUIRE(noa::all(b.circular_shift(4) == b));
        REQUIRE(b.pop_front()[0] == b[1]);
        REQUIRE(b.pop_back()[0] == b[0]);
        REQUIRE(noa::all(b.template pop_back<2>() == vec2_t{123, 43}));

        REQUIRE(noa::all(b.push_front(1) == vec5_t{1, 123, 43, 32, 12}));
        REQUIRE(noa::all(b.template push_front<2>(1) == vec6_t{1, 1, 123, 43, 32, 12}));
        REQUIRE(noa::all(b.template push_back<2>(1) == vec6_t{123, 43, 32, 12, 1, 1}));
        REQUIRE(noa::all(b.push_front(vec2_t{1, 2}) == vec6_t{1, 2, 123, 43, 32, 12}));
        REQUIRE(noa::all(b.push_back(vec2_t{1, 2}) == vec6_t{123, 43, 32, 12, 1, 2}));

        REQUIRE(noa::all(b.reorder({1, 0, 3, 2}) == b.filter(1, 0, 3, 2)));
        REQUIRE(noa::all(b.filter(0, 1) == vec2_t{123, 43}));
        REQUIRE(b.filter(2)[0] == b[2]);
        REQUIRE(b.template set<3>(0)[3] == 0);

        const std::array<TestType, 4> b_array = {123, 43, 32, 12};
        REQUIRE(fmt::format("{}", b) == fmt::format("{}", b_array));
    }
}
