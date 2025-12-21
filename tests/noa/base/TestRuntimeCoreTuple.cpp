#include <noa/base/Tuple.hpp>
#include <noa/base/Utils.hpp>

#include "Catch.hpp"

namespace {
    struct Tracked {
        std::array<int, 3> count{};
        Tracked() { count[0] += 1; }
        Tracked(const Tracked& t) : count(t.count) { count[1] += 1; }
        Tracked(Tracked&& t)  noexcept : count(t.count) { count[2] += 1; }
    };
}

using namespace noa::types;

NOA_NV_DIAG_SUPPRESS(445)
TEST_CASE("base::Tuple") {
    // Element access.
    using tt = std::tuple<int, int&, int&&, const int, const int&, const int&&>;
    using tn = Tuple<int, int&, int&&, const int, const int&, const int&&>;
    noa::static_for_each<6>([]<size_t I>() {
        static_assert(std::is_same_v<
                decltype(std::get<I>(std::declval<tt>())),
                decltype(noa::get<I>(std::declval<tn>()))>);

        static_assert(std::is_same_v<
                decltype(std::get<I>(std::move(std::declval<tt>()))),
                decltype(noa::get<I>(std::move(std::declval<tn>())))>);
    });

    static_assert(std::is_same_v<decltype(std::tuple{std::declval<int&>()}), std::tuple<int>>);
    static_assert(std::is_same_v<decltype(std::tuple{std::declval<int&&>()}), std::tuple<int>>);
    static_assert(std::is_same_v<decltype(std::make_tuple(std::ref(std::declval<int&>()))), std::tuple<int&>>);

    static_assert(std::is_same_v<decltype(Tuple{std::declval<int&>()}), Tuple<int>>);
    static_assert(std::is_same_v<decltype(Tuple{std::declval<int&&>()}), Tuple<int>>);
    static_assert(std::is_same_v<decltype(Tuple{std::ref(std::declval<int&>())}), Tuple<int&>>);
    static_assert(std::is_same_v<decltype(noa::make_tuple(std::ref(std::declval<int&>()))), Tuple<int&>>);

    {
        int a{};
        std::vector<int> b{};
        const float c{};

        [[maybe_unused]] auto t0 = noa::make_tuple(1, a, std::move(b), std::ref(a), c);
        [[maybe_unused]] auto s0 = std::make_tuple(1, a, std::move(b), std::ref(a), c);
        static_assert(std::is_same_v<decltype(t0), Tuple<int, int, std::vector<int>, int&, float>>);
        static_assert(std::is_same_v<decltype(s0), std::tuple<int, int, std::vector<int>, int&, float>>);

        [[maybe_unused]] auto t1 = noa::forward_as_tuple(1, a, std::move(b), c, std::move(c));
        [[maybe_unused]] auto s1 = std::forward_as_tuple(1, a, std::move(b), c, std::move(c));
        static_assert(std::is_same_v<decltype(t1), Tuple<int&&, int&, std::vector<int>&&, const float&, const float&&>>);
        static_assert(std::is_same_v<decltype(s1), std::tuple<int&&, int&, std::vector<int>&&, const float&, const float&&>>);

        [[maybe_unused]] auto t2 = noa::forward_as_final_tuple(1, a, std::move(b), c, std::move(c));
        static_assert(std::is_same_v<decltype(t2), Tuple<int, int&, std::vector<int>, const float&, float>>);
    }

    Tuple t0 = noa::make_tuple(1, 2);
    Tuple t1 = t0.apply([](int& a, int& b) {
        a += 1;
        b -= 1;
        return noa::make_tuple(a + 1, b + 2);
    });
    REQUIRE(noa::get<0>(t0) == 2);
    REQUIRE(noa::get<1>(t0) == 1);

    t1.for_each_enumerate([]<int I>(int& a) { a += 10 + I; });
    t1[Tag<1>{}] += 2;
    REQUIRE(noa::get<0>(t1) == 13);
    REQUIRE(noa::get<1>(t1) == 16);

    {
        Tracked u0{};
        std::tuple<Tracked&> t2{u0};
        Tuple<Tracked&> t3{u0};
        Tracked u1(std::get<0>(t2));
        Tracked u2(noa::get<0>(t3));
        REQUIRE((u1.count[0] == 1 and u1.count[1] == 1 and u1.count[2] == 0));
        REQUIRE((u1.count[0] == u2.count[0] and u1.count[1] == u2.count[1] and u1.count[2] == u2.count[2]));

    }

    {
        Tracked u0{}, u1{};
        std::tuple<Tracked&&> t2{std::move(u0)};
        Tuple<Tracked&&> t3{std::move(u1)};
        Tracked u2(std::get<0>(t2));
        Tracked u3(noa::get<0>(t3));
        REQUIRE((u2.count[0] == 1 and u2.count[1] == 1 and u2.count[2] == 0));
        REQUIRE((u3.count[0] == u2.count[0] and u3.count[1] == u2.count[1] and u3.count[2] == u2.count[2]));
    }

    {
        Tracked u0{}, u1{};
        std::tuple<Tracked&&> t2{std::move(u0)};
        Tuple<Tracked&&> t3{std::move(u1)};
        Tracked u2(std::get<0>(std::move(t2)));
        Tracked u3(noa::get<0>(std::move(t3)));
        REQUIRE((u2.count[0] == 1 and u2.count[1] == 0 and u2.count[2] == 1));
        REQUIRE((u3.count[0] == u2.count[0] and u3.count[1] == u2.count[1] and u3.count[2] == u2.count[2]));
    }
}
NOA_NV_DIAG_DEFAULT(445)

TEST_CASE("base::tuple_cat") {
    constexpr Tuple<i32, i32> a = noa::make_tuple(1, 2);
    constexpr Tuple<f32, f64> b = noa::make_tuple(3.f, 4.);
    constexpr Tuple<i32, i32, f32, f64> c = noa::tuple_cat(a, b);
    static_assert(noa::get<0>(a) == noa::get<0>(c) and
                  noa::get<1>(a) == noa::get<1>(c) and
                  noa::get<0>(b) == noa::get<2>(c) and
                  noa::get<1>(b) == noa::get<3>(c)
    );

    f64 aa{1.4};
    Tuple<f64&&> t{std::move(aa)};
    auto tt = noa::tuple_cat(std::move(t));
    REQUIRE(noa::get<0>(tt) == 1.4);

    static_assert(std::is_same_v<decltype(std::declval<Tuple<int, int&, f64&&>&>()[Tag<2>{}]), f64&>);
    static_assert(std::is_same_v<decltype(noa::tuple_cat(std::declval<Tuple<int, int&, f64&&>>())),
                                 Tuple<int, int&, f64&&>>);
}

TEST_CASE("base::Tuple of c-arrays") {
    using t0 = std::tuple<int[10]>;
    using t1 = Tuple<int[10]>;
    static_assert(std::is_same_v<std::tuple_element_t<0, t0>, std::tuple_element_t<0, t1>>);
    static_assert(std::is_same_v<std::tuple_element_t<0, t0>, int[10]>);

    static_assert(std::is_same_v<decltype(std::get<0>(std::declval<t0&>())), int(&)[10]>);
    static_assert(std::is_same_v<decltype(std::declval<t1&>()[Tag<0>{}]), int(&)[10]>);
    static_assert(std::is_same_v<decltype(std::get<0>(std::declval<t0&&>())), int(&&)[10]>);
    static_assert(std::is_same_v<decltype(std::declval<t1&&>()[Tag<0>{}]), int(&&)[10]>);
    static_assert(std::is_same_v<decltype(std::get<0>(std::declval<const t0&>())), const int(&)[10]>);
    static_assert(std::is_same_v<decltype(std::declval<const t1&>()[Tag<0>{}]), const int(&)[10]>);

    t1 t{};
    t[Tag<0>{}][5] = 1;
    REQUIRE((t[Tag<0>{}][0] == 0 and t[Tag<0>{}][5] == 1));
}
