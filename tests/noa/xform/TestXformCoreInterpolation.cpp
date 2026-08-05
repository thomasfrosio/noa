#include "noa/xform/core/Interpolation.hpp"
#include "noa/runtime/core/Span.hpp"

#include "Catch.hpp"
#include "Utils.hpp"

TEST_CASE("xform::Interpolator") {
    using namespace noa;

    auto shape = Shape3{3, 64, 64};
    auto buffer = test::random<f32>(shape.n_elements(), test::Randomizer<f32>(-10, 10));
    auto data = Span<f32, 3>{buffer.get(), shape};

    using interpolator_t = const nx::Interpolator<nx::Interp::CUBIC, Border::ZERO, f64, 2, isize>;
    auto op = interpolator_t(data.shape().pop_front());

    [[maybe_unused]] auto coordinate = Vec<f64, 2>{1, 1};
    [[maybe_unused]] auto interpolated_value_batch0 = op(data[0], coordinate);
    [[maybe_unused]] auto interpolated_value_batch2 = op(data[1], data.shape().pop_front(), coordinate);
}

TEMPLATE_TEST_CASE("xform::interpolation_weight<LANCZOS>", "", float, double) {
    using namespace noa;
    {
        Vec<TestType, 1> a{std::numeric_limits<TestType>::epsilon()};
        for ([[maybe_unused]] auto _: noa::irange(10)) {
            for (auto c: nx::interpolation_weights<nx::Interp::LANCZOS4, decltype(a)>(a))
                REQUIRE(noa::is_finite(c[0]));

            for (auto c: nx::interpolation_weights<nx::Interp::LANCZOS6, decltype(a)>(a))
                REQUIRE(noa::is_finite(c[0]));

            for (auto c: nx::interpolation_weights<nx::Interp::LANCZOS8, decltype(a)>(a))
                REQUIRE(noa::is_finite(c[0]));

            a[0] = std::nextafter(a[0], TestType{1.});
        }
    }
    {
        Vec<TestType, 1> a{1};
        for ([[maybe_unused]] auto _: noa::irange(10)) {
            for (auto c: nx::interpolation_weights<nx::Interp::LANCZOS4, decltype(a)>(a))
                REQUIRE(noa::is_finite(c[0]));

            for (auto c: nx::interpolation_weights<nx::Interp::LANCZOS6, decltype(a)>(a))
                REQUIRE(noa::is_finite(c[0]));

            for (auto c: nx::interpolation_weights<nx::Interp::LANCZOS8, decltype(a)>(a))
                REQUIRE(noa::is_finite(c[0]));

            a[0] = std::nextafter(a[0], TestType{0.});
        }
    }
}
