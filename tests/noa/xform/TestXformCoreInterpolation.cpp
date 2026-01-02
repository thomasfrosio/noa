#include "noa/xform/core/Interpolation.hpp"
#include "noa/runtime/core/Accessor.hpp"
#include "noa/runtime/core/Span.hpp"

#include "Catch.hpp"
#include "Utils.hpp"

TEST_CASE("xform::Interpolator") {
    using namespace noa;

    {
        auto shape = Shape3{3, 64, 64};
        auto buffer = test::random<f32>(shape.n_elements(), test::Randomizer<f32>(-10, 10));
        auto data_2d = Span<f32, 3>{buffer.get(), shape};
        auto accessor = Accessor<const f32, 3>{data_2d.get(), data_2d.strides()};

        using interpolator_t = const nx::Interpolator<2, nx::Interp::CUBIC, Border::ZERO, decltype(accessor)>;
        auto op = interpolator_t(accessor, data_2d.shape().pop_front());

        [[maybe_unused]] auto coordinate = Vec<f64, 2>{1, 1};
        [[maybe_unused]] auto interpolated_value_batch0 = op.interpolate_at(coordinate);
        [[maybe_unused]] auto interpolated_value_batch2 = op.interpolate_at(coordinate, 2);

        [[maybe_unused]] auto value0 = op(6, 7); // batch=0, height=6, width=7
        [[maybe_unused]] auto value1 = op(2, 6, 7); // batch=2, height=6, width=7
    }
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
