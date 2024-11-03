#include "noa/core/Interpolation.hpp"
#include "noa/core/types/Accessor.hpp"
#include "noa/core/types/Span.hpp"

#include <catch2/catch.hpp>
#include "Utils.hpp"

TEST_CASE("core, Interpolator", "[noa][core]") {
    using namespace noa;

    {
        auto shape = Shape3<i64>{3, 64, 64};
        auto buffer = test::random<f32>(shape.n_elements(), test::Randomizer<f32>(-10, 10));
        auto data_2d = Span<f32, 3, i64>{buffer.get(), shape};
        auto accessor = Accessor<const f32, 3, i64>{data_2d.get(), data_2d.strides()};

        using interpolator_t = const Interpolator<2, Interp::CUBIC, Border::ZERO, decltype(accessor)>;
        auto op = interpolator_t(accessor, data_2d.shape().pop_front());

        auto coordinate = Vec<f64, 2>{1, 1};
        auto interpolated_value_batch0 = op.interpolate_at(coordinate);
        auto interpolated_value_batch2 = op.interpolate_at(coordinate, 2);

        auto value0 = op(6, 7); // batch=0, height=6, width=7
        auto value1 = op(2, 6, 7); // batch=2, height=6, width=7
    }
}

TEMPLATE_TEST_CASE("core, interpolation_weight<LANCZOS>, ", "[noa][core]", float, double) {
    using namespace noa::types;
    {
        Vec<TestType, 1> a{std::numeric_limits<TestType>::epsilon()};
        for (auto i: noa::irange(10)) {
            for (auto c: noa::interpolation_weights<noa::Interp::LANCZOS4, decltype(a)>(a))
                REQUIRE(noa::is_finite(c[0]));

            for (auto c: noa::interpolation_weights<noa::Interp::LANCZOS6, decltype(a)>(a))
                REQUIRE(noa::is_finite(c[0]));

            for (auto c: noa::interpolation_weights<noa::Interp::LANCZOS8, decltype(a)>(a))
                REQUIRE(noa::is_finite(c[0]));

            a[0] = std::nextafter(a[0], TestType{1.});
        }
    }
    {
        Vec<TestType, 1> a{1};
        for (auto i: noa::irange(10)) {
            for (auto c: noa::interpolation_weights<noa::Interp::LANCZOS4, decltype(a)>(a))
                REQUIRE(noa::is_finite(c[0]));

            for (auto c: noa::interpolation_weights<noa::Interp::LANCZOS6, decltype(a)>(a))
                REQUIRE(noa::is_finite(c[0]));

            for (auto c: noa::interpolation_weights<noa::Interp::LANCZOS8, decltype(a)>(a))
                REQUIRE(noa::is_finite(c[0]));

            a[0] = std::nextafter(a[0], TestType{0.});
        }
    }
}
