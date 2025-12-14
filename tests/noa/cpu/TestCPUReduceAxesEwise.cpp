#include <noa/cpu/ReduceAxesEwise.hpp>
#include <noa/core/utils/Irange.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

namespace {
    struct Tracked {
        std::array<int, 2> count{};
        Tracked() = default;
        Tracked(const Tracked& t) : count(t.count) { count[0] += 1; }
        Tracked(Tracked&& t) noexcept: count(t.count) { count[1] += 1; }
    };
}

TEST_CASE("cpu::reduce_axes_ewise") {
    using namespace noa::types;
    using noa::cpu::reduce_axes_ewise;
    using noa::cpu::ReduceAxesEwiseConfig;

    AND_THEN("sum over one axis") {
        const auto input_shape = Shape<i64, 4>{6, 7, 8, 9};
        const auto input_elements = input_shape.n_elements();

        const auto input_buffer = std::make_unique<f32[]>(static_cast<size_t>(input_elements));
        std::fill_n(input_buffer.get(), input_elements, 1);

        auto input = noa::make_tuple(Accessor<f32, 4, i64>(input_buffer.get(), input_shape.strides()));
        auto init = noa::make_tuple(AccessorValue<f32>(0.));
        auto reduce_op = [](f32 to_reduce, f32& reduced) { reduced += to_reduce; };

        for (auto i: noa::irange(4)) {
            INFO("axis=" << i);
            auto output_shape = input_shape;
            output_shape[i] = 1;
            const auto output_elements = output_shape.n_elements();
            const auto output_buffer = std::make_unique<f32[]>(static_cast<size_t>(output_elements));
            auto output = noa::make_tuple(Accessor<f32, 4, i64>(output_buffer.get(), output_shape.strides()));

            reduce_axes_ewise(input_shape, output_shape, reduce_op, input, init, output);

            const f32 expected_value = static_cast<f32>(input_shape[i]);
            REQUIRE(test::allclose_abs(output_buffer.get(), expected_value, output_elements, 1e-5));
        }
    }

    AND_THEN("sum per batch") {
        const auto input_shape = Shape<i64, 4>{6, 7, 8, 9};
        const auto output_shape = Shape<i64, 4>{6, 1, 1, 1};
        const auto input_elements = input_shape.n_elements();
        const auto output_elements = output_shape.n_elements();

        const auto input_buffer = std::make_unique<f32[]>(static_cast<size_t>(input_elements));
        const auto output_buffer = std::make_unique<f32[]>(static_cast<size_t>(output_elements));
        std::fill_n(input_buffer.get(), input_elements, 1);

        auto input = noa::make_tuple(Accessor<f32, 4, i64>(input_buffer.get(), input_shape.strides()));
        auto output = noa::make_tuple(Accessor<f32, 4, i64>(output_buffer.get(), output_shape.strides()));
        auto init = noa::make_tuple(AccessorValue<f32>(0.));
        auto reduce_op = [](f32 to_reduce, f32& reduced) { reduced += to_reduce; };

        reduce_axes_ewise(input_shape, output_shape, reduce_op, input, init, output);

        const f32 expected_value = static_cast<f32>(input_shape.pop_front().n_elements());
        REQUIRE(test::allclose_abs(output_buffer.get(), expected_value, output_elements, 1e-5));
    }
}
