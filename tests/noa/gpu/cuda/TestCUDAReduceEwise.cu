#include <noa/gpu/cuda/ReduceEwise.hpp>
#include <noa/gpu/cuda/AllocatorManaged.hpp>
#include <catch2/catch.hpp>

#include "Helpers.h"

namespace {
    using namespace noa::types;

    struct SumMaxOp {
        constexpr void init(f32 input, f64& sum, f32& max) const {
            sum += static_cast<f64>(input);
            max = std::max(max, input);
        }
        constexpr void join(f64 current_sum, f32 current_max, f64& sum, f32& max) const {
            sum += current_sum;
            max = std::max(max, current_max);
        }
        constexpr void final(f64& sum, f32& max, Tuple<f64&, i32&>& output) const {
            auto& [a, b] = output;
            a = sum + 1;
            b = static_cast<i32>(max + 1);
        }
    };
}

TEST_CASE("cuda::reduce_ewise") {
    using namespace noa::types;
    using noa::cuda::reduce_ewise;
    using noa::cuda::ReduceEwiseConfig;
    using noa::cuda::Stream;
    using noa::cuda::Device;
    using noa::cuda::AllocatorManaged;

    Stream stream(Device::current());

    AND_THEN("simple sum, contiguous") {
        const std::array shapes = {Shape4<i64>{1, 50, 1, 100}, test::get_random_shape4_batched(4)};
        for (const auto& shape: shapes) {
            INFO("shape=" << shape);
            const auto n_elements = shape.elements();

            const auto b0 = AllocatorManaged<i64>::allocate(n_elements, stream);
            const auto b1 = AllocatorManaged<i64>::allocate(1, stream);
            test::arange(b0.get(), n_elements);

            auto reduced = noa::make_tuple(AccessorValue<i64>(0.));
            auto output = noa::make_tuple(AccessorRestrictContiguousI64<i64, 1>(b1.get()));
            auto reduce_op = []__device__(i64 to_reduce, i64& reduced) { reduced += to_reduce; };
            const auto expected_sum = (n_elements / 2) * (n_elements - 1);

            { // no vectorization
                auto input = noa::make_tuple(AccessorI64<i64, 4>(b0.get(), shape.strides()));
                reduce_ewise(shape, reduce_op, input, reduced, output, stream);
                stream.synchronize();
                REQUIRE(b1[0] == expected_sum);
            }
            { // vectorization
                auto input = noa::make_tuple(AccessorI64<const i64, 4>(b0.get(), shape.strides()));
                reduce_ewise(shape, reduce_op, input, reduced, output, stream);
                stream.synchronize();
                REQUIRE(b1[0] == expected_sum);
            }
        }
    }

    AND_THEN("simple sum, small strided") {
        const std::array shapes = {Shape4<i64>{1, 8, 8, 100}, test::get_random_shape4(3, true)};
        for (const auto& shape: shapes) {
            INFO("shape=" << shape);
            const auto n_elements = shape.elements();

            const auto b0 = AllocatorManaged<i64>::allocate(n_elements, stream);
            const auto b1 = AllocatorManaged<i64>::allocate(1, stream);
            test::arange(b0.get(), n_elements);

            // Repeat the batch. This breaks the batch contiguity.
            const auto broadcasted_shape = shape.set<0>(2);
            const auto broadcasted_strides = shape.strides().set<0>(0);

            auto reduced = noa::make_tuple(AccessorValue<i64>(0.));
            auto output = noa::make_tuple(AccessorRestrictContiguousI64<i64, 1>(b1.get()));
            auto reduce_op = []__device__(i64 to_reduce, i64& reduced) { reduced += to_reduce; };
            const auto expected_sum = (n_elements / 2) * (n_elements - 1);

            { // no vectorization
                auto input = noa::make_tuple(AccessorI64<i64, 4>(b0.get(), broadcasted_strides));
                reduce_ewise(broadcasted_shape, reduce_op, input, reduced, output, stream);
                stream.synchronize();
                REQUIRE(b1[0] == expected_sum * 2);
            }

            { // no vectorization
                auto input = noa::make_tuple(AccessorI64<const i64, 4>(b0.get(), broadcasted_strides));
                reduce_ewise(broadcasted_shape, reduce_op, input, reduced, output, stream);
                stream.synchronize();
                REQUIRE(b1[0] == expected_sum * 2);
            }
        }
    }

    AND_THEN("sum-max") {
        const auto shape = test::get_random_shape4(4);
        const auto n_elements = shape.elements();

        const auto b0 = AllocatorManaged<f32>::allocate(n_elements, stream);
        const auto b1 = AllocatorManaged<f64>::allocate(1, stream);
        const auto b2 = AllocatorManaged<i32>::allocate(1, stream);
        std::fill_n(b0.get(), n_elements, 1);
        b0[234] += 12.f;
        auto input = noa::make_tuple(AccessorI64<f32, 4>(b0.get(), shape.strides()));
        auto reduced = noa::make_tuple(AccessorValue<f64>(0.), AccessorValue<f32>(0.));
        auto output = noa::make_tuple(
                AccessorRestrictContiguousI32<f64, 1>(b1.get()),
                AccessorRestrictContiguousI32<i32, 1>(b2.get())
        );

        using config = ReduceEwiseConfig<false, false, true>;
        reduce_ewise<config>(shape, SumMaxOp{}, input, reduced, output, stream);
        stream.synchronize();
        REQUIRE_THAT(b1[0], Catch::WithinAbs(static_cast<f64>(n_elements + 12 + 1), 1e-8));
        REQUIRE(b2[0] == 13 + 1);
    }
}
