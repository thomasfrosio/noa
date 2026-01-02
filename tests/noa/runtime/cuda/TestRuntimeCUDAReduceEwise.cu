#include <noa/runtime/cuda/ReduceEwise.cuh>
#include <noa/runtime/cuda/Allocators.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

namespace {
    using namespace noa::types;

    struct SumMaxOp {
        static constexpr void init(f32 input, f64& sum, f32& max) {
            sum += static_cast<f64>(input);
            max = std::max(max, input);
        }
        static constexpr void join(f64 current_sum, f32 current_max, f64& sum, f32& max) {
            sum += current_sum;
            max = std::max(max, current_max);
        }
        static constexpr void final(f64& sum, f32& max, Tuple<f64&, i32&>& output) {
            auto& [a, b] = output;
            a = sum + 1;
            b = static_cast<i32>(max + 1);
        }
    };
}

TEST_CASE("runtime::cuda::reduce_ewise") {
    using namespace noa::types;
    using noa::cuda::reduce_ewise;
    using noa::cuda::ReduceEwiseConfig;
    using noa::cuda::Stream;
    using noa::cuda::Device;
    using noa::cuda::AllocatorManaged;

    Stream stream(Device::current());

    SECTION("simple sum, contiguous") {
        const std::array shapes = {Shape4{1, 50, 1, 100}, test::random_shape_batched(4, {.only_even_sizes = true})};
        for (const auto& shape: shapes) {
            INFO("shape=" << shape);
            const auto n_elements = shape.n_elements();

            const auto b0 = AllocatorManaged::allocate<isize>(n_elements, stream);
            const auto b1 = AllocatorManaged::allocate<isize>(1, stream);
            test::arange(b0.get(), n_elements);

            auto reduced = noa::make_tuple(noa::AccessorValue<isize>(0.));
            auto output = noa::make_tuple(noa::AccessorRestrictContiguous<isize, 1>(b1.get()));
            auto reduce_op = []__device__(isize to_reduce, isize& reduced) { reduced += to_reduce; };
            const auto expected_sum = static_cast<isize>(static_cast<f64>(n_elements) / 2) * (n_elements - 1);

            { // no vectorization
                auto input = noa::make_tuple(noa::Accessor<isize, 4>(b0.get(), shape.strides()));
                reduce_ewise(shape, reduce_op, input, reduced, output, stream);
                stream.synchronize();
                REQUIRE(b1[0] == expected_sum);
            }
            { // vectorization
                auto input = noa::make_tuple(noa::Accessor<const isize, 4>(b0.get(), shape.strides()));
                reduce_ewise(shape, reduce_op, input, reduced, output, stream);
                stream.synchronize();
                REQUIRE(b1[0] == expected_sum);
            }
        }
    }

    SECTION("simple sum, small strided") {
        const std::array shapes{Shape<isize, 4>{1, 8, 8, 100}, test::random_shape(3, {.only_even_sizes = true})};
        for (const auto& shape: shapes) {
            INFO("shape=" << shape);
            const auto n_elements = shape.n_elements();

            const auto b0 = AllocatorManaged::allocate<isize>(n_elements, stream);
            const auto b1 = AllocatorManaged::allocate<isize>(1, stream);
            test::arange(b0.get(), n_elements);

            // Repeat the batch. This breaks the batch contiguity.
            const auto batch = 2;
            const auto broadcasted_shape = shape.set<0>(2);
            const auto broadcasted_strides = shape.strides().set<0>(0);

            auto reduced = noa::make_tuple(noa::AccessorValue<isize>(0.));
            auto output = noa::make_tuple(noa::AccessorRestrictContiguous<isize, 1>(b1.get()));
            auto reduce_op = []__device__(isize to_reduce, isize& reduced) { reduced += to_reduce; };

            const auto n_elements_per_batch = shape.pop_front().n_elements();
            const auto expected_sum =
                static_cast<isize>(static_cast<f64>(n_elements_per_batch) / 2) * (n_elements_per_batch - 1) * batch;

            { // no vectorization
                auto input = noa::make_tuple(noa::Accessor<isize, 4>(b0.get(), broadcasted_strides));
                reduce_ewise(broadcasted_shape, reduce_op, input, reduced, output, stream);
                stream.synchronize();
                REQUIRE(b1[0] == expected_sum);
            }

            { // no vectorization
                auto input = noa::make_tuple(noa::Accessor<const isize, 4>(b0.get(), broadcasted_strides));
                reduce_ewise(broadcasted_shape, reduce_op, input, reduced, output, stream);
                stream.synchronize();
                REQUIRE(b1[0] == expected_sum);
            }
        }
    }

    SECTION("sum-max") {
        const auto shape = test::random_shape(4);
        const auto n_elements = shape.n_elements();

        const auto b0 = AllocatorManaged::allocate<f32>(n_elements, stream);
        const auto b1 = AllocatorManaged::allocate<f64>(1, stream);
        const auto b2 = AllocatorManaged::allocate<i32>(1, stream);
        std::fill_n(b0.get(), n_elements, 1);
        b0[234] += 12.f;
        auto input = noa::make_tuple(noa::Accessor<f32, 4>(b0.get(), shape.strides()));
        auto reduced = noa::make_tuple(noa::AccessorValue(0.), noa::AccessorValue(0.f));
        auto output = noa::make_tuple(
            noa::AccessorRestrictContiguous<f64, 1, i32>(b1.get()),
            noa::AccessorRestrictContiguous<i32, 1, i32>(b2.get())
        );

        using config = ReduceEwiseConfig<false, false, true>;
        reduce_ewise<config>(shape, SumMaxOp{}, input, reduced, output, stream);
        stream.synchronize();
        REQUIRE_THAT(b1[0], Catch::Matchers::WithinAbs(static_cast<f64>(n_elements + 12 + 1), 1e-8));
        REQUIRE(b2[0] == 13 + 1);
    }
}
