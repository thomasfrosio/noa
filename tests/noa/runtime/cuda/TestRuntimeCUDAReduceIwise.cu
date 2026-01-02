#include <noa/runtime/core/Accessor.hpp>
#include <noa/runtime/cuda/ReduceIwise.cuh>
#include <noa/runtime/cuda/Allocators.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

namespace {
    using namespace noa::types;

    // 1d sum and argmax fused into one operation.
    // In this example, we don't pay attention to taking the first or last max...
    template<usize N>
    struct SumArgmax {
        noa::AccessorRestrictContiguous<f64, N, i32> input;

        constexpr void init(const auto& indices, f64& sum, Pair<f64, i32>& argmax) const {
            const auto value = input(indices);
            sum += value;
            if (value > argmax.first) {
                argmax.first = value;
                argmax.second = input.offset_at(indices);
            }
        }

        constexpr void join(
            f64 isum, Pair<f64, i32> iargmax,
            f64& osum, Pair<f64, i32>& oargmax
        ) const {
            osum += isum;
            if (iargmax.first > oargmax.first)
                oargmax = iargmax;
        }
    };
}

TEST_CASE("runtime::cuda::reduce_iwise") {
    using namespace noa::types;
    using noa::cuda::reduce_iwise;
    using noa::cuda::ReduceIwiseConfig;
    using noa::cuda::Stream;
    using noa::cuda::Device;
    using noa::cuda::AllocatorManaged;

    Stream stream(Device::current());

    const Tuple shapes = noa::make_tuple(
        Shape1{2451}, Shape1{524289},
        Shape2{72, 130}, Shape2{542, 845},
        Shape3{4, 45, 35}, Shape3{64, 64, 64},
        Shape4{2, 3, 25, 35}, Shape4{3, 64, 64, 64},
        Shape4{2, 128, 128, 128}
    );
    shapes.for_each([&]<usize N>(const Shape<isize, N>& shape) {
        INFO("shape=" << shape);
        const auto n_elements = shape.n_elements();
        const auto b0 = AllocatorManaged::allocate<f64>(n_elements, stream);
        const auto b1 = AllocatorManaged::allocate<f64>(1, stream);
        const auto b2 = AllocatorManaged::allocate<Pair<f64, i32>>(1, stream);
        test::randomize(b0.get(), n_elements, test::Randomizer<f64>(-5, 10));

        const i32 expected_index = 1751;
        const f64 expected_max = 16.23;
        b0[expected_index] = expected_max;
        const auto expected_sum = std::accumulate(b0.get(), b0.get() + n_elements, 0.);

        auto reduced = noa::make_tuple(
            noa::AccessorValue<f64>(0.),
            noa::AccessorValue<Pair<f64, i32>>({-10., 0})
        );
        auto output = noa::make_tuple(
            noa::AccessorRestrictContiguous<f64, 1, i32>(b1.get()),
            noa::AccessorRestrictContiguous<Pair<f64, i32>, 1, i32>(b2.get())
        );

        const auto shape_i32 = shape.template as<i32>();
        auto op = SumArgmax{.input=noa::AccessorRestrictContiguous<f64, N, i32>(b0.get(), shape_i32.strides())};
        reduce_iwise(shape_i32, op, reduced, output, stream);
        stream.synchronize();

        REQUIRE_THAT(b1[0], Catch::Matchers::WithinRel(static_cast<f64>(expected_sum), 1e-8));
        REQUIRE((b2[0].first == expected_max and b2[0].second == expected_index));
    });
}
