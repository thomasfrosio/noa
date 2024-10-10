#include <noa/core/types/Accessor.hpp>
#include <noa/gpu/cuda/ReduceIwise.cuh>
#include <noa/gpu/cuda/Allocators.hpp>
#include <catch2/catch.hpp>

#include "Utils.hpp"

namespace {
    using namespace noa::types;

    // 1d sum and argmax fused into one operation.
    // In this example, we don't pay attention to taking the first or last max...
    template<size_t N>
    struct SumArgmax {
        AccessorRestrictContiguousI32<f64, N> input;

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

TEST_CASE("cuda::reduce_iwise") {
    using namespace noa::types;
    using noa::cuda::reduce_iwise;
    using noa::cuda::ReduceIwiseConfig;
    using noa::cuda::Stream;
    using noa::cuda::Device;
    using noa::cuda::AllocatorManaged;

    Stream stream(Device::current());

    const Tuple shapes = noa::make_tuple(
        Shape1<i64>{2451}, Shape1<i64>{524289},
        Shape2<i64>{72, 130}, Shape2<i64>{542, 845},
        Shape3<i64>{4, 45, 35}, Shape3<i64>{64, 64, 64},
        Shape4<i64>{2, 3, 25, 35}, Shape4<i64>{3, 64, 64, 64},
        Shape4<i64>{2, 128, 128, 128}
    );
    shapes.for_each([&]<size_t N>(const Shape<i64, N>& shape) {
        INFO("shape=" << shape);
        const auto n_elements = shape.n_elements();
        const auto b0 = AllocatorManaged<f64>::allocate(n_elements, stream);
        const auto b1 = AllocatorManaged<f64>::allocate(1, stream);
        const auto b2 = AllocatorManaged<Pair<f64, i32>>::allocate(1, stream);
        test::randomize(b0.get(), n_elements, test::Randomizer<f64>(-5, 10));

        const i32 expected_index = 1751;
        const f64 expected_max = 16.23;
        b0[expected_index] = expected_max;
        const auto expected_sum = std::accumulate(b0.get(), b0.get() + n_elements, 0.);

        auto reduced = noa::make_tuple(
            AccessorValue<f64>(0.),
            AccessorValue<Pair<f64, i32>>({-10., 0})
        );
        auto output = noa::make_tuple(
            AccessorRestrictContiguousI32<f64, 1>(b1.get()),
            AccessorRestrictContiguousI32<Pair<f64, i32>, 1>(b2.get())
        );

        const auto shape_i32 = shape.template as<i32>();
        auto op = SumArgmax{.input=AccessorRestrictContiguousI32<f64, N>(b0.get(), shape_i32.strides())};
        reduce_iwise(shape_i32, op, reduced, output, stream);
        stream.synchronize();

        REQUIRE_THAT(b1[0], Catch::WithinRel(static_cast<f64>(expected_sum), 1e-8));
        REQUIRE((b2[0].first == expected_max and b2[0].second == expected_index));
    });
}
