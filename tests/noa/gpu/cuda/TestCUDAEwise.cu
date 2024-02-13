#include <catch2/catch.hpp>
#include <noa/core/Operators.hpp>
#include <noa/core/utils/Interfaces.hpp>
#include <noa/gpu/cuda/AllocatorManaged.hpp>
#include <noa/gpu/cuda/Ewise.hpp>

#include <noa/gpu/cuda/ReduceIwise.hpp>
#include <noa/gpu/cuda/ReduceEwise.hpp>
#include <noa/gpu/cuda/ReduceAxesEwise.hpp>

#include "Helpers.h"

namespace {
    using namespace noa::types;

    struct Tracked {
        Vec2<int> count{};
         constexpr Tracked() = default;
         constexpr Tracked(const Tracked& t) : count(t.count) { count[0] += 1; }
         constexpr Tracked(Tracked&& t)  noexcept : count(t.count) { count[1] += 1; }
         constexpr Tracked& operator=(const Tracked& t) { count[0] += t.count[0] + 1; return *this; }
    };

    struct Op {
        Tracked t1{};
        constexpr void operator()(const Tracked& t0, Vec<int, 2>& o0, Vec<int, 2>& o1) const {
            o0 = t0.count;
            o1 = t1.count;
        }
    };
}

TEST_CASE("cuda::ewise") {
    using namespace noa::types;
    using noa::cuda::ewise;
    using noa::cuda::EwiseConfig;
    using noa::cuda::Stream;
    using noa::cuda::Device;
    using noa::cuda::AllocatorManaged;

    Stream stream(Device::current());

    AND_THEN("check operator") {
        const auto shape = Shape4<i64>{1, 1, 1, 1};
        auto value = AllocatorManaged<Vec2<int>>::allocate(2, stream);
        Tuple<AccessorValue<Tracked>> input{};
        auto output_contiguous = noa::make_tuple(
                AccessorI64<Vec<int, 2>, 4>(value.get() + 0, shape.strides()),
                AccessorI64<Vec<int, 2>, 4>(value.get() + 1, shape.strides()));

        auto op0 = Op{};
        auto op1 = Op{};

        // Operator is copied once to the kernel
        ewise(shape, op0, input, output_contiguous, stream);
        stream.synchronize();
        REQUIRE((value[0][0] == 1 and value[0][1] == 1)); // input is copied once and moved once*
        REQUIRE((value[1][0] == 1 and value[1][1] == 0)); // operator is copied once
        //* in the contiguous case, the input/output are forwarded to a 1d accessor (+1 copy|move),
        //  which is then moved into the kernel

        ewise(shape, std::move(op0), std::move(input), output_contiguous, stream);
        stream.synchronize();
        REQUIRE((value[0][0] == 0 and value[0][1] == 2)); // input is moved twice
        REQUIRE((value[1][0] == 0 and value[1][1] == 1)); // operator is moved once

        // Create a non-contiguous case by broadcasting.
        auto shape_strided = Shape4<i64>{1, 1, 2, 1};
        auto output_strided = noa::make_tuple(
                AccessorI64<Vec<int, 2>, 4>(value.get() + 0, Strides4<i64>{}),
                AccessorI64<Vec<int, 2>, 4>(value.get() + 1, Strides4<i64>{}));

        ewise(shape_strided, op1, input, output_strided, stream);
        stream.synchronize();
        REQUIRE((value[0][0] == 1 and value[0][1] == 1)); // input is copied once*
        REQUIRE((value[1][0] == 1 and value[1][1] == 0)); // operator is copied once
        //* in the non-contiguous case, things are simpler, and the input/output are simply forwarded once

        ewise(shape_strided, std::move(op1), std::move(input), output_strided, stream);
        stream.synchronize();
        REQUIRE((value[0][0] == 0 and value[0][1] == 2)); // input is moved once
        REQUIRE((value[1][0] == 0 and value[1][1] == 1)); // operator is moved once
    }

    AND_THEN("simply fill and copy") {
        const auto shape = Shape4<i64>{1, 1, 1, 512}; // test::get_random_shape4_batched(4);
        const auto elements = shape.elements();
        const auto value = 3.1415;

        const auto buffer = AllocatorManaged<f64>::allocate(elements, stream);
        const auto expected = AllocatorManaged<f64>::allocate(elements, stream);
        std::fill_n(expected.get(), elements, value);

        {
            // This is not vectorized because the input is mutable.
            const auto input = noa::make_tuple(Accessor<f64, 4, i64>(buffer.get(), shape.strides()));
            ewise(shape, [value]__device__(f64& i) { i = value; }, input, Tuple<>{}, stream);
            stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS, buffer.get(), expected.get(), elements, 1e-8));

            std::fill_n(buffer.get(), elements, 0.);

            // This is vectorized.
            ewise(shape, [value]__device__(f64& i) { i = value; }, Tuple<>{}, input, stream);
            stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS, buffer.get(), expected.get(), elements, 1e-8));
        }

        {
            // This is vectorized.
            const auto input = noa::make_tuple(Accessor<const f64, 4, i64>(buffer.get(), shape.strides()));
            const auto output = noa::make_tuple(Accessor<f64, 4, i64>(expected.get(), shape.strides()));

            std::fill_n(expected.get(), elements, 0.);
            ewise(shape, []__device__(f64 i, f64& o) { o = i; }, input, output, stream);
            stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS, buffer.get(), expected.get(), elements, 1e-8));
        }

        {
            // Same as above, but not vectorized.
            const auto input = noa::make_tuple(Accessor<const f64, 4, i64>(buffer.get(), shape.strides()));
            const auto output = noa::make_tuple(Accessor<f64, 4, i64>(expected.get(), shape.strides()));

            std::fill_n(expected.get(), elements, 0.);

            using config = noa::cuda::EwiseConfig<true, true>;
            ewise<config>(shape, []__device__(Tuple<const f64&> i, Tuple<f64&> o) { o = i; }, input, output, stream);
            stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS, buffer.get(), expected.get(), elements, 1e-8));
        }

        {
            const auto input = noa::make_tuple(AccessorValue<const f64>(1.1));
            const auto output = noa::make_tuple(Accessor<f64, 4, i64>(buffer.get(), shape.strides()));

            using config = noa::cuda::EwiseConfig<true, false>;
            ewise<config>(shape, []__device__(Tuple<const f64&> i, f64& o) { o = i[Tag<0>{}]; }, input, output, stream);
            stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS, buffer.get(), 1.1, elements, 1e-8));
        }
    }
}
