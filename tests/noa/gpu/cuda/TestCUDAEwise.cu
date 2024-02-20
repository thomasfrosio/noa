#include <catch2/catch.hpp>
#include <noa/gpu/cuda/AllocatorManaged.hpp>
#include <noa/gpu/cuda/Ewise.cuh>

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
        REQUIRE((value[0][0] == 1 and value[0][1] == 1));
        REQUIRE((value[1][0] == 1 and value[1][1] == 0));

        ewise(shape, std::move(op0), std::move(input), output_contiguous, stream);
        stream.synchronize();
        REQUIRE((value[0][0] == 0 and value[0][1] == 2));
        REQUIRE((value[1][0] == 0 and value[1][1] == 1));

        // Create a non-contiguous case by broadcasting.
        auto shape_strided = Shape4<i64>{1, 1, 2, 1};
        auto output_strided = noa::make_tuple(
                AccessorI64<Vec<int, 2>, 4>(value.get() + 0, Strides4<i64>{}),
                AccessorI64<Vec<int, 2>, 4>(value.get() + 1, Strides4<i64>{}));

        ewise(shape_strided, op1, input, output_strided, stream);
        stream.synchronize();
        REQUIRE((value[0][0] == 1 and value[0][1] == 1));
        REQUIRE((value[1][0] == 1 and value[1][1] == 0));

        ewise(shape_strided, std::move(op1), std::move(input), output_strided, stream);
        stream.synchronize();
        REQUIRE((value[0][0] == 0 and value[0][1] == 2));
        REQUIRE((value[1][0] == 0 and value[1][1] == 1));
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

    AND_THEN("more complex example") {
        const auto shape = test::get_random_shape4_batched(4);
        const auto n_elements = shape.elements();
        const auto value = 3.1415;

        const auto b0 = AllocatorManaged<f32>::allocate(n_elements, stream);
        const auto b1 = AllocatorManaged<f64>::allocate(n_elements, stream);
        const auto b2 = AllocatorManaged<f16>::allocate(n_elements, stream);
        test::randomize(b0.get(), n_elements, test::Randomizer<f32>(-10, 10));

        // Generate expected.
        const auto e1 = AllocatorManaged<f64>::allocate(n_elements, stream);
        const auto e2 = AllocatorManaged<f16>::allocate(n_elements, stream);
        for (size_t i = 0; i < static_cast<size_t>(n_elements); ++i) {
            e1[i] = static_cast<f64>(b0[i] + static_cast<f32>(7));
            e2[i] = static_cast<f16>(e1[i]);
        }

        const auto input = noa::make_tuple(
                AccessorRestrictContiguousI32<f32, 4>(b0.get(), shape.as<i32>().strides()),
                AccessorValue<const i32>(7));
        const auto output = noa::make_tuple(
                AccessorI64<f64, 4>(b1.get(), shape.strides()),
                AccessorI64<f16, 4>(b2.get(), shape.strides())
        );

        // lhs+rhs->output, lhs->zero
        const auto op = [value]__device__(Tuple<f32&, const i32&> i, f64& o0, f16& o1) {
            auto& [lhs, rhs] = i;
            o0 = static_cast<f64>(lhs + static_cast<f32>(rhs));
            o1 = static_cast<f16>(o0);
            lhs = 0;
        };

        ewise<EwiseConfig<true, false>>(shape, op, input, output, stream);
        stream.synchronize();

        REQUIRE(test::Matcher(test::MATCH_ABS, b0.get(), 0.f, n_elements, 1e-8));
        REQUIRE(test::Matcher(test::MATCH_ABS, b1.get(), e1.get(), n_elements, 1e-8));
        REQUIRE(test::Matcher(test::MATCH_ABS, b2.get(), e2.get(), n_elements, 1e-2));
    }

    AND_THEN("more complex example, vectorized") {
        const auto shape = test::get_random_shape4_batched(4);
        const auto strides_u32 = shape.as<u32>().strides();
        const auto n_elements = shape.elements();

        const auto b0 = AllocatorManaged<f32>::allocate(n_elements, stream);
        const auto b1 = AllocatorManaged<f32>::allocate(n_elements, stream);
        const auto b2 = AllocatorManaged<f32>::allocate(n_elements, stream);
        const auto b3 = AllocatorManaged<i32>::allocate(n_elements, stream);
        test::randomize(b0.get(), n_elements, test::Randomizer<f32>(-10, 10));
        test::randomize(b1.get(), n_elements, test::Randomizer<f32>(-10, 10));

        // Generate expected.
        const auto e2 = AllocatorManaged<f32>::allocate(n_elements, stream);
        const auto e3 = AllocatorManaged<i32>::allocate(n_elements, stream);
        for (size_t i = 0; i < static_cast<size_t>(n_elements); ++i) {
            e2[i] = b0[i] + b1[i];
            e3[i] = static_cast<i32>(b0[i] - b1[i]);
        }

        const auto input = noa::make_tuple(
                AccessorRestrictContiguousU32<const f32, 4>(b0.get(), strides_u32),
                AccessorRestrictContiguousU32<const f32, 4>(b1.get(), strides_u32)
        );
        const auto output = noa::make_tuple(
                AccessorRestrictContiguousU32<f32, 4>(b2.get(), strides_u32),
                AccessorRestrictContiguousU32<i32, 4>(b3.get(), strides_u32)
        );

        const auto op = []__device__(const f32& i0, const f32& i1, Tuple<f32&, i32&> o) {
            auto& [lhs, rhs] = o;
            lhs = i0 + i1;
            rhs = static_cast<i32>(i0 - i1);
        };

        ewise<EwiseConfig<false, true>>(shape, op, input, output, stream); // expected vec size of 4
        stream.synchronize();

        REQUIRE(test::Matcher(test::MATCH_ABS, b2.get(), e2.get(), n_elements, 1e-6));
        REQUIRE(test::Matcher(test::MATCH_ABS, b3.get(), e3.get(), n_elements, 1e-6));
    }
}
