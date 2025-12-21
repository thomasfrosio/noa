#include <noa/runtime/core/Ewise.hpp>
#include <noa/runtime/cuda/Allocators.hpp>
#include <noa/runtime/cuda/Ewise.cuh>

#include "Catch.hpp"
#include "Utils.hpp"

namespace {
    using namespace noa::types;

    struct Tracked {
        Vec<i32, 2> count{};
         constexpr Tracked() = default;
         constexpr Tracked(const Tracked& t) : count(t.count) { count[0] += 1; }
         constexpr Tracked(Tracked&& t) noexcept : count(t.count) { count[1] += 1; }
    };

    struct Op {
        Tracked t1{};
        constexpr void operator()(const Tracked& t0, Vec<i32, 2>& o0, Vec<i32, 2>& o1) const {
            o0 = t0.count;
            o1 = t1.count;
        }
    };

    struct Op1 {
        using enable_vectorization = bool;
        constexpr void operator()(const f32& i0, const f32& i1, Tuple<f32&, i32&> o) const {
            auto& [lhs, rhs] = o;
            lhs = i0 + i1;
            rhs = static_cast<i32>(i0 - i1);
        };
    };
}

TEST_CASE("runtime::cuda::ewise") {
    using namespace noa::types;
    using noa::cuda::ewise;
    using noa::cuda::EwiseConfig;
    using noa::cuda::Stream;
    using noa::cuda::Device;
    using noa::cuda::AllocatorManaged;

    Stream stream(Device::current());

    SECTION("check operator") {
        constexpr auto shape = Shape<i64, 4>{1, 1, 1, 1};
        auto value = AllocatorManaged::allocate<Vec<i32, 2>>(2, stream);
        Tuple<AccessorValue<Tracked>> input{};
        auto output_contiguous = noa::make_tuple(
            AccessorI64<Vec<i32, 2>, 4>(value.get() + 0, shape.strides()),
            AccessorI64<Vec<i32, 2>, 4>(value.get() + 1, shape.strides()));

        auto op0 = Op{};
        auto op1 = Op{};

        // Operator is copied once to the kernel
        ewise(shape, op0, input, output_contiguous, stream);
        stream.synchronize();
        REQUIRE((value[0][0] == 2 and value[0][1] == 0));
        REQUIRE((value[1][0] == 1 and value[1][1] == 0));

        ewise(shape, std::move(op0), std::move(input), output_contiguous, stream);
        stream.synchronize();
        REQUIRE((value[0][0] == 1 and value[0][1] == 1));
        REQUIRE((value[1][0] == 1 and value[1][1] == 0));

        // Create a non-contiguous case by broadcasting.
        auto shape_strided = Shape<i64, 4>{1, 1, 2, 1};
        auto output_strided = noa::make_tuple(
            AccessorI64<Vec<i32, 2>, 4>(value.get() + 0, Strides<i64, 4>{}),
            AccessorI64<Vec<i32, 2>, 4>(value.get() + 1, Strides<i64, 4>{}));

        ewise(shape_strided, op1, input, output_strided, stream);
        stream.synchronize();
        REQUIRE((value[0][0] == 2 and value[0][1] == 1));
        REQUIRE((value[1][0] == 1 and value[1][1] == 0));

        ewise(shape_strided, std::move(op1), std::move(input), output_strided, stream);
        stream.synchronize();
        REQUIRE((value[0][0] == 1 and value[0][1] == 2));
        REQUIRE((value[1][0] == 1 and value[1][1] == 0));
    }

    SECTION("simply fill and copy") {
        const auto shape = test::random_shape<i64, 4>(4);
        const auto elements = shape.n_elements();
        constexpr auto value = 3.1415;

        const auto buffer = AllocatorManaged::allocate<f64>(elements, stream);
        const auto expected = AllocatorManaged::allocate<f64>(elements, stream);
        std::fill_n(expected.get(), elements, value);

        {
            // This is not vectorized.
            const auto input = noa::make_tuple(Accessor<f64, 4, i64>(buffer.get(), shape.strides()));
            ewise(shape, [value]__device__(f64& i) { i = value; }, input, Tuple{}, stream);
            stream.synchronize();
            REQUIRE(test::allclose_abs(buffer.get(), expected.get(), elements, 1e-8));

            std::fill_n(buffer.get(), elements, 0.);

            // This is vectorized.
            ewise(shape, noa::Fill{value}, Tuple{}, input, stream);
            stream.synchronize();
            REQUIRE(test::allclose_abs(buffer.get(), expected.get(), elements, 1e-8));
        }

        {
            // This is vectorized.
            const auto input = noa::make_tuple(Accessor<const f64, 4, i64>(buffer.get(), shape.strides()));
            const auto output = noa::make_tuple(Accessor<f64, 4, i64>(expected.get(), shape.strides()));

            std::fill_n(expected.get(), elements, 0.);
            ewise(shape, noa::Copy{}, input, output, stream);
            stream.synchronize();
            REQUIRE(test::allclose_abs(buffer.get(), expected.get(), elements, 1e-8));
        }

        {
            // Same as above, but not vectorized.
            const auto input = noa::make_tuple(Accessor<const f64, 4, i64>(buffer.get(), shape.strides()));
            const auto output = noa::make_tuple(Accessor<f64, 4, i64>(expected.get(), shape.strides()));

            std::fill_n(expected.get(), elements, 0.);

            using config = EwiseConfig<true, true>;
            ewise<config>(shape, []__device__(Tuple<const f64&> i, Tuple<f64&> o) { o[Tag<0>{}] = i[Tag<0>{}]; }, input, output, stream);
            stream.synchronize();
            REQUIRE(test::allclose_abs(buffer.get(), expected.get(), elements, 1e-8));
        }

        {
            constexpr auto input = noa::make_tuple(AccessorValue<const f64>(1.1));
            const auto output = noa::make_tuple(Accessor<f64, 4, i64>(buffer.get(), shape.strides()));

            using config = EwiseConfig<true, false>;
            ewise<config>(shape, []__device__(Tuple<const f64&> i, f64& o) { o = i[Tag<0>{}]; }, input, output, stream);
            stream.synchronize();
            REQUIRE(test::allclose_abs(buffer.get(), 1.1, elements, 1e-8));
        }
    }

    SECTION("more complex example") {
        const auto shape = test::random_shape_batched(4);
        const auto n_elements = shape.n_elements();

        const auto b0 = AllocatorManaged::allocate<f32>(n_elements, stream);
        const auto b1 = AllocatorManaged::allocate<f64>(n_elements, stream);
        const auto b2 = AllocatorManaged::allocate<f16>(n_elements, stream);
        test::randomize(b0.get(), n_elements, test::Randomizer<f32>(-10, 10));

        // Generate expected.
        const auto e1 = AllocatorManaged::allocate<f64>(n_elements, stream);
        const auto e2 = AllocatorManaged::allocate<f16>(n_elements, stream);
        for (size_t i{}; i < static_cast<size_t>(n_elements); ++i) {
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
        const auto op = []__device__(Tuple<f32&, const i32&> i, f64& o0, f16& o1) {
            auto& [lhs, rhs] = i;
            o0 = static_cast<f64>(lhs + static_cast<f32>(rhs));
            o1 = static_cast<f16>(o0);
            lhs = 0;
        };

        ewise<EwiseConfig<true, false>>(shape, op, input, output, stream);
        stream.synchronize();

        REQUIRE(test::allclose_abs(b0.get(), 0.f, n_elements, 1e-8));
        REQUIRE(test::allclose_abs(b1.get(), e1.get(), n_elements, 1e-8));
        REQUIRE(test::allclose_abs(b2.get(), e2.get(), n_elements, 1e-2));
    }

    SECTION("more complex example, vectorized") {
        const auto shape = test::random_shape_batched(4);
        const auto strides_u32 = shape.as<u32>().strides();
        const auto n_elements = shape.n_elements();

        const auto b0 = AllocatorManaged::allocate<f32>(n_elements, stream);
        const auto b1 = AllocatorManaged::allocate<f32>(n_elements, stream);
        const auto b2 = AllocatorManaged::allocate<f32>(n_elements, stream);
        const auto b3 = AllocatorManaged::allocate<i32>(n_elements, stream);
        test::randomize(b0.get(), n_elements, test::Randomizer<f32>(-10, 10));
        test::randomize(b1.get(), n_elements, test::Randomizer<f32>(-10, 10));

        // Generate expected.
        const auto e2 = AllocatorManaged::allocate<f32>(n_elements, stream);
        const auto e3 = AllocatorManaged::allocate<i32>(n_elements, stream);
        for (size_t i{}; i < static_cast<size_t>(n_elements); ++i) {
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

        ewise<EwiseConfig<false, true>>(shape, Op1{}, input, output, stream); // expected vec size of 4
        stream.synchronize();

        REQUIRE(test::allclose_abs(b2.get(), e2.get(), n_elements, 1e-6));
        REQUIRE(test::allclose_abs(b3.get(), e3.get(), n_elements, 1e-6));
    }
}

TEMPLATE_TEST_CASE("runtime::cuda::ewise - copy", "", i8, i16, i32, i64, c16, c32, c64) {
    using namespace noa::types;
    using noa::cuda::ewise;
    using noa::cuda::EwiseConfig;
    using noa::cuda::Stream;
    using noa::cuda::Device;
    using noa::cuda::AllocatorManaged;
    using value_t = TestType;

    Stream stream(Device::current());

    const auto shapes = std::array{
        Shape<i64, 4>{1, 1, 1, 512},
        Shape<i64, 4>{2, 6, 40, 65},
        test::random_shape_batched(1),
        test::random_shape_batched(2),
        test::random_shape_batched(3),
        test::random_shape_batched(4),
    };

    for (const auto& shape: shapes) {
        const auto n_elements = shape.n_elements();

        const auto buffer = AllocatorManaged::allocate<value_t>(n_elements, stream);
        const auto expected = AllocatorManaged::allocate<value_t>(n_elements, stream);
        test::randomize(buffer.get(), n_elements, test::Randomizer<value_t>(-128, 127));

        auto input = noa::make_tuple(Accessor<const value_t, 4, i64>(buffer.get(), shape.strides()));
        auto output = noa::make_tuple(Accessor<value_t, 4, i64>(expected.get(), shape.strides()));

        ewise(shape, noa::Copy{}, input, output, stream);
        stream.synchronize();
        REQUIRE(test::allclose_abs(buffer.get(), expected.get(), n_elements, 1e-6));

        test::fill(buffer.get(), n_elements, value_t{});
        ewise(shape, []__device__(auto i, auto& o) { o = i; }, input, output, stream);
        stream.synchronize();
        REQUIRE(test::allclose_abs(buffer.get(), expected.get(), n_elements, 1e-6));

        // Trigger strided implementation.
        input[Tag<0>{}].strides()[0] = 0;
        output[Tag<0>{}].strides()[0] = 0;

        test::fill(buffer.get(), n_elements, value_t{});
        ewise(shape, noa::Copy{}, input, output, stream);
        stream.synchronize();
        REQUIRE(test::allclose_abs(buffer.get(), expected.get(), n_elements, 1e-6));
    }
}

TEST_CASE("runtime::cuda::ewise - multi-grid - 2d") {
    using namespace noa::types;
    using namespace noa::cuda;
    namespace ni = noa::indexing;

    const auto shape = Shape<i64, 4>{140'000, 1, 1, 512};
    const auto n_elements = shape.n_elements();

    auto stream = Stream(Device::current());
    const auto buffer = AllocatorManaged::allocate<f32>(n_elements, stream);
    test::fill(buffer.get(), n_elements, 0.f);

    const auto original = Span(buffer.get(), shape);

    {
        const auto strided = original.subregion(ni::Slice{0, shape[0], 2});
        const auto accessors = noa::make_tuple(AccessorI64<f32, 4>(strided.get(), strided.strides_full()));
        ewise(strided.shape(), []NOA_DEVICE(f32& e){ e += 1; }, accessors, Tuple{}, stream);
    }
    {
        const auto strided = original.subregion(ni::Slice{1, shape[0], 2});
        const auto accessors = noa::make_tuple(AccessorI64<f32, 4>(strided.get(), strided.strides_full()));
        ewise(strided.shape(), []NOA_DEVICE(f32& e){ e += 1; }, accessors, Tuple{}, stream);
    }

    stream.synchronize();
    REQUIRE(test::allclose_abs(buffer.get(), 1.f, n_elements, 1e-8));
}

TEST_CASE("runtime::cuda::ewise - multi-grid - 4d") {
    using namespace noa::types;
    using namespace noa::cuda;
    namespace ni = noa::indexing;

    const auto shape = Shape<i64, 4>{1, 250'000, 1, 512};
    const auto n_elements = shape.n_elements();

    auto stream = Stream(Device::current());
    const auto buffer = AllocatorManaged::allocate<f32>(n_elements, stream);
    test::fill(buffer.get(), n_elements, 0.f);

    const auto original = Span(buffer.get(), shape);

    {
        const auto strided = original.subregion(ni::Full{}, ni::Slice{0, shape[1], 2});
        const auto accessors = noa::make_tuple(AccessorI64<f32, 4>(strided.get(), strided.strides_full()));
        ewise(strided.shape(), []NOA_DEVICE(f32& e){ e += 1; }, accessors, Tuple{}, stream);
    }
    {
        const auto strided = original.subregion(ni::Full{}, ni::Slice{1, shape[1], 2});
        const auto accessors = noa::make_tuple(AccessorI64<f32, 4>(strided.get(), strided.strides_full()));
        ewise(strided.shape(), []NOA_DEVICE(f32& e){ e += 1; }, accessors, Tuple{}, stream);
    }

    stream.synchronize();
    REQUIRE(test::allclose_abs(buffer.get(), 1.f, n_elements, 1e-8));
}
