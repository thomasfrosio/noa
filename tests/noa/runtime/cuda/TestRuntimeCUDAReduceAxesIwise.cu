#include <noa/runtime/core/Reduce.hpp>
#include <noa/runtime/core/Accessor.hpp>
#include <noa/runtime/cuda/ReduceAxesIwise.cuh>
#include <noa/runtime/cuda/Allocators.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

namespace {
    using namespace noa::types;

    template<size_t N>
    struct SumOp {
        noa::AccessorContiguous<const isize, N> accessor;

        NOA_HD void operator()(const Vec<isize, N>& indices, isize& reduced) const {
            reduced += accessor(indices);
        }
        NOA_HD static void join(isize to_reduce, isize& reduced) {
            reduced += to_reduce;
        }
        NOA_HD static void post(isize reduced, isize& output) {
            output += reduced;
        }
    };

    template<size_t N>
    struct SumOp2 {
        SpanContiguous<const isize, N> span;

        NOA_HD void operator()(const Vec<isize, N>& indices, isize& reduced) const {
            reduced += span(indices);
        }
        NOA_HD static void join(isize to_reduce, isize& reduced) {
            reduced += to_reduce;
        }
        NOA_HD static void post(isize reduced, isize& output) {
            output += reduced;
        }
    };
}

TEMPLATE_TEST_CASE("runtime::cuda::reduce_axes_iwise", "", Tag<1>, Tag<2>, Tag<3>, Tag<4>, Tag<5>, Tag<6>) {
    using namespace noa::types;
    using noa::cuda::reduce_axes_iwise;
    using noa::cuda::AllocatorManaged;
    using noa::cuda::Device;
    using noa::cuda::Stream;
    constexpr usize N = TestType::value;;

    Stream stream(Device{});

    SECTION("sum one axis") {
        auto input_shape = test::random_shape_batched<isize, N>(3, {.batch_range = {2, 6}});
        const auto input_strides = input_shape.strides();
        const auto n_elements = input_shape.n_elements();

        const auto buffer = AllocatorManaged::allocate<isize>(n_elements, stream);
        test::arange(buffer.get(), n_elements);

        auto reduced = noa::make_tuple(noa::AccessorValue<isize>(0));
        auto sum_op = SumOp<N>{.accessor={buffer.get(), input_strides}};

        for (auto axis: noa::irange(N)) {
            INFO("axis=" << axis << ", shape=" << input_shape);
            auto output_shape = input_shape;
            output_shape[axis] = 1;
            const auto output_strides = output_shape.strides();
            const auto output_elements = output_shape.n_elements();

            const auto expected_buffer = AllocatorManaged::allocate<isize>(output_elements, stream);
            auto compute_expected_reduction = [](Shape<isize, N> shape, const auto& input, const auto& expected) {
                auto reduce = [](const auto& src, auto& dst, isize n) {
                    isize tmp = 0;
                    for (isize i = 0; i < n; ++i)
                        tmp += src[i];
                    dst = tmp + 1; // +1: add to output
                };
                if constexpr (N == 1) {
                    reduce(input, expected(0), shape[0]);
                } else if constexpr (N == 2) {
                    for (isize i = 0; i < shape[0]; ++i)
                        reduce(input[i], expected(i), shape[1]);
                } else if constexpr (N == 3) {
                    for (isize i = 0; i < shape[0]; ++i)
                        for (isize j = 0; j < shape[1]; ++j)
                            reduce(input[i][j], expected(i, j), shape[2]);
                } else if constexpr (N == 4) {
                    for (isize i = 0; i < shape[0]; ++i)
                        for (isize j = 0; j < shape[1]; ++j)
                            for (isize k = 0; k < shape[2]; ++k)
                                reduce(input[i][j][k], expected(i, j, k), shape[3]);
                } else if constexpr (N == 5) {
                    for (isize i = 0; i < shape[0]; ++i)
                        for (isize j = 0; j < shape[1]; ++j)
                            for (isize k = 0; k < shape[2]; ++k)
                                for (isize l = 0; l < shape[3]; ++l)
                                    reduce(input[Vec{i, j, k, l}], expected(i, j, k, l), shape[4]);
                } else {
                    for (isize i = 0; i < shape[0]; ++i)
                        for (isize j = 0; j < shape[1]; ++j)
                            for (isize k = 0; k < shape[2]; ++k)
                                for (isize l = 0; l < shape[3]; ++l)
                                    for (isize m = 0; m < shape[4]; ++m)
                                        reduce(input[Vec{i, j, k, l, m}], expected(i, j, k, l, m), shape[5]);
                }
            };
            using expected_t = noa::Accessor<const isize, N>;

            auto axis_to_reduce = Vec<i32, N>::from_value(1);
            axis_to_reduce[axis] = 2;
            auto input_order = noa::squeeze_empty_dimensions_left(axis_to_reduce);
            auto expected_order = input_order.template pop_back<N == 1 ? 0 : 1>();

            compute_expected_reduction(
                input_shape.filter(input_order),
                expected_t(buffer.get(), input_strides.filter(input_order)),
                noa::Accessor<isize, (N == 1 ? 1 : N - 1)>(expected_buffer.get(), output_strides.filter(expected_order))
            );

            const auto output_buffer = AllocatorManaged::allocate<isize>(output_elements, stream);
            auto output = noa::make_tuple(noa::Accessor<isize, N>(output_buffer.get(), output_strides));
            std::fill_n(output_buffer.get(), output_elements, 1);
            reduce_axes_iwise(input_shape, output_shape, sum_op, reduced, output, stream);
            stream.synchronize();
            REQUIRE(test::allclose_abs(output_buffer.get(), expected_buffer.get(), output_elements, 0));
        }
    }

    SECTION("per batch") {
        const auto input_shapes = std::array{
            test::random_shape<isize, N>(1),
            test::random_shape<isize, N>(2),
            test::random_shape<isize, N>(3),
            test::random_shape_batched<isize, N>(1),
            test::random_shape_batched<isize, N>(2),
            test::random_shape_batched<isize, N>(3)
        };
        for (const auto& input_shape: input_shapes) {
            const auto input_strides = input_shape.strides();
            const auto n_elements = input_shape.n_elements();

            const auto buffer = AllocatorManaged::allocate<isize>(n_elements, stream);
            test::arange(buffer.get(), n_elements);

            auto reduced = noa::make_tuple(noa::AccessorValue<isize>(0));
            auto sum_op = SumOp<N>{.accessor={buffer.get(), input_strides}};

            const auto expected_buffer = AllocatorManaged::allocate<isize>(input_shape[0], stream);
            const auto per_batch_n_elements = input_shape.pop_front().n_elements();
            for (isize i = 0; i < input_shape[0]; ++i) {
                const isize* input = buffer.get() + input_strides[0] * i;
                isize tmp = 0;
                for (isize j = 0; j < per_batch_n_elements; ++j)
                    tmp += input[j];
                expected_buffer.get()[i] = tmp + 1;
            }

            const auto output_shape = Shape<isize, N>::from_value(1).template set<0>(input_shape[0]);
            const auto output_elements = output_shape.n_elements();
            const auto output_buffer = AllocatorManaged::allocate<isize>(output_elements, stream);
            const auto output = noa::make_tuple(noa::Accessor<isize, N>(output_buffer.get(), output_shape.strides()));
            INFO(input_shape);

            std::fill_n(output_buffer.get(), output_elements, 1);
            reduce_axes_iwise(input_shape, output_shape, sum_op, reduced, output, stream);
            stream.synchronize();
            REQUIRE(test::allclose_abs(output_buffer.get(), expected_buffer.get(), output_elements, 0));
        }
    }

    SECTION("reduce height") {
        const auto input_shape = Shape<isize, 3>{70000, 32, 64};
        const auto output_shape = input_shape.set<1>(1);

        const auto input_buffer = AllocatorManaged::allocate<isize>(input_shape.n_elements(), stream);
        const auto input = Span(input_buffer.get(), input_shape);
        const auto op = SumOp2{.span=input.as_const()};
        std::ranges::fill(input.as_1d(), 1);

        const auto reduced = noa::make_tuple(noa::AccessorValue<isize>(0));

        const auto output_buffer = AllocatorManaged::allocate<isize>(output_shape.n_elements(), stream);
        const auto output = noa::Accessor<isize, 3>(output_buffer.get(), output_shape.strides());
        const auto output_tuple = noa::make_tuple(output);

        reduce_axes_iwise(input_shape, output_shape, op, reduced, output_tuple, stream);

        stream.synchronize();
        REQUIRE(test::allclose_abs(output_buffer.get(), input_shape[1], output_shape.n_elements(), 0));
    }
}
