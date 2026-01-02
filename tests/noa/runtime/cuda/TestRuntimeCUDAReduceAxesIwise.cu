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

        NOA_HD void init(const Vec<isize, N>& indices, isize& reduced) const {
            reduced += accessor(indices);
        }
        NOA_HD static void join(isize to_reduce, isize& reduced) {
            reduced += to_reduce;
        }
        NOA_HD static void final(isize reduced, isize& output) {
            output += reduced;
        }
    };

    template<size_t N>
    struct SumOp2 {
        SpanContiguous<const isize, N> span;

        NOA_HD void init(const Vec<isize, N>& indices, isize& reduced) const {
            reduced += span(indices);
        }
        NOA_HD static void join(isize to_reduce, isize& reduced) {
            reduced += to_reduce;
        }
        NOA_HD static void final(isize reduced, isize& output) {
            output += reduced;
        }
    };
}

TEST_CASE("runtime::cuda::reduce_axes_iwise - 4d") {
    using namespace noa::types;
    using noa::cuda::reduce_axes_iwise;
    using noa::cuda::AllocatorManaged;
    using noa::cuda::Device;
    using noa::cuda::Stream;

    Stream stream(Device{});

    SECTION("sum one axis") {
        auto input_shape = test::random_shape_batched(4);
        input_shape[0] += 1; // make sure there's something to reduce
        const auto input_strides = input_shape.strides();
        const auto n_elements = input_shape.n_elements();

        const auto buffer = AllocatorManaged::allocate<isize>(n_elements, stream);
        test::arange(buffer.get(), n_elements);

        auto reduced = noa::make_tuple(noa::AccessorValue<isize>(0));
        auto sum_op = SumOp<4>{.accessor={buffer.get(), input_strides}};

        for (auto axis: noa::irange(4)) {
            INFO("axis=" << axis << ", shape=" << input_shape);
            auto output_shape = input_shape;
            output_shape[axis] = 1;
            const auto output_strides = output_shape.strides();
            const auto output_elements = output_shape.n_elements();

            const auto expected_buffer = AllocatorManaged::allocate<isize>(output_elements, stream);
            auto compute_expected_reduction = [](Shape<isize, 4> shape, const auto& input, const auto& expected) {
                for (isize i = 0; i < shape[0]; ++i) {
                    for (isize j = 0; j < shape[1]; ++j) {
                        for (isize k = 0; k < shape[2]; ++k) {
                            isize tmp = 0;
                            for (isize l = 0; l < shape[3]; ++l)
                                tmp += input(i, j, k, l);
                            expected(i, j, k) = tmp + 1; // +1: add to output
                        }
                    }
                }
            };
            using expected_t = noa::Accessor<const isize, 4>;
            if (axis == 0) {
                compute_expected_reduction(
                    input_shape.filter(1, 2, 3, 0),
                    expected_t(buffer.get(), input_strides.filter(1, 2, 3, 0)),
                    noa::Accessor<isize, 3>(expected_buffer.get(), output_strides.filter(1, 2, 3)));
            } else if (axis == 1) {
                compute_expected_reduction(
                    input_shape.filter(0, 2, 3, 1),
                    expected_t(buffer.get(), input_strides.filter(0, 2, 3, 1)),
                    noa::Accessor<isize, 3>(expected_buffer.get(), output_strides.filter(0, 2, 3)));
            } else if (axis == 2) {
                compute_expected_reduction(
                    input_shape.filter(0, 1, 3, 2),
                    expected_t(buffer.get(), input_strides.filter(0, 1, 3, 2)),
                    noa::Accessor<isize, 3>(expected_buffer.get(), output_strides.filter(0, 1, 3)));
            } else {
                compute_expected_reduction(
                    input_shape,
                    expected_t(buffer.get(), input_strides),
                    noa::Accessor<isize, 3>(expected_buffer.get(), output_strides.filter(0, 1, 2)));
            }

            const auto output_buffer = AllocatorManaged::allocate<isize>(output_elements, stream);
            auto output = noa::make_tuple(noa::Accessor<isize, 4>(output_buffer.get(), output_strides));
            std::fill(output_buffer.get(), output_buffer.get() + output_elements, 1);
            reduce_axes_iwise(input_shape, output_shape, sum_op, reduced, output, stream);
            stream.synchronize();
            REQUIRE(test::allclose_abs(output_buffer.get(), expected_buffer.get(), output_elements, 0));
        }
    }

    SECTION("per batch") {
        const auto input_shapes = std::array{
            test::random_shape(1),
            test::random_shape(2),
            test::random_shape(3),
            test::random_shape(4),
            test::random_shape_batched(1),
            test::random_shape_batched(2),
            test::random_shape_batched(3),
            test::random_shape_batched(4)
        };
        for (const auto& input_shape: input_shapes) {
            const auto input_strides = input_shape.strides();
            const auto n_elements = input_shape.n_elements();

            const auto buffer = AllocatorManaged::allocate<isize>(n_elements, stream);
            test::arange(buffer.get(), n_elements);

            auto reduced = noa::make_tuple(noa::AccessorValue<isize>(0));
            auto sum_op = SumOp<4>{.accessor={buffer.get(), input_strides}};

            const auto expected_buffer = AllocatorManaged::allocate<isize>(input_shape[0], stream);
            const auto per_batch_n_elements = input_shape.pop_front().n_elements();
            for (isize i = 0; i < input_shape[0]; ++i) {
                const isize* input = buffer.get() + input_strides[0] * i;
                isize tmp = 0;
                for (isize j = 0; j < per_batch_n_elements; ++j)
                    tmp += input[j];
                expected_buffer.get()[i] = tmp + 1;
            }

            const auto output_shape = Shape4{input_shape[0], 1, 1, 1};
            const auto output_elements = output_shape.n_elements();
            const auto output_buffer = AllocatorManaged::allocate<isize>(output_elements, stream);
            const auto output = noa::make_tuple(noa::Accessor<isize, 4>(output_buffer.get(), output_shape.strides()));

            std::fill(output_buffer.get(), output_buffer.get() + output_elements, 1);
            reduce_axes_iwise(input_shape, output_shape, sum_op, reduced, output, stream);
            stream.synchronize();
            REQUIRE(test::allclose_abs(output_buffer.get(), expected_buffer.get(), output_elements, 0));
        }
    }
}

TEST_CASE("runtime::cuda::reduce_axes_iwise - 3d") {
    using namespace noa::types;
    using noa::cuda::reduce_axes_iwise;
    using noa::cuda::AllocatorManaged;
    using noa::cuda::Device;
    using noa::cuda::Stream;

    Stream stream(Device{});

    SECTION("sum one axis") {
        const auto input_shape = test::random_shape<isize, 3>(3);
        const auto input_strides = input_shape.strides();
        const auto n_elements = input_shape.n_elements();

        const auto buffer = AllocatorManaged::allocate<isize>(n_elements, stream);
        test::arange(buffer.get(), n_elements);

        auto reduced = noa::make_tuple(noa::AccessorValue<isize>(0));
        auto sum_op = SumOp<3>{.accessor={buffer.get(), input_strides}};

        for (auto axis: noa::irange(3)) {
            INFO("axis=" << axis << ", shape=" << input_shape);
            auto output_shape = input_shape;
            output_shape[axis] = 1;
            const auto output_strides = output_shape.strides();
            const auto output_elements = output_shape.n_elements();

            const auto expected_buffer = AllocatorManaged::allocate<isize>(output_elements, stream);
            auto compute_expected_reduction = [](Shape<isize, 3> shape, const auto& input, const auto& expected) {
                for (isize i = 0; i < shape[0]; ++i) {
                    for (isize j = 0; j < shape[1]; ++j) {
                        isize tmp = 0;
                        for (isize k = 0; k < shape[2]; ++k)
                            tmp += input(i, j, k);
                        expected(i, j) = tmp + 1; // +1: add to output
                    }
                }
            };
            using expected_t = noa::Accessor<const isize, 3>;
            if (axis == 0) {
                compute_expected_reduction(
                    input_shape.filter(1, 2, 0),
                    expected_t(buffer.get(), input_strides.filter(1, 2, 0)),
                    noa::Accessor<isize, 2>(expected_buffer.get(), output_strides.filter(1, 2)));
            } else if (axis == 1) {
                compute_expected_reduction(
                    input_shape.filter(0, 2, 1),
                    expected_t(buffer.get(), input_strides.filter(0, 2, 1)),
                    noa::Accessor<isize, 2>(expected_buffer.get(), output_strides.filter(0, 2)));
            } else {
                compute_expected_reduction(
                    input_shape,
                    expected_t(buffer.get(), input_strides),
                    noa::Accessor<isize, 2>(expected_buffer.get(), output_strides.filter(0, 1)));
            }

            const auto output_buffer = AllocatorManaged::allocate<isize>(output_elements, stream);
            auto output = noa::make_tuple(noa::Accessor<isize, 3>(output_buffer.get(), output_strides));

            std::fill(output_buffer.get(), output_buffer.get() + output_elements, 1);
            reduce_axes_iwise(input_shape, output_shape, sum_op, reduced, output, stream);
            stream.synchronize();
            REQUIRE(test::allclose_abs(output_buffer.get(), expected_buffer.get(), output_elements, 0));
        }
    }

    SECTION("per batch") {
        const auto input_shapes = std::array{
            test::random_shape<isize, 3>(1),
            test::random_shape<isize, 3>(2),
            test::random_shape<isize, 3>(3),
            test::random_shape<isize, 3>(4),
        };
        for (const auto& input_shape: input_shapes) {
            const auto input_strides = input_shape.strides();
            const auto n_elements = input_shape.n_elements();

            const auto buffer = AllocatorManaged::allocate<isize>(n_elements, stream);
            test::arange(buffer.get(), n_elements);

            auto reduced = noa::make_tuple(noa::AccessorValue<isize>(0));
            auto sum_op = SumOp<3>{.accessor={buffer.get(), input_strides}};

            const auto expected_buffer = AllocatorManaged::allocate<isize>(input_shape[0], stream);
            const auto per_batch_n_elements = input_shape.pop_front().n_elements();
            for (isize i = 0; i < input_shape[0]; ++i) {
                const isize* input = buffer.get() + input_strides[0] * i;
                isize tmp = 0;
                for (isize j = 0; j < per_batch_n_elements; ++j)
                    tmp += input[j];
                expected_buffer.get()[i] = tmp + 1;
            }

            const auto output_shape = Shape<isize, 3>{input_shape[0], 1, 1};
            const auto output_elements = output_shape.n_elements();
            const auto output_buffer = AllocatorManaged::allocate<isize>(output_elements, stream);
            const auto output = noa::make_tuple(noa::Accessor<isize, 3>(output_buffer.get(), output_shape.strides()));

            std::fill(output_buffer.get(), output_buffer.get() + output_elements, 1);
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

TEST_CASE("runtime::cuda::reduce_axes_iwise - 2d") {
    using namespace noa::types;
    using noa::cuda::reduce_axes_iwise;
    using noa::cuda::AllocatorManaged;
    using noa::cuda::Device;
    using noa::cuda::Stream;

    Stream stream(Device{});

    SECTION("sum one axis") {
        const auto input_shape = test::random_shape<isize, 2>(2);
        const auto input_strides = input_shape.strides();
        const auto n_elements = input_shape.n_elements();

        const auto buffer = AllocatorManaged::allocate<isize>(n_elements, stream);
        test::arange(buffer.get(), n_elements);

        auto reduced = noa::make_tuple(noa::AccessorValue<isize>(0));
        auto sum_op = SumOp<2>{.accessor={buffer.get(), input_strides}};

        for (auto axis: noa::irange(2)) {
            INFO("axis=" << axis << ", shape=" << input_shape);
            auto output_shape = input_shape;
            output_shape[axis] = 1;
            const auto output_strides = output_shape.strides();
            const auto output_elements = output_shape.n_elements();

            const auto expected_buffer = AllocatorManaged::allocate<isize>(output_elements, stream);
            auto compute_expected_reduction = [](Shape<isize, 2> shape, const auto& input, const auto& expected) {
                for (isize i = 0; i < shape[0]; ++i) {
                    isize tmp = 0;
                    for (isize j = 0; j < shape[1]; ++j)
                        tmp += input(i, j);
                    expected(i) = tmp + 1; // +1: add to output
                }
            };
            using expected_t = noa::Accessor<const isize, 2>;
            if (axis == 0) {
                compute_expected_reduction(
                    input_shape.filter(1, 0),
                    expected_t(buffer.get(), input_strides.filter(1, 0)),
                    noa::Accessor<isize, 1>(expected_buffer.get(), output_strides.filter(1)));
            } else {
                compute_expected_reduction(
                    input_shape,
                    expected_t(buffer.get(), input_strides),
                    noa::Accessor<isize, 1>(expected_buffer.get(), output_strides.filter(0)));
            }

            const auto output_buffer = AllocatorManaged::allocate<isize>(output_elements, stream);
            auto output = noa::make_tuple(noa::Accessor<isize, 2>(output_buffer.get(), output_strides));

            std::fill_n(output_buffer.get(), output_elements, 1);
            reduce_axes_iwise(input_shape, output_shape, sum_op, reduced, output, stream);
            stream.synchronize();
            REQUIRE(test::allclose_abs(output_buffer.get(), expected_buffer.get(), output_elements, 0));
        }
    }
}

TEST_CASE("runtime::cuda::reduce_axes_iwise - 1d") {
    using namespace noa::types;
    using noa::cuda::reduce_axes_iwise;
    using noa::cuda::AllocatorManaged;
    using noa::cuda::Device;
    using noa::cuda::Stream;

    Stream stream(Device{});

    const auto input_shape = test::random_shape<isize, 1>(1) * 50;
    const auto input_strides = input_shape.strides();
    const auto n_elements = input_shape.n_elements();

    const auto buffer = AllocatorManaged::allocate<isize>(n_elements, stream);
    test::arange(buffer.get(), n_elements);

    auto reduced = noa::make_tuple(noa::AccessorValue<isize>(0));
    auto sum_op = SumOp<1>{.accessor={buffer.get(), input_strides}};

    auto output_shape = Shape1{1};

    isize expected{1};
    for (isize i = 0; i < input_shape[0]; ++i)
        expected += sum_op.accessor(i);

    auto output = AllocatorManaged::allocate<isize>(1, stream);
    output[0] = 1;
    auto output_accessor = noa::make_tuple(noa::AccessorContiguous<isize, 1>(output.get()));
    reduce_axes_iwise(input_shape, output_shape, sum_op, reduced, output_accessor, stream);
    stream.synchronize();
    REQUIRE(expected == output[0]);
}
