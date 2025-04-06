#include <noa/core/types/Accessor.hpp>
#include <noa/gpu/cuda/ReduceAxesIwise.cuh>
#include <noa/gpu/cuda/Allocators.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

namespace {
    using namespace noa::types;

    template<size_t N>
    struct SumOp {
        AccessorContiguousI64<const i64, N> accessor;

        NOA_HD void init(const Vec<i64, N>& indices, i64& reduced) const {
            reduced += accessor(indices);
        }
        NOA_HD static void join(i64 to_reduce, i64& reduced) {
            reduced += to_reduce;
        }
        NOA_HD static void final(i64 reduced, i64& output) {
            output += reduced;
        }
    };
}

TEST_CASE("cuda::reduce_axes_iwise - 4d") {
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

        const auto buffer = AllocatorManaged<i64>::allocate(n_elements, stream);
        test::arange(buffer.get(), n_elements);

        auto reduced = noa::make_tuple(AccessorValue<i64>(0));
        auto sum_op = SumOp<4>{.accessor={buffer.get(), input_strides}};

        for (auto axis: noa::irange(4)) {
            INFO("axis=" << axis << ", shape=" << input_shape);
            auto output_shape = input_shape;
            output_shape[axis] = 1;
            const auto output_strides = output_shape.strides();
            const auto output_elements = output_shape.n_elements();

            const auto expected_buffer = AllocatorManaged<i64>::allocate(output_elements, stream);
            auto compute_expected_reduction = [](Shape4<i64> shape, const auto& input, const auto& expected) {
                for (i64 i = 0; i < shape[0]; ++i) {
                    for (i64 j = 0; j < shape[1]; ++j) {
                        for (i64 k = 0; k < shape[2]; ++k) {
                            i64 tmp = 0;
                            for (i64 l = 0; l < shape[3]; ++l)
                                tmp += input(i, j, k, l);
                            expected(i, j, k) = tmp + 1; // +1: add to output
                        }
                    }
                }
            };
            using expected_t = AccessorI64<const i64, 4>;
            if (axis == 0) {
                compute_expected_reduction(
                    input_shape.filter(1, 2, 3, 0),
                    expected_t(buffer.get(), input_strides.filter(1, 2, 3, 0)),
                    AccessorI64<i64, 3>(expected_buffer.get(), output_strides.filter(1, 2, 3)));
            } else if (axis == 1) {
                compute_expected_reduction(
                    input_shape.filter(0, 2, 3, 1),
                    expected_t(buffer.get(), input_strides.filter(0, 2, 3, 1)),
                    AccessorI64<i64, 3>(expected_buffer.get(), output_strides.filter(0, 2, 3)));
            } else if (axis == 2) {
                compute_expected_reduction(
                    input_shape.filter(0, 1, 3, 2),
                    expected_t(buffer.get(), input_strides.filter(0, 1, 3, 2)),
                    AccessorI64<i64, 3>(expected_buffer.get(), output_strides.filter(0, 1, 3)));
            } else {
                compute_expected_reduction(
                    input_shape,
                    expected_t(buffer.get(), input_strides),
                    AccessorI64<i64, 3>(expected_buffer.get(), output_strides.filter(0, 1, 2)));
            }

            const auto output_buffer = AllocatorManaged<i64>::allocate(output_elements, stream);
            auto output = noa::make_tuple(AccessorI64<i64, 4>(output_buffer.get(), output_strides));
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

            const auto buffer = AllocatorManaged<i64>::allocate(n_elements, stream);
            test::arange(buffer.get(), n_elements);

            auto reduced = noa::make_tuple(AccessorValue<i64>(0));
            auto sum_op = SumOp<4>{.accessor={buffer.get(), input_strides}};

            const auto expected_buffer = AllocatorManaged<i64>::allocate(input_shape[0], stream);
            const auto per_batch_n_elements = input_shape.pop_front().n_elements();
            for (i64 i = 0; i < input_shape[0]; ++i) {
                const i64* input = buffer.get() + input_strides[0] * i;
                i64 tmp = 0;
                for (i64 j = 0; j < per_batch_n_elements; ++j)
                    tmp += input[j];
                expected_buffer.get()[i] = tmp + 1;
            }

            const auto output_shape = Shape4<i64>{input_shape[0], 1, 1, 1};
            const auto output_elements = output_shape.n_elements();
            const auto output_buffer = AllocatorManaged<i64>::allocate(output_elements, stream);
            const auto output = noa::make_tuple(AccessorI64<i64, 4>(output_buffer.get(), output_shape.strides()));

            std::fill(output_buffer.get(), output_buffer.get() + output_elements, 1);
            reduce_axes_iwise(input_shape, output_shape, sum_op, reduced, output, stream);
            stream.synchronize();
            REQUIRE(test::allclose_abs(output_buffer.get(), expected_buffer.get(), output_elements, 0));
        }
    }
}

TEST_CASE("cuda::reduce_axes_iwise - 3d") {
    using namespace noa::types;
    using noa::cuda::reduce_axes_iwise;
    using noa::cuda::AllocatorManaged;
    using noa::cuda::Device;
    using noa::cuda::Stream;

    Stream stream(Device{});

    SECTION("sum one axis") {
        const auto input_shape = test::random_shape<i64, 3>(3);
        const auto input_strides = input_shape.strides();
        const auto n_elements = input_shape.n_elements();

        const auto buffer = AllocatorManaged<i64>::allocate(n_elements, stream);
        test::arange(buffer.get(), n_elements);

        auto reduced = noa::make_tuple(AccessorValue<i64>(0));
        auto sum_op = SumOp<3>{.accessor={buffer.get(), input_strides}};

        for (auto axis: noa::irange(3)) {
            INFO("axis=" << axis << ", shape=" << input_shape);
            auto output_shape = input_shape;
            output_shape[axis] = 1;
            const auto output_strides = output_shape.strides();
            const auto output_elements = output_shape.n_elements();

            const auto expected_buffer = AllocatorManaged<i64>::allocate(output_elements, stream);
            auto compute_expected_reduction = [](Shape3<i64> shape, const auto& input, const auto& expected) {
                for (i64 i = 0; i < shape[0]; ++i) {
                    for (i64 j = 0; j < shape[1]; ++j) {
                        i64 tmp = 0;
                        for (i64 k = 0; k < shape[2]; ++k)
                            tmp += input(i, j, k);
                        expected(i, j) = tmp + 1; // +1: add to output
                    }
                }
            };
            using expected_t = AccessorI64<const i64, 3>;
            if (axis == 0) {
                compute_expected_reduction(
                    input_shape.filter(1, 2, 0),
                    expected_t(buffer.get(), input_strides.filter(1, 2, 0)),
                    AccessorI64<i64, 2>(expected_buffer.get(), output_strides.filter(1, 2)));
            } else if (axis == 1) {
                compute_expected_reduction(
                    input_shape.filter(0, 2, 1),
                    expected_t(buffer.get(), input_strides.filter(0, 2, 1)),
                    AccessorI64<i64, 2>(expected_buffer.get(), output_strides.filter(0, 2)));
            } else {
                compute_expected_reduction(
                    input_shape,
                    expected_t(buffer.get(), input_strides),
                    AccessorI64<i64, 2>(expected_buffer.get(), output_strides.filter(0, 1)));
            }

            const auto output_buffer = AllocatorManaged<i64>::allocate(output_elements, stream);
            auto output = noa::make_tuple(AccessorI64<i64, 3>(output_buffer.get(), output_strides));

            std::fill(output_buffer.get(), output_buffer.get() + output_elements, 1);
            reduce_axes_iwise(input_shape, output_shape, sum_op, reduced, output, stream);
            stream.synchronize();
            REQUIRE(test::allclose_abs(output_buffer.get(), expected_buffer.get(), output_elements, 0));
        }
    }

    SECTION("per batch") {
        const auto input_shapes = std::array{
            test::random_shape<i64, 3>(1),
            test::random_shape<i64, 3>(2),
            test::random_shape<i64, 3>(3),
            test::random_shape<i64, 3>(4),
        };
        for (const auto& input_shape: input_shapes) {
            const auto input_strides = input_shape.strides();
            const auto n_elements = input_shape.n_elements();

            const auto buffer = AllocatorManaged<i64>::allocate(n_elements, stream);
            test::arange(buffer.get(), n_elements);

            auto reduced = noa::make_tuple(AccessorValue<i64>(0));
            auto sum_op = SumOp<3>{.accessor={buffer.get(), input_strides}};

            const auto expected_buffer = AllocatorManaged<i64>::allocate(input_shape[0], stream);
            const auto per_batch_n_elements = input_shape.pop_front().n_elements();
            for (i64 i = 0; i < input_shape[0]; ++i) {
                const i64* input = buffer.get() + input_strides[0] * i;
                i64 tmp = 0;
                for (i64 j = 0; j < per_batch_n_elements; ++j)
                    tmp += input[j];
                expected_buffer.get()[i] = tmp + 1;
            }

            const auto output_shape = Shape3<i64>{input_shape[0], 1, 1};
            const auto output_elements = output_shape.n_elements();
            const auto output_buffer = AllocatorManaged<i64>::allocate(output_elements, stream);
            const auto output = noa::make_tuple(AccessorI64<i64, 3>(output_buffer.get(), output_shape.strides()));

            std::fill(output_buffer.get(), output_buffer.get() + output_elements, 1);
            reduce_axes_iwise(input_shape, output_shape, sum_op, reduced, output, stream);
            stream.synchronize();
            REQUIRE(test::allclose_abs(output_buffer.get(), expected_buffer.get(), output_elements, 0));
        }
    }
}

TEST_CASE("cuda::reduce_axes_iwise - 2d") {
    using namespace noa::types;
    using noa::cuda::reduce_axes_iwise;
    using noa::cuda::AllocatorManaged;
    using noa::cuda::Device;
    using noa::cuda::Stream;

    Stream stream(Device{});

    SECTION("sum one axis") {
        const auto input_shape = test::random_shape<i64, 2>(2);
        const auto input_strides = input_shape.strides();
        const auto n_elements = input_shape.n_elements();

        const auto buffer = AllocatorManaged<i64>::allocate(n_elements, stream);
        test::arange(buffer.get(), n_elements);

        auto reduced = noa::make_tuple(AccessorValue<i64>(0));
        auto sum_op = SumOp<2>{.accessor={buffer.get(), input_strides}};

        for (auto axis: noa::irange(2)) {
            INFO("axis=" << axis << ", shape=" << input_shape);
            auto output_shape = input_shape;
            output_shape[axis] = 1;
            const auto output_strides = output_shape.strides();
            const auto output_elements = output_shape.n_elements();

            const auto expected_buffer = AllocatorManaged<i64>::allocate(output_elements, stream);
            auto compute_expected_reduction = [](Shape2<i64> shape, const auto& input, const auto& expected) {
                for (i64 i = 0; i < shape[0]; ++i) {
                    i64 tmp = 0;
                    for (i64 j = 0; j < shape[1]; ++j)
                        tmp += input(i, j);
                    expected(i) = tmp + 1; // +1: add to output
                }
            };
            using expected_t = AccessorI64<const i64, 2>;
            if (axis == 0) {
                compute_expected_reduction(
                    input_shape.filter(1, 0),
                    expected_t(buffer.get(), input_strides.filter(1, 0)),
                    AccessorI64<i64, 1>(expected_buffer.get(), output_strides.filter(1)));
            } else {
                compute_expected_reduction(
                    input_shape,
                    expected_t(buffer.get(), input_strides),
                    AccessorI64<i64, 1>(expected_buffer.get(), output_strides.filter(0)));
            }

            const auto output_buffer = AllocatorManaged<i64>::allocate(output_elements, stream);
            auto output = noa::make_tuple(AccessorI64<i64, 2>(output_buffer.get(), output_strides));

            std::fill_n(output_buffer.get(), output_elements, 1);
            reduce_axes_iwise(input_shape, output_shape, sum_op, reduced, output, stream);
            stream.synchronize();
            REQUIRE(test::allclose_abs(output_buffer.get(), expected_buffer.get(), output_elements, 0));
        }
    }
}

TEST_CASE("cuda::reduce_axes_iwise - 1d") {
    using namespace noa::types;
    using noa::cuda::reduce_axes_iwise;
    using noa::cuda::AllocatorManaged;
    using noa::cuda::Device;
    using noa::cuda::Stream;

    Stream stream(Device{});

    const auto input_shape = test::random_shape<i64, 1>(1) * 50;
    const auto input_strides = input_shape.strides();
    const auto n_elements = input_shape.n_elements();

    const auto buffer = AllocatorManaged<i64>::allocate(n_elements, stream);
    test::arange(buffer.get(), n_elements);

    auto reduced = noa::make_tuple(AccessorValue<i64>(0));
    auto sum_op = SumOp<1>{.accessor={buffer.get(), input_strides}};

    auto output_shape = Shape1<i64>{1};

    i64 expected{1};
    for (i64 i = 0; i < input_shape[0]; ++i)
        expected += sum_op.accessor(i);

    auto output = AllocatorManaged<i64>::allocate(1, stream);
    output[0] = 1;
    auto output_accessor = noa::make_tuple(AccessorContiguousI64<i64, 1>(output.get()));
    reduce_axes_iwise(input_shape, output_shape, sum_op, reduced, output_accessor, stream);
    stream.synchronize();
    REQUIRE(expected == output[0]);
}
