#include <noa/cpu/ReduceAxesIwise.hpp>
#include <noa/core/utils/Irange.hpp>
#include <catch2/catch.hpp>

#include "Utils.hpp"

TEST_CASE("cpu::reduce_axes_iwise - 4d") {
    using namespace noa::types;
    using noa::cpu::reduce_axes_iwise;

    struct SumOp {
        AccessorContiguousI64<const i64, 4> accessor;

        void init(const Vec4<i64>& indices, i64& reduced) const {
            reduced += accessor(indices);
        }
        void join(i64 to_reduce, i64& reduced) const {
            reduced += to_reduce;
        }
        void final(i64 reduced, i64& output) const {
            output += reduced;
        }
    };

    AND_THEN("sum one axis") {
        auto input_shape = test::random_shape(3, {.batch_range{2, 10}});
        const auto input_strides = input_shape.strides();
        const auto n_elements = input_shape.n_elements();

        const auto buffer = std::make_unique<i64[]>(static_cast<size_t>(n_elements));
        test::arange(buffer.get(), n_elements);

        auto reduced = noa::make_tuple(AccessorValue<i64>(0));
        auto sum_op = SumOp{.accessor={buffer.get(), input_strides}};

        for (auto axis: noa::irange(4)) {
            INFO("axis=" << axis << ", shape=" << input_shape);
            auto output_shape = input_shape;
            output_shape[axis] = 1;
            const auto output_strides = output_shape.strides();
            const auto output_elements = output_shape.n_elements();

            const auto expected_buffer = std::make_unique<i64[]>(static_cast<size_t>(output_elements));
            auto compute_expected_reduction = [](Shape4<i64> shape, const auto& input, const auto& expected) {
                for (i64 i{}; i < shape[0]; ++i) {
                    for (i64 j{}; j < shape[1]; ++j) {
                        for (i64 k{}; k < shape[2]; ++k) {
                            i64 tmp{};
                            for (i64 l{}; l < shape[3]; ++l)
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

            const auto output_buffer = std::make_unique<i64[]>(static_cast<size_t>(output_elements));
            auto output = noa::make_tuple(AccessorI64<i64, 4>(output_buffer.get(), output_strides));

            for (i64 n_threads: std::array{1, 4}) {
                std::fill(output_buffer.get(), output_buffer.get() + output_elements, 1);
                reduce_axes_iwise(input_shape, output_shape, sum_op, reduced, output, n_threads);
                REQUIRE(test::allclose_abs(output_buffer.get(), expected_buffer.get(), output_elements, 0));
            }
        }
    }

    AND_THEN("per batch") {
        const auto input_shapes = std::array{
                test::random_shape(1),
                test::random_shape(2),
                test::random_shape(3),
                test::random_shape(4),
                test::random_shape(1, {.batch_range={2, 10}}),
                test::random_shape(2, {.batch_range={2, 10}}),
                test::random_shape(3, {.batch_range={2, 10}}),
                test::random_shape(4, {.batch_range={2, 10}})
        };
        for (const auto& input_shape: input_shapes) {
            const auto input_strides = input_shape.strides();
            const auto n_elements = input_shape.n_elements();

            const auto buffer = std::make_unique<i64[]>(static_cast<size_t>(n_elements));
            test::arange(buffer.get(), n_elements);

            auto reduced = noa::make_tuple(AccessorValue<i64>(0));
            auto sum_op = SumOp{.accessor={buffer.get(), input_strides}};

            const auto expected_buffer = std::make_unique<i64[]>(static_cast<size_t>(input_shape[0]));
            const auto per_batch_n_elements = input_shape.pop_front().n_elements();
            for (i64 i{}; i < input_shape[0]; ++i) {
                const i64* input = buffer.get() + input_strides[0] * i;
                i64 tmp{};
                for (i64 j{}; j < per_batch_n_elements; ++j)
                    tmp += input[j];
                expected_buffer.get()[i] = tmp + 1;
            }

            const auto output_shape = Shape4<i64>{input_shape[0], 1, 1, 1};
            const auto output_elements = output_shape.n_elements();
            const auto output_buffer = std::make_unique<i64[]>(static_cast<size_t>(output_elements));
            const auto output = noa::make_tuple(AccessorI64<i64, 4>(output_buffer.get(), output_shape.strides()));

            for (i64 n_threads: std::array{1, 4}) {
                std::fill(output_buffer.get(), output_buffer.get() + output_elements, 1);
                reduce_axes_iwise(input_shape, output_shape, sum_op, reduced, output, n_threads);
                REQUIRE(test::allclose_abs(output_buffer.get(), expected_buffer.get(), output_elements, 0));
            }
        }
    }
}

TEST_CASE("cpu::reduce_axes_iwise - 3d") {
    using namespace noa::types;
    using noa::cpu::reduce_axes_iwise;

    struct SumOp {
        AccessorContiguousI64<const i64, 3> accessor;

        void init(const Vec3<i64>& indices, i64& reduced) const {
            reduced += accessor(indices);
        }
        void join(i64 to_reduce, i64& reduced) const {
            reduced += to_reduce;
        }
        void final(i64 reduced, i64& output) const {
            output += reduced;
        }
    };

    AND_THEN("sum one axis") {
        const auto input_shape = test::random_shape<i64, 3>(3);
        const auto input_strides = input_shape.strides();
        const auto n_elements = input_shape.n_elements();

        const auto buffer = std::make_unique<i64[]>(static_cast<size_t>(n_elements));
        test::arange(buffer.get(), n_elements);

        auto reduced = noa::make_tuple(AccessorValue<i64>(0));
        auto sum_op = SumOp{.accessor={buffer.get(), input_strides}};

        for (auto axis: noa::irange(3)) {
            INFO("axis=" << axis << ", shape=" << input_shape);
            auto output_shape = input_shape;
            output_shape[axis] = 1;
            const auto output_strides = output_shape.strides();
            const auto output_elements = output_shape.n_elements();

            const auto expected_buffer = std::make_unique<i64[]>(static_cast<size_t>(output_elements));
            auto compute_expected_reduction = [](Shape3<i64> shape, const auto& input, const auto& expected) {
                for (i64 i{}; i < shape[0]; ++i) {
                    for (i64 j{}; j < shape[1]; ++j) {
                        i64 tmp{};
                        for (i64 k{}; k < shape[2]; ++k)
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

            const auto output_buffer = std::make_unique<i64[]>(static_cast<size_t>(output_elements));
            auto output = noa::make_tuple(AccessorI64<i64, 3>(output_buffer.get(), output_strides));

            for (i64 n_threads: std::array{1, 4}) {
                std::fill(output_buffer.get(), output_buffer.get() + output_elements, 1);
                reduce_axes_iwise(input_shape, output_shape, sum_op, reduced, output, n_threads);
                REQUIRE(test::allclose_abs(output_buffer.get(), expected_buffer.get(), output_elements, 0));
            }
        }
    }

    AND_THEN("per batch") {
        const auto input_shapes = std::array{
                test::random_shape<i64, 3>(1),
                test::random_shape<i64, 3>(2),
                test::random_shape<i64, 3>(3)
        };
        for (const auto& input_shape: input_shapes) {
            const auto input_strides = input_shape.strides();
            const auto n_elements = input_shape.n_elements();

            const auto buffer = std::make_unique<i64[]>(static_cast<size_t>(n_elements));
            test::arange(buffer.get(), n_elements);

            auto reduced = noa::make_tuple(AccessorValue<i64>(0));
            auto sum_op = SumOp{.accessor={buffer.get(), input_strides}};

            const auto expected_buffer = std::make_unique<i64[]>(static_cast<size_t>(input_shape[0]));
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
            const auto output_buffer = std::make_unique<i64[]>(static_cast<size_t>(output_elements));
            const auto output = noa::make_tuple(AccessorI64<i64, 3>(output_buffer.get(), output_shape.strides()));

            for (i64 n_threads: std::array{1, 4}) {
                std::fill(output_buffer.get(), output_buffer.get() + output_elements, 1);
                reduce_axes_iwise(input_shape, output_shape, sum_op, reduced, output, n_threads);
                REQUIRE(test::allclose_abs(output_buffer.get(), expected_buffer.get(), output_elements, 0));
            }
        }
    }
}

TEST_CASE("cpu::reduce_axes_iwise - 2d") {
    using namespace noa::types;
    using noa::cpu::reduce_axes_iwise;

    struct SumOp {
        AccessorContiguousI64<const i64, 2> accessor;

        void init(const Vec2<i64>& indices, i64& reduced) const {
            reduced += accessor(indices);
        }
        void join(i64 to_reduce, i64& reduced) const {
            reduced += to_reduce;
        }
        void final(i64 reduced, i64& output) const {
            output += reduced;
        }
    };

    AND_THEN("sum one axis") {
        const auto input_shape = test::random_shape<i64, 2>(2);
        const auto input_strides = input_shape.strides();
        const auto n_elements = input_shape.n_elements();

        const auto buffer = std::make_unique<i64[]>(static_cast<size_t>(n_elements));
        test::arange(buffer.get(), n_elements);

        auto reduced = noa::make_tuple(AccessorValue<i64>(0));
        auto sum_op = SumOp{.accessor={buffer.get(), input_strides}};

        for (auto axis: noa::irange(2)) {
            INFO("axis=" << axis << ", shape=" << input_shape);
            auto output_shape = input_shape;
            output_shape[axis] = 1;
            const auto output_strides = output_shape.strides();
            const auto output_elements = output_shape.n_elements();

            const auto expected_buffer = std::make_unique<i64[]>(static_cast<size_t>(output_elements));
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

            const auto output_buffer = std::make_unique<i64[]>(static_cast<size_t>(output_elements));
            auto output = noa::make_tuple(AccessorI64<i64, 2>(output_buffer.get(), output_strides));

            for (i64 n_threads: std::array{1, 4}) {
                std::fill(output_buffer.get(), output_buffer.get() + output_elements, 1);
                reduce_axes_iwise(input_shape, output_shape, sum_op, reduced, output, n_threads);
                REQUIRE(test::allclose_abs(output_buffer.get(), expected_buffer.get(), output_elements, 0));
            }
        }
    }
}

TEST_CASE("cpu::reduce_axes_iwise - 1d") {
    using namespace noa::types;
    using noa::cpu::reduce_axes_iwise;

    struct SumOp {
        AccessorContiguousI64<const i64, 1> accessor;

        void init(const Vec1<i64>& indices, i64& reduced) const {
            reduced += accessor(indices);
        }
        void join(i64 to_reduce, i64& reduced) const {
            reduced += to_reduce;
        }
        void final(i64 reduced, i64& output) const {
            output += reduced;
        }
    };

    const auto input_shape = test::random_shape<i64, 1>(1) * 50;
    const auto input_strides = input_shape.strides();
    const auto n_elements = input_shape.n_elements();

    const auto buffer = std::make_unique<i64[]>(static_cast<size_t>(n_elements));
    test::arange(buffer.get(), n_elements);

    auto reduced = noa::make_tuple(AccessorValue<i64>(0));
    auto sum_op = SumOp{.accessor={buffer.get(), input_strides}};

    auto output_shape = Shape1<i64>{1};

    i64 expected{1};
    for (i64 i = 0; i < input_shape[0]; ++i)
        expected += sum_op.accessor(i);

    for (i64 n_threads: std::array{1, 4}) {
        i64 output{1};
        auto output_accessor = noa::make_tuple(AccessorContiguousI64<i64, 1>(&output));
        reduce_axes_iwise(input_shape, output_shape, sum_op, reduced, output_accessor, n_threads);
        REQUIRE(expected == output);
    }
}
