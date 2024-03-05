#include <noa/cpu/ReduceAxesIwise.hpp>
#include <catch2/catch.hpp>

#include "Helpers.h"

TEST_CASE("cpu::reduce_axes_iwise") {
    using namespace noa::types;
    using noa::cpu::reduce_axes_iwise;

    AND_THEN("sum 4d") {
        const auto input_shape = test::get_random_shape4_batched(4);
        const auto n_elements = input_shape.elements();

        const auto buffer = std::make_unique<i64[]>(static_cast<size_t>(n_elements));
        std::fill_n(buffer.get(), n_elements, 1);
        auto reduced = noa::make_tuple(AccessorValue<i64>(0));

        using accessor_t = AccessorContiguousI64<const i64, 4>;
        struct reduce_op_t {
            accessor_t accessor;

            void init(const Vec4<i64>& indices, i64& reduced) const {
                reduced += accessor(indices);
            }
            void join(i64 to_reduce, i64& reduced) const {
                reduced += to_reduce;
            }
        } reduce_op{accessor_t(buffer.get(), input_shape.strides())};

        for (auto i: noa::irange(4)) {
            INFO("axis=" << i);
            auto output_shape = input_shape;
            output_shape[i] = 1;
            const auto output_elements = output_shape.elements();
            const auto expected_sum = input_shape[i];

            const auto output_buffer = std::make_unique<i64[]>(static_cast<size_t>(output_elements));
            auto output = noa::make_tuple(AccessorI64<i64, 4>(output_buffer.get(), output_shape.strides()));

            reduce_axes_iwise(input_shape, output_shape, reduce_op, reduced, output, 4);

            REQUIRE(test::Matcher(test::MATCH_ABS, output_buffer.get(), expected_sum, output_elements, 1e-5));
        }
    }
}
