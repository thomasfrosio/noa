#include <noa/runtime/core/Reduce.hpp>
#include <noa/runtime/Array.hpp>
#include <noa/runtime/ReduceIwise.hpp>
#include <noa/runtime/Random.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace noa::types;

namespace {
    template<typename A>
    struct SumAdd {
        A accessor;
        constexpr void operator()(const auto& indices, f64& sum) const {
            sum += static_cast<f64>(accessor(indices));
        }
        constexpr void join(f64 isum, f64& sum) const {
            sum += isum;
        }
        constexpr void post(f64 sum, auto& output) const {
            output += static_cast<std::decay_t<decltype(output)>>(sum);
        }
    };
}

TEMPLATE_TEST_CASE("runtime::reduce_iwise - simple", "", i32, f64) {
    const auto shapes = noa::make_tuple(
        test::random_shape<isize, 1>(1),
        test::random_shape<isize, 2>(2),
        test::random_shape<isize, 3>(3),
        test::random_shape<isize, 4>(3, {.batch_range = {10, 20}}),
        test::random_shape<isize, 5>(3, {.size_range = {15, 20}, .batch_range = {15, 20}}),
        test::random_shape<isize, 6>(3, {.size_range = {15, 20}, .batch_range = {15, 20}})
    );

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    shapes.for_each([&]<usize N>(const Shape<isize, N>& shape) {
        auto min_indices = shape.vec / 2 - 2;
        auto max_indices = shape.vec / 2 + 2;

        auto input = noa::random(noa::Uniform<TestType>{-50, 50}, shape);
        input.span()(min_indices) = -51;
        input.span()(max_indices) = 51;

        f64 sum{};
        for (auto& value: input.span_1d())
            sum += static_cast<f64>(value);

        for (auto& device: devices) {
            auto stream = StreamGuard(device, Stream::DEFAULT);
            const auto options = ArrayOption{device, "managed"};
            INFO(device);

            if (device != input.device())
                input = input.to(options);

            using accessor_t = noa::Accessor<const TestType, N>;
            using reduce_t = Pair<TestType, i64>;
            using op_t = noa::ReduceFirstMax<accessor_t, reduce_t, true, true>;
            auto op = op_t{noa::details::to_accessor(input)};
            auto initial = reduce_t{std::numeric_limits<TestType>::lowest(), 0};

            i64 output_max_offset{};
            TestType output_max_value{};
            noa::reduce_iwise(shape, device, initial, noa::wrap(output_max_value, output_max_offset), op);

            REQUIRE(noa::offset2index(output_max_offset, input) == max_indices);
            REQUIRE(output_max_value == 51);

            TestType output_sum{1};
            noa::reduce_iwise(shape, device, f64{}, noa::wrap(output_sum), SumAdd{input.span()});
            REQUIRE_THAT(output_sum, Catch::Matchers::WithinRel(sum + 1, 1e-6));
        }
    });
}
