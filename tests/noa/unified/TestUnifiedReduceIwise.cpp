#include <noa/core/Reduce.hpp>
#include <noa/unified/Array.hpp>
#include <noa/unified/Factory.hpp>
#include <noa/unified/ReduceIwise.hpp>

#include <catch2/catch.hpp>

#include "Helpers.h"

using namespace noa::types;

namespace {
    template<typename A>
    struct SumAdd {
        A accessor;
        constexpr void init(const auto& indices, f64& sum) const {
            sum += static_cast<f64>(accessor(indices));
        }
        constexpr void join(f64 isum, f64& sum) const {
            sum += isum;
        }
        constexpr void final(f64 sum, auto& output) const {
            output += static_cast<std::decay_t<decltype(output)>>(sum);
        }
    };
}

TEMPLATE_TEST_CASE("unified::reduce_iwise - simple", "[noa][unified]", i32, f64) {
    auto shape = Shape4<i64>{5, 35, 64, 81};
    auto min_indices = Vec4<i64>{2, 12, 43, 56};
    auto max_indices = Vec4<i64>{1, 6, 12, 34};

    auto input = noa::empty<TestType>(shape);
    auto randomizer = test::Randomizer<TestType>(-50, 50);
    for (auto& value: input.span())
        value = randomizer.get();
    input(min_indices) = -51;
    input(max_indices) = 51;

    f64 sum{};
    for (auto& value: input.span())
        sum += static_cast<f64>(value);

    std::vector<Device> devices{"cpu"};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        auto stream = StreamGuard(device, StreamMode::DEFAULT);
        const auto options = ArrayOption{device, "managed"};
        INFO(device);

        if (device != input.device())
            input = input.to(options);

        using accessor_t = AccessorI64<const TestType, 4>;
        using op_t = noa::ReduceFirstMax<accessor_t>;
        using reduce_t = op_t::reduced_type;
        auto op = op_t{input.accessor()};
        auto initial = reduce_t{std::numeric_limits<TestType>::lowest(), 0};

        i64 output_max_offset{};
        TestType output_max_value{};
        noa::reduce_iwise(shape, device, initial, noa::wrap(output_max_value, output_max_offset), op);

        REQUIRE(all(noa::indexing::offset2index(output_max_offset, input) == max_indices));
        REQUIRE(output_max_value == 51);

        TestType output_sum{1};
        noa::reduce_iwise(Shape{shape.elements()}, device, f64{}, noa::wrap(output_sum),
                          SumAdd{input.accessor_contiguous_1d()});
        REQUIRE_THAT(output_sum, Catch::WithinRel(sum + 1, 1e-6));
    }
}
