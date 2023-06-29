#include <noa/unified/Array.hpp>
#include <noa/unified/Ewise.hpp>
#include <noa/unified/math/Random.hpp>
#include <noa/unified/math/Reduce.hpp>
#include <noa/unified/memory/Factory.hpp>
#include <noa/unified/memory/Index.hpp>

#include <catch2/catch.hpp>

#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::memory::extract_{unary|binary}(), insert_elements()", "[noa][unified]", f32, i32) {
    const auto shape = test::get_random_shape4_batched(3);

    // Initialize data.
    auto data = noa::math::random<TestType>(noa::math::uniform_t{}, shape, 1, 100);

    // Prepare expected data.
    const auto mask_shape = shape.pop_front().push_front(1);
    auto mask = noa::math::random<TestType>(noa::math::uniform_t{}, mask_shape, 0, 4);

    // Extract elements from data only if mask isn't 0.
    std::vector<size_t> expected_indexes;
    std::vector<TestType> expected_values;
    const auto expected_data_reinsert = noa::memory::zeros<TestType>(data.shape()).eval();

    const auto data_1d = data.accessor_contiguous_1d();
    const auto mask_1d = mask.accessor_contiguous_1d();
    const auto expected_data_reinsert_1d = expected_data_reinsert.accessor_contiguous_1d();

    for (i64 batch = 0; batch < shape[0]; ++batch) {
        const auto elements = mask_shape.elements();
        const auto batch_offset = batch * elements;
        for (i64 i = 0; i < elements; ++i) {
            if (mask_1d[i] == 0)
                continue;
            const i64 index = batch_offset + i;
            expected_indexes.emplace_back(index);
            expected_values.emplace_back(data_1d[index]);
            expected_data_reinsert_1d[index] = data_1d[index];
        }
    }

    // Sse same mask for every batch. We have to explicitly broadcast
    // The extract_{unary|binary} functions cannot broadcast because there's no output.
    mask = noa::indexing::broadcast(mask, shape);

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    data.eval(); // sync before changing the current stream
    for (const auto& device: devices) {
        auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        if (device != data.device())
            data = data.to(options);
        if (device != mask.device())
            mask = mask.to(options);

        THEN("contiguous") {
            const auto extracted = noa::memory::extract_unary<TestType, u64>(data, mask, nonzero_t{});
            const auto extracted_count = extracted.values.elements();
            REQUIRE((extracted_count == extracted.offsets.elements() &&
                     extracted_count == static_cast<i64>(expected_indexes.size())));

            stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS, expected_indexes.data(), extracted.offsets.get(), extracted_count, 1e-8));
            REQUIRE(test::Matcher(test::MATCH_ABS, expected_values.data(), extracted.values.get(), extracted_count, 1e-8));

            const auto reinsert = noa::memory::zeros<TestType>(shape, options);
            noa::memory::insert_elements(extracted.values, extracted.offsets, reinsert);
            REQUIRE(test::Matcher(test::MATCH_ABS, expected_data_reinsert, reinsert, 0));
        }

        THEN("padded") {
            const auto physical_shape = shape + test::Randomizer<i64>(5, 10).get() * Shape4<i64>{shape != 1};
            auto padded = noa::memory::fill<TestType>(physical_shape, 2, options);
            padded = padded.subregion(
                    noa::indexing::Slice{0, shape[0]},
                    noa::indexing::Slice{0, shape[1]},
                    noa::indexing::Slice{0, shape[2]},
                    noa::indexing::Slice{0, shape[3]});

            const auto extracted = noa::memory::extract_binary<TestType, i64>(
                    padded, padded, TestType{1}, greater_t{}, false, true);

            REQUIRE(extracted.values.is_empty());
            const auto expected_count = shape.elements();
            REQUIRE(extracted.offsets.elements() == expected_count); // padded elements should not be selected
            const i64 last = noa::indexing::at((shape - 1).vec(), physical_shape.strides());
            REQUIRE(extracted.offsets.data()[expected_count - 1] == last);
        }
    }
}

TEMPLATE_TEST_CASE("unified::memory::extract_binary() and insert_elements()", "[noa][unified]",
                   i32, f32, f64) {
    std::vector<Device> devices = {Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::DEFAULT_ASYNC);
        const auto data = noa::math::random<TestType>(noa::math::uniform_t{}, {2, 100, 200, 300}, -5, 5, options);

        auto[values, indexes] = noa::memory::extract_binary<TestType, u64>(data, data, TestType{0}, less_t{});
        noa::ewise_unary(values, values, abs_t{});
        noa::memory::insert_elements(values, indexes, data);

        TestType min = noa::math::min(data);
        REQUIRE(min >= 0);
        if constexpr (noa::traits::is_real_v<TestType>) {
            TestType mean = noa::math::mean(data);
            REQUIRE_THAT(mean, Catch::WithinAbs(2.5, 0.1));
        }
    }
}
