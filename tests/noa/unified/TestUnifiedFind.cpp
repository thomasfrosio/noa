#include <noa/unified/Array.hpp>
#include <noa/unified/Find.hpp>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("unified::find_offset()", "[noa][unified]",
                   (std::tuple<i32, i32, first_min_t>),
                   (std::tuple<f32, u64, last_min_t>),
                   (std::tuple<i32, i64, first_max_t>),
                   (std::tuple<f64, u32, last_max_t>)) {
    using value_t = std::tuple_element_t<0, TestType>;
    using offset_t = std::tuple_element_t<1, TestType>;
    using op_t = std::tuple_element_t<2, TestType>;

    const bool small = GENERATE(true, false);
    const auto shape = small ? Shape4<i64>{3, 64, 128, 128} : Shape4<i64>{3, 256, 256, 300};
    const auto elements_per_batch = shape.pop_front().template as<offset_t>().elements();
    INFO(small);

    Array<value_t> input(shape);
    Array<offset_t> offset_results(shape.batch());
    const Array<offset_t> offset_expected(shape.batch());

    test::Randomizer<value_t> randomizer(-100., 100.);
    test::randomize(input.get(), input.elements(), randomizer);

    test::Randomizer<offset_t> randomizer_expected_offset(offset_t{0}, elements_per_batch - offset_t{200});
    test::randomize(offset_expected.get(), offset_expected.elements(), randomizer_expected_offset);

    const auto input_2d = input.reshape({shape.batch(), 1, 1, elements_per_batch});
    const value_t expected_value = std::is_same_v<op_t, first_min_t> || std::is_same_v<op_t, last_min_t> ? -101 : 101;

    for (i64 batch = 0; batch < shape.batch(); ++batch) {
        auto& offset = offset_expected(0, 0, 0, batch);
        input_2d(batch, 0, 0, offset + 0) = expected_value;
        input_2d(batch, 0, 0, offset + 50) = expected_value;
        input_2d(batch, 0, 0, offset + 100) = expected_value;
        input_2d(batch, 0, 0, offset + 150) = expected_value;
        if constexpr (noa::traits::is_any_v<op_t, last_min_t, last_max_t>)
            offset += 150;
    }

    std::vector<Device> devices = {Device{}};
    if (Device::any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        INFO(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        input = input.device().is_cpu() ? input : input.to(options);
        offset_results = offset_results.device().is_cpu() ? offset_results : offset_results.to(options);

        find_offsets(op_t{}, input, offset_results);
        REQUIRE(test::Matcher(test::MATCH_ABS, offset_expected, offset_results, 1e-7));

        const auto offset = static_cast<offset_t>(find_offset(op_t{}, input));
        if constexpr (noa::traits::is_any_v<op_t, last_min_t, last_max_t>) {
            // By default, find_offsets returns the offsets relative to each batch.
            // Here, we expect to get the last batch, so offset to that.
            const auto index_last_batch = static_cast<offset_t>(shape.batch() - 1);
            const auto offset_to_last_batch = elements_per_batch * index_last_batch;
            REQUIRE(offset == offset_expected(0, 0, 0, 2) + offset_to_last_batch);
        } else {
            REQUIRE(offset == offset_expected(0, 0, 0, 0));
        }
    }
}

TEMPLATE_TEST_CASE("unified::find_offset(), padded", "[noa][unified]",
                   (std::tuple<i32, i32, first_min_t>),
                   (std::tuple<f32, u64, last_min_t>),
                   (std::tuple<i32, i64, first_max_t>),
                   (std::tuple<f64, u32, last_max_t>)) {
    using value_t = std::tuple_element_t<0, TestType>;
    using offset_t = std::tuple_element_t<1, TestType>;
    using op_t = std::tuple_element_t<2, TestType>;

    const auto subregion_shape = test::get_random_shape4(3) + 10;
    auto shape = subregion_shape;
    shape[1] += 20;
    shape[2] += 20;
    shape[3] += 20;

    const bool reduce_batch = GENERATE(true, false);
    const i64 output_size = reduce_batch ? 1 : shape.batch();
    INFO(output_size);
    INFO(subregion_shape);

    std::vector<Device> devices = {Device{}};
    if (Device::any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        INFO(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        Array<value_t> data(shape, options);
        const auto data_padded = data.subregion(
                noa::indexing::full_extent_t{},
                noa::indexing::slice_t{0, subregion_shape[1]},
                noa::indexing::slice_t{0, subregion_shape[2]},
                noa::indexing::slice_t{0, subregion_shape[3]});
        test::Randomizer<value_t> randomizer(-100., 100.);
        test::randomize(data.get(), data.elements(), randomizer);
        data = data_padded.copy();
        data.eval();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, data, data_padded, 1e-6));

        const Array<offset_t> offset_expected(output_size, options);
        const Array<offset_t> offset_result(output_size, options);
        find_offsets(op_t{}, data, offset_expected, reduce_batch);
        find_offsets(op_t{}, data_padded, offset_result, reduce_batch);
        data.eval();

        for (i64 i = 0; i < output_size; ++i) {
            const auto index_expected = indexing::offset2index(static_cast<i64>(offset_expected(0, 0, 0, i)), data);
            const auto index_result = indexing::offset2index(static_cast<i64>(offset_result(0, 0, 0, i)), data_padded);
            INFO(offset_expected(0, 0, 0, i));
            INFO(offset_result(0, 0, 0, i));
            INFO(index_expected);
            INFO(index_result);
            REQUIRE(noa::all(index_expected == index_result));
        }
    }
}
