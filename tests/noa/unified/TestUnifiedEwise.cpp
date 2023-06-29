#include <noa/unified/Array.hpp>
#include <noa/unified/Ewise.hpp>
#include <noa/unified/math/Random.hpp>
#include <noa/unified/math/Reduce.hpp>
#include <noa/unified/memory/Factory.hpp>

#include <catch2/catch.hpp>

#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::ewise_unary", "[noa][unified]",
                   (std::tuple<i32, i32, noa::square_t>),
                   (std::tuple<f32, f32, noa::square_t>),
                   (std::tuple<c32, c32, noa::square_t>),
                   (std::tuple<f64, f64, noa::cos_t>),
                   (std::tuple<c64, f64, noa::abs_squared_t>),
                   (std::tuple<c64, f64, noa::abs_one_log_t>),
                   (std::tuple<i64, i64, noa::nonzero_t>),
                   (std::tuple<i32, bool, noa::nonzero_t>)) {
    using value_t = std::tuple_element_t<0, TestType>;
    using result_t = std::tuple_element_t<1, TestType>;
    using op_t = std::tuple_element_t<2, TestType>;

    // Test with padding too.
    const i32 pad = GENERATE(0, 1);
    INFO(pad);
    auto shape = test::get_random_shape4_batched(3);
    auto subregion_shape = shape;
    if (pad)
        shape[3] += 20;

    const Array<value_t> input(subregion_shape);
    const Array<result_t> expected(subregion_shape);
    const auto input_1d = input.template accessor_contiguous_1d();
    const auto expected_1d = expected.template accessor_contiguous_1d();

    test::Randomizer<value_t> randomizer(-5, 5);
    test::randomize(input_1d.get(), input.elements(), randomizer); // FIXME
    for (i64 i = 0; i < input.elements(); ++i)
        expected_1d[i] = static_cast<result_t>(op_t{}(input_1d[i]));

    std::vector<Device> devices = {Device{}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const StreamGuard stream(device);
        const ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        const Array i_input = device.is_cpu() ? input : input.to(options);

        // Create output and ewise.
        const Array<result_t> result_padded(shape, options);
        auto result = result_padded.subregion(
                noa::indexing::Ellipsis{},
                noa::indexing::Slice{0, subregion_shape[3]});

        ewise_unary(i_input, result, op_t{});
        REQUIRE(test::Matcher(test::MATCH_ABS, result, expected, 5e-6));

        // Let ewise allocate the output.
        result = ewise_unary<result_t>(i_input, op_t{});
        REQUIRE(result.device() == device);
        REQUIRE(result.allocator() == i_input.options().allocator());
        REQUIRE(test::Matcher(test::MATCH_ABS, result, expected, 5e-6));
    }
}

TEMPLATE_TEST_CASE("unified::ewise_unary, large", "[noa][unified]",
                   (std::tuple<f32, f32, noa::sin_t>),
                   (std::tuple<f64, f64, noa::sqrt_t>)) {
    using value_t = std::tuple_element_t<0, TestType>;
    using result_t = std::tuple_element_t<1, TestType>;
    using op_t = std::tuple_element_t<2, TestType>;

    // Test with padding too.
    const i32 pad = GENERATE(0, 1);
    INFO(pad);
    auto shape = Shape4<i64>{1, 1, 8192, 8192};
    auto subregion_shape = shape;
    if (pad)
        shape[3] += 20;

    const Array<value_t> input(subregion_shape);
    const Array<result_t> expected(subregion_shape);
    const auto input_1d = input.template accessor_contiguous_1d();
    const auto expected_1d = expected.template accessor_contiguous_1d();

    test::Randomizer<value_t> randomizer(1, 5);
    test::randomize(input_1d.get(), input.elements(), randomizer); // FIXME
    for (i64 i = 0; i < input.elements(); ++i)
        expected_1d[i] = static_cast<result_t>(op_t{}(input_1d[i]));

    std::vector<Device> devices = {Device{}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const StreamGuard stream(device);
        const ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        const Array i_input = device.is_cpu() ? input : input.to(options);
        const Array<result_t> result_padded(shape, options);
        auto result = result_padded.subregion(
                noa::indexing::Ellipsis{},
                noa::indexing::Slice{0, subregion_shape[3]});
        ewise_unary(i_input, result, op_t{});
        REQUIRE(test::Matcher(test::MATCH_ABS, result, expected, 5e-6));
    }
}

TEMPLATE_TEST_CASE("unified::ewise_binary", "[noa][unified]",
                   (std::tuple<i32, i32, i32, noa::plus_t>),
                   (std::tuple<f32, f32, f32, noa::minus_t>),
                   (std::tuple<i64, i64, bool, noa::greater_equal_t>),
                   (std::tuple<c64, c64, c64, noa::multiply_conj_t>)) {
    using lhs_t = std::tuple_element_t<0, TestType>;
    using rhs_t = std::tuple_element_t<1, TestType>;
    using result_t = std::tuple_element_t<2, TestType>;
    using op_t = std::tuple_element_t<3, TestType>;

    // Test with padding too.
    const i32 pad = GENERATE(0, 1);
    INFO(pad);
    auto shape = test::get_random_shape4_batched(3);
    auto subregion_shape = shape;
    if (pad)
        shape[3] += 20;

    const Array<lhs_t> lhs(subregion_shape);
    const Array<rhs_t> rhs(subregion_shape);
    const Array<result_t> expected(subregion_shape);
    const auto lhs_1d = lhs.accessor_contiguous_1d();
    const auto rhs_1d = rhs.accessor_contiguous_1d();
    const auto expected_1d = expected.accessor_contiguous_1d();

    test::Randomizer<lhs_t> randomizer_lhs(-5, 5);
    test::randomize(lhs_1d.get(), lhs.elements(), randomizer_lhs);
    test::Randomizer<rhs_t> randomizer_rhs(-5, 5);
    test::randomize(rhs_1d.get(), rhs.elements(), randomizer_rhs);

    for (i64 i = 0; i < expected.elements(); ++i)
        expected_1d[i] = static_cast<result_t>(op_t{}(lhs_1d[i], rhs_1d[i]));

    std::vector<Device> devices = {Device{}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const StreamGuard stream(device);
        const ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        const Array i_lhs = device.is_cpu() ? lhs : lhs.to(options);
        const Array i_rhs = device.is_cpu() ? rhs : rhs.to(options);

        // Create output and ewise.
        const Array<result_t> result_padded(shape, options);
        auto result = result_padded.subregion(
                noa::indexing::Ellipsis{},
                noa::indexing::Slice{0, subregion_shape[3]});

        ewise_binary(i_lhs, i_rhs, result, op_t{});
        REQUIRE(test::Matcher(test::MATCH_ABS, result, expected, 5e-6));

        // Let ewise allocate the output.
        result = ewise_binary<result_t>(i_lhs, i_rhs, op_t{});
        REQUIRE(result.device() == device);
        REQUIRE(result.allocator() == i_lhs.options().allocator());
        REQUIRE(test::Matcher(test::MATCH_ABS, result, expected, 5e-6));
    }
}

TEMPLATE_TEST_CASE("unified::ewise_binary, large", "[noa][unified]",
                   (std::tuple<i32, i32, i32, noa::multiply_t>),
                   (std::tuple<f32, f32, f32, noa::dist2_t>)) {
    using lhs_t = std::tuple_element_t<0, TestType>;
    using rhs_t = std::tuple_element_t<1, TestType>;
    using result_t = std::tuple_element_t<2, TestType>;
    using op_t = std::tuple_element_t<3, TestType>;

    // Test with padding too.
    const i32 pad = GENERATE(0, 1);
    INFO(pad);
    auto shape = Shape4<i64>{1, 256, 256, 300};
    auto subregion_shape = shape;
    if (pad)
        shape[3] += 20;

    const Array<lhs_t> lhs(subregion_shape);
    const Array<rhs_t> rhs(subregion_shape);
    const Array<result_t> expected(subregion_shape);
    const auto lhs_1d = lhs.accessor_contiguous_1d();
    const auto rhs_1d = rhs.accessor_contiguous_1d();
    const auto expected_1d = expected.accessor_contiguous_1d();

    test::Randomizer<lhs_t> randomizer_lhs(-5, 5);
    test::randomize(lhs_1d.get(), lhs.elements(), randomizer_lhs);
    test::Randomizer<rhs_t> randomizer_rhs(-5, 5);
    test::randomize(rhs_1d.get(), rhs.elements(), randomizer_rhs);

    for (i64 i = 0; i < expected.elements(); ++i)
        expected_1d[i] = static_cast<result_t>(op_t{}(lhs_1d[i], rhs_1d[i]));

    std::vector<Device> devices = {Device{}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const StreamGuard stream(device);
        const ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        const Array i_lhs = device.is_cpu() ? lhs : lhs.to(options);
        const Array i_rhs = device.is_cpu() ? rhs : rhs.to(options);

        // Create output and ewise.
        const Array<result_t> result_padded(shape, options);
        auto result = result_padded.subregion(
                noa::indexing::Ellipsis{},
                noa::indexing::Slice{0, subregion_shape[3]});

        ewise_binary(i_lhs, i_rhs, result, op_t{});
        REQUIRE(test::Matcher(test::MATCH_ABS, result, expected, 5e-6));
    }
}


TEMPLATE_TEST_CASE("unified::ewise_trinary", "[noa][unified]",
                   (std::tuple<i32, i32, i32, i32, noa::plus_t>),
                   (std::tuple<f32, f32, f32, f32, noa::multiply_t>),
                   (std::tuple<i64, i64, i64, bool, noa::within_equal_t>),
                   (std::tuple<c64, f64, c64, c64, noa::multiply_t>)) {
    using lhs_t = std::tuple_element_t<0, TestType>;
    using mhs_t = std::tuple_element_t<1, TestType>;
    using rhs_t = std::tuple_element_t<2, TestType>;
    using result_t = std::tuple_element_t<3, TestType>;
    using op_t = std::tuple_element_t<4, TestType>;

    // Test with padding too.
    const i32 pad = GENERATE(0, 1);
    INFO(pad);
    auto shape = test::get_random_shape4_batched(3);
    auto subregion_shape = shape;
    if (pad)
        shape[3] += 20;

    const Array<lhs_t> lhs(subregion_shape);
    const Array<mhs_t> mhs(subregion_shape);
    const Array<rhs_t> rhs(subregion_shape);
    const Array<result_t> expected(subregion_shape);
    const auto lhs_1d = lhs.accessor_contiguous_1d();
    const auto mhs_1d = mhs.accessor_contiguous_1d();
    const auto rhs_1d = rhs.accessor_contiguous_1d();
    const auto expected_1d = expected.accessor_contiguous_1d();

    test::Randomizer<lhs_t> randomizer_lhs(-5, 5);
    test::randomize(lhs_1d.get(), lhs.elements(), randomizer_lhs);
    test::Randomizer<mhs_t> randomizer_mhs(-5, 5);
    test::randomize(mhs_1d.get(), mhs.elements(), randomizer_mhs);
    test::Randomizer<rhs_t> randomizer_rhs(-5, 5);
    test::randomize(rhs_1d.get(), rhs.elements(), randomizer_rhs);

    for (i64 i = 0; i < expected.elements(); ++i)
        expected_1d[i] = static_cast<result_t>(op_t{}(lhs_1d[i], mhs_1d[i], rhs_1d[i]));

    std::vector<Device> devices = {Device{}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const StreamGuard stream(device);
        const ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        const Array i_lhs = device.is_cpu() ? lhs : lhs.to(options);
        const Array i_mhs = device.is_cpu() ? mhs : mhs.to(options);
        const Array i_rhs = device.is_cpu() ? rhs : rhs.to(options);

        // Create output and ewise.
        const Array<result_t> result_padded(shape, options);
        auto result = result_padded.subregion(
                noa::indexing::Ellipsis{},
                noa::indexing::Slice{0, subregion_shape[3]});

        ewise_trinary(i_lhs, i_mhs, i_rhs, result, op_t{});
        REQUIRE(test::Matcher(test::MATCH_ABS, result, expected, 1e-6));

        // Let ewise allocate the output.
        result = ewise_trinary<result_t>(i_lhs, i_mhs, i_rhs, op_t{});
        REQUIRE(result.device() == device);
        REQUIRE(result.allocator() == i_lhs.options().allocator());
        REQUIRE(test::Matcher(test::MATCH_ABS, result, expected, 1e-6));
    }
}

TEMPLATE_TEST_CASE("unified::ewise_trinary, large", "[noa][unified]",
                   (std::tuple<i32, i32, i32, i32, noa::minus_t>),
                   (std::tuple<f32, f32, f32, f32, noa::clamp_t>)) {
    using lhs_t = std::tuple_element_t<0, TestType>;
    using mhs_t = std::tuple_element_t<1, TestType>;
    using rhs_t = std::tuple_element_t<2, TestType>;
    using result_t = std::tuple_element_t<3, TestType>;
    using op_t = std::tuple_element_t<4, TestType>;

    // Test with padding too.
    const i32 pad = GENERATE(0, 1);
    INFO(pad);
    auto shape = Shape4<i64>{1, 256, 256, 300};
    auto subregion_shape = shape;
    if (pad)
        shape[3] += 20;

    const Array<lhs_t> lhs(subregion_shape);
    const Array<mhs_t> mhs(subregion_shape);
    const Array<rhs_t> rhs(subregion_shape);
    const Array<result_t> expected(subregion_shape);
    const auto lhs_1d = lhs.accessor_contiguous_1d();
    const auto mhs_1d = mhs.accessor_contiguous_1d();
    const auto rhs_1d = rhs.accessor_contiguous_1d();
    const auto expected_1d = expected.accessor_contiguous_1d();

    test::Randomizer<lhs_t> randomizer_lhs(-12, 12);
    test::randomize(lhs_1d.get(), lhs.elements(), randomizer_lhs);
    test::Randomizer<mhs_t> randomizer_mhs(-10, 0);
    test::randomize(mhs_1d.get(), mhs.elements(), randomizer_mhs);
    test::Randomizer<rhs_t> randomizer_rhs(0, 10);
    test::randomize(rhs_1d.get(), rhs.elements(), randomizer_rhs);

    for (i64 i = 0; i < expected.elements(); ++i)
        expected_1d[i] = static_cast<result_t>(op_t{}(lhs_1d[i], mhs_1d[i], rhs_1d[i]));

    std::vector<Device> devices = {Device{}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const StreamGuard stream(device);
        const ArrayOption options(device, Allocator::MANAGED);
        INFO(device);

        const Array i_lhs = device.is_cpu() ? lhs : lhs.to(options);
        const Array i_mhs = device.is_cpu() ? mhs : mhs.to(options);
        const Array i_rhs = device.is_cpu() ? rhs : rhs.to(options);

        // Create output and ewise.
        const Array<result_t> result_padded(shape, options);
        auto result = result_padded.subregion(
                noa::indexing::Ellipsis{},
                noa::indexing::Slice{0, subregion_shape[3]});

        ewise_trinary(i_lhs, i_mhs, i_rhs, result, op_t{});
        REQUIRE(test::Matcher(test::MATCH_ABS, result, expected, 1e-6));
    }
}

TEMPLATE_TEST_CASE("unified::ewise_binary, broadcast", "[noa][unified]", i32, f32, f64, c32) {
    std::vector<Device> devices = {Device{"cpu"}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const auto shape = test::get_random_shape4_batched(3);
    for (auto& device: devices) {
        const auto stream = StreamGuard(device, StreamMode::DEFAULT);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        const Array<TestType> lhs(shape, options);
        const Array<TestType> rhs(shape, options);
        test::memset(lhs.get(), lhs.elements(), TestType{1});
        test::memset(rhs.get(), rhs.elements(), TestType{2});
        const Array<TestType> expected = ewise_binary(lhs, rhs, minus_t{});

        // Test passing a single value.
        const Array<TestType> output(shape, options);
        ewise_binary(lhs, TestType{2}, output, minus_t{});
        REQUIRE(test::Matcher(test::MATCH_ABS, expected, output, 1e-6));
        test::memset(output.get(), output.elements(), TestType{0});

        // Test broadcasting all dimensions.
        const Array<TestType> rhs_value(1, options);
        rhs_value(0, 0, 0, 0) = TestType{2};
        const View<const TestType> rhs_value_view = rhs_value.view();
        ewise_binary(lhs, rhs_value_view, output, minus_t{});
        REQUIRE(test::Matcher(test::MATCH_ABS, expected, output, 1e-6));
        test::memset(output.get(), output.elements(), TestType{0});

        // Test passing a single value.
        ewise_binary(TestType{1}, rhs, output, minus_t{});
        REQUIRE(test::Matcher(test::MATCH_ABS, expected, output, 1e-6));
        test::memset(output.get(), output.elements(), TestType{0});

        // Test broadcasting all dimensions.
        const Array<TestType> lhs_value(1, options);
        lhs_value(0, 0, 0, 0) = TestType{1};
        const View<const TestType> lhs_value_view = lhs_value.view();
        ewise_binary(lhs_value_view, rhs, output, minus_t{});
        REQUIRE(test::Matcher(test::MATCH_ABS, expected, output, 1e-6));
    }
}

TEMPLATE_TEST_CASE("unified::ewise_trinary, broadcast", "[noa][unified]", i32, f32, f64, c32) {
    std::vector<Device> devices = {Device{"cpu"}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const auto shape = test::get_random_shape4_batched(3);
    for (auto& device: devices) {
        const auto stream = StreamGuard(device, StreamMode::DEFAULT);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        const Array<TestType> lhs(shape, options);
        const Array<TestType> mhs(shape, options);
        const Array<TestType> rhs(shape, options);
        test::memset(lhs.get(), lhs.elements(), TestType{1});
        test::memset(mhs.get(), mhs.elements(), TestType{-3});
        test::memset(rhs.get(), rhs.elements(), TestType{2});
        const Array<TestType> expected = ewise_trinary(lhs, mhs, rhs, minus_t{});

        // Test passing a single value.
        const Array<TestType> output(shape, options);
        ewise_trinary(lhs, TestType{-3}, rhs, output, minus_t{});
        REQUIRE(test::Matcher(test::MATCH_ABS, expected, output, 1e-6));
        test::memset(output.get(), output.elements(), TestType{0});

        // Test broadcasting all dimensions.
        const Array<TestType> rhs_value(1, options);
        rhs_value(0, 0, 0, 0) = TestType{2};
        const View<const TestType> rhs_value_view = rhs_value.view();
        ewise_trinary(lhs, mhs, rhs_value_view, output, minus_t{});
        REQUIRE(test::Matcher(test::MATCH_ABS, expected, output, 1e-6));
        test::memset(output.get(), output.elements(), TestType{0});

        // Test passing a single value.
        ewise_trinary(TestType{1}, TestType{-3}, rhs, output, minus_t{});
        REQUIRE(test::Matcher(test::MATCH_ABS, expected, output, 1e-6));
        test::memset(output.get(), output.elements(), TestType{0});

        // Test broadcasting all dimensions.
        const Array<TestType> lhs_value(1, options);
        lhs_value(0, 0, 0, 0) = TestType{1};
        const View<const TestType> lhs_value_view = lhs_value.view();
        ewise_trinary(lhs_value_view, mhs, rhs, output, minus_t{});
        REQUIRE(test::Matcher(test::MATCH_ABS, expected, output, 1e-6));
    }
}
