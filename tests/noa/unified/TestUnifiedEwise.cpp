#include <noa/unified/Array.hpp>
#include <noa/unified/Ewise.hpp>
#include <noa/unified/Factory.hpp>

#include <catch2/catch.hpp>

#include "Helpers.h"

using namespace noa::types;

TEMPLATE_TEST_CASE("unified::ewise - simple", "[noa][unified]", i32, f64) {
    auto shape = test::get_random_shape4_batched(3);
    auto lhs = noa::empty<TestType>(shape);
    auto rhs = noa::like(lhs);
    auto expected = noa::like<i64>(lhs);

    auto randomizer = test::Randomizer<TestType>(-50, 50);
    auto lhs_span = lhs.span();
    auto rhs_span = rhs.span();
    auto expected_span = expected.span();
    for (i64 i = 0; i < lhs.elements(); ++i) {
        lhs_span[i] = randomizer.get();
        rhs_span[i] = randomizer.get();
        expected_span[i] = static_cast<i64>(lhs_span[i] + rhs_span[i]);
    }

    std::vector<Device> devices{"cpu"};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        auto stream = StreamGuard(device, StreamMode::DEFAULT);
        const auto options = ArrayOption{device, "managed"};
        INFO(device);

        if (device != lhs.device()) {
            lhs = lhs.to(options);
            rhs = rhs.to(options);
        }
        auto output = noa::like<i64>(lhs);

        noa::ewise(noa::wrap(lhs, rhs), output, noa::Plus{});
        REQUIRE(test::Matcher<i64>(test::MATCH_ABS, output, expected, 0));
    }
}

TEMPLATE_TEST_CASE("unified::ewise - broadcast", "[noa][unified]", i32, f32, f64, c32) {
    std::vector<Device> devices{"cpu"};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    using value_t = TestType;
    value_t lhs_value, rhs_value;
    if constexpr (noa::traits::is_complex_v<value_t>) {
        lhs_value = {1, 1};
        rhs_value = {2, 2};
    } else {
        lhs_value = 1;
        rhs_value = 2;
    }

    const auto shapes = std::array{
            test::get_random_shape4_batched(1),
            test::get_random_shape4_batched(2),
            test::get_random_shape4_batched(3),
            test::get_random_shape4_batched(4)};

    for (const auto& shape: shapes) {
        for (auto& device: devices) {
            const auto stream = StreamGuard(device, StreamMode::DEFAULT);
            const auto options = ArrayOption(device, "managed");

            const Array<TestType> lhs(shape, options);
            const Array<TestType> rhs(shape, options);
            test::memset(lhs.get(), lhs.elements(), lhs_value);
            test::memset(rhs.get(), rhs.elements(), rhs_value);
            const auto expected = noa::like(lhs);
            noa::ewise(noa::wrap(lhs, rhs), expected, noa::Minus{});

            // Test passing a single value.
            const Array<TestType> output(shape, options);
            noa::ewise(noa::wrap(lhs, rhs_value), output, noa::Minus{});
            REQUIRE(test::Matcher<value_t>(test::MATCH_ABS, expected, output, 1e-6));
            test::memset(output.get(), output.elements(), value_t{});

            // Test passing a single value.
            noa::ewise(noa::wrap(lhs_value, rhs), output, noa::Minus{});
            REQUIRE(test::Matcher<value_t>(test::MATCH_ABS, expected, output, 1e-6));
            test::memset(output.get(), output.elements(), value_t{});

            // Test broadcasting all dimensions.
            noa::ewise(noa::wrap(lhs, View(&rhs_value, 1).to(options)), output, noa::Minus{});
            REQUIRE(test::Matcher<value_t>(test::MATCH_ABS, expected, output, 1e-6));
            test::memset(output.get(), output.elements(), value_t{});

            // Test broadcasting all dimensions.
            const Array<TestType> lhs_(1, options);
            lhs_.get()[0] = lhs_value;
            noa::ewise(noa::wrap(lhs_.view(), rhs), output, noa::Minus{});
            REQUIRE(test::Matcher<value_t>(test::MATCH_ABS, expected, output, 1e-6));
        }
    }
}

TEST_CASE("unified::ewise - zip vs wrap", "[noa][unified]") {
    std::vector<Device> devices{"cpu"};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const auto shapes = std::array{
            test::get_random_shape4_batched(1),
            test::get_random_shape4_batched(2),
            test::get_random_shape4_batched(3),
            test::get_random_shape4_batched(4)};

    for (const auto& shape: shapes) {
        auto lhs = noa::empty<i64>(shape);
        auto mhs = noa::like(lhs);
        auto rhs = noa::like(lhs);

        test::Randomizer<i64> randomizer(-100, 30);
        test::randomize(lhs.get(), lhs.elements(), randomizer);
        test::randomize(mhs.get(), mhs.elements(), randomizer);
        test::randomize(rhs.get(), rhs.elements(), randomizer);

        // Generate the expected results.
        auto expected0 = noa::like<f64>(lhs);
        auto expected1 = noa::like<f64>(lhs);
        for (i64 i{0}; i < lhs.elements(); ++i) {
            auto sum = lhs.get()[i] + mhs.get()[i] + rhs.get()[i];
            expected0.get()[i] = static_cast<f64>(sum);
            expected1.get()[i] = static_cast<f64>(noa::abs(sum));
        }

        for (auto& device: devices) {
            const auto stream = StreamGuard(device, StreamMode::DEFAULT);
            const auto options = ArrayOption(device, "managed");

            if (device != lhs.device()) {
                lhs = lhs.to(options);
                mhs = mhs.to(options);
                rhs = rhs.to(options);
            }

            auto output0 = noa::like<f64>(lhs);
            auto output1 = noa::like(output0);
            noa::ewise(noa::zip(lhs, mhs, rhs), noa::wrap(output0, output1),
                       []NOA_HD(const Tuple<i64&, i64&, i64&>& inputs, f64& o0, f64& o1) {
                           const auto sum = inputs.apply([](auto& ... i) {
                               return (i + ...);
                           });
                           o0 = static_cast<f64>(sum);
                           o1 = static_cast<f64>(noa::abs(sum));
                       });

            test::Matcher<f64>(test::MATCH_ABS_SAFE, output0, expected0, 1e-8);
            test::Matcher<f64>(test::MATCH_ABS_SAFE, output1, expected1, 1e-8);
        }
    }
}

TEST_CASE("unified::ewise - no inputs/outputs", "[noa][unified]") {
    std::vector<Device> devices{"cpu"};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const auto shape = test::get_random_shape4_batched(4);
    for (auto& device: devices) {
        const auto stream = StreamGuard(device, StreamMode::ASYNC);
        const auto options = ArrayOption(device, "managed");
        INFO(device);

        noa::ewise({}, {}, noa::Copy{}); // valid, do nothing

        const auto array = noa::empty<i64>(shape, options);
        noa::ewise(2, array, noa::Copy{});
        REQUIRE(test::Matcher<i64>(test::MATCH_ABS_SAFE, array, 2, 1e-8));

        noa::ewise({}, array, noa::Fill{3});
        REQUIRE(test::Matcher<i64>(test::MATCH_ABS_SAFE, array, 3, 1e-8));

        noa::ewise(array,  {}, []NOA_HD(i64& i) { i = 4; });
        REQUIRE(test::Matcher<i64>(test::MATCH_ABS_SAFE, array, 4, 1e-8));
    }
}
