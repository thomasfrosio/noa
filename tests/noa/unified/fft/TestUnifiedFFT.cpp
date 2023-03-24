#include <noa/unified/fft/Transform.hpp>
#include <noa/unified/fft/Factory.hpp>
#include <noa/unified/math/Random.hpp>
#include <noa/unified/memory/Factory.hpp>

#include <catch2/catch.hpp>
#include "Helpers.h"
using namespace ::noa;

TEMPLATE_TEST_CASE("unified::fft::r2c() -> c2r()", "[noa][unified]", f32, f64) {
    const i64 ndim = GENERATE(1, 2, 3);
    const bool pad = GENERATE(true, false);
    INFO("ndim: " << ndim);
    INFO("pad: " << pad);

    const f64 abs_epsilon = std::is_same_v<TestType, f32> ? 1e-5 : 1e-9;
    auto subregion_shape = test::get_random_shape4_batched(ndim);
    auto shape = subregion_shape;
    if (pad)
        shape += {0, ndim == 3 ? 11 : 0, ndim >= 2 ? 12 : 0, 13}; // don't change the dimensionality

    subregion_shape = noa::fft::next_fast_shape(subregion_shape);
    shape = noa::fft::next_fast_shape(shape);

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (const auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        AND_THEN("out-of-place") {
            Array expected = noa::math::random<TestType>(noa::math::uniform_t{}, shape, -5, 5, options);
            expected = expected.subregion(
                    noa::indexing::ellipsis_t{},
                    noa::indexing::slice_t{0, subregion_shape[1]},
                    noa::indexing::slice_t{0, subregion_shape[2]},
                    noa::indexing::slice_t{0, subregion_shape[3]});

            Array fft = noa::memory::empty<Complex<TestType>>(shape.fft(), options);
            fft = fft.subregion(
                    noa::indexing::ellipsis_t{},
                    noa::indexing::slice_t{0, subregion_shape[1]},
                    noa::indexing::slice_t{0, subregion_shape[2]},
                    noa::indexing::slice_t{0, subregion_shape[3] / 2 + 1});

            noa::fft::r2c(expected, fft);
            const auto result = noa::fft::c2r(fft, expected.shape());
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, result, abs_epsilon));
        }

        AND_THEN("in-place") {
            const auto [real, fft] = noa::fft::empty<TestType>(shape, options);
            noa::math::randomize(noa::math::uniform_t{}, real, -5, 5);
            const auto expected = real.copy();
            noa::fft::r2c(real, fft);
            noa::fft::c2r(fft, real);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, real, abs_epsilon));
        }
    }
}

TEMPLATE_TEST_CASE("unified::fft::r2c/c2r(), cpu vs gpu", "[noa][unified]", f32, f64) {
    if (!Device::is_any(DeviceType::GPU))
        return;

    const i64 ndim = GENERATE(1, 2, 3);
    const bool pad = GENERATE(true, false);
    INFO("ndim: " << ndim);
    INFO("pad: " << pad);

    // most are okay at 1e-5 but there are some outliers...
    const f64 abs_epsilon = std::is_same_v<TestType, f32> ? 1e-3 : 1e-9;
    auto subregion_shape = test::get_random_shape4_batched(ndim);
    auto shape = subregion_shape;
    if (pad)
        shape += {0, ndim == 3 ? 11 : 0, ndim >= 2 ? 12 : 0, 13}; // don't change the dimensionality

    subregion_shape = noa::fft::next_fast_shape(subregion_shape);
    shape = noa::fft::next_fast_shape(shape);

    // Ensure CPU and GPU compute the FFTs concurrently.
    const auto guard = StreamGuard(Device{}, StreamMode::ASYNC);

    AND_THEN("out-of-place") {
        const auto cpu_real = noa::math::random<TestType>(noa::math::uniform_t{}, subregion_shape, -5, 5);
        const auto gpu_buffer = noa::memory::empty<TestType>(shape, ArrayOption(Device("gpu"), Allocator::MANAGED));
        const auto gpu_real = gpu_buffer.view().subregion(
                noa::indexing::ellipsis_t{},
                noa::indexing::slice_t{0, subregion_shape[1]},
                noa::indexing::slice_t{0, subregion_shape[2]},
                noa::indexing::slice_t{0, subregion_shape[3]});
        cpu_real.to(gpu_real);

        const auto cpu_fft = noa::fft::r2c(cpu_real);
        auto gpu_fft = noa::fft::r2c(gpu_real);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, cpu_fft, gpu_fft, abs_epsilon));

        // c2r:
        gpu_fft = cpu_fft.to(ArrayOption(Device("gpu"), Allocator::MANAGED)).eval(); // wait because c2r overwrites cpu_fft
        const auto cpu_result = noa::fft::c2r(cpu_fft, cpu_real.shape());
        const auto gpu_result = noa::fft::c2r(gpu_fft, cpu_real.shape());
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, cpu_result, gpu_result, abs_epsilon));
    }

    AND_THEN("in-place") {
        const auto [cpu_real, cpu_fft] = noa::fft::empty<TestType>(shape);
        const auto [gpu_real, gpu_fft] = noa::fft::empty<TestType>(shape, ArrayOption(Device("gpu"), Allocator::MANAGED));
        noa::math::randomize(noa::math::uniform_t{}, cpu_real, -5, 5);
        cpu_real.to(gpu_real);

        noa::fft::r2c(cpu_real, cpu_fft);
        noa::fft::r2c(gpu_real, gpu_fft);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, cpu_fft, gpu_fft, abs_epsilon));

        cpu_fft.to(gpu_fft);
        gpu_fft.eval(); // c2r overwrites cpu_fft
        noa::fft::c2r(cpu_fft, cpu_real);
        noa::fft::c2r(gpu_fft, gpu_real);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, cpu_real, gpu_real, abs_epsilon));
    }
}

TEMPLATE_TEST_CASE("unified::fft::c2c()", "[noa][unified]", f32, f64) {
    const i64 ndim = GENERATE(1, 2, 3);
    const bool pad = GENERATE(true, false);
    INFO("ndim: " << ndim);
    INFO("pad: " << pad);

    const f64 abs_epsilon = std::is_same_v<TestType, f32> ? 1e-5 : 1e-9;
    auto subregion_shape = test::get_random_shape4_batched(ndim);
    auto shape = subregion_shape;
    if (pad)
        shape += {0, ndim == 3 ? 11 : 0, ndim >= 2 ? 12 : 0, 13}; // don't change the dimensionality

    subregion_shape = noa::fft::next_fast_shape(subregion_shape);
    shape = noa::fft::next_fast_shape(shape);

    std::vector<Device> devices{Device("cpu")};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (const auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        using complex_t = Complex<TestType>;

        AND_THEN("out-of-place") {
            Array expected = noa::math::random<complex_t>(noa::math::uniform_t{}, shape, -5, 5, options);
            expected = expected.subregion(
                    noa::indexing::ellipsis_t{},
                    noa::indexing::slice_t{0, subregion_shape[1]},
                    noa::indexing::slice_t{0, subregion_shape[2]},
                    noa::indexing::slice_t{0, subregion_shape[3]});
            const auto fft = noa::fft::c2c(expected, noa::fft::Sign::FORWARD);
            const auto result = noa::fft::c2c(fft, noa::fft::Sign::BACKWARD);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected, result, abs_epsilon));
        }

        AND_THEN("in-place") {
            const Array input = noa::math::random<complex_t>(noa::math::uniform_t{}, shape, -5, 5, options);
            const auto expected = input.copy();
            noa::fft::c2c(expected, expected, noa::fft::Sign::FORWARD);
            noa::fft::c2c(expected, expected, noa::fft::Sign::BACKWARD);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, input, expected, abs_epsilon));
        }
    }
}
