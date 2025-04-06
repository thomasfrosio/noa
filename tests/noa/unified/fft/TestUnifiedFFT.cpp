#include <noa/unified/fft/Transform.hpp>
#include <noa/unified/fft/Factory.hpp>
#include <noa/unified/Random.hpp>
#include <noa/unified/Factory.hpp>
#include <noa/unified/IO.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;

TEMPLATE_TEST_CASE("unified::fft::r2c() -> c2r()", "", f32, f64) {
    const i64 ndim = GENERATE(1, 2, 3);
    const bool pad = GENERATE(true, false);
    INFO("ndim: " << ndim);
    INFO("pad: " << pad);

    const f64 abs_epsilon = std::is_same_v<TestType, f32> ? 1e-5 : 1e-9;
    auto subregion_shape = test::random_shape_batched(ndim);
    auto shape = subregion_shape;
    if (pad)
        shape += {0, ndim == 3 ? 11 : 0, ndim >= 2 ? 12 : 0, 13}; // don't change the dimensionality

    subregion_shape = noa::fft::next_fast_shape(subregion_shape);
    shape = noa::fft::next_fast_shape(shape);

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (const auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        SECTION("out-of-place") {
            Array expected = noa::random(noa::Uniform<TestType>{-5, 5}, shape, options);
            expected = expected.subregion(
                noa::indexing::Ellipsis{},
                noa::indexing::Slice{0, subregion_shape[1]},
                noa::indexing::Slice{0, subregion_shape[2]},
                noa::indexing::Slice{0, subregion_shape[3]});

            Array fft = noa::empty<Complex<TestType>>(shape.rfft(), options);
            fft = fft.subregion(
                noa::indexing::Ellipsis{},
                noa::indexing::Slice{0, subregion_shape[1]},
                noa::indexing::Slice{0, subregion_shape[2]},
                noa::indexing::Slice{0, subregion_shape[3] / 2 + 1});

            noa::fft::r2c(expected, fft);
            const auto result = noa::fft::c2r(fft, expected.shape());
            REQUIRE(test::allclose_abs_safe(expected, result, abs_epsilon));
        }

        SECTION("in-place") {
            const auto [real, fft] = noa::fft::empty<TestType>(shape, options);
            noa::randomize(noa::Uniform<TestType>{-5, 5}, real);
            const auto expected = real.copy();
            noa::fft::r2c(real, fft);
            noa::fft::c2r(fft, real);
            REQUIRE(test::allclose_abs_safe(expected, real, abs_epsilon));
        }
    }
}

TEMPLATE_TEST_CASE("unified::fft::r2c/c2r(), cpu vs gpu", "", f32, f64) {
    if (not Device::is_any_gpu())
        return;

    const i64 ndim = GENERATE(1, 2, 3);
    const bool pad = GENERATE(true, false);
    INFO("ndim: " << ndim);
    INFO("pad: " << pad);

    // most are okay at 1e-5 but there are some outliers...
    const f64 abs_epsilon = std::is_same_v<TestType, f32> ? 1e-3 : 1e-9;
    auto subregion_shape = test::random_shape_batched(ndim);
    auto shape = subregion_shape;
    if (pad)
        shape += {0, ndim == 3 ? 11 : 0, ndim >= 2 ? 12 : 0, 13}; // don't change the dimensionality

    subregion_shape = noa::fft::next_fast_shape(subregion_shape);
    shape = noa::fft::next_fast_shape(shape);

    // Ensure CPU and GPU compute the FFTs concurrently.
    const auto guard = StreamGuard(Device{}, Stream::ASYNC);

    { // SECTION("out-of-place")
        const auto cpu_real = noa::random(noa::Uniform<TestType>{-5, 5}, subregion_shape);
        const auto gpu_buffer = noa::empty<TestType>(shape, ArrayOption("gpu", Allocator::MANAGED));
        const auto gpu_real = gpu_buffer.view().subregion(
            noa::indexing::Ellipsis{},
            noa::indexing::Slice{0, subregion_shape[1]},
            noa::indexing::Slice{0, subregion_shape[2]},
            noa::indexing::Slice{0, subregion_shape[3]});
        cpu_real.to(gpu_real);

        const auto cpu_rfft = noa::fft::r2c(cpu_real);
        auto gpu_rfft = noa::fft::r2c(gpu_real);
        REQUIRE(test::allclose_abs_safe(cpu_rfft, gpu_rfft, abs_epsilon));

        // c2r:
        gpu_rfft = cpu_rfft.to(ArrayOption("gpu", Allocator::MANAGED)).eval(); // wait because c2r overwrites cpu_rfft
        const auto cpu_result = noa::fft::c2r(cpu_rfft, cpu_real.shape());
        const auto gpu_result = noa::fft::c2r(gpu_rfft, cpu_real.shape());
        noa::io::write(cpu_result.subregion(0), test::NOA_DATA_PATH / "test_cpu.mrc");
        noa::io::write(gpu_result.subregion(0), test::NOA_DATA_PATH / "test_gpu.mrc");
        REQUIRE(test::allclose_abs_safe(cpu_result, gpu_result, abs_epsilon));
    }

    { // SECTION("in-place")
        const auto [cpu_real, cpu_fft] = noa::fft::empty<TestType>(shape);
        const auto [gpu_real, gpu_fft] = noa::fft::empty<TestType>(shape, {"gpu", Allocator::MANAGED});
        noa::randomize(noa::Uniform<TestType>{-5, 5}, cpu_real);
        cpu_real.to(gpu_real);

        noa::fft::r2c(cpu_real, cpu_fft);
        noa::fft::r2c(gpu_real, gpu_fft);
        REQUIRE(test::allclose_abs_safe(cpu_fft, gpu_fft, abs_epsilon));

        cpu_fft.to(gpu_fft);
        gpu_fft.eval(); // c2r overwrites cpu_fft
        noa::fft::c2r(cpu_fft, cpu_real);
        noa::fft::c2r(gpu_fft, gpu_real);
        REQUIRE(test::allclose_abs_safe(cpu_real, gpu_real, abs_epsilon));
    }
}

TEMPLATE_TEST_CASE("unified::fft::c2c()", "", f32, f64) {
    const i64 ndim = GENERATE(1, 2, 3);
    const bool pad = GENERATE(true, false);
    INFO("ndim: " << ndim);
    INFO("pad: " << pad);

    const f64 abs_epsilon = std::is_same_v<TestType, f32> ? 1e-5 : 1e-9;
    auto subregion_shape = test::random_shape_batched(ndim);
    auto shape = subregion_shape;
    if (pad)
        shape += {0, ndim == 3 ? 11 : 0, ndim >= 2 ? 12 : 0, 13}; // don't change the dimensionality

    subregion_shape = noa::fft::next_fast_shape(subregion_shape);
    shape = noa::fft::next_fast_shape(shape);

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (const auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);
        using complex_t = Complex<TestType>;

        SECTION("out-of-place") {
            Array expected = noa::random(noa::Uniform<complex_t>{-5, 5}, shape, options);
            expected = expected.subregion(
                    noa::indexing::Ellipsis{},
                    noa::indexing::Slice{0, subregion_shape[1]},
                    noa::indexing::Slice{0, subregion_shape[2]},
                    noa::indexing::Slice{0, subregion_shape[3]});
            const auto fft = noa::fft::c2c(expected, noa::fft::Sign::FORWARD);
            const auto result = noa::fft::c2c(fft, noa::fft::Sign::BACKWARD);
            REQUIRE(test::allclose_abs_safe(expected, result, abs_epsilon));
        }

        SECTION("in-place") {
            const Array input = noa::random(noa::Uniform<complex_t>{-5, 5}, shape, options);
            const auto expected = input.copy();
            noa::fft::c2c(expected, expected, noa::fft::Sign::FORWARD);
            noa::fft::c2c(expected, expected, noa::fft::Sign::BACKWARD);
            REQUIRE(test::allclose_abs_safe(input, expected, abs_epsilon));
        }
    }
}
