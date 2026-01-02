#include <noa/runtime/Random.hpp>
#include <noa/runtime/Factory.hpp>
#include <noa/io/IO.hpp>

#include <noa/fft/Transform.hpp>
#include <noa/fft/Factory.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;
namespace nf = noa::fft;

TEMPLATE_TEST_CASE("fft::r2c(), c2r()", "", f32, f64) {
    const i64 ndim = GENERATE(1, 2, 3);
    const bool pad = GENERATE(true, false);
    INFO("ndim: " << ndim);
    INFO("pad: " << pad);

    const f64 abs_epsilon = std::is_same_v<TestType, f32> ? 1e-5 : 1e-9;
    auto subregion_shape = test::random_shape_batched(ndim);
    auto shape = subregion_shape;
    if (pad)
        shape += {0, ndim == 3 ? 11 : 0, ndim >= 2 ? 12 : 0, 13}; // don't change the dimensionality

    subregion_shape = nf::next_fast_shape(subregion_shape);
    shape = nf::next_fast_shape(shape);

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (const auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        SECTION("out-of-place") {
            Array expected = noa::random(noa::Uniform<TestType>{-5, 5}, shape, options);
            expected = expected.subregion(
                noa::Ellipsis{},
                noa::Slice{0, subregion_shape[1]},
                noa::Slice{0, subregion_shape[2]},
                noa::Slice{0, subregion_shape[3]});

            Array fft = noa::empty<Complex<TestType>>(shape.rfft(), options);
            fft = fft.subregion(
                noa::Ellipsis{},
                noa::Slice{0, subregion_shape[1]},
                noa::Slice{0, subregion_shape[2]},
                noa::Slice{0, subregion_shape[3] / 2 + 1});

            nf::r2c(expected, fft);
            const auto result = nf::c2r(fft, expected.shape());
            REQUIRE(test::allclose_abs_safe(expected, result, abs_epsilon));
        }

        SECTION("in-place") {
            const auto [real, fft] = nf::empty<TestType>(shape, options);
            noa::randomize(noa::Uniform<TestType>{-5, 5}, real);
            const auto expected = real.copy();
            nf::r2c(real, fft);
            nf::c2r(fft, real);
            REQUIRE(test::allclose_abs_safe(expected, real, abs_epsilon));
        }
    }
}

TEMPLATE_TEST_CASE("fft::r2c/c2r(), cpu vs gpu", "", f32, f64) {
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

    subregion_shape = nf::next_fast_shape(subregion_shape);
    shape = nf::next_fast_shape(shape);

    // Ensure CPU and GPU compute the FFTs concurrently.
    const auto guard = StreamGuard(Device{}, Stream::ASYNC);

    { // SECTION("out-of-place")
        const auto cpu_real = noa::random(noa::Uniform<TestType>{-5, 5}, subregion_shape);
        const auto gpu_buffer = noa::empty<TestType>(shape, ArrayOption("gpu", Allocator::MANAGED));
        const auto gpu_real = gpu_buffer.view().subregion(
            noa::Ellipsis{},
            noa::Slice{0, subregion_shape[1]},
            noa::Slice{0, subregion_shape[2]},
            noa::Slice{0, subregion_shape[3]});
        cpu_real.to(gpu_real);

        const auto cpu_rfft = nf::r2c(cpu_real);
        auto gpu_rfft = nf::r2c(gpu_real);
        REQUIRE(test::allclose_abs_safe(cpu_rfft, gpu_rfft, abs_epsilon));

        // c2r:
        gpu_rfft = cpu_rfft.to(ArrayOption("gpu", Allocator::MANAGED)).eval(); // wait because c2r overwrites cpu_rfft
        const auto cpu_result = nf::c2r(cpu_rfft, cpu_real.shape());
        const auto gpu_result = nf::c2r(gpu_rfft, cpu_real.shape());
        REQUIRE(test::allclose_abs_safe(cpu_result, gpu_result, abs_epsilon));
    }

    { // SECTION("in-place")
        const auto [cpu_real, cpu_fft] = nf::empty<TestType>(shape);
        const auto [gpu_real, gpu_fft] = nf::empty<TestType>(shape, {"gpu", Allocator::MANAGED});
        noa::randomize(noa::Uniform<TestType>{-5, 5}, cpu_real);
        cpu_real.to(gpu_real);

        nf::r2c(cpu_real, cpu_fft);
        nf::r2c(gpu_real, gpu_fft);
        REQUIRE(test::allclose_abs_safe(cpu_fft, gpu_fft, abs_epsilon));

        cpu_fft.to(gpu_fft);
        gpu_fft.eval(); // c2r overwrites cpu_fft
        nf::c2r(cpu_fft, cpu_real);
        nf::c2r(gpu_fft, gpu_real);
        REQUIRE(test::allclose_abs_safe(cpu_real, gpu_real, abs_epsilon));
    }
}

TEMPLATE_TEST_CASE("fft::c2c()", "", f32, f64) {
    const i64 ndim = GENERATE(1, 2, 3);
    const bool pad = GENERATE(true, false);
    INFO("ndim: " << ndim);
    INFO("pad: " << pad);

    const f64 abs_epsilon = std::is_same_v<TestType, f32> ? 1e-5 : 1e-9;
    auto subregion_shape = test::random_shape_batched(ndim);
    auto shape = subregion_shape;
    if (pad)
        shape += {0, ndim == 3 ? 11 : 0, ndim >= 2 ? 12 : 0, 13}; // don't change the dimensionality

    subregion_shape = nf::next_fast_shape(subregion_shape);
    shape = nf::next_fast_shape(shape);

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
                    noa::Ellipsis{},
                    noa::Slice{0, subregion_shape[1]},
                    noa::Slice{0, subregion_shape[2]},
                    noa::Slice{0, subregion_shape[3]});
            const auto fft = nf::c2c(expected, nf::Sign::FORWARD);
            const auto result = nf::c2c(fft, nf::Sign::BACKWARD);
            REQUIRE(test::allclose_abs_safe(expected, result, abs_epsilon));
        }

        SECTION("in-place") {
            const Array input = noa::random(noa::Uniform<complex_t>{-5, 5}, shape, options);
            const auto expected = input.copy();
            nf::c2c(expected, expected, nf::Sign::FORWARD);
            nf::c2c(expected, expected, nf::Sign::BACKWARD);
            REQUIRE(test::allclose_abs_safe(input, expected, abs_epsilon));
        }
    }
}

TEST_CASE("fft, caching plans") {
    auto devices = std::vector<Device>{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto device: devices) {
        INFO(device);
        nf::clear_cache(device);
        nf::set_cache_limit(5, device);

        auto a = Array<f32>(64, {.device = device});
        auto a_rfft = nf::r2c(a).eval();
        auto a_s = nf::cache_size(device);
        REQUIRE((a_s == -1 or a_s > 0));

        auto b = Array<f32>(128, {.device = device});
        auto b_rfft = nf::r2c(b).eval();
        auto b_s = nf::cache_size(device);
        REQUIRE((b_s == -1 or b_s >= a_s));

        nf::clear_cache(device);
        REQUIRE(nf::cache_size(device) <= 0);

        auto c = Array<f32>({2, 1, 1024, 1024}, {.device = device});
        auto c_rfft = nf::r2c(c).eval();
        auto c_s = nf::cache_size(device);

        auto d = Array<f32>({2, 1, 1024, 1024}, {.device = device});
        auto d_rfft = nf::r2c(d).eval();
        auto d_s = nf::cache_size(device);

        REQUIRE(c_s == d_s);
    }
}

TEST_CASE("fft, caching plans, shared workspace") {
    if (not Device::is_any_gpu())
        return;

    const auto devices = std::vector<Device>{"gpu"};
    const auto shape = Shape<isize, 4>{15, 1, 512, 512};

    for (auto device: devices) {
        INFO(device);

        auto a0 = noa::random<f32>(noa::Uniform{-1.f, 1.f}, shape, {.device = device});
        auto a1 = a0.copy();
        auto b0 = Array<c32>(shape.rfft(), {.device = device});
        auto b1 = Array<c32>(shape.rfft(), {.device = device});
        nf::set_cache_limit(50, device);

        // auto print_memory_usage = [init = device.memory_capacity().free, previous = device.memory_capacity().free, device] () mutable {
        //     auto current = device.memory_capacity().free;
        //     auto diff1 = (static_cast<f64>(init) - static_cast<f64>(current)) * 10e-6;
        //     auto diff2 = (static_cast<f64>(previous) - static_cast<f64>(current)) * 10e-6;
        //     fmt::println("diff={:.2f}, {:.2f}, n_cached={}", diff1, diff2, nf::cache_size(device));
        //     previous = current;
        // };

        // Create plans the normal way.
        nf::clear_cache(device);
        auto init0 = device.memory_capacity().free;
        auto max0 = init0;

        for (auto i: noa::irange(shape[0])) {
            auto offset = noa::Offset{i};
            nf::r2c(a0.subregion(offset), b0.subregion(offset), {.plan_only = true});
            max0 = std::min(max0, device.memory_capacity().free);
        }
        nf::r2c(a0, b0);
        a0.eval();
        max0 = std::min(max0, device.memory_capacity().free);

        // Shared-workspace.
        nf::clear_cache(device);
        auto init1 = device.memory_capacity().free;
        auto max1 = init1;

        for (auto i: noa::irange(shape[0])) {
            auto offset = noa::Offset{i};
            nf::r2c(a1.subregion(offset), b1.subregion(offset), {.record_and_share_workspace = true});
            max1 = std::min(max1, device.memory_capacity().free);
        }
        nf::r2c(a1, b1);
        a1.eval();
        max1 = std::min(max1, device.memory_capacity().free);

        nf::clear_cache(device);
        nf::set_cache_limit(8, device);

        REQUIRE(max0 <= max1);

        // Set the workspace.
        nf::clear_cache(device);
        REQUIRE(nf::workspace_left_to_allocate(device) == 0);
        nf::r2c(a0, b0, {.record_and_share_workspace = true});
        auto n_bytes = nf::workspace_left_to_allocate(device);
        if (n_bytes == 0)
            return;
        auto buffer = Array<std::byte>(n_bytes, {.device = device, .allocator = Allocator::DEFAULT});
        nf::set_workspace(device, buffer);
        REQUIRE(nf::workspace_left_to_allocate(device) == 0);
    }
}
