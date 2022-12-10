#include <noa/FFT.h>
#include <noa/Signal.h>
#include <noa/Math.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::signal::fft, correlation peak 2D", "[noa][unified]", float, double) {
    using namespace ::noa;
    dim4_t shape = test::getRandomShape(2);
    shape[2] += 200;
    shape[3] += 200;

    test::Randomizer<float> randomizer{-30, 30};
    const float2_t radius{25, 25};
    const float taper = 7;
    const float2_t lhs_center{shape[2] / 2, shape[3] / 2};
    const float2_t rhs_shift{randomizer.get(), randomizer.get()};
    const float2_t rhs_center = lhs_center + rhs_shift;

    std::vector<Device> devices{Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const std::array<signal::CorrelationMode, 4> modes{
            signal::CONVENTIONAL_CORRELATION,
            signal::MUTUAL_CORRELATION,
            signal::PHASE_CORRELATION,
            signal::DOUBLE_PHASE_CORRELATION};

    for (auto correlation_mode: modes) {
        for (auto& device: devices) {
            INFO(device);
            INFO(correlation_mode);
            StreamGuard stream(device);
            ArrayOption options(device, Allocator::MANAGED);

            auto [lhs, lhs_fft] = fft::empty<TestType>(shape, options);
            auto [rhs, rhs_fft] = fft::empty<TestType>(shape, options);
            Array xmap = memory::empty<TestType>(shape, options);
            Array buffer = memory::like(lhs_fft);

            signal::rectangle({}, lhs, lhs_center, radius, taper);
            signal::rectangle({}, rhs, rhs_center, radius, taper);
            fft::r2c(lhs, lhs_fft);
            fft::r2c(rhs, rhs_fft);

            float2_t shift_centered;
            float2_t shift_not_centered;

            {
                signal::fft::xmap<fft::H2FC>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, fft::NORM_DEFAULT, buffer);

                shift_centered = signal::fft::xpeak2D<fft::FC2FC>(xmap);
                shift_centered -= lhs_center;
                if (correlation_mode == signal::DOUBLE_PHASE_CORRELATION)
                    shift_centered /= 2;

                INFO("Excepted shift: " << rhs_shift);
                INFO("Computed shift: " << -shift_centered);
                REQUIRE_THAT(shift_centered[0], Catch::WithinAbs(static_cast<double>(-rhs_shift[0]), 5e-2));
                REQUIRE_THAT(shift_centered[1], Catch::WithinAbs(static_cast<double>(-rhs_shift[1]), 5e-2));
            }

            {
                signal::fft::xmap<fft::H2F>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, fft::NORM_DEFAULT, buffer);

                shift_not_centered = signal::fft::xpeak2D<fft::F2F>(xmap);
                shift_not_centered -= lhs_center;
                if (correlation_mode == signal::DOUBLE_PHASE_CORRELATION)
                    shift_not_centered /= 2;

                INFO("Excepted shift: " << rhs_shift);
                INFO("Computed shift: " << -shift_not_centered);
                REQUIRE_THAT(shift_not_centered[0], Catch::WithinAbs(static_cast<double>(-rhs_shift[0]), 5e-2));
                REQUIRE_THAT(shift_not_centered[1], Catch::WithinAbs(static_cast<double>(-rhs_shift[1]), 5e-2));
            }

            REQUIRE_THAT(shift_not_centered[0], Catch::WithinAbs(static_cast<double>(shift_centered[0]), 5e-5));
            REQUIRE_THAT(shift_not_centered[1], Catch::WithinAbs(static_cast<double>(shift_centered[1]), 5e-5));
        }
    }
}

TEMPLATE_TEST_CASE("unified::signal::fft, correlation peak 2D batched", "[noa][unified]", float, double) {
    using namespace ::noa;
    dim4_t shape = test::getRandomShapeBatched(2);
    shape[2] += 200;
    shape[3] += 200;

    test::Randomizer<float> randomizer{-30, 30};
    const float2_t radius{25, 25};
    const float taper = 7;
    const float2_t lhs_center{shape[2] / 2, shape[3] / 2};
    const float2_t rhs_shift{randomizer.get(), randomizer.get()};
    const float2_t rhs_center = lhs_center + rhs_shift;

    std::vector<Device> devices{Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const std::array<signal::CorrelationMode, 4> modes{
            signal::CONVENTIONAL_CORRELATION,
            signal::MUTUAL_CORRELATION,
            signal::PHASE_CORRELATION,
            signal::DOUBLE_PHASE_CORRELATION};

    for (auto correlation_mode: modes) {
        for (auto& device: devices) {
            StreamGuard stream(device);
            ArrayOption options(device, Allocator::MANAGED);

            auto [lhs, lhs_fft] = fft::empty<TestType>(shape, options);
            auto [rhs, rhs_fft] = fft::empty<TestType>(shape, options);
            Array xmap = memory::empty<TestType>(shape, options);
            Array buffer = memory::like(lhs_fft);
            Array shifts = memory::empty<float2_t>({1, 1, 1, shape[0]}, options);

            signal::rectangle({}, lhs, lhs_center, radius, taper);
            signal::rectangle({}, rhs, rhs_center, radius, taper);
            fft::r2c(lhs, lhs_fft);
            fft::r2c(rhs, rhs_fft);

            // Centered:
            {
                signal::fft::xmap<fft::H2FC>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, fft::NORM_DEFAULT, buffer);
                signal::fft::xpeak2D<fft::FC2FC>(xmap, shifts);
                shifts.eval();

                for (dim_t i = 0; i < shape[0]; ++i) {
                    float2_t shift = shifts[i];
                    shift -= lhs_center;
                    if (correlation_mode == signal::DOUBLE_PHASE_CORRELATION)
                        shift /= 2;

                    INFO("Excepted shift: " << rhs_shift);
                    INFO("Computed shift: " << -shift);
                    REQUIRE_THAT(shift[0], Catch::WithinAbs(static_cast<double>(-rhs_shift[0]), 5e-2));
                    REQUIRE_THAT(shift[1], Catch::WithinAbs(static_cast<double>(-rhs_shift[1]), 5e-2));
                }
            }

            // Non-centered:
            {
                signal::fft::xmap<fft::H2F>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, fft::NORM_DEFAULT, buffer);
                signal::fft::xpeak2D<fft::F2F>(xmap, shifts);
                shifts.eval();

                for (dim_t i = 0; i < shape[0]; ++i) {
                    float2_t shift = shifts[i];
                    shift -= lhs_center;
                    if (correlation_mode == signal::DOUBLE_PHASE_CORRELATION)
                        shift /= 2;

                    INFO("Excepted shift: " << rhs_shift);
                    INFO("Computed shift: " << -shift);
                    REQUIRE_THAT(shift[0], Catch::WithinAbs(static_cast<double>(-rhs_shift[0]), 5e-2));
                    REQUIRE_THAT(shift[1], Catch::WithinAbs(static_cast<double>(-rhs_shift[1]), 5e-2));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::signal::fft, correlation peak 2D, cpu vs gpu", "[noa][unified]", float, double) {
    using namespace ::noa;
    dim4_t shape = test::getRandomShape(2);
    shape[2] += 100;
    shape[3] += 100;

    test::Randomizer<float> float_randomizer{-20, 20};
    const float2_t radius{25, 25};
    const float taper = 7;
    const float2_t lhs_center{shape[2] / 2, shape[3] / 2};
    const float2_t rhs_shift{float_randomizer.get(), float_randomizer.get()};
    const float2_t rhs_center = lhs_center + rhs_shift;

    if (!Device::any(Device::GPU))
        return;

    const auto peak_mode = GENERATE(signal::PEAK_PARABOLA_1D, signal::PEAK_COM);
    test::Randomizer<long> long_randomizer{1, peak_mode == signal::PEAK_COM ? 8 : 32};
    const long2_t peak_window = {long_randomizer.get(), long_randomizer.get()};

    INFO(peak_mode);
    INFO(peak_window);

    // Centered:
    {
        auto [lhs, lhs_fft] = fft::empty<TestType>(shape);
        auto [rhs, rhs_fft] = fft::empty<TestType>(shape);
        Array xmap = memory::empty<TestType>(shape);

        signal::rectangle({}, lhs, lhs_center, radius, taper);
        signal::rectangle({}, rhs, rhs_center, radius, taper);
        fft::r2c(lhs, lhs_fft);
        fft::r2c(rhs, rhs_fft);
        signal::fft::xmap<fft::H2FC>(lhs_fft, rhs_fft, xmap);
        const auto xmap_gpu = xmap.to(Device("gpu"));

        const float2_t cpu_shift = signal::fft::xpeak2D<fft::FC2FC>(xmap, {}, peak_mode, peak_window) - lhs_center;
        const float2_t gpu_shift = signal::fft::xpeak2D<fft::FC2FC>(xmap_gpu, {}, peak_mode, peak_window) - lhs_center;

        INFO("Excepted shift: " << rhs_shift);
        INFO("Computed cpu shift: " << -cpu_shift);
        INFO("Computed gpu shift: " << -gpu_shift);
        REQUIRE_THAT(cpu_shift[0], Catch::WithinAbs(static_cast<double>(gpu_shift[0]), 5e-5));
        REQUIRE_THAT(cpu_shift[1], Catch::WithinAbs(static_cast<double>(gpu_shift[1]), 5e-5));
    }

    // Non-centered:
    {
        auto [lhs, lhs_fft] = fft::empty<TestType>(shape);
        auto [rhs, rhs_fft] = fft::empty<TestType>(shape);
        Array xmap = memory::empty<TestType>(shape);

        signal::rectangle({}, lhs, lhs_center, radius, taper);
        signal::rectangle({}, rhs, rhs_center, radius, taper);
        fft::r2c(lhs, lhs_fft);
        fft::r2c(rhs, rhs_fft);
        signal::fft::xmap<fft::H2F>(lhs_fft, rhs_fft, xmap);
        const auto xmap_gpu = xmap.to(Device("gpu"));

        const float2_t cpu_shift = signal::fft::xpeak2D<fft::F2F>(xmap, {}, peak_mode, peak_window) - lhs_center;
        const float2_t gpu_shift = signal::fft::xpeak2D<fft::F2F>(xmap_gpu, {}, peak_mode, peak_window) - lhs_center;

        INFO("Excepted shift: " << rhs_shift);
        INFO("Computed cpu shift: " << -cpu_shift);
        INFO("Computed gpu shift: " << -gpu_shift);
        REQUIRE_THAT(cpu_shift[0], Catch::WithinAbs(static_cast<double>(gpu_shift[0]), 5e-5));
        REQUIRE_THAT(cpu_shift[1], Catch::WithinAbs(static_cast<double>(gpu_shift[1]), 5e-5));
    }
}

TEMPLATE_TEST_CASE("unified::signal::fft, correlation peak 3D", "[noa][unified]", float, double) {
    using namespace ::noa;
    dim4_t shape = test::getRandomShape(3);
    shape[1] += 50;
    shape[2] += 50;
    shape[3] += 50;

    test::Randomizer<float> randomizer{-10, 10};
    const float3_t radius{20, 20, 20};
    const float taper = 7;
    const float3_t lhs_center{shape[1] / 2, shape[2] / 2, shape[3] / 2};
    const float3_t rhs_shift{randomizer.get(), randomizer.get(), randomizer.get()};
    const float3_t rhs_center = lhs_center + rhs_shift;

    std::vector<Device> devices{Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const std::array<signal::CorrelationMode, 4> modes{
            signal::CONVENTIONAL_CORRELATION,
            signal::MUTUAL_CORRELATION,
            signal::PHASE_CORRELATION,
            signal::DOUBLE_PHASE_CORRELATION};

    for (auto correlation_mode: modes) {
        for (auto& device: devices) {
            StreamGuard stream(device);
            ArrayOption options(device, Allocator::MANAGED);

            auto [lhs, lhs_fft] = fft::empty<TestType>(shape, options);
            auto [rhs, rhs_fft] = fft::empty<TestType>(shape, options);
            Array xmap = memory::empty<TestType>(shape, options);
            Array buffer = memory::like(lhs_fft);

            signal::rectangle({}, lhs, lhs_center, radius, taper);
            signal::rectangle({}, rhs, rhs_center, radius, taper);
            fft::r2c(lhs, lhs_fft);
            fft::r2c(rhs, rhs_fft);

            float3_t shift_centered;
            float3_t shift_not_centered;

            // Centered:
            {
                signal::fft::xmap<fft::H2FC>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, fft::NORM_DEFAULT, buffer);

                shift_centered = signal::fft::xpeak3D<fft::FC2FC>(xmap);
                shift_centered -= lhs_center;
                if (correlation_mode == signal::DOUBLE_PHASE_CORRELATION)
                    shift_centered /= 2;

                INFO("Excepted shift: " << rhs_shift);
                INFO("Computed shift: " << -shift_centered);
                REQUIRE_THAT(shift_centered[0], Catch::WithinAbs(static_cast<double>(-rhs_shift[0]), 5e-2));
                REQUIRE_THAT(shift_centered[1], Catch::WithinAbs(static_cast<double>(-rhs_shift[1]), 5e-2));
                REQUIRE_THAT(shift_centered[2], Catch::WithinAbs(static_cast<double>(-rhs_shift[2]), 5e-2));
            }

            // Non-centered:
            {
                signal::fft::xmap<fft::H2F>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, fft::NORM_DEFAULT, buffer);

                shift_not_centered = signal::fft::xpeak3D<fft::F2F>(xmap);
                shift_not_centered -= lhs_center;
                if (correlation_mode == signal::DOUBLE_PHASE_CORRELATION)
                    shift_not_centered /= 2;

                INFO("Excepted shift: " << rhs_shift);
                INFO("Computed shift: " << -shift_not_centered);
                REQUIRE_THAT(shift_not_centered[0], Catch::WithinAbs(static_cast<double>(-rhs_shift[0]), 5e-2));
                REQUIRE_THAT(shift_not_centered[1], Catch::WithinAbs(static_cast<double>(-rhs_shift[1]), 5e-2));
                REQUIRE_THAT(shift_not_centered[2], Catch::WithinAbs(static_cast<double>(-rhs_shift[2]), 5e-2));
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::signal::fft, correlation peak 3D batched", "[noa][unified]", double) {
    using namespace ::noa;
    dim4_t shape = test::getRandomShapeBatched(3);
    shape[1] += 50;
    shape[2] += 50;
    shape[3] += 50;

    test::Randomizer<float> randomizer{-10, 10};
    const float3_t radius{20, 20, 20};
    const float taper = 7;
    const float3_t lhs_center{shape[1] / 2, shape[2] / 2, shape[3] / 2};
    const float3_t rhs_shift{randomizer.get(), randomizer.get(), randomizer.get()};
    const float3_t rhs_center = lhs_center + rhs_shift;

    std::vector<Device> devices{Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const std::array<signal::CorrelationMode, 4> modes{
            signal::CONVENTIONAL_CORRELATION,
            signal::MUTUAL_CORRELATION,
            signal::PHASE_CORRELATION,
            signal::DOUBLE_PHASE_CORRELATION};

    for (auto correlation_mode: modes) {
        for (auto& device: devices) {
            StreamGuard stream(device);
            ArrayOption options(device, Allocator::MANAGED);

            auto [lhs, lhs_fft] = fft::empty<TestType>(shape, options);
            auto [rhs, rhs_fft] = fft::empty<TestType>(shape, options);
            Array xmap = memory::empty<TestType>(shape, options);
            Array buffer = memory::like(lhs_fft);
            Array shifts = memory::empty<float3_t>({1, 1, 1, shape[0]}, options);

            signal::rectangle({}, lhs, lhs_center, radius, taper);
            signal::rectangle({}, rhs, rhs_center, radius, taper);
            fft::r2c(lhs, lhs_fft);
            fft::r2c(rhs, rhs_fft);

            // Centered:
            {
                signal::fft::xmap<fft::H2FC>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, fft::NORM_DEFAULT, buffer);
                signal::fft::xpeak3D<fft::FC2FC>(xmap, shifts);
                shifts.eval();

                for (dim_t i = 0; i < shape[0]; ++i) {
                    float3_t shift = shifts[i];
                    shift -= lhs_center;
                    if (correlation_mode == signal::DOUBLE_PHASE_CORRELATION)
                        shift /= 2;

                    INFO("Excepted shift: " << rhs_shift);
                    INFO("Computed shift: " << -shift);
                    REQUIRE_THAT(shift[0], Catch::WithinAbs(static_cast<double>(-rhs_shift[0]), 5e-2));
                    REQUIRE_THAT(shift[1], Catch::WithinAbs(static_cast<double>(-rhs_shift[1]), 5e-2));
                    REQUIRE_THAT(shift[2], Catch::WithinAbs(static_cast<double>(-rhs_shift[2]), 5e-2));
                }
            }

            // Non-centered:
            {
                signal::fft::xmap<fft::H2F>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, fft::NORM_DEFAULT, buffer);
                signal::fft::xpeak3D<fft::F2F>(xmap, shifts);
                shifts.eval();

                for (dim_t i = 0; i < shape[0]; ++i) {
                    float3_t shift = shifts[i];
                    shift -= lhs_center;
                    if (correlation_mode == signal::DOUBLE_PHASE_CORRELATION)
                        shift /= 2;

                    INFO("Excepted shift: " << rhs_shift);
                    INFO("Computed shift: " << -shift);
                    REQUIRE_THAT(shift[0], Catch::WithinAbs(static_cast<double>(-rhs_shift[0]), 5e-2));
                    REQUIRE_THAT(shift[1], Catch::WithinAbs(static_cast<double>(-rhs_shift[1]), 5e-2));
                    REQUIRE_THAT(shift[2], Catch::WithinAbs(static_cast<double>(-rhs_shift[2]), 5e-2));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::signal::fft, correlation peak 3D, cpu vs gpu", "[noa][unified]", float, double) {
    using namespace ::noa;
    dim4_t shape = test::getRandomShape(3);
    shape[1] += 50;
    shape[2] += 50;
    shape[3] += 50;

    test::Randomizer<float> randomizer{-10, 10};
    const float3_t radius{20, 20, 20};
    const float taper = 7;
    const float3_t lhs_center{shape[1] / 2, shape[2] / 2, shape[3] / 2};
    const float3_t rhs_shift{randomizer.get(), randomizer.get(), randomizer.get()};
    const float3_t rhs_center = lhs_center + rhs_shift;

    if (!Device::any(Device::GPU))
        return;

    const auto peak_mode = GENERATE(signal::PEAK_PARABOLA_1D, signal::PEAK_COM);
    test::Randomizer<long> long_randomizer{1, peak_mode == signal::PEAK_COM ? 2 : 32};
    const long3_t peak_window = {long_randomizer.get(), long_randomizer.get(), long_randomizer.get()};

    INFO(peak_mode);
    INFO(peak_window);

    // Centered:
    {
        auto [lhs, lhs_fft] = fft::empty<TestType>(shape);
        auto [rhs, rhs_fft] = fft::empty<TestType>(shape);
        Array xmap = memory::empty<TestType>(shape);

        signal::rectangle({}, lhs, lhs_center, radius, taper, {}, {}, TestType(-1));
        signal::rectangle({}, rhs, rhs_center, radius, taper, {}, {}, TestType(-1));
        fft::r2c(lhs, lhs_fft);
        fft::r2c(rhs, rhs_fft);
        signal::fft::xmap<fft::H2FC>(lhs_fft, rhs_fft, xmap);
        const auto xmap_gpu = xmap.to(Device("gpu"));

        const float3_t cpu_shift = signal::fft::xpeak3D<fft::FC2FC>(xmap, {}, peak_mode, peak_window) - lhs_center;
        const float3_t gpu_shift = signal::fft::xpeak3D<fft::FC2FC>(xmap_gpu, {}, peak_mode, peak_window) - lhs_center;

        INFO("Excepted shift: " << rhs_shift);
        INFO("Computed cpu shift: " << -cpu_shift);
        INFO("Computed gpu shift: " << -gpu_shift);
        REQUIRE_THAT(cpu_shift[0], Catch::WithinAbs(static_cast<double>(gpu_shift[0]), 5e-5));
        REQUIRE_THAT(cpu_shift[1], Catch::WithinAbs(static_cast<double>(gpu_shift[1]), 5e-5));
        REQUIRE_THAT(cpu_shift[2], Catch::WithinAbs(static_cast<double>(gpu_shift[2]), 5e-5));
    }

    // Non-centered:
    {
        auto [lhs, lhs_fft] = fft::empty<TestType>(shape);
        auto [rhs, rhs_fft] = fft::empty<TestType>(shape);
        Array xmap = memory::empty<TestType>(shape);

        signal::rectangle({}, lhs, lhs_center, radius, taper);
        signal::rectangle({}, rhs, rhs_center, radius, taper);
        fft::r2c(lhs, lhs_fft);
        fft::r2c(rhs, rhs_fft);
        signal::fft::xmap<fft::H2F>(lhs_fft, rhs_fft, xmap);
        const auto xmap_gpu = xmap.to(Device("gpu"));

        const float3_t cpu_shift = signal::fft::xpeak3D<fft::F2F>(xmap, {}, peak_mode, peak_window) - lhs_center;
        const float3_t gpu_shift = signal::fft::xpeak3D<fft::F2F>(xmap_gpu, {}, peak_mode, peak_window) - lhs_center;

        INFO("Excepted shift: " << rhs_shift);
        INFO("Computed cpu shift: " << -cpu_shift);
        INFO("Computed gpu shift: " << -gpu_shift);
        REQUIRE_THAT(cpu_shift[0], Catch::WithinAbs(static_cast<double>(gpu_shift[0]), 5e-5));
        REQUIRE_THAT(cpu_shift[1], Catch::WithinAbs(static_cast<double>(gpu_shift[1]), 5e-3));
        REQUIRE_THAT(cpu_shift[2], Catch::WithinAbs(static_cast<double>(gpu_shift[2]), 5e-5));
    }
}

TEST_CASE("unified::signal::fft::autocorrelate", "[.]") {
    std::vector<Device> devices{Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const auto shape = test::getRandomShape(3);
    const auto center = double3_t(dim3_t(shape.get(1)) / 2);

    for (auto& device: devices) {
        StreamGuard stream(device);
        ArrayOption options(device, Allocator::MANAGED);

        Array lhs = math::random<float>(math::uniform_t{}, shape, -50, 50, options);
        Array lhs_fft = fft::r2c(lhs);
        Array rhs_fft = lhs_fft.copy();
        Array xmap = memory::like(lhs);
        signal::fft::xmap<fft::H2F>(lhs_fft, rhs_fft, xmap);
        const float3_t shift = signal::fft::xpeak3D<fft::F2F>(xmap);
        REQUIRE_THAT(shift[0], Catch::WithinAbs(center[0], 5e-2));
        REQUIRE_THAT(shift[1], Catch::WithinAbs(center[1], 5e-2));
        REQUIRE_THAT(shift[2], Catch::WithinAbs(center[2], 5e-2));
    }
}
