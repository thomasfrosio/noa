#include <noa/FFT.h>
#include <noa/Signal.h>
#include <noa/Math.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::signal::fft, correlation peak 2D", "[noa][unified]", float, double) {
    dim4_t shape = test::getRandomShape(2);
    shape[2] += 200;
    shape[3] += 200;

    test::Randomizer<float> randomizer{-30, 30};
    const float2_t radius{25, 25};
    const float taper = 7;
    const float2_t lhs_center{shape[2] / 2, shape[3] / 2};
    const float2_t rhs_shift{randomizer.get(), randomizer.get()};
    const float2_t rhs_center = lhs_center + rhs_shift;

    std::vector<noa::Device> devices{noa::Device("cpu")};
    if (noa::Device::any(noa::Device::GPU))
        devices.emplace_back("gpu");

    const std::array<noa::signal::CorrelationMode, 4> modes{
            noa::signal::CONVENTIONAL_CORRELATION,
            noa::signal::MUTUAL_CORRELATION,
            noa::signal::PHASE_CORRELATION,
            noa::signal::DOUBLE_PHASE_CORRELATION};

    for (auto correlation_mode: modes) {
        for (auto& device: devices) {
            INFO(device);
            INFO(correlation_mode);
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);

            auto [lhs, lhs_fft] = noa::fft::empty<TestType>(shape, options);
            auto [rhs, rhs_fft] = noa::fft::empty<TestType>(shape, options);
            noa::Array xmap = noa::memory::empty<TestType>(shape, options);
            noa::Array buffer = noa::memory::like(lhs_fft);

            noa::signal::rectangle({}, lhs, lhs_center, radius, taper);
            noa::signal::rectangle({}, rhs, rhs_center, radius, taper);
            noa::fft::r2c(lhs, lhs_fft);
            noa::fft::r2c(rhs, rhs_fft);

            float2_t shift_centered;
            float2_t shift_not_centered;

            {
                noa::signal::fft::xmap<noa::fft::H2FC>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, noa::fft::NORM_DEFAULT, buffer);

                auto [coordinate, value] = noa::signal::fft::xpeak2D<noa::fft::FC2FC>(xmap);
                coordinate -= lhs_center;
                if (correlation_mode == noa::signal::DOUBLE_PHASE_CORRELATION)
                    coordinate /= 2;

                INFO("Excepted shift: " << rhs_shift);
                INFO("Computed shift: " << -coordinate);
                REQUIRE_THAT(coordinate[0], Catch::WithinAbs(static_cast<double>(-rhs_shift[0]), 5e-2));
                REQUIRE_THAT(coordinate[1], Catch::WithinAbs(static_cast<double>(-rhs_shift[1]), 5e-2));
                shift_centered = coordinate;

                const auto max_value = noa::math::max(xmap);
                REQUIRE(max_value <= value);
            }

            {
                noa::signal::fft::xmap<noa::fft::H2F>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, noa::fft::NORM_DEFAULT, buffer);

                auto [coordinate, value] = noa::signal::fft::xpeak2D<noa::fft::F2F>(xmap);
                coordinate -= lhs_center;
                if (correlation_mode == noa::signal::DOUBLE_PHASE_CORRELATION)
                    coordinate /= 2;

                INFO("Excepted shift: " << rhs_shift);
                INFO("Computed shift: " << -coordinate);
                REQUIRE_THAT(coordinate[0], Catch::WithinAbs(static_cast<double>(-rhs_shift[0]), 5e-2));
                REQUIRE_THAT(coordinate[1], Catch::WithinAbs(static_cast<double>(-rhs_shift[1]), 5e-2));
                shift_not_centered = coordinate;

                const auto max_value = noa::math::max(xmap);
                REQUIRE(max_value <= value);
            }

            REQUIRE_THAT(shift_not_centered[0], Catch::WithinAbs(static_cast<double>(shift_centered[0]), 1e-4));
            REQUIRE_THAT(shift_not_centered[1], Catch::WithinAbs(static_cast<double>(shift_centered[1]), 1e-4));
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

    std::vector<noa::Device> devices{noa::Device("cpu")};
    if (noa::Device::any(noa::Device::GPU))
        devices.emplace_back("gpu");

    const std::array<noa::signal::CorrelationMode, 4> modes{
            noa::signal::CONVENTIONAL_CORRELATION,
            noa::signal::MUTUAL_CORRELATION,
            noa::signal::PHASE_CORRELATION,
            noa::signal::DOUBLE_PHASE_CORRELATION};

    for (auto correlation_mode: modes) {
        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, Allocator::MANAGED);

            auto [lhs, lhs_fft] = noa::fft::empty<TestType>(shape, options);
            auto [rhs, rhs_fft] = noa::fft::empty<TestType>(shape, options);
            noa::Array xmap = noa::memory::empty<TestType>(shape, options);
            noa::Array buffer = noa::memory::like(lhs_fft);
            noa::Array shifts = noa::memory::empty<float2_t>({shape[0], 1, 1, 1}, options);
            noa::Array values = noa::memory::empty<TestType>({shape[0], 1, 1, 1}, options);

            noa::signal::rectangle({}, lhs, lhs_center, radius, taper);
            noa::signal::rectangle({}, rhs, rhs_center, radius, taper);
            noa::fft::r2c(lhs, lhs_fft);
            noa::fft::r2c(rhs, rhs_fft);

            // Centered:
            {
                noa::signal::fft::xmap<noa::fft::H2FC>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, noa::fft::NORM_DEFAULT, buffer);
                noa::signal::fft::xpeak2D<noa::fft::FC2FC>(xmap, shifts, values);
                shifts.eval();

                for (dim_t i = 0; i < shape[0]; ++i) {
                    float2_t shift = shifts[i];
                    shift -= lhs_center;
                    if (correlation_mode == noa::signal::DOUBLE_PHASE_CORRELATION)
                        shift /= 2;

                    INFO("Excepted shift: " << rhs_shift);
                    INFO("Computed shift: " << -shift);
                    REQUIRE_THAT(shift[0], Catch::WithinAbs(static_cast<double>(-rhs_shift[0]), 5e-2));
                    REQUIRE_THAT(shift[1], Catch::WithinAbs(static_cast<double>(-rhs_shift[1]), 5e-2));

                    const auto max = noa::math::max(xmap.subregion(i));
                    REQUIRE(max <= values[i]);
                }
            }

            noa::memory::fill(shifts, float2_t{0});
            noa::memory::fill(values, TestType{0});

            // Non-centered:
            {
                noa::signal::fft::xmap<noa::fft::H2F>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, noa::fft::NORM_DEFAULT, buffer);
                noa::signal::fft::xpeak2D<noa::fft::F2F>(xmap, shifts, values);
                shifts.eval();

                for (dim_t i = 0; i < shape[0]; ++i) {
                    float2_t shift = shifts[i];
                    shift -= lhs_center;
                    if (correlation_mode == noa::signal::DOUBLE_PHASE_CORRELATION)
                        shift /= 2;

                    INFO("Excepted shift: " << rhs_shift);
                    INFO("Computed shift: " << -shift);
                    REQUIRE_THAT(shift[0], Catch::WithinAbs(static_cast<double>(-rhs_shift[0]), 5e-2));
                    REQUIRE_THAT(shift[1], Catch::WithinAbs(static_cast<double>(-rhs_shift[1]), 5e-2));

                    const auto max = noa::math::max(xmap.subregion(i));
                    REQUIRE(max <= values[i]);
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

    INFO(shape);
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

        auto [cpu_shift, cpu_value] = signal::fft::xpeak2D<fft::FC2FC>(xmap, {}, peak_mode, peak_window);
        auto [gpu_shift, gpu_value] = signal::fft::xpeak2D<fft::FC2FC>(xmap_gpu.eval(), {}, peak_mode, peak_window);
        cpu_shift -= lhs_center;
        gpu_shift -= lhs_center;

        INFO("Excepted shift: " << rhs_shift);
        INFO("Computed cpu shift: " << -cpu_shift);
        INFO("Computed gpu shift: " << -gpu_shift);
        REQUIRE_THAT(cpu_shift[0], Catch::WithinAbs(static_cast<double>(gpu_shift[0]), 5e-5));
        REQUIRE_THAT(cpu_shift[1], Catch::WithinAbs(static_cast<double>(gpu_shift[1]), 5e-5));
        REQUIRE_THAT(cpu_value, Catch::WithinAbs(static_cast<double>(gpu_value), 5e-6));
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

        auto [cpu_shift, cpu_value] = signal::fft::xpeak2D<fft::F2F>(xmap, {}, peak_mode, peak_window);
        auto [gpu_shift, gpu_value] = signal::fft::xpeak2D<fft::F2F>(xmap_gpu, {}, peak_mode, peak_window);
        cpu_shift -= lhs_center;
        gpu_shift -= lhs_center;

        INFO("Excepted shift: " << rhs_shift);
        INFO("Computed cpu shift: " << -cpu_shift);
        INFO("Computed gpu shift: " << -gpu_shift);
        REQUIRE_THAT(cpu_shift[0], Catch::WithinAbs(static_cast<double>(gpu_shift[0]), 5e-5));
        REQUIRE_THAT(cpu_shift[1], Catch::WithinAbs(static_cast<double>(gpu_shift[1]), 5e-5));
        REQUIRE_THAT(cpu_value, Catch::WithinAbs(static_cast<double>(gpu_value), 5e-6));
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

    std::vector<noa::Device> devices{noa::Device("cpu")};
    if (noa::Device::any(noa::Device::GPU))
        devices.emplace_back("gpu");

    const std::array<noa::signal::CorrelationMode, 4> modes{
            noa::signal::CONVENTIONAL_CORRELATION,
            noa::signal::MUTUAL_CORRELATION,
            noa::signal::PHASE_CORRELATION,
            noa::signal::DOUBLE_PHASE_CORRELATION};

    for (auto correlation_mode: modes) {
        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);

            auto [lhs, lhs_fft] = noa::fft::empty<TestType>(shape, options);
            auto [rhs, rhs_fft] = noa::fft::empty<TestType>(shape, options);
            noa::Array xmap = noa::memory::empty<TestType>(shape, options);
            noa::Array buffer = noa::memory::like(lhs_fft);

            noa::signal::rectangle({}, lhs, lhs_center, radius, taper);
            noa::signal::rectangle({}, rhs, rhs_center, radius, taper);
            noa::fft::r2c(lhs, lhs_fft);
            noa::fft::r2c(rhs, rhs_fft);

            float3_t shift_centered;
            float3_t shift_not_centered;

            // Centered:
            {
                noa::signal::fft::xmap<noa::fft::H2FC>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, noa::fft::NORM_DEFAULT, buffer);

                auto [shift, value] = noa::signal::fft::xpeak3D<noa::fft::FC2FC>(xmap);
                shift -= lhs_center;
                if (correlation_mode == noa::signal::DOUBLE_PHASE_CORRELATION)
                    shift /= 2;

                INFO("Excepted shift: " << rhs_shift);
                INFO("Computed shift: " << -shift);
                REQUIRE_THAT(shift[0], Catch::WithinAbs(static_cast<double>(-rhs_shift[0]), 5e-2));
                REQUIRE_THAT(shift[1], Catch::WithinAbs(static_cast<double>(-rhs_shift[1]), 5e-2));
                REQUIRE_THAT(shift[2], Catch::WithinAbs(static_cast<double>(-rhs_shift[2]), 5e-2));
                shift_centered = shift;

                const auto max = noa::math::max(xmap);
                REQUIRE(max <= value);
            }

            // Non-centered:
            {
                noa::signal::fft::xmap<noa::fft::H2F>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, noa::fft::NORM_DEFAULT, buffer);

                auto [shift, value] = noa::signal::fft::xpeak3D<noa::fft::F2F>(xmap);
                shift -= lhs_center;
                if (correlation_mode == noa::signal::DOUBLE_PHASE_CORRELATION)
                    shift /= 2;

                INFO("Excepted shift: " << rhs_shift);
                INFO("Computed shift: " << -shift);
                REQUIRE_THAT(shift[0], Catch::WithinAbs(static_cast<double>(-rhs_shift[0]), 5e-2));
                REQUIRE_THAT(shift[1], Catch::WithinAbs(static_cast<double>(-rhs_shift[1]), 5e-2));
                REQUIRE_THAT(shift[2], Catch::WithinAbs(static_cast<double>(-rhs_shift[2]), 5e-2));
                shift_not_centered = shift;

                const auto max = noa::math::max(xmap);
                REQUIRE(max <= value);
            }

            REQUIRE_THAT(shift_not_centered[0], Catch::WithinAbs(static_cast<double>(shift_centered[0]), 1e-4));
            REQUIRE_THAT(shift_not_centered[1], Catch::WithinAbs(static_cast<double>(shift_centered[1]), 1e-4));
            REQUIRE_THAT(shift_not_centered[2], Catch::WithinAbs(static_cast<double>(shift_centered[2]), 1e-4));
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

    std::vector<noa::Device> devices{noa::Device("cpu")};
    if (noa::Device::any(noa::Device::GPU))
        devices.emplace_back("gpu");

    const std::array<noa::signal::CorrelationMode, 4> modes{
            noa::signal::CONVENTIONAL_CORRELATION,
            noa::signal::MUTUAL_CORRELATION,
            noa::signal::PHASE_CORRELATION,
            noa::signal::DOUBLE_PHASE_CORRELATION};

    for (auto correlation_mode: modes) {
        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);

            auto [lhs, lhs_fft] = noa::fft::empty<TestType>(shape, options);
            auto [rhs, rhs_fft] = noa::fft::empty<TestType>(shape, options);
            noa::Array xmap = noa::memory::empty<TestType>(shape, options);
            noa::Array buffer = noa::memory::like(lhs_fft);
            noa::Array shifts = noa::memory::empty<float3_t>({shape[0], 1, 1, 1}, options);
            noa::Array values = noa::memory::empty<TestType>({shape[0], 1, 1, 1}, options);

            noa::signal::rectangle({}, lhs, lhs_center, radius, taper);
            noa::signal::rectangle({}, rhs, rhs_center, radius, taper);
            noa::fft::r2c(lhs, lhs_fft);
            noa::fft::r2c(rhs, rhs_fft);

            // Centered:
            {
                noa::signal::fft::xmap<noa::fft::H2FC>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, noa::fft::NORM_DEFAULT, buffer);
                noa::signal::fft::xpeak3D<noa::fft::FC2FC>(xmap, shifts, values);
                shifts.eval();

                for (dim_t i = 0; i < shape[0]; ++i) {
                    float3_t shift = shifts[i];
                    shift -= lhs_center;
                    if (correlation_mode == noa::signal::DOUBLE_PHASE_CORRELATION)
                        shift /= 2;

                    INFO("Excepted shift: " << rhs_shift);
                    INFO("Computed shift: " << -shift);
                    REQUIRE_THAT(shift[0], Catch::WithinAbs(static_cast<double>(-rhs_shift[0]), 5e-2));
                    REQUIRE_THAT(shift[1], Catch::WithinAbs(static_cast<double>(-rhs_shift[1]), 5e-2));
                    REQUIRE_THAT(shift[2], Catch::WithinAbs(static_cast<double>(-rhs_shift[2]), 5e-2));

                    const auto max = noa::math::max(xmap.subregion(i));
                    REQUIRE(max <= values[i]);
                }
            }

            noa::memory::fill(shifts, float3_t{0});
            noa::memory::fill(values, TestType{0});

            // Non-centered:
            {
                noa::signal::fft::xmap<noa::fft::H2F>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, noa::fft::NORM_DEFAULT, buffer);
                noa::signal::fft::xpeak3D<noa::fft::F2F>(xmap, shifts, values);
                shifts.eval();

                for (dim_t i = 0; i < shape[0]; ++i) {
                    float3_t shift = shifts[i];
                    shift -= lhs_center;
                    if (correlation_mode == noa::signal::DOUBLE_PHASE_CORRELATION)
                        shift /= 2;

                    INFO("Excepted shift: " << rhs_shift);
                    INFO("Computed shift: " << -shift);
                    REQUIRE_THAT(shift[0], Catch::WithinAbs(static_cast<double>(-rhs_shift[0]), 5e-2));
                    REQUIRE_THAT(shift[1], Catch::WithinAbs(static_cast<double>(-rhs_shift[1]), 5e-2));
                    REQUIRE_THAT(shift[2], Catch::WithinAbs(static_cast<double>(-rhs_shift[2]), 5e-2));

                    const auto max = noa::math::max(xmap.subregion(i));
                    REQUIRE(max <= values[i]);
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

        auto [cpu_shift, cpu_value] = signal::fft::xpeak3D<fft::FC2FC>(xmap, {}, peak_mode, peak_window);
        auto [gpu_shift, gpu_value] = signal::fft::xpeak3D<fft::FC2FC>(xmap_gpu, {}, peak_mode, peak_window);
        cpu_shift -= lhs_center;
        gpu_shift -= lhs_center;

        INFO("Excepted shift: " << rhs_shift);
        INFO("Computed cpu shift: " << -cpu_shift);
        INFO("Computed gpu shift: " << -gpu_shift);
        REQUIRE_THAT(cpu_shift[0], Catch::WithinAbs(static_cast<double>(gpu_shift[0]), 5e-5));
        REQUIRE_THAT(cpu_shift[1], Catch::WithinAbs(static_cast<double>(gpu_shift[1]), 5e-5));
        REQUIRE_THAT(cpu_shift[2], Catch::WithinAbs(static_cast<double>(gpu_shift[2]), 5e-5));
        REQUIRE_THAT(cpu_value, Catch::WithinAbs(static_cast<double>(gpu_value), 5e-6));
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

        auto [cpu_shift, cpu_value] = signal::fft::xpeak3D<fft::F2F>(xmap, {}, peak_mode, peak_window);
        auto [gpu_shift, gpu_value] = signal::fft::xpeak3D<fft::F2F>(xmap_gpu, {}, peak_mode, peak_window);
        cpu_shift -= lhs_center;
        gpu_shift -= lhs_center;

        INFO("Excepted shift: " << rhs_shift);
        INFO("Computed cpu shift: " << -cpu_shift);
        INFO("Computed gpu shift: " << -gpu_shift);
        REQUIRE_THAT(cpu_shift[0], Catch::WithinAbs(static_cast<double>(gpu_shift[0]), 5e-5));
        REQUIRE_THAT(cpu_shift[1], Catch::WithinAbs(static_cast<double>(gpu_shift[1]), 5e-3));
        REQUIRE_THAT(cpu_shift[2], Catch::WithinAbs(static_cast<double>(gpu_shift[2]), 5e-5));
        REQUIRE_THAT(cpu_value, Catch::WithinAbs(static_cast<double>(gpu_value), 5e-6));
    }
}

TEST_CASE("unified::signal::fft::autocorrelate", "[.]") {
    std::vector<noa::Device> devices{noa::Device("cpu")};
    if (noa::Device::any(noa::Device::GPU))
        devices.emplace_back("gpu");

    const auto shape = test::getRandomShape(3);
    const auto center = double3_t(dim3_t(shape.get(1)) / 2);

    for (auto& device: devices) {
        const auto stream = noa::StreamGuard(device);
        const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);

        noa::Array lhs = noa::math::random<float>(noa::math::uniform_t{}, shape, -50, 50, options);
        noa::Array lhs_fft = noa::fft::r2c(lhs);
        noa::Array rhs_fft = lhs_fft.copy();
        noa::Array xmap = noa::memory::like(lhs);
        noa::signal::fft::xmap<noa::fft::H2F>(lhs_fft, rhs_fft, xmap);
        const auto [shift, _] = noa::signal::fft::xpeak3D<noa::fft::F2F>(xmap);
        REQUIRE_THAT(shift[0], Catch::WithinAbs(center[0], 5e-2));
        REQUIRE_THAT(shift[1], Catch::WithinAbs(center[1], 5e-2));
        REQUIRE_THAT(shift[2], Catch::WithinAbs(center[2], 5e-2));
    }
}
