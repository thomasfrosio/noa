#include <noa/unified/fft/Transform.hpp>
#include <noa/unified/fft/Factory.hpp>
#include <noa/unified/fft/Remap.hpp>
#include <noa/unified/geometry/Shape.hpp>
#include <noa/unified/memory/Factory.hpp>
#include <noa/unified/math/Random.hpp>
#include <noa/unified/math/Reduce.hpp>
#include <noa/unified/signal/fft/Correlate.hpp>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::signal::fft, correlation peak 2d", "[noa][unified]", f32, f64) {
    auto shape = test::get_random_shape4(2);
    shape[2] += 200;
    shape[3] += 200;

    test::Randomizer<f32> randomizer{-30, 30};
    const Vec2<f32> radius{25, 25};
    const float taper = 7;
    const Vec2<f32> lhs_center{shape[2] / 2, shape[3] / 2};
    const Vec2<f32> rhs_shift{randomizer.get(), randomizer.get()};
    const Vec2<f32> rhs_center = lhs_center + rhs_shift;

    std::vector<noa::Device> devices{noa::Device("cpu")};
    if (noa::Device::is_any(noa::DeviceType::GPU))
        devices.emplace_back("gpu");

    const std::array<noa::signal::CorrelationMode, 4> modes{
            noa::signal::CorrelationMode::CONVENTIONAL,
            noa::signal::CorrelationMode::MUTUAL,
            noa::signal::CorrelationMode::PHASE,
            noa::signal::CorrelationMode::DOUBLE_PHASE};

    for (auto correlation_mode: modes) {
        for (auto& device: devices) {
            INFO(device);
            INFO(correlation_mode);
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);

            auto [lhs, lhs_fft] = noa::fft::empty<TestType>(shape, options);
            auto [rhs, rhs_fft] = noa::fft::empty<TestType>(shape, options);
            const noa::Array xmap = noa::memory::empty<TestType>(shape, options);
            const noa::Array buffer = noa::memory::like(lhs_fft);

            noa::geometry::rectangle({}, lhs.view(), lhs_center, radius, taper);
            noa::geometry::rectangle({}, rhs.view(), rhs_center, radius, taper);
            noa::fft::r2c(lhs.view(), lhs_fft.view());
            noa::fft::r2c(rhs.view(), rhs_fft.view());

            Vec2<f32> shift_centered;
            Vec2<f32> shift_not_centered;

            {
                noa::signal::fft::xmap<noa::fft::H2FC>(
                        lhs_fft.view(), rhs_fft.view(), xmap,
                        correlation_mode, noa::fft::NORM_DEFAULT, buffer.view());

                auto [coordinate, value] = noa::signal::fft::xpeak_2d<noa::fft::FC2FC>(xmap.view());
                coordinate -= lhs_center;
                if (correlation_mode == noa::signal::CorrelationMode::DOUBLE_PHASE)
                    coordinate /= 2;

                INFO("Excepted shift: " << rhs_shift);
                INFO("Computed shift: " << -coordinate);
                REQUIRE_THAT(coordinate[0], Catch::WithinAbs(static_cast<f64>(-rhs_shift[0]), 5e-2));
                REQUIRE_THAT(coordinate[1], Catch::WithinAbs(static_cast<f64>(-rhs_shift[1]), 5e-2));
                shift_centered = coordinate;

                const auto max_value = noa::math::max(xmap.view());
                REQUIRE(max_value <= value);
            }

            {
                noa::signal::fft::xmap<noa::fft::H2F>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, noa::fft::NORM_DEFAULT, buffer);

                auto [coordinate, value] = noa::signal::fft::xpeak_2d<noa::fft::F2F>(xmap);
                coordinate -= lhs_center;
                if (correlation_mode == noa::signal::CorrelationMode::DOUBLE_PHASE)
                    coordinate /= 2;

                INFO("Excepted shift: " << rhs_shift);
                INFO("Computed shift: " << -coordinate);
                REQUIRE_THAT(coordinate[0], Catch::WithinAbs(static_cast<f64>(-rhs_shift[0]), 5e-2));
                REQUIRE_THAT(coordinate[1], Catch::WithinAbs(static_cast<f64>(-rhs_shift[1]), 5e-2));
                shift_not_centered = coordinate;

                const auto max_value = noa::math::max(xmap);
                REQUIRE(max_value <= value);
            }

            REQUIRE_THAT(shift_not_centered[0], Catch::WithinAbs(static_cast<f64>(shift_centered[0]), 1e-4));
            REQUIRE_THAT(shift_not_centered[1], Catch::WithinAbs(static_cast<f64>(shift_centered[1]), 1e-4));
        }
    }
}

TEMPLATE_TEST_CASE("unified::signal::fft, correlation peak 2D batched", "[noa][unified]", f32, f64) {
    using namespace ::noa;
    auto shape = test::get_random_shape4_batched(2);
    shape[2] += 200;
    shape[3] += 200;

    test::Randomizer<float> randomizer{-30, 30};
    const Vec2<f32> radius{25, 25};
    const float taper = 7;
    const Vec2<f32> lhs_center{shape[2] / 2, shape[3] / 2};
    const Vec2<f32> rhs_shift{randomizer.get(), randomizer.get()};
    const Vec2<f32> rhs_center = lhs_center + rhs_shift;

    std::vector<noa::Device> devices{noa::Device("cpu")};
    if (noa::Device::is_any(noa::DeviceType::GPU))
        devices.emplace_back("gpu");

    const std::array<noa::signal::CorrelationMode, 4> modes{
            noa::signal::CorrelationMode::CONVENTIONAL,
            noa::signal::CorrelationMode::MUTUAL,
            noa::signal::CorrelationMode::PHASE,
            noa::signal::CorrelationMode::DOUBLE_PHASE};

    for (auto correlation_mode: modes) {
        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, Allocator::MANAGED);

            auto [lhs, lhs_fft] = noa::fft::empty<TestType>(shape, options);
            auto [rhs, rhs_fft] = noa::fft::empty<TestType>(shape, options);
            noa::Array xmap = noa::memory::empty<TestType>(shape, options);
            noa::Array buffer = noa::memory::like(lhs_fft);
            noa::Array shifts = noa::memory::empty<Vec2<f32>>({shape[0], 1, 1, 1}, options);
            noa::Array values = noa::memory::empty<TestType>({shape[0], 1, 1, 1}, options);

            noa::geometry::rectangle({}, lhs, lhs_center, radius, taper);
            noa::geometry::rectangle({}, rhs, rhs_center, radius, taper);
            noa::fft::r2c(lhs, lhs_fft);
            noa::fft::r2c(rhs, rhs_fft);

            // Centered:
            {
                noa::signal::fft::xmap<noa::fft::H2FC>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, noa::fft::NORM_DEFAULT, buffer);
                noa::signal::fft::xpeak_2d<noa::fft::FC2FC>(xmap, shifts, values);
                shifts.eval();

                for (i64 i = 0; i < shape[0]; ++i) {
                    Vec2<f32> shift = shifts(i, 0, 0, 0);
                    shift -= lhs_center;
                    if (correlation_mode == noa::signal::CorrelationMode::DOUBLE_PHASE)
                        shift /= 2;

                    INFO("Excepted shift: " << rhs_shift);
                    INFO("Computed shift: " << -shift);
                    REQUIRE_THAT(shift[0], Catch::WithinAbs(static_cast<f64>(-rhs_shift[0]), 5e-2));
                    REQUIRE_THAT(shift[1], Catch::WithinAbs(static_cast<f64>(-rhs_shift[1]), 5e-2));

                    const auto max = noa::math::max(xmap.subregion(i));
                    REQUIRE(max <= values(i, 0, 0, 0));
                }
            }

            noa::memory::fill(shifts, Vec2<f32>{0});
            noa::memory::fill(values, TestType{0});

            // Non-centered:
            {
                noa::signal::fft::xmap<noa::fft::H2F>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, noa::fft::NORM_DEFAULT, buffer);
                noa::signal::fft::xpeak_2d<noa::fft::F2F>(xmap, shifts, values);
                shifts.eval();

                for (i64 i = 0; i < shape[0]; ++i) {
                    Vec2<f32> shift = shifts(i, 0, 0, 0);
                    shift -= lhs_center;
                    if (correlation_mode == noa::signal::CorrelationMode::DOUBLE_PHASE)
                        shift /= 2;

                    INFO("Excepted shift: " << rhs_shift);
                    INFO("Computed shift: " << -shift);
                    REQUIRE_THAT(shift[0], Catch::WithinAbs(static_cast<f64>(-rhs_shift[0]), 5e-2));
                    REQUIRE_THAT(shift[1], Catch::WithinAbs(static_cast<f64>(-rhs_shift[1]), 5e-2));

                    const auto max = noa::math::max(xmap.subregion(i));
                    REQUIRE(max <= values(i, 0, 0, 0));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::signal::fft, correlation peak 2D, cpu vs gpu", "[noa][unified]", float, double) {
    using namespace ::noa;
    auto shape = test::get_random_shape4(2);
    shape[2] += 100;
    shape[3] += 100;

    test::Randomizer<f32> float_randomizer{-20, 20};
    const Vec2<f32> radius{25, 25};
    const f32 taper = 7;
    const Vec2<f32> lhs_center{shape[2] / 2, shape[3] / 2};
    const Vec2<f32> rhs_shift{float_randomizer.get(), float_randomizer.get()};
    const Vec2<f32> rhs_center = lhs_center + rhs_shift;

    if (!Device::is_any(DeviceType::GPU))
        return;

    const auto peak_mode = GENERATE(signal::PeakMode::PARABOLA_1D, signal::PeakMode::COM);
    test::Randomizer<i64> long_randomizer{1, peak_mode == signal::PeakMode::COM ? 8 : 32};
    const Vec2<i64> peak_window = {long_randomizer.get(), long_randomizer.get()};

    INFO(shape);
    INFO(peak_mode);
    INFO(peak_window);

    // Centered:
    {
        auto [lhs, lhs_fft] = fft::empty<TestType>(shape);
        auto [rhs, rhs_fft] = fft::empty<TestType>(shape);
        Array xmap = memory::empty<TestType>(shape);

        noa::geometry::rectangle({}, lhs, lhs_center, radius, taper);
        noa::geometry::rectangle({}, rhs, rhs_center, radius, taper);
        noa::fft::r2c(lhs, lhs_fft);
        noa::fft::r2c(rhs, rhs_fft);
        noa::signal::fft::xmap<fft::H2FC>(lhs_fft, rhs_fft, xmap);
        const auto xmap_gpu = xmap.to(Device("gpu"));

        auto [cpu_shift, cpu_value] = noa::signal::fft::xpeak_2d<fft::FC2FC>(xmap, {}, peak_mode, peak_window);
        auto [gpu_shift, gpu_value] = noa::signal::fft::xpeak_2d<fft::FC2FC>(xmap_gpu.eval(), {}, peak_mode, peak_window);
        cpu_shift -= lhs_center;
        gpu_shift -= lhs_center;

        INFO("Excepted shift: " << rhs_shift);
        INFO("Computed cpu shift: " << -cpu_shift);
        INFO("Computed gpu shift: " << -gpu_shift);
        REQUIRE_THAT(cpu_shift[0], Catch::WithinAbs(static_cast<f64>(gpu_shift[0]), 5e-5));
        REQUIRE_THAT(cpu_shift[1], Catch::WithinAbs(static_cast<f64>(gpu_shift[1]), 5e-5));
        REQUIRE_THAT(cpu_value, Catch::WithinAbs(static_cast<f64>(gpu_value), 5e-6));
    }

    // Non-centered:
    {
        auto [lhs, lhs_fft] = noa::fft::empty<TestType>(shape);
        auto [rhs, rhs_fft] = noa::fft::empty<TestType>(shape);
        Array xmap = noa::memory::empty<TestType>(shape);

        geometry::rectangle({}, lhs, lhs_center, radius, taper);
        geometry::rectangle({}, rhs, rhs_center, radius, taper);
        noa::fft::r2c(lhs, lhs_fft);
        noa::fft::r2c(rhs, rhs_fft);
        noa::signal::fft::xmap<fft::H2F>(lhs_fft, rhs_fft, xmap);
        const auto xmap_gpu = xmap.to(Device("gpu"));

        auto [cpu_shift, cpu_value] = noa::signal::fft::xpeak_2d<fft::F2F>(xmap, {}, peak_mode, peak_window);
        auto [gpu_shift, gpu_value] = noa::signal::fft::xpeak_2d<fft::F2F>(xmap_gpu, {}, peak_mode, peak_window);
        cpu_shift -= lhs_center;
        gpu_shift -= lhs_center;

        INFO("Excepted shift: " << rhs_shift);
        INFO("Computed cpu shift: " << -cpu_shift);
        INFO("Computed gpu shift: " << -gpu_shift);
        REQUIRE_THAT(cpu_shift[0], Catch::WithinAbs(static_cast<f64>(gpu_shift[0]), 5e-5));
        REQUIRE_THAT(cpu_shift[1], Catch::WithinAbs(static_cast<f64>(gpu_shift[1]), 5e-5));
        REQUIRE_THAT(cpu_value, Catch::WithinAbs(static_cast<f64>(gpu_value), 5e-6));
    }
}

TEMPLATE_TEST_CASE("unified::signal::fft, correlation peak 3D", "[noa][unified]", float, double) {
    using namespace ::noa;
    auto shape = test::get_random_shape4(3);
    shape[1] += 50;
    shape[2] += 50;
    shape[3] += 50;

    test::Randomizer<f32> randomizer{-10, 10};
    const Vec3<f32> radius{20, 20, 20};
    const f32 taper = 7;
    const Vec3<f32> lhs_center{shape[1] / 2, shape[2] / 2, shape[3] / 2};
    const Vec3<f32> rhs_shift{randomizer.get(), randomizer.get(), randomizer.get()};
    const Vec3<f32> rhs_center = lhs_center + rhs_shift;

    std::vector<noa::Device> devices{noa::Device("cpu")};
    if (noa::Device::is_any(noa::DeviceType::GPU))
        devices.emplace_back("gpu");

    const std::array<noa::signal::CorrelationMode, 4> modes{
            noa::signal::CorrelationMode::CONVENTIONAL,
            noa::signal::CorrelationMode::MUTUAL,
            noa::signal::CorrelationMode::PHASE,
            noa::signal::CorrelationMode::DOUBLE_PHASE};

    for (auto correlation_mode: modes) {
        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);

            auto [lhs, lhs_fft] = noa::fft::empty<TestType>(shape, options);
            auto [rhs, rhs_fft] = noa::fft::empty<TestType>(shape, options);
            noa::Array xmap = noa::memory::empty<TestType>(shape, options);
            noa::Array buffer = noa::memory::like(lhs_fft);

            noa::geometry::rectangle({}, lhs, lhs_center, radius, taper);
            noa::geometry::rectangle({}, rhs, rhs_center, radius, taper);
            noa::fft::r2c(lhs, lhs_fft);
            noa::fft::r2c(rhs, rhs_fft);

            Vec3<f32> shift_centered;
            Vec3<f32> shift_not_centered;

            // Centered:
            {
                noa::signal::fft::xmap<noa::fft::H2FC>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, noa::fft::NORM_DEFAULT, buffer);

                auto [shift, value] = noa::signal::fft::xpeak_3d<noa::fft::FC2FC>(xmap);
                shift -= lhs_center;
                if (correlation_mode == noa::signal::CorrelationMode::DOUBLE_PHASE)
                    shift /= 2;

                INFO("Excepted shift: " << rhs_shift);
                INFO("Computed shift: " << -shift);
                REQUIRE_THAT(shift[0], Catch::WithinAbs(static_cast<f64>(-rhs_shift[0]), 5e-2));
                REQUIRE_THAT(shift[1], Catch::WithinAbs(static_cast<f64>(-rhs_shift[1]), 5e-2));
                REQUIRE_THAT(shift[2], Catch::WithinAbs(static_cast<f64>(-rhs_shift[2]), 5e-2));
                shift_centered = shift;

                const auto max = noa::math::max(xmap);
                REQUIRE(max <= value);
            }

            // Non-centered:
            {
                noa::signal::fft::xmap<noa::fft::H2F>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, noa::fft::NORM_DEFAULT, buffer);

                auto [shift, value] = noa::signal::fft::xpeak_3d<noa::fft::F2F>(xmap);
                shift -= lhs_center;
                if (correlation_mode == noa::signal::CorrelationMode::DOUBLE_PHASE)
                    shift /= 2;

                INFO("Excepted shift: " << rhs_shift);
                INFO("Computed shift: " << -shift);
                REQUIRE_THAT(shift[0], Catch::WithinAbs(static_cast<f64>(-rhs_shift[0]), 5e-2));
                REQUIRE_THAT(shift[1], Catch::WithinAbs(static_cast<f64>(-rhs_shift[1]), 5e-2));
                REQUIRE_THAT(shift[2], Catch::WithinAbs(static_cast<f64>(-rhs_shift[2]), 5e-2));
                shift_not_centered = shift;

                const auto max = noa::math::max(xmap);
                REQUIRE(max <= value);
            }

            REQUIRE_THAT(shift_not_centered[0], Catch::WithinAbs(static_cast<f64>(shift_centered[0]), 1e-4));
            REQUIRE_THAT(shift_not_centered[1], Catch::WithinAbs(static_cast<f64>(shift_centered[1]), 1e-4));
            REQUIRE_THAT(shift_not_centered[2], Catch::WithinAbs(static_cast<f64>(shift_centered[2]), 1e-4));
        }
    }
}

TEMPLATE_TEST_CASE("unified::signal::fft, correlation peak 3D batched", "[noa][unified]", f64) {
    using namespace ::noa;
    auto shape = test::get_random_shape4_batched(3);
    shape[1] += 50;
    shape[2] += 50;
    shape[3] += 50;

    test::Randomizer<f32> randomizer{-10, 10};
    const Vec3<f32> radius{20, 20, 20};
    const f32 taper = 7;
    const Vec3<f32> lhs_center{shape[1] / 2, shape[2] / 2, shape[3] / 2};
    const Vec3<f32> rhs_shift{randomizer.get(), randomizer.get(), randomizer.get()};
    const Vec3<f32> rhs_center = lhs_center + rhs_shift;

    std::vector<noa::Device> devices{noa::Device("cpu")};
    if (noa::Device::is_any(noa::DeviceType::GPU))
        devices.emplace_back("gpu");

    const std::array<noa::signal::CorrelationMode, 4> modes{
            noa::signal::CorrelationMode::CONVENTIONAL,
            noa::signal::CorrelationMode::MUTUAL,
            noa::signal::CorrelationMode::PHASE,
            noa::signal::CorrelationMode::DOUBLE_PHASE};

    for (auto correlation_mode: modes) {
        for (auto& device: devices) {
            const auto stream = noa::StreamGuard(device);
            const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);

            auto [lhs, lhs_fft] = noa::fft::empty<TestType>(shape, options);
            auto [rhs, rhs_fft] = noa::fft::empty<TestType>(shape, options);
            noa::Array xmap = noa::memory::empty<TestType>(shape, options);
            noa::Array buffer = noa::memory::like(lhs_fft);
            noa::Array shifts = noa::memory::empty<Vec3<f32>>({shape[0], 1, 1, 1}, options);
            noa::Array values = noa::memory::empty<TestType>({shape[0], 1, 1, 1}, options);

            noa::geometry::rectangle({}, lhs, lhs_center, radius, taper);
            noa::geometry::rectangle({}, rhs, rhs_center, radius, taper);
            noa::fft::r2c(lhs, lhs_fft);
            noa::fft::r2c(rhs, rhs_fft);

            // Centered:
            {
                noa::signal::fft::xmap<noa::fft::H2FC>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, noa::fft::NORM_DEFAULT, buffer);
                noa::signal::fft::xpeak_3d<noa::fft::FC2FC>(xmap, shifts, values);
                shifts.eval();

                for (i64 i = 0; i < shape[0]; ++i) {
                    Vec3<f32> shift = shifts(i, 0, 0, 0);
                    shift -= lhs_center;
                    if (correlation_mode == noa::signal::CorrelationMode::DOUBLE_PHASE)
                        shift /= 2;

                    INFO("Excepted shift: " << rhs_shift);
                    INFO("Computed shift: " << -shift);
                    REQUIRE_THAT(shift[0], Catch::WithinAbs(static_cast<f64>(-rhs_shift[0]), 5e-2));
                    REQUIRE_THAT(shift[1], Catch::WithinAbs(static_cast<f64>(-rhs_shift[1]), 5e-2));
                    REQUIRE_THAT(shift[2], Catch::WithinAbs(static_cast<f64>(-rhs_shift[2]), 5e-2));

                    const auto max = noa::math::max(xmap.subregion(i));
                    REQUIRE(max <= values(i, 0, 0, 0));
                }
            }

            noa::memory::fill(shifts, Vec3<f32>{0});
            noa::memory::fill(values, TestType{0});

            // Non-centered:
            {
                noa::signal::fft::xmap<noa::fft::H2F>(
                        lhs_fft, rhs_fft, xmap,
                        correlation_mode, noa::fft::NORM_DEFAULT, buffer);
                noa::signal::fft::xpeak_3d<noa::fft::F2F>(xmap, shifts, values);
                shifts.eval();

                for (i64 i = 0; i < shape[0]; ++i) {
                    Vec3<f32> shift = shifts(i, 0, 0, 0);
                    shift -= lhs_center;
                    if (correlation_mode == noa::signal::CorrelationMode::DOUBLE_PHASE)
                        shift /= 2;

                    INFO("Excepted shift: " << rhs_shift);
                    INFO("Computed shift: " << -shift);
                    REQUIRE_THAT(shift[0], Catch::WithinAbs(static_cast<f64>(-rhs_shift[0]), 5e-2));
                    REQUIRE_THAT(shift[1], Catch::WithinAbs(static_cast<f64>(-rhs_shift[1]), 5e-2));
                    REQUIRE_THAT(shift[2], Catch::WithinAbs(static_cast<f64>(-rhs_shift[2]), 5e-2));

                    const auto max = noa::math::max(xmap.subregion(i));
                    REQUIRE(max <= values(i, 0, 0, 0));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("unified::signal::fft, correlation peak 3D, cpu vs gpu", "[noa][unified]", f32, f64) {
    using namespace ::noa;
    auto shape = test::get_random_shape4(3);
    shape[1] += 50;
    shape[2] += 50;
    shape[3] += 50;

    test::Randomizer<float> randomizer{-10, 10};
    const Vec3<f32> radius{20, 20, 20};
    const float taper = 7;
    const Vec3<f32> lhs_center{shape[1] / 2, shape[2] / 2, shape[3] / 2};
    const Vec3<f32> rhs_shift{randomizer.get(), randomizer.get(), randomizer.get()};
    const Vec3<f32> rhs_center = lhs_center + rhs_shift;

    if (!Device::is_any(DeviceType::GPU))
        return;

    const auto peak_mode = GENERATE(signal::PeakMode::PARABOLA_1D, signal::PeakMode::COM);
    test::Randomizer<i64> long_randomizer{1, peak_mode == signal::PeakMode::COM ? 2 : 32};
    const Vec3<i64> peak_window = {long_randomizer.get(), long_randomizer.get(), long_randomizer.get()};

    INFO(peak_mode);
    INFO(peak_window);

    // Centered:
    {
        auto [lhs, lhs_fft] = fft::empty<TestType>(shape);
        auto [rhs, rhs_fft] = fft::empty<TestType>(shape);
        Array xmap = memory::empty<TestType>(shape);

        noa::geometry::rectangle({}, lhs, lhs_center, radius, taper, {}, {}, TestType(-1));
        noa::geometry::rectangle({}, rhs, rhs_center, radius, taper, {}, {}, TestType(-1));
        noa::fft::r2c(lhs, lhs_fft);
        noa::fft::r2c(rhs, rhs_fft);
        noa::signal::fft::xmap<fft::H2FC>(lhs_fft, rhs_fft, xmap);
        const auto xmap_gpu = xmap.to(Device("gpu"));

        auto [cpu_shift, cpu_value] = noa::signal::fft::xpeak_3d<fft::FC2FC>(xmap, {}, peak_mode, peak_window);
        auto [gpu_shift, gpu_value] = noa::signal::fft::xpeak_3d<fft::FC2FC>(xmap_gpu, {}, peak_mode, peak_window);
        cpu_shift -= lhs_center;
        gpu_shift -= lhs_center;

        INFO("Excepted shift: " << rhs_shift);
        INFO("Computed cpu shift: " << -cpu_shift);
        INFO("Computed gpu shift: " << -gpu_shift);
        REQUIRE_THAT(cpu_shift[0], Catch::WithinAbs(static_cast<f64>(gpu_shift[0]), 5e-5));
        REQUIRE_THAT(cpu_shift[1], Catch::WithinAbs(static_cast<f64>(gpu_shift[1]), 5e-5));
        REQUIRE_THAT(cpu_shift[2], Catch::WithinAbs(static_cast<f64>(gpu_shift[2]), 5e-5));
        REQUIRE_THAT(cpu_value, Catch::WithinAbs(static_cast<f64>(gpu_value), 5e-6));
    }

    // Non-centered:
    {
        auto [lhs, lhs_fft] = fft::empty<TestType>(shape);
        auto [rhs, rhs_fft] = fft::empty<TestType>(shape);
        Array xmap = memory::empty<TestType>(shape);

        noa::geometry::rectangle({}, lhs, lhs_center, radius, taper);
        noa::geometry::rectangle({}, rhs, rhs_center, radius, taper);
        noa::fft::r2c(lhs, lhs_fft);
        noa::fft::r2c(rhs, rhs_fft);
        noa::signal::fft::xmap<fft::H2F>(lhs_fft, rhs_fft, xmap);
        const auto xmap_gpu = xmap.to(Device("gpu"));

        auto [cpu_shift, cpu_value] = signal::fft::xpeak_3d<fft::F2F>(xmap, {}, peak_mode, peak_window);
        auto [gpu_shift, gpu_value] = signal::fft::xpeak_3d<fft::F2F>(xmap_gpu, {}, peak_mode, peak_window);
        cpu_shift -= lhs_center;
        gpu_shift -= lhs_center;

        INFO("Excepted shift: " << rhs_shift);
        INFO("Computed cpu shift: " << -cpu_shift);
        INFO("Computed gpu shift: " << -gpu_shift);
        REQUIRE_THAT(cpu_shift[0], Catch::WithinAbs(static_cast<f64>(gpu_shift[0]), 5e-5));
        REQUIRE_THAT(cpu_shift[1], Catch::WithinAbs(static_cast<f64>(gpu_shift[1]), 5e-3));
        REQUIRE_THAT(cpu_shift[2], Catch::WithinAbs(static_cast<f64>(gpu_shift[2]), 5e-5));
        REQUIRE_THAT(cpu_value, Catch::WithinAbs(static_cast<f64>(gpu_value), 5e-6));
    }
}

TEST_CASE("unified::signal::fft, autocorrelate", "[.]") {
    std::vector<noa::Device> devices{noa::Device("cpu")};
    if (noa::Device::is_any(noa::DeviceType::GPU))
        devices.emplace_back("gpu");

    const auto shape = test::get_random_shape4(3);
    const auto center = (shape.pop_front() / 2).vec().as<f64>();

    for (auto& device: devices) {
        const auto stream = noa::StreamGuard(device);
        const auto options = noa::ArrayOption(device, noa::Allocator::MANAGED);

        const auto lhs = noa::math::random<f32>(noa::math::uniform_t{}, shape, -50, 50, options);
        const auto lhs_fft = noa::fft::r2c(lhs);
        const auto rhs_fft = lhs_fft.copy();
        const auto xmap = noa::memory::like(lhs);
        noa::signal::fft::xmap<noa::fft::H2F>(lhs_fft, rhs_fft, xmap);
        const auto [shift, _] = noa::signal::fft::xpeak_3d<noa::fft::F2F>(xmap);
        REQUIRE_THAT(shift[0], Catch::WithinAbs(center[0], 5e-2));
        REQUIRE_THAT(shift[1], Catch::WithinAbs(center[1], 5e-2));
        REQUIRE_THAT(shift[2], Catch::WithinAbs(center[2], 5e-2));
    }
}
