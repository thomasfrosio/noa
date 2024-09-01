#include <noa/unified/fft/Transform.hpp>
#include <noa/unified/fft/Factory.hpp>
#include <noa/unified/fft/Remap.hpp>
#include <noa/unified/geometry/DrawShape.hpp>
#include <noa/unified/Factory.hpp>
#include <noa/unified/Random.hpp>
#include <noa/unified/Reduce.hpp>
#include <noa/unified/signal/Correlate.hpp>

#include <noa/unified/io/ImageFile.hpp>

#include <catch2/catch.hpp>
#include "Utils.hpp"

using namespace ::noa::types;

namespace {
    constexpr std::array cross_correlation_modes{
        noa::signal::Correlation::CONVENTIONAL,
        noa::signal::Correlation::MUTUAL,
        noa::signal::Correlation::PHASE,
        noa::signal::Correlation::DOUBLE_PHASE};

    template<size_t N>
    struct TestData {
        Vec<f64, N> radius;
        f64 smoothness;
        Vec<f64, N> lhs_center;
        Vec<f64, N> rhs_center;
        Vec<f64, N> expected_shift;
    };

    template<size_t N>
    auto generate_data(const auto& shape) {
        TestData<N> data{};
        data.radius = Vec<f64, N>::from_value(N == 2 ? 25.: 20.);
        data.smoothness = 7.;
        data.lhs_center = (shape.vec.template pop_front<4 - N>() / 2).template as<f64>();

        constexpr auto max_shift = N == 2 ? 30.: 10.;
        test::Randomizer<f64> randomizer{-max_shift, max_shift};
        test::randomize(data.expected_shift.data(), N, randomizer);
        data.rhs_center = data.lhs_center + data.expected_shift;
        return data;
    }
}

TEST_CASE("unified::signal, correlation peak", "[noa][unified]") { // TEMPLATE_ , f32, f64
    using TestType = f32;
    constexpr size_t N = 2;

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    auto shape = test::random_shape(N) + Shape<i64, N>::from_value(N == 2 ? 200 : 50).push_front<4 - N>(0);
    fmt::print("shape={}\n", shape);

    TestData data = generate_data<N>(shape);
    fmt::print("Excepted shift: {}\n", data.expected_shift);

    for (auto correlation_mode: cross_correlation_modes) {
        const auto xmap_options = noa::signal::CrossCorrelationMapOptions{
            .mode=correlation_mode, .ifft_norm=noa::fft::NORM_DEFAULT};

        for (auto device: devices) {
            const auto stream = StreamGuard(device);
            const auto options = ArrayOption(device, Allocator::MANAGED);

            auto [lhs, lhs_rfft] = noa::fft::empty<TestType>(shape, options);
            auto [rhs, rhs_rfft] = noa::fft::empty<TestType>(shape, options);
            const Array xmap = noa::empty<TestType>(shape, options);
            const Array buffer = noa::like(lhs_rfft);

            noa::geometry::draw_shape({}, lhs, noa::geometry::Rectangle{data.lhs_center, data.radius, data.smoothness});
            noa::geometry::draw_shape({}, rhs, noa::geometry::Rectangle{data.rhs_center, data.radius, data.smoothness});
            noa::fft::r2c(lhs, lhs_rfft);
            noa::fft::r2c(rhs, rhs_rfft);

            auto run = [&]<Remap REMAP>(auto xpeak_options) {
                noa::signal::cross_correlation_map<REMAP>(lhs_rfft, rhs_rfft, xmap, xmap_options, buffer);

                constexpr Remap REMAP_ = REMAP.flip().erase_output();
                auto [peak_coordinate, peak_value] = noa::signal::cross_correlation_peak<REMAP_>(xmap, xpeak_options);
                auto computed_shift = -(peak_coordinate - data.lhs_center);
                if (correlation_mode == noa::signal::Correlation::DOUBLE_PHASE)
                    computed_shift /= 2;
                fmt::print("Computed shift={}, diff={}\n", computed_shift, abs(computed_shift - data.expected_shift));

                const auto max_value = noa::max(xmap);
                REQUIRE(max_value <= peak_value);

                for (size_t i: noa::irange(N))
                    REQUIRE_THAT(computed_shift[i], Catch::WithinAbs(data.expected_shift[i], 5e-2));
                return computed_shift;
            };

            const auto xpeak_options = noa::signal::CrossCorrelationPeakOptions<N>{};
            const auto shift_centered = run.operator()<Remap::H2FC>(xpeak_options);
            const auto shift_not_centered = run.operator()<Remap::H2F>(xpeak_options);

            auto xpeak_options_max = noa::signal::CrossCorrelationPeakOptions{
                .maximum_lag = abs(data.expected_shift) + 5};
            if (correlation_mode == noa::signal::Correlation::DOUBLE_PHASE)
                xpeak_options_max.maximum_lag *= 2;

            fmt::print("shape={}, maximum_lag={}\n", shape, xpeak_options_max.maximum_lag);
            const auto shift_centered_max = run.operator()<Remap::H2FC>(xpeak_options_max);
            const auto shift_not_centered_max = run.operator()<Remap::H2F>(xpeak_options_max);

            for (size_t i: noa::irange(N)) {
                REQUIRE_THAT(shift_not_centered[i], Catch::WithinAbs(shift_centered[i], 1e-4));
                REQUIRE_THAT(shift_not_centered[i], Catch::WithinAbs(shift_not_centered_max[i], 1e-6));
                REQUIRE_THAT(shift_centered[i], Catch::WithinAbs(shift_centered_max[i], 1e-6));
            }
        }
    }
}

TEST_CASE("unified::signal, correlation peak batched", "[noa][unified]") {
    // f32, f64
    using TestType = f32;
    constexpr size_t N = 2;

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    auto shape = test::random_shape_batched(N, {.batch_range = {2, 5}});
    shape[2] += 200;
    shape[3] += 200;
    INFO(shape);

    std::vector<TestData<N>> data;
    const auto lhs_inverse_affine_matrices = noa::empty<Mat<f32, N + 1, N + 1>>(shape[0]);
    const auto rhs_inverse_affine_matrices = noa::empty<Mat<f32, N + 1, N + 1>>(shape[0]);
    for (auto i: noa::irange(shape[0])) {
        auto tmp = generate_data<N>(shape);
        data.emplace_back(tmp);
        lhs_inverse_affine_matrices(0, 0, 0, i) = noa::geometry::translate(-tmp.lhs_center).as<f32>();
        rhs_inverse_affine_matrices(0, 0, 0, i) = noa::geometry::translate(-tmp.rhs_center).as<f32>();
    }

    for (auto correlation_mode: cross_correlation_modes) {
        const auto xmap_options = noa::signal::CrossCorrelationMapOptions{
            .mode=correlation_mode, .ifft_norm=noa::fft::NORM_DEFAULT};
        const auto xpeak_options = noa::signal::CrossCorrelationPeakOptions<N>{};

        for (auto& device: devices) {
            const auto stream = StreamGuard(device, Stream::DEFAULT);
            const auto options = ArrayOption(device, Allocator::MANAGED);

            auto [lhs, lhs_rfft] = noa::fft::empty<TestType>(shape, options);
            auto [rhs, rhs_rfft] = noa::fft::empty<TestType>(shape, options);
            const auto xmap = noa::empty<TestType>(shape, options);
            const auto buffer = noa::like(lhs_rfft);

            noa::geometry::draw_shape({}, lhs, noa::geometry::Rectangle{
                .radius=data[0].radius, .smoothness=data[0].smoothness},
                 lhs_inverse_affine_matrices.to({device}));
            noa::geometry::draw_shape({}, rhs, noa::geometry::Rectangle{
                .radius=data[0].radius, .smoothness=data[0].smoothness},
                 rhs_inverse_affine_matrices.to({device}));

            noa::fft::r2c(lhs, lhs_rfft);
            noa::fft::r2c(rhs, rhs_rfft);

            auto run = [&]<Remap REMAP>{
                noa::signal::cross_correlation_map<REMAP>(lhs_rfft, rhs_rfft, xmap, xmap_options, buffer);

                const auto shifts = noa::empty<Vec<f32, N>>(shape[0]);
                const auto values = noa::empty<TestType>(shape[0]);
                constexpr Remap REMAP_ = REMAP.flip().erase_output();
                noa::signal::cross_correlation_peak<REMAP_>(xmap, shifts, values, xpeak_options);

                for (size_t i{}; i < shifts.eval().size(); ++i) {
                    INFO(i);
                    auto computed_shift = -(shifts(0, 0, 0, i).as<f64>() - data[i].lhs_center);
                    if (correlation_mode == noa::signal::Correlation::DOUBLE_PHASE)
                        computed_shift /= 2;

                    for (size_t j: noa::irange(N))
                        REQUIRE_THAT(computed_shift[j], Catch::WithinAbs(data[i].expected_shift[j], 5e-2));

                    const auto max = noa::max(xmap.subregion(i));
                    REQUIRE(max <= values(0, 0, 0, i));
                }
            };
            run.operator()<Remap::H2FC>();
            run.operator()<Remap::H2F>();
        }
    }
}

// TEMPLATE_TEST_CASE("unified::signal, correlation peak 2d, cpu vs gpu", "[noa][unified]", f32, f64) {
//     if (not Device::is_any_gpu())
//         return;
//
//     auto shape = test::random_shape(2);
//     shape[2] += 100;
//     shape[3] += 100;
//
//     test::Randomizer<f64> float_randomizer{-20, 20};
//     const auto radius = Vec{25., 25.};
//     const auto taper = 7.;
//     const auto lhs_center = (shape.vec.filter(2, 3) / 2).as<f64>();
//     const auto expected_shift = Vec{float_randomizer.get(), float_randomizer.get()};
//     const auto rhs_center = lhs_center + expected_shift;
//
//     const auto gpu_options = ArrayOption{.device=Device(Device::GPU)};
//
//     test::Randomizer<i64> long_randomizer{1, 4};
//     const auto peak_radius = Vec{long_randomizer.get(), long_randomizer.get()};
//
//     INFO(shape);
//     INFO(peak_radius);
//
//     // Centered:
//     {
//         const auto [lhs, lhs_fft] = noa::fft::empty<TestType>(shape);
//         const auto [rhs, rhs_fft] = noa::fft::empty<TestType>(shape);
//         const auto xmap = noa::empty<TestType>(shape);
//
//         noa::geometry::draw_shape({}, lhs, noa::geometry::Rectangle{
//             .center=lhs_center, .radius=radius, .smoothness=taper});
//         noa::geometry::draw_shape({}, rhs, noa::geometry::Rectangle{
//             .center=rhs_center, .radius=radius, .smoothness=taper});
//         noa::fft::r2c(lhs, lhs_fft);
//         noa::fft::r2c(rhs, rhs_fft);
//
//         noa::signal::cross_correlation_map<Remap::H2FC>(lhs_fft, rhs_fft, xmap);
//         const auto xmap_gpu = xmap.to(gpu_options).eval();
//
//         auto [cpu_peak_coords, cpu_peak_value] = noa::signal::cross_correlation_peak_2d<"fc2fc">(
//             xmap, {.registration_radius=peak_radius});
//         auto [gpu_peak_coords, gpu_peak_value] = noa::signal::cross_correlation_peak_2d<"fc2fc">(
//             xmap_gpu, {.registration_radius=peak_radius});
//         const auto cpu_computed_shift = -(cpu_peak_coords - lhs_center);
//         const auto gpu_computed_shift = -(gpu_peak_coords - lhs_center);
//
//         INFO("Excepted shift: " << expected_shift);
//         INFO("Computed cpu shift: " << cpu_computed_shift);
//         INFO("Computed gpu shift: " << gpu_computed_shift);
//         REQUIRE_THAT(cpu_computed_shift[0], Catch::WithinAbs(gpu_computed_shift[0], 5e-5));
//         REQUIRE_THAT(cpu_computed_shift[1], Catch::WithinAbs(gpu_computed_shift[1], 5e-5));
//         REQUIRE_THAT(cpu_peak_value, Catch::WithinAbs(static_cast<f64>(gpu_peak_value), 5e-6));
//     }
//
//     // Non-centered:
//     {
//         const auto [lhs, lhs_fft] = noa::fft::empty<TestType>(shape);
//         const auto [rhs, rhs_fft] = noa::fft::empty<TestType>(shape);
//         const auto xmap = noa::empty<TestType>(shape);
//
//         noa::geometry::draw_shape({}, lhs, noa::geometry::Rectangle{
//             .center=lhs_center, .radius=radius, .smoothness=taper});
//         noa::geometry::draw_shape({}, rhs, noa::geometry::Rectangle{
//             .center=rhs_center, .radius=radius, .smoothness=taper});
//         noa::fft::r2c(lhs, lhs_fft);
//         noa::fft::r2c(rhs, rhs_fft);
//         noa::signal::cross_correlation_map<Remap::H2F>(lhs_fft, rhs_fft, xmap);
//         const auto xmap_gpu = xmap.to(gpu_options).eval();
//
//         auto [cpu_peak_coords, cpu_peak_value] = noa::signal::cross_correlation_peak_2d<"f2f">(
//             xmap, {.registration_radius=peak_radius});
//         auto [gpu_peak_coords, gpu_peak_value] = noa::signal::cross_correlation_peak_2d<"f2f">(
//             xmap_gpu, {.registration_radius=peak_radius});
//         const auto cpu_computed_shift = -(cpu_peak_coords - lhs_center);
//         const auto gpu_computed_shift = -(gpu_peak_coords - lhs_center);
//
//         INFO("Excepted shift: " << expected_shift);
//         INFO("Computed cpu shift: " << cpu_computed_shift);
//         INFO("Computed gpu shift: " << gpu_computed_shift);
//         REQUIRE_THAT(cpu_computed_shift[0], Catch::WithinAbs(gpu_computed_shift[0], 5e-5));
//         REQUIRE_THAT(cpu_computed_shift[1], Catch::WithinAbs(gpu_computed_shift[1], 5e-5));
//         REQUIRE_THAT(cpu_peak_value, Catch::WithinAbs(static_cast<f64>(gpu_peak_value), 5e-6));
//     }
// }



// TEMPLATE_TEST_CASE("unified::signal, correlation peak 3D, cpu vs gpu", "[noa][unified]", f32, f64) {
//     auto shape = test::random_shape(3);
//     shape[1] += 50;
//     shape[2] += 50;
//     shape[3] += 50;
//
//     test::Randomizer<f64> float_randomizer{-10, 10};
//     const auto radius = Vec{20., 20., 20.};
//     const auto taper = 7.;
//     const auto lhs_center = (shape.vec.filter(1, 2, 3) / 2).as<f64>();
//     const auto expected_shift = Vec{float_randomizer.get(), float_randomizer.get(), float_randomizer.get()};
//     const auto rhs_center = lhs_center + expected_shift;
//
//     if (not Device::is_any_gpu())
//         return;
//
//     test::Randomizer<i64> long_randomizer{1, 4};;
//     const auto peak_radius = Vec{long_randomizer.get(), long_randomizer.get(), long_randomizer.get()};
//     INFO(peak_radius);
//
//     // Centered:
//     {
//         const auto [lhs, lhs_fft] = noa::fft::empty<TestType>(shape);
//         const auto [rhs, rhs_fft] = noa::fft::empty<TestType>(shape);
//         const auto xmap = noa::empty<TestType>(shape);
//
//         noa::geometry::draw_shape({}, lhs, noa::geometry::Rectangle{
//             .center=lhs_center, .radius=radius, .smoothness=taper, .cvalue=-1.});
//         noa::geometry::draw_shape({}, rhs, noa::geometry::Rectangle{
//             .center=rhs_center, .radius=radius, .smoothness=taper, .cvalue=-1.});
//         noa::fft::r2c(lhs, lhs_fft);
//         noa::fft::r2c(rhs, rhs_fft);
//         noa::signal::cross_correlation_map<Remap::H2FC>(lhs_fft, rhs_fft, xmap);
//         const auto xmap_gpu = xmap.to({.device=Device("gpu")});
//
//         const auto [cpu_peak_coords, cpu_peak_value] = noa::signal::cross_correlation_peak_3d<Remap::FC2FC>(
//             xmap, {.registration_radius=peak_radius});
//         const auto [gpu_peak_coords, gpu_peak_value] = noa::signal::cross_correlation_peak_3d<Remap::FC2FC>(
//             xmap_gpu, {.registration_radius=peak_radius});
//         const auto cpu_computed_shift = -(cpu_peak_coords - lhs_center);
//         const auto gpu_computed_shift = -(gpu_peak_coords - lhs_center);
//
//         INFO("Excepted shift: " << expected_shift);
//         INFO("Computed cpu shift: " << cpu_computed_shift);
//         INFO("Computed gpu shift: " << gpu_computed_shift);
//         REQUIRE_THAT(cpu_computed_shift[0], Catch::WithinAbs(static_cast<f64>(gpu_computed_shift[0]), 5e-5));
//         REQUIRE_THAT(cpu_computed_shift[1], Catch::WithinAbs(static_cast<f64>(gpu_computed_shift[1]), 5e-5));
//         REQUIRE_THAT(cpu_computed_shift[2], Catch::WithinAbs(static_cast<f64>(gpu_computed_shift[2]), 5e-5));
//         REQUIRE_THAT(cpu_peak_value, Catch::WithinAbs(static_cast<f64>(gpu_peak_value), 5e-6));
//     }
//
//     // Non-centered:
//     {
//         const auto [lhs, lhs_fft] = noa::fft::empty<TestType>(shape);
//         const auto [rhs, rhs_fft] = noa::fft::empty<TestType>(shape);
//         const auto xmap = noa::empty<TestType>(shape);
//
//         noa::geometry::draw_shape({}, lhs, noa::geometry::Rectangle{
//             .center=lhs_center, .radius=radius, .smoothness=taper});
//         noa::geometry::draw_shape({}, rhs, noa::geometry::Rectangle{
//             .center=rhs_center, .radius=radius, .smoothness=taper});
//         noa::fft::r2c(lhs, lhs_fft);
//         noa::fft::r2c(rhs, rhs_fft);
//         noa::signal::cross_correlation_map<Remap::H2F>(lhs_fft, rhs_fft, xmap);
//         const auto xmap_gpu = xmap.to({.device=Device("gpu")});
//
//         const auto [cpu_peak_coords, cpu_peak_value] = noa::signal::cross_correlation_peak_3d<Remap::FC2FC>(
//             xmap, {.registration_radius=peak_radius});
//         const auto [gpu_peak_coords, gpu_peak_value] = noa::signal::cross_correlation_peak_3d<Remap::FC2FC>(
//             xmap_gpu, {.registration_radius=peak_radius});
//         const auto cpu_computed_shift = -(cpu_peak_coords - lhs_center);
//         const auto gpu_computed_shift = -(gpu_peak_coords - lhs_center);
//
//         INFO("Excepted shift: " << expected_shift);
//         INFO("Computed cpu shift: " << cpu_computed_shift);
//         INFO("Computed gpu shift: " << gpu_computed_shift);
//         REQUIRE_THAT(cpu_computed_shift[0], Catch::WithinAbs(static_cast<f64>(gpu_computed_shift[0]), 5e-5));
//         REQUIRE_THAT(cpu_computed_shift[1], Catch::WithinAbs(static_cast<f64>(gpu_computed_shift[1]), 5e-5));
//         REQUIRE_THAT(cpu_computed_shift[2], Catch::WithinAbs(static_cast<f64>(gpu_computed_shift[2]), 5e-5));
//         REQUIRE_THAT(cpu_peak_value, Catch::WithinAbs(static_cast<f64>(gpu_peak_value), 5e-6));
//     }
// }

TEST_CASE("unified::signal, autocorrelate", "[.]") {
    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    const auto shape = test::random_shape(3);
    const auto center = (shape.pop_front() / 2).vec.as<f64>();

    for (auto& device: devices) {
        const auto stream = noa::StreamGuard(device);
        const auto options = noa::ArrayOption(device, Allocator::MANAGED);

        const auto lhs = noa::random(noa::Uniform{-50.f, 50.f}, shape, options);
        const auto lhs_rfft = noa::fft::r2c(lhs);
        const auto rhs_rfft = lhs_rfft.copy();
        const auto xmap = noa::like<f32>(lhs);
        noa::signal::cross_correlation_map<Remap::H2F>(lhs_rfft, rhs_rfft, xmap);
        const auto [shift, _] = noa::signal::cross_correlation_peak_3d<Remap::F2F>(xmap);
        REQUIRE_THAT(shift[0], Catch::WithinAbs(center[0], 5e-2));
        REQUIRE_THAT(shift[1], Catch::WithinAbs(center[1], 5e-2));
        REQUIRE_THAT(shift[2], Catch::WithinAbs(center[2], 5e-2));
    }
}
