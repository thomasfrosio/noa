#include <noa/Array.h>
#include <noa/FFT.h>
#include <noa/Math.h>
#include <noa/Memory.h>
#include <noa/Geometry.h>
#include <noa/Signal.h>
#include <noa/IO.h>
#include <noa/Utils.h>

#include <iostream>

#include <catch2/catch.hpp>

#include "Helpers.h"

using namespace ::noa;

TEST_CASE("unified::signal::fft, Fourier-Mellin", "[.]") {
    const size4_t shape{1, 1, 512, 512};
    const float2_t center{shape[2] / 2, shape[3] / 2};
    const float2_t radius{128, 128};
    const float rotation = math::deg2rad(45.f);
    const float33_t rotation_matrix = math::inverse(
            geometry::translate(center) *
            float33_t(geometry::rotate(rotation)) *
            geometry::translate(-center)
    );

    constexpr float PI = math::Constants<float>::PI;
    const float2_t frequency_range{0, 0.5};
    const float2_t angle_range{-PI / 2, PI / 2};
    const size4_t polar_shape{1, 1, 1024, 256};

    std::vector<Device> devices = {Device{"cpu"}};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    io::MRCFile file;
    const path_t directory = test::NOA_DATA_PATH / "signal" / "fft";

    for (auto& device: devices) {
        StreamGuard stream{device};
        ArrayOption options{device, Allocator::MANAGED};
        INFO(device);

        // Generate the input images:
        Array<float> lhs{shape, options}, rhs{shape, options};
        signal::rectangle(Array<float>{}, lhs, center, radius, 10);
        geometry::transform2D(lhs, rhs, rotation_matrix);

        // Compute the non-redundant centered FFT:
        Array<cfloat_t> lhs_fft{shape.fft(), options}, rhs_fft{shape.fft(), options};
        fft::r2c(lhs, lhs_fft);
        fft::r2c(rhs, rhs_fft);
        fft::remap(fft::H2HC, lhs_fft, lhs_fft, shape);
        fft::remap(fft::H2HC, rhs_fft, rhs_fft, shape);

        signal::fft::highpass<fft::HC2HC>(lhs_fft, lhs_fft, shape, 0.4f, 0.4f);
        signal::fft::highpass<fft::HC2HC>(rhs_fft, rhs_fft, shape, 0.4f, 0.4f);

        // Compute the abs log-polar transforms:
        Array<float> lhs_cart = Array{lhs.share(), lhs_fft.shape(), lhs_fft.strides(), options};
        Array<float> rhs_cart = Array{rhs.share(), rhs_fft.shape(), rhs_fft.strides(), options};
        math::ewise(lhs_fft.release(), lhs_cart, math::abs_t{});
        math::ewise(rhs_fft.release(), rhs_cart, math::abs_t{});

        file.open(directory / string::format("test_{}_xmap_mellin_lhs.mrc", device), io::WRITE);
        file.shape(shape.fft());
        file.writeAll(lhs_cart.eval().get());
        file.close();

        file.open(directory / string::format("test_{}_xmap_mellin_rhs.mrc", device), io::WRITE);
        file.shape(shape.fft());
        file.writeAll(rhs_cart.eval().get());
        file.close();

        Array<float> lhs_polar{polar_shape, options}, rhs_polar{polar_shape, options};
        geometry::fft::cartesian2polar<fft::HC2FC>(lhs_cart, shape, lhs_polar, frequency_range, angle_range, true);
        geometry::fft::cartesian2polar<fft::HC2FC>(rhs_cart, shape, rhs_polar, frequency_range, angle_range, true);

        file.open(directory / string::format("test_{}_xmap_mellin_lhs_polar.mrc", device), io::WRITE);
        file.shape(polar_shape);
        file.writeAll(lhs_polar.eval().get());
        file.close();

        file.open(directory / string::format("test_{}_xmap_mellin_rhs_polar.mrc", device), io::WRITE);
        file.shape(polar_shape);
        file.writeAll(rhs_polar.eval().get());
        file.close();

        // Phase correlate:
        Array<cfloat_t> lhs_polar_fft{polar_shape.fft(), options}, rhs_polar_fft{polar_shape.fft(), options};
        fft::r2c(lhs_polar, lhs_polar_fft);
        fft::r2c(rhs_polar, rhs_polar_fft);
        signal::fft::xmap<fft::H2FC>(lhs_polar_fft, rhs_polar_fft, lhs_polar, true);

        file.open(directory / string::format("test_{}_xmap_mellin.mrc", device), io::WRITE);
        file.shape(polar_shape);
        file.writeAll(lhs_polar.eval().get());
        file.close();

        const float2_t peak = signal::fft::xpeak2D<fft::FC2FC>(lhs_polar);
        std::cout << peak;
    }
}

TEMPLATE_TEST_CASE("unified::signal::fft, Fourier-Mellin cryoEM", "[.]", float) {
    constexpr float PI = math::Constants<float>::PI;
    const float2_t frequency_range{0, 0.5};
    const float2_t angle_range{0, PI * 2};

    std::vector<Device> devices = {Device{"cpu"}};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const path_t directory = test::NOA_DATA_PATH / "signal" / "fft";
    const fft::Norm norm = fft::NORM_FORWARD;

    io::MRCFile file;
    for (auto& device: devices) {
        StreamGuard stream{device};
        ArrayOption options{device, Allocator::MANAGED};
        INFO(device);

        file.open(directory / "tilt1_slice21_subregion_cpu_rs5.mrc", io::READ);
        const size4_t cartesian_shape = file.shape();
        Array<float> lhs{cartesian_shape, options};
        file.readAll(lhs.get(), false);

        file.open(directory / "tilt1_slice21_subregion_rotated_cpu_rs5.mrc", io::READ);
        Array<float> rhs{cartesian_shape, options};
        file.readAll(rhs.get(), false);

        // Compute the non-redundant centered FFT:
        Array<cfloat_t> lhs_fft{cartesian_shape.fft(), options};
        Array<cfloat_t> rhs_fft{cartesian_shape.fft(), options};
        fft::r2c(lhs.release(), lhs_fft, norm);
        fft::r2c(rhs.release(), rhs_fft, norm);
        fft::remap(fft::H2HC, lhs_fft, lhs_fft, cartesian_shape);
        fft::remap(fft::H2HC, rhs_fft, rhs_fft, cartesian_shape);

        signal::fft::bandpass<fft::HC2HC>(lhs_fft, lhs_fft, cartesian_shape, 0.25, 0.5f, 0.20f, 0.1f);
        signal::fft::bandpass<fft::HC2HC>(rhs_fft, rhs_fft, cartesian_shape, 0.25, 0.5f, 0.20f, 0.1f);

        // Compute the abs log-polar transforms (reuse the real inputs):
        Array<float> lhs_cart{cartesian_shape.fft(), options};
        Array<float> rhs_cart{cartesian_shape.fft(), options};
        math::ewise(lhs_fft.release(), lhs_cart, math::abs_t{});
        math::ewise(rhs_fft.release(), rhs_cart, math::abs_t{});

        file.open(directory / string::format("test_{}_xmap_mellin_lhs.mrc", device), io::WRITE);
        file.shape(lhs_cart.shape());
        file.writeAll(lhs_cart.eval().get());
        file.close();

        file.open(directory / string::format("test_{}_xmap_mellin_rhs.mrc", device), io::WRITE);
        file.shape(rhs_cart.shape());
        file.writeAll(rhs_cart.eval().get());
        file.close();

        const size4_t polar_shape = cartesian_shape;
        Array<float> lhs_polar{polar_shape, options};
        Array<float> rhs_polar{polar_shape, options};
        geometry::fft::cartesian2polar<fft::HC2FC>(lhs_cart, cartesian_shape, lhs_polar, frequency_range, angle_range, true);
        geometry::fft::cartesian2polar<fft::HC2FC>(rhs_cart, cartesian_shape, rhs_polar, frequency_range, angle_range, true);

//        Array<float> radial_average{size4_t{1, 1, 1, polar_shape[3]}, options};
//        auto inverse_spectrum = [](float x) {
//            if (math::abs(x) > 0.f)
//                return 1.f / x;
//            else
//                return 1.f;
//        };
//        math::sum(lhs_polar, radial_average);
//        math::ewise(radial_average, radial_average, inverse_spectrum);
//        math::ewise(lhs_polar, radial_average, lhs_polar, math::multiply_t{});
//        math::sum(rhs_polar, radial_average);
//        math::ewise(radial_average, radial_average, inverse_spectrum);
//        math::ewise(rhs_polar, radial_average, rhs_polar, math::multiply_t{});

        file.open(directory / string::format("test_{}_xmap_mellin_lhs_polar1.mrc", device), io::WRITE);
        file.shape(polar_shape);
        file.writeAll(lhs_polar.eval().get());
        file.close();

        file.open(directory / string::format("test_{}_xmap_mellin_rhs_polar1.mrc", device), io::WRITE);
        file.shape(polar_shape);
        file.writeAll(rhs_polar.eval().get());
        file.close();

        // Phase correlate:
        Array<cfloat_t> lhs_polar_fft{polar_shape.fft(), options};
        Array<cfloat_t> rhs_polar_fft{polar_shape.fft(), options};
        fft::r2c(lhs_polar, lhs_polar_fft, norm);
        fft::r2c(rhs_polar, rhs_polar_fft, norm);
        signal::fft::xmap<fft::H2FC>(lhs_polar_fft, rhs_polar_fft, lhs_polar, true, norm);

        file.open(directory / string::format("test_{}_xmap_mellin.mrc", device), io::WRITE);
        file.shape(polar_shape);
        file.writeAll(lhs_polar.eval().get());
        file.close();

        const float2_t peak0 = signal::fft::xpeak2D<fft::FC2FC>(lhs_polar);
        std::cout << peak0 << '\n';

        Array<float> line{size4_t{1, 1, polar_shape[2], 1}, options};
        math::sum(lhs_polar, line);
        const float peak1 = signal::fft::xpeak1D<fft::FC2FC>(line.reshape({1, 1, 1, polar_shape[2]}));
        std::cout << peak1 << '\n';
    }
}
