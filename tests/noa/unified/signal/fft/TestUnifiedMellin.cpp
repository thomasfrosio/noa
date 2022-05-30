#include <noa/unified/Array.h>
#include <noa/unified/FFT.h>
#include <noa/unified/Math.h>
#include <noa/unified/Geometry.h>
#include <noa/unified/Signal.h>

#include <noa/common/io/ImageFile.h>
#include <iostream>

#include <catch2/catch.hpp>

#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::signal::fft, Fourier-Mellin", "[noa][unified]", float) {
    const size4_t shape{1, 1, 512, 512};
    const float2_t center{shape[2] / 2, shape[3] / 2};
    const float2_t radius{128, 128};
    const float rotation = math::toRad(45.f);

    constexpr float PI = math::Constants<float>::PI;
    const float2_t frequency_range{0, 0.5};
    const float2_t angle_range{-PI / 2, PI / 2};
    const size4_t polar_shape{1, 1, 1024, 256};

    std::vector<Device> devices = {Device{"cpu"}};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    io::ImageFile file;
    const path_t directory = test::NOA_DATA_PATH / "signal" / "fft";

    for (auto& device: devices) {
        StreamGuard stream{device};
        ArrayOption options{device, Allocator::MANAGED};
        INFO(device);

        // Generate the input images:
        Array<float> lhs{shape, options}, rhs{shape, options};
        signal::rectangle(Array<float>{}, lhs, center, radius, 10);
        geometry::rotate2D(lhs, rhs, rotation, center);

        // Compute the non-redundant centered FFT:
        Array<cfloat_t> lhs_fft{shape.fft(), options}, rhs_fft{shape.fft(), options};
        fft::r2c(lhs, lhs_fft);
        fft::r2c(rhs, rhs_fft);
        fft::remap(fft::H2HC, lhs_fft, lhs_fft, shape);
        fft::remap(fft::H2HC, rhs_fft, rhs_fft, shape);

        signal::fft::highpass<fft::HC2HC>(lhs_fft, lhs_fft, shape, 0.4f, 0.4f);
        signal::fft::highpass<fft::HC2HC>(rhs_fft, rhs_fft, shape, 0.4f, 0.4f);

        // Compute the abs log-polar transforms:
        Array<float> lhs_cart = Array{lhs.share(), lhs_fft.shape(), lhs_fft.stride(), options};
        Array<float> rhs_cart = Array{rhs.share(), rhs_fft.shape(), rhs_fft.stride(), options};
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
        signal::fft::xmap<fft::H2FC>(lhs_polar_fft, rhs_polar_fft, lhs_polar, polar_shape, true);

        file.open(directory / string::format("test_{}_xmap_mellin.mrc", device), io::WRITE);
        file.shape(polar_shape);
        file.writeAll(lhs_polar.eval().get());
        file.close();

        const float2_t peak = signal::fft::xpeak2D<fft::FC2FC>(lhs_polar);
        std::cout << peak;
    }
}
