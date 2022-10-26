#include <noa/FFT.h>
#include <noa/Geometry.h>
#include <noa/IO.h>
#include <noa/Math.h>
#include <noa/Memory.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

TEST_CASE("unified::geometry::fft::transform2D, check redundancy", ".") {
    const dim4_t shape = {1, 1, 128, 128};
    const path_t output_path = test::NOA_DATA_PATH / "geometry" / "fft";
    ArrayOption option(Device("gpu"), Allocator::MANAGED);

    Array input = memory::linspace<float>(shape, -10, 10, true, option);
    Array output0 = fft::r2c(input);
    fft::remap(fft::H2HC, output0, output0, shape);
    io::save(math::real(output0), output_path / "test_output0_real.mrc");
    io::save(math::imag(output0), output_path / "test_output0_imag.mrc");

    Array output1 = memory::like(output0);
    const float22_t rotation = geometry::rotate(math::deg2rad(45.f));
    geometry::fft::transform2D<fft::HC2HC>(output0, output1, shape, rotation, float2_t{});
    io::save(math::real(output1), output_path / "test_output1_real.mrc");
    io::save(math::imag(output1), output_path / "test_output1_imag.mrc");
}
