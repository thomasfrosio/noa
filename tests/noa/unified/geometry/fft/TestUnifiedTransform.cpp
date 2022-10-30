#include <noa/FFT.h>
#include <noa/Geometry.h>
#include <noa/IO.h>
#include <noa/Math.h>
#include <noa/Memory.h>
#include <noa/cpu/geometry/fft/Project.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

// The hermitian symmetry isn't broken by transform2D and transform3D.
TEST_CASE("unified::geometry::fft::transform2D, check redundancy", "[.]") {
    const dim4_t shape = {1, 1, 128, 128};
    const path_t output_path = test::NOA_DATA_PATH / "geometry" / "fft";
    ArrayOption option(Device("gpu"), Allocator::MANAGED);

    Array input = memory::linspace<float>(shape, -10, 10, true, option);
    Array output0 = fft::r2c(input);
    fft::remap(fft::H2HC, output0, output0, shape);

    Array output1 = memory::like(output0);
    const float22_t rotation = geometry::rotate(math::deg2rad(45.f));
    geometry::fft::transform2D<fft::HC2HC>(output0, output1, shape, rotation, float2_t{});
    io::save(math::real(output1), output_path / "test_output1_real.mrc");
    io::save(math::imag(output1), output_path / "test_output1_imag.mrc");
}

TEST_CASE("unified::geometry::fft::transform3D, check redundancy", "[.]") {
    const dim4_t shape = {1, 128, 128, 128};
    const path_t output_path = test::NOA_DATA_PATH / "geometry" / "fft";
    ArrayOption option(Device("gpu"), Allocator::MANAGED);

    Array input = memory::linspace<float>(shape, -10, 10, true, option);
    Array output0 = fft::r2c(input);
    fft::remap(fft::H2HC, output0, output0, shape);

    Array output1 = memory::like(output0);
    const float33_t rotation = geometry::euler2matrix(math::deg2rad(float3_t{45.f, 0, 0}), "ZYX", false);
    geometry::fft::transform3D<fft::HC2HC>(output0, output1, shape, rotation, float3_t{});
    io::save(math::real(output1), output_path / "test_output1_real.mrc");
    io::save(math::imag(output1), output_path / "test_output1_imag.mrc");
}

TEST_CASE("unified::geometry::fft::insert2D, check redundancy", "[.]") {
    const dim4_t slice_shape = {1, 1, 128, 128};
    const dim4_t grid_shape = {1, 128, 128, 128};
    const path_t output_path = test::NOA_DATA_PATH / "geometry" / "fft";
    ArrayOption option(Device("cpu"), Allocator::MANAGED);

    Array input = memory::linspace<float>(slice_shape, -10, 10, true, option);
    Array slice = fft::r2c(input).eval();
    fft::remap(fft::H2HC, slice, slice, slice_shape);
//    memory::arange(slice);
    io::save(math::real(slice), output_path / "test_slice_real.mrc");
//    io::save(math::imag(slice), output_path / "test_grid_imag.mrc");

    Array grid = memory::zeros<cfloat_t>(grid_shape.fft(), option);
    const float33_t rotation = geometry::euler2matrix(math::deg2rad(float3_t{45, 0, 0}), "ZYX", false);

//    geometry::fft::insert3D<fft::HC2HC>(slice, slice_shape, grid, grid_shape, float22_t{}, rotation);
    cpu::Stream stream = Stream::current(option.device()).cpu();
    noa::cpu::geometry::fft::insert3D<fft::HC2HC>(
            slice.share(), slice.strides(), slice_shape,
            grid.share(), grid.strides(), grid_shape,
            float22_t{}, rotation,
            0.5f, dim4_t{}, float2_t{}, stream);

//    io::save(math::real(grid), output_path / "test_grid_real.mrc");
    io::save(math::imag(grid), output_path / "test_grid_imag.mrc");
}
