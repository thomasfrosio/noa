#include <noa/common/geometry/Euler.h>
#include <noa/common/geometry/Transform.h>
#include <noa/common/io/MRCFile.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Set.h>
#include <noa/cpu/memory/Linspace.h>
#include <noa/cpu/geometry/fft/Project.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::geometry::fft::insert3D", "[.]") {
    const size4_t slices_shape{1, 1, 256, 256};
    const size4_t grid_shape{1, 256, 256, 256};
    const size4_t target_shape{0, 256, 256, 256};
    const size4_t slices_stride = slices_shape.fft().strides();
    const size4_t grid_stride = grid_shape.fft().strides();

    cpu::memory::PtrHost<float> slices(slices_shape.fft().elements());
    cpu::memory::PtrHost<float> grid(grid_shape.fft().elements());

    cpu::memory::set(slices.begin(), slices.end(), 1.f);
    cpu::memory::set(grid.begin(), grid.end(), 0.f);

    cpu::memory::PtrHost<float22_t> scaling_factors(slices_shape[0]);
    cpu::memory::PtrHost<float33_t> rotations(slices_shape[0]);

    for (uint i = 0; i < slices_shape[0]; ++i) {
        scaling_factors[i] = geometry::scale(float2_t{1, 1});
        rotations[i] = geometry::euler2matrix(math::deg2rad(float3_t{176, 20, 0}), "ZYX", false);
    }

    const bool do_ews = false;
    const float cutoff = 0.5f;
    const float wavelength = 0.01968761530923358f; // A
    const float2_t pixel_size{1, 1}; // A/pix
    const float2_t ews_radius = do_ews ? pixel_size / wavelength : float2_t{}; // 1/pix

    cpu::Stream stream(cpu::Stream::DEFAULT);
    stream.threads(1);

    for (int i = 0; i < (1 + do_ews) ; ++i) {
        const float sign = i == 0 ? 1 : -1;
        cpu::geometry::fft::insert3D<fft::HC2HC>(
                slices.share(), slices_stride, slices_shape,
                grid.share(), grid_stride, grid_shape,
                scaling_factors.share(), rotations.share(),
                cutoff, target_shape, sign * ews_radius, stream);

        cpu::geometry::fft::extract3D<fft::HC2HC>(
                grid.share(), grid_stride, grid_shape,
                slices.share(), slices_stride, slices_shape,
                scaling_factors.share(), rotations.share(),
                cutoff, target_shape, sign * ews_radius, stream);
    }

    stream.synchronize();
    io::MRCFile file(test::NOA_DATA_PATH / "geometry" / "fft" / "test_insert3D.mrc", io::WRITE);
    file.shape(grid_shape.fft());
    file.writeAll(grid.get(), false);

    file.open(test::NOA_DATA_PATH / "geometry" / "fft" / "test_extract3D.mrc", io::WRITE);
    file.shape(slices_shape.fft());
    file.writeAll(slices.get(), false);
}

TEST_CASE("cpu::geometry::fft::insert3D, thickness", "[.]") {
    const size4_t slices_shape{1, 1, 512, 512};
    const size4_t grid_shape{1, 512, 512, 512};
    const size4_t target_shape{0, 512, 512, 512};
    const size4_t slices_stride = slices_shape.fft().strides();
    const size4_t grid_stride = grid_shape.fft().strides();

    cpu::memory::PtrHost<float> slices(slices_shape.fft().elements());
    cpu::memory::PtrHost<float> grid(grid_shape.fft().elements());

    cpu::memory::set(slices.begin(), slices.end(), 1.f);
    cpu::memory::set(grid.begin(), grid.end(), 0.f);

    cpu::memory::PtrHost<float22_t> scaling_factors(slices_shape[0]);
    cpu::memory::PtrHost<float33_t> rotations(slices_shape[0]);

    for (uint i = 0; i < slices_shape[0]; ++i) {
        scaling_factors[i] = geometry::scale(float2_t{1, 1});
        rotations[i] = geometry::euler2matrix(math::deg2rad(float3_t{0, 0, 0}), "ZYX", false);
    }

    const float cutoff = 0.5f;
    const float slice_z_radius = 0.001953125f;

    const bool do_ews = true;
    const float wavelength = 0.01968761530923358f; // A
    const float2_t pixel_size{1, 1}; // A/pix
    const float2_t ews_radius = do_ews ? pixel_size / wavelength : float2_t{}; // 1/pix

    cpu::Stream stream(cpu::Stream::DEFAULT);
    stream.threads(1);

//    cpu::geometry::fft::insert3D<fft::HC2HC>(
//            slices.share(), slices_stride, slices_shape,
//            grid.share(), grid_stride, grid_shape,
//            scaling_factors[0], rotations[0],
//            cutoff, target_shape, float2_t{}, stream);
    cpu::geometry::fft::insert3D<fft::HC2HC>(
            slices.share(), slices_stride, slices_shape,
            grid.share(), grid_stride, grid_shape,
            scaling_factors[0], rotations[0],
            cutoff, target_shape, ews_radius, stream);
    stream.synchronize();
    io::MRCFile file(test::NOA_DATA_PATH / "geometry" / "fft" / "test_insert3D_thickness0.mrc", io::WRITE);
    file.shape(grid_shape.fft());
    file.writeAll(grid.get(), false);

    cpu::memory::set(grid.begin(), grid.end(), 0.f);
//    cpu::geometry::fft::insert3D<fft::HC2HC>(
//            slices.share(), slices_stride, slices_shape,
//            grid.share(), grid_stride, grid_shape,
//            scaling_factors[0], rotations[0],
//            cutoff, target_shape, float2_t{}, slice_z_radius, stream);
    cpu::geometry::fft::insert3D<fft::HC2HC>(
            slices.share(), slices_stride, slices_shape,
            grid.share(), grid_stride, grid_shape,
            scaling_factors[0], rotations[0],
            cutoff, target_shape, ews_radius, slice_z_radius, stream);
    stream.synchronize();
    file.open(test::NOA_DATA_PATH / "geometry" / "fft" / "test_insert3D_thickness1.mrc", io::WRITE);
    file.shape(grid_shape.fft());
    file.writeAll(grid.get(), false);
}

TEST_CASE("cpu::geometry::fft::insert3D and extract3D", "[.]") {
    const dim4_t input_slice_shape{5, 1, 512, 512};
    const dim4_t output_slice_shape{1, 1, 512, 512};
    const dim4_t grid_shape{1, 512, 512, 512};

    const dim4_t input_slice_strides = input_slice_shape.fft().strides();
    const dim4_t output_slice_strides = output_slice_shape.fft().strides();
    const dim4_t grid_strides = grid_shape.fft().strides();

    cpu::memory::PtrHost<float> input_slice(input_slice_shape.fft().elements());
    cpu::memory::PtrHost<float> output_slice(output_slice_shape.fft().elements());
    cpu::memory::PtrHost<float> grid(grid_shape.fft().elements());

    cpu::memory::PtrHost<float22_t> input_scaling_matrices(input_slice_shape[0]);
    cpu::memory::PtrHost<float33_t> input_rotation_matrices(input_slice_shape[0]);

    float count = 0;
    for (uint i = 0; i < input_slice_shape[0]; ++i) {
        input_scaling_matrices[i] = geometry::scale(float2_t{1, 1});
        input_rotation_matrices[i] = geometry::euler2matrix(math::deg2rad(float3_t{count, 0, 0}), "YZX", false);
        count += 2;
    }
    const float22_t output_scaling_matrix{1};
    const auto output_rotation_matrix = geometry::euler2matrix(math::deg2rad(float3_t{count, 0, 0}), "YZX", false);

    const float cutoff = 0.5f;
    const float slice_z_radius = 0.004f;

    const bool do_ews = false;
    const float wavelength = 0.01968761530923358f; // A
    const float2_t pixel_size{1, 1}; // A/pix
    const float2_t ews_radius = do_ews ? pixel_size / wavelength : float2_t{}; // 1/pix

    cpu::Stream stream(cpu::Stream::DEFAULT);
    cpu::memory::linspace(input_slice.share(), input_slice_strides, input_slice_shape, 1.f, 5.f, false, stream);

    {
        cpu::memory::set(output_slice.begin(), output_slice.end(), 1.f);
        cpu::memory::set(grid.begin(), grid.end(), 0.f);

        cpu::geometry::fft::insert3D<fft::HC2HC>(
                input_slice.share(), input_slice_strides, input_slice_shape,
                grid.share(), grid_strides, grid_shape,
                input_scaling_matrices.share(), input_rotation_matrices.share(),
                cutoff, {}, ews_radius, stream);

        cpu::geometry::fft::extract3D<fft::HC2HC>(
                grid.share(), grid_strides, grid_shape,
                output_slice.share(), output_slice_strides, output_slice_shape,
                output_scaling_matrix, output_rotation_matrix,
                cutoff, {}, ews_radius, stream);

        stream.synchronize();
        io::MRCFile file(test::NOA_DATA_PATH / "geometry" / "fft" / "test_insert_extract0.mrc", io::WRITE);
        file.shape(output_slice_shape.fft());
        file.writeAll(output_slice.get(), false);

        file.open(test::NOA_DATA_PATH / "geometry" / "fft" / "test_insert_extract0_grid.mrc", io::WRITE);
        file.shape(grid_shape.fft());
        file.writeAll(grid.get(), false);
    }

    {
        cpu::memory::set(output_slice.begin(), output_slice.end(), 0.f);

        cpu::geometry::fft::extract3D<fft::HC2HC>(
                input_slice.share(), input_slice_strides, input_slice_shape,
                output_slice.share(), output_slice_strides, output_slice_shape,
                input_scaling_matrices.share(), input_rotation_matrices.share(),
                output_scaling_matrix, output_rotation_matrix,
                cutoff, ews_radius, slice_z_radius, stream);

        stream.synchronize();
        io::MRCFile file(test::NOA_DATA_PATH / "geometry" / "fft" / "test_insert_extract1.mrc", io::WRITE);
        file.shape(output_slice_shape.fft());
        file.writeAll(output_slice.get(), false);
    }
}
