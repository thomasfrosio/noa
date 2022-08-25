#include <noa/geometry/fft/Project.h>
#include <noa/math/Ewise.h>
#include <noa/math/Reduce.h>
#include <noa/memory/Factory.h>
#include <noa/memory/Resize.h>
#include <noa/fft/Transform.h>
#include <noa/fft/Remap.h>
#include <noa/signal/fft/Shift.h>
#include <noa/signal/fft/Bandpass.h>

#include <noa/common/geometry/Euler.h>
#include <noa/common/io/ImageFile.h>
#include <noa/common/io/TextFile.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

namespace {
    using namespace ::noa;

    Array<float33_t> loadTltAngles_(const path_t& filename) {
        std::vector<float3_t> tilt_angles;
        io::TextFile<std::ifstream> file(filename, io::READ);
        std::string line;
        while(file.getLine(line))
            tilt_angles.emplace_back(float3_t{0, math::deg2rad(-string::toFloat(line)), 0});
        if (file.bad())
            throw std::runtime_error("failed reading the tlt file");

        // Convert to 3x3 rotation matrices:
        Array<float33_t> rotm(tilt_angles.size());
        geometry::euler2matrix(tilt_angles.data(), rotm.get(), tilt_angles.size());
        return rotm;
    }

    Array<float> loadStack_(const path_t& filename) {
        io::ImageFile file(filename, io::READ);
        Array<float> output(file.shape());
        file.readAll(output.get());
        return output;
    }

    void saveStack_(const path_t& filename, const Array<float>& array) {
        io::ImageFile file(filename, io::WRITE);
        file.shape(array.shape());
        file.dtype(io::FLOAT32);
        Array<float> output = array.contiguous() ? array : array.copy();
        file.writeAll(output.eval().get());
    }

    void testInsert3D_(const path_t& filename, size_t size, const Array<float33_t>& tilt_angles) {
        const size4_t slice_shape{1, 1, size, size};
        const size4_t slices_shape{tilt_angles.shape().elements(), 1, size, size};
        const size4_t recons_shape{1, size, size, size};

        Array weights_fft = memory::ones<float>(slice_shape.fft());
        weights_fft = indexing::broadcast(weights_fft, slices_shape.fft());

        Array recons_weights_fft = memory::zeros<float>(recons_shape.fft());
        geometry::fft::insert3D<fft::H2H>(weights_fft, slices_shape,
                                          recons_weights_fft, recons_shape,
                                          {}, tilt_angles);
        saveStack_(filename, recons_weights_fft);
    }

    Array<float> reconstruct_(Array<float>&& stack, const Array<float33_t>& tilt_angles, size_t thickness) {
        const size4_t original_shape = stack.shape();
        const size2_t optimal_size{fft::nextFastSize(original_shape[2]),
                                   fft::nextFastSize(original_shape[3])};
        const size4_t optimal_shape{original_shape[0], 1, optimal_size[0], optimal_size[1]};
        const size4_t recons_shape{1, thickness * 4, optimal_size[0] * 2, optimal_size[1] * 2};
        const ArrayOption options = stack.options();

        // Center:
        Array<float> means({original_shape[0], 1, 1, 1}, options);
        math::mean(stack, means);
        math::ewise(stack, means, stack, math::minus_t{});

        // Pad to next fast shape:
        Array<cfloat_t> slices_fft(optimal_shape.fft(), options);
        Array slices = fft::alias(slices_fft, optimal_shape);
        memory::resize(stack.release(), slices);

        // Get the FFT of slices:
        fft::r2c(slices, slices_fft);
        signal::fft::highpass<fft::H2H>(slices_fft, slices_fft, slices.shape(), 0.08f, 0.08f);

        // Insert the slices:
        Array recons_fft = memory::zeros<cfloat_t>(recons_shape.fft(), options);
        signal::fft::shift2D<fft::H2H>(slices_fft, slices_fft, optimal_shape, -float2_t{optimal_size} / 2);
        geometry::fft::insert3D<fft::H2H>(slices_fft.release(), optimal_shape,
                                          recons_fft, recons_shape,
                                          {}, tilt_angles);
        signal::fft::shift3D<fft::H2H>(recons_fft, recons_fft, recons_shape, float3_t{recons_shape.get(1)} / 2);

        // Insert the weights:
        Array weights_fft = memory::ones<float>(size4_t{1, 1, optimal_size[0], optimal_size[1]}.fft(), options);
        weights_fft = indexing::broadcast(weights_fft, optimal_shape.fft());
        Array recons_weights_fft = memory::zeros<float>(recons_shape.fft(), options);
        geometry::fft::insert3D<fft::H2H>(weights_fft.release(), optimal_shape,
                                          recons_weights_fft, recons_shape,
                                          {}, tilt_angles);
        // Normalize and gridding correction:
        recons_weights_fft += 1e-3f;
        recons_fft /= recons_weights_fft.release();
        Array recons = fft::alias(recons_fft, recons_shape);
        fft::c2r(recons_fft, recons);
        geometry::fft::griddingCorrection(recons, recons, true);

        // Trim and go back to original shape:
        Array<float> recons_trimmed(size4_t{1, thickness, original_shape[2], original_shape[3]});
        memory::resize(recons.release(), recons_trimmed);
        return recons_trimmed;
    }

    Array<float> extract_(Array<float>&& stack) {
        const size4_t original_shape = stack.shape();
        const size_t optimal_size{fft::nextFastSize(math::max(original_shape[2], original_shape[3]))};
        const size4_t optimal_shape{1, 1, optimal_size, optimal_size};
        const size4_t recons_shape{1, optimal_size, optimal_size, optimal_size};
        const ArrayOption options = stack.options();

        // Insert the 0deg and extract at 3deg
        const Array<float33_t> angles(2);
        angles.get()[0] = geometry::euler2matrix(float3_t{0, math::deg2rad(0.f), 0});
        angles.get()[1] = geometry::euler2matrix(float3_t{0, math::deg2rad(-3.f), 0});
        Array angle_0 = angles.subregion(indexing::ellipsis_t{}, 0).copy();
        Array angle_3 = angles.subregion(indexing::ellipsis_t{}, 1).copy();

        // Center:
        Array<float> means({original_shape[0], 1, 1, 1}, options);
        math::mean(stack, means);
        math::ewise(stack, means, stack, math::minus_t{});

        // Insert the 0 deg:
        Array view = stack.subregion(20);

        Array<cfloat_t> slice_fft(optimal_shape.fft(), options);
        Array slice = fft::alias(slice_fft, optimal_shape);
        memory::resize(view.release(), slice);

        fft::r2c(slice, slice_fft);
//        signal::fft::highpass<fft::H2H>(slice_fft, slice_fft, slice.shape(), 0.08f, 0.08f);

        Array recons_fft = memory::zeros<cfloat_t>(recons_shape.fft(), options);
        signal::fft::shift2D<fft::H2H>(slice_fft, slice_fft, optimal_shape, -float2_t{optimal_size} / 2);
        geometry::fft::insert3D<fft::H2HC>(slice_fft, optimal_shape,
                                           recons_fft, recons_shape,
                                           {}, angle_0);

        // Extract the 3 deg:
        memory::fill(slice_fft, cfloat_t{0});
        geometry::fft::extract3D<fft::HC2HC>(recons_fft.release(), recons_shape,
                                             slice_fft, optimal_shape,
                                             {}, angle_3);


        signal::fft::shift2D<fft::HC2HC>(slice_fft, slice_fft, optimal_shape, float2_t{optimal_size} / 2);

        Array<float> tmp(slice_fft.shape(), options);
        math::ewise(slice_fft, tmp, [](cfloat_t x) { return math::log(math::abs(x) + 1); });
        saveStack_(test::NOA_DATA_PATH / "tilt1" / "tilt1_3deg_reprojected_fft.rec", tmp);

        fft::remap(fft::HC2H, slice_fft.copy(), slice_fft, slice.shape());
        fft::c2r(slice_fft, slice);

        // Trim and go back to original shape:
        Array<float> slice_projected(size4_t{1, 1, original_shape[2], original_shape[3]});
        memory::resize(slice, slice_projected);
        return slice_projected;
    }
}

TEST_CASE("geometry::fft, back-projection", "[.]") {
    const path_t directory = test::NOA_DATA_PATH / "tilt1";
    Array<float33_t> tilt_angles = loadTltAngles_(directory / "tilt1.tlt");
    Array<float> stack = loadStack_(directory / "tilt1_cropped_ali.mrc");

    // ignore first image
    stack = stack.subregion(indexing::slice_t{1, -1});
    tilt_angles = tilt_angles.subregion(indexing::ellipsis_t{}, indexing::slice_t{1, -1}).copy();

//    Array<float> tomogram = reconstruct_(std::move(stack), tilt_angles, 300);
//    saveStack_(directory / "tilt1_cropped_ali_noa3.rec", tomogram);

    Array<float> reprojected_view = extract_(std::move(stack));
    saveStack_(directory / "tilt1_3deg_reprojected.rec", reprojected_view);

    // Insert the weights:
    const size4_t shape{1, 1, 512, 512};
    const size4_t shape_batched{tilt_angles.shape().elements(), 1, 512, 512};
    const size4_t shape_recons{1, 512, 512, 512};
    Array weights_fft = memory::ones<float>(shape.fft());
    weights_fft = indexing::broadcast(weights_fft, shape_batched.fft());
    Array recons_weights_fft = memory::zeros<float>(shape_recons.fft());
    geometry::fft::insert3D<fft::H2HC>(weights_fft.release(), shape_batched,
                                       recons_weights_fft, shape_recons,
                                       {}, tilt_angles);

//    fft::remap(fft::H2HC, recons_weights_fft, recons_weights_fft, shape);
    saveStack_(test::NOA_DATA_PATH / "tilt1" / "tilt1_reprojected_weights.mrc", recons_weights_fft);
}
