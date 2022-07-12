#include <noa/unified/geometry/fft/Project.h>
#include <noa/unified/math/Ewise.h>
#include <noa/unified/math/Reduce.h>
#include <noa/unified/memory/Factory.h>
#include <noa/unified/memory/Resize.h>
#include <noa/unified/fft/Transform.h>
#include <noa/unified/signal/fft/Shift.h>

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
            tilt_angles.emplace_back(float3_t{0, math::toRad(-string::toFloat(line)), 0});
        if (file.bad())
            throw std::runtime_error("failed reading the tlt file");

        // Convert to 3x3 rotation matrices:
        Array<float33_t> rotm(tilt_angles.size());
        geometry::euler2matrix(tilt_angles.data(), rotm.get(), tilt_angles.size());
        return rotm;
    }

    Array<float> loadStack_(const path_t& filename) {
        io::ImageFile file(filename, io::READ);
        const size4_t shape = file.shape();
        Array<float> output(shape);
        file.readAll(output.get());
        return output;
    }

    void saveStack_(const path_t& filename, const Array<float>& array) {
        io::ImageFile file(filename, io::WRITE);
        file.shape(array.shape());
        file.dtype(io::FLOAT32);
        Array<float> output = all(array.contiguous()) ? array : array.copy();
        file.writeAll(output.eval().get());
    }

    void testInsert3D_(const path_t& filename, size_t size, const Array<float33_t>& tilt_angles) {
        const size4_t slice_shape{1, 1, size, size};
        const size4_t slices_shape{tilt_angles.shape().elements(), 1, size, size};
        const size4_t recons_shape{1, size, size, size};

        Array<float> weights_fft = memory::ones<float>(slice_shape.fft());
        weights_fft = indexing::broadcast(weights_fft, slices_shape.fft());

        Array<float> recons_weights_fft = memory::zeros<float>(recons_shape.fft());
        geometry::fft::insert3D<fft::H2H>(weights_fft, slices_shape,
                                          recons_weights_fft, recons_shape,
                                          {}, tilt_angles);
        saveStack_(filename, recons_weights_fft);
    }

    Array<float> reconstruct_(Array<float>&& stack, const Array<float33_t>& tilt_angles, size_t thickness) {
        const size4_t original_shape = stack.shape();
        const size_t optimal_size = fft::nextFastSize(math::max(original_shape[2], original_shape[3]));
        const size4_t optimal_shape{original_shape[0], 1, optimal_size, optimal_size};
        const size4_t recons_shape{1, optimal_size, optimal_size, optimal_size};
        const ArrayOption options = stack.options();

        // Center:
        Array<float> means({original_shape[0], 1, 1, 1}, options);
        math::mean(stack, means);
        math::ewise(stack, means, stack, math::minus_t{});

        // Pad to next fast shape:
        Array<cfloat_t> slices_fft(optimal_shape.fft());
        Array slices = fft::alias(slices_fft, optimal_shape);
        memory::resize(stack.release(), slices);

        // Insert the slices:
        fft::r2c(slices, slices_fft);
        Array recons_fft = memory::zeros<cfloat_t>(recons_shape.fft());
        signal::fft::shift2D<fft::H2H>(slices_fft, slices_fft, optimal_shape, -float2_t{optimal_size / 2});
        geometry::fft::insert3D<fft::H2H>(slices_fft.release(), optimal_shape,
                                          recons_fft, recons_shape,
                                          {}, tilt_angles);
        signal::fft::shift3D<fft::H2H>(recons_fft, recons_fft, recons_shape, float3_t{optimal_size / 2});

        // Insert the weights:
        Array weights_fft = memory::ones<float>(size4_t{1, 1, optimal_size, optimal_size}.fft());
        weights_fft = indexing::broadcast(weights_fft, optimal_shape.fft());
        Array recons_weights_fft = memory::zeros<float>(recons_shape.fft());
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

//    Array<float> extract_(Array<cfloat_t>&& fft_data, Array<cfloat_t> fft_weight, size4_t logial_shape, float angle) {
//        Array<cfloat_t> tmp()
//        geometry::fft::extract3D<fft::H2H>(fft_data, );
//    }
}

TEST_CASE("geometry::fft, back-projection", "[.]") {
    const path_t directory = test::NOA_DATA_PATH / "geometry" / "fft" / "tilt1";
    const Array<float33_t> tilt_angles = loadTltAngles_(directory / "tilt1.tlt");
//    testInsert3D_(directory / "test_recons_tilt1_insert_weights.mrc", 1024, tilt_angles);

    Array<float> stack = loadStack_(directory / "tilt1_ali.mrc");

    // Divide the stack in 8

}
