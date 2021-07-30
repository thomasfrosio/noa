#include <noa/common/files/MRCFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/math/Arithmetics.h>
#include <noa/cpu/math/Reductions.h>

#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/transform/Translate.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cuda::transform::translate2D()", "[noa][cuda][transform]") {
    int test_number = GENERATE(0, 2, 4, 5, 6, 8, 10, 11);
    INFO(test_number);

    path_t filename_data;
    path_t filename_expected;
    float value;
    float2_t shift;
    InterpMode interp;
    BorderMode border;
    test::assets::transform::getTranslate2DParams(test_number, &filename_data, &filename_expected,
                                                  &interp, &border, &value, &shift);

    // Get input.
    MRCFile file(filename_data, io::READ);
    size3_t shape = file.getShape();
    size2_t shape_2d(shape.x, shape.y);
    size_t elements = getElements(shape);
    memory::PtrHost<float> input(elements);
    file.readAll(input.get());

    // Get expected.
    memory::PtrHost<float> expected(elements);
    file.open(filename_expected, io::READ);
    file.readAll(expected.get());

    cuda::Stream stream;
    memory::PtrHost<float> output(elements);
    cuda::memory::PtrDevicePadded<float> d_input(shape);
    cuda::memory::copy(input.get(), shape.x, d_input.get(), d_input.pitch(), shape, stream);
    cuda::transform::translate2D(d_input.get(), d_input.pitch(), shape_2d,
                                 d_input.get(), d_input.pitch(), shape_2d,
                                 shift, interp, border, stream);
    cuda::memory::copy(d_input.get(), d_input.pitch(), output.get(), shape.x, shape, stream);
    stream.synchronize();

    if (interp == INTERP_LINEAR) {
        math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
        float min, max, mean;
        math::minMaxSumMean<float>(output.get(), &min, &max, nullptr, &mean, elements, 1);
        REQUIRE(math::abs(min) < 1e-2f); // don't know what to think about these values
        REQUIRE(math::abs(max) < 1e-2f);
        REQUIRE(math::abs(mean) < 1e-4f);
    } else {
        float diff = test::getDifference(expected.get(), output.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
    }
}

TEST_CASE("cuda::transform::translate3D()", "[noa][cuda][transform]") {
    int test_number = GENERATE(0, 2, 4, 5, 6, 8, 10, 11);
    INFO(test_number);

    path_t filename_data;
    path_t filename_expected;
    float value;
    float3_t shift;
    InterpMode interp;
    BorderMode border;
    test::assets::transform::getTranslate3DParams(test_number, &filename_data, &filename_expected,
                                                  &interp, &border, &value, &shift);

    // Get input.
    MRCFile file(filename_data, io::READ);
    size3_t shape = file.getShape();
    size_t elements = getElements(shape);
    memory::PtrHost<float> input(elements);
    file.readAll(input.get());

    // Get expected.
    memory::PtrHost<float> expected(elements);
    file.open(filename_expected, io::READ);
    file.readAll(expected.get());

    cuda::Stream stream;
    memory::PtrHost<float> output(elements);
    cuda::memory::PtrDevicePadded<float> d_input(shape);
    cuda::memory::copy(input.get(), shape.x, d_input.get(), d_input.pitch(), shape, stream);
    cuda::transform::translate3D(d_input.get(), d_input.pitch(), shape,
                                 d_input.get(), d_input.pitch(), shape,
                                 shift, interp, border, stream);
    cuda::memory::copy(d_input.get(), d_input.pitch(), output.get(), shape.x, shape, stream);
    stream.synchronize();

    if (interp == INTERP_LINEAR) {
        math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
        float min, max, mean;
        math::minMaxSumMean<float>(output.get(), &min, &max, nullptr, &mean, elements, 1);
        REQUIRE(math::abs(min) < 1e-2f); // don't know what to think about these values
        REQUIRE(math::abs(max) < 1e-2f);
        REQUIRE(math::abs(mean) < 1e-4f);
    } else {
        float diff = test::getDifference(expected.get(), output.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
    }
}
