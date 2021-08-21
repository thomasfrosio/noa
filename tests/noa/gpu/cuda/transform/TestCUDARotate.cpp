#include <noa/common/files/MRCFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Set.h>
#include <noa/cpu/math/Arithmetics.h>
#include <noa/cpu/math/Reductions.h>
#include <noa/cpu/transform/Rotate.h>

#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/transform/Rotate.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cuda::transform::rotate2D()", "[noa][cuda][transform]") {
    int test_number = GENERATE(0, 2, 4, 5, 6, 8, 10, 11);
    INFO(test_number);

    path_t filename_data;
    path_t filename_expected;
    float value;
    float rotation;
    float2_t rotation_center;
    InterpMode interp;
    BorderMode border;
    test::assets::transform::getRotate2DParams(test_number, &filename_data, &filename_expected,
                                               &interp, &border, &value, &rotation, &rotation_center);

    // Get input.
    MRCFile file(filename_data, io::READ);
    size3_t shape = file.getShape();
    size_t elements = getElements(shape);
    cpu::memory::PtrHost<float> input(elements);
    file.readAll(input.get());

    // Get expected.
    cpu::memory::PtrHost<float> expected(elements);
    file.open(filename_expected, io::READ);
    file.readAll(expected.get());

    cuda::Stream stream;
    cpu::memory::PtrHost<float> output(elements);
    cuda::memory::PtrDevicePadded<float> d_input(shape);
    cuda::memory::copy(input.get(), shape.x, d_input.get(), d_input.pitch(), shape, stream);
    cuda::transform::rotate2D(d_input.get(), d_input.pitch(), d_input.get(), d_input.pitch(),
                              size2_t(shape.x, shape.y), rotation, rotation_center, interp, border, stream);
    cuda::memory::copy(d_input.get(), d_input.pitch(), output.get(), shape.x, shape, stream);
    stream.synchronize();

    if (interp == INTERP_LINEAR) {
        cpu::math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
        float min, max, mean;
        cpu::math::minMaxSumMean<float>(output.get(), &min, &max, nullptr, &mean, elements, 1);
        REQUIRE(math::abs(min) < 1e-2f); // don't know what to think about these values
        REQUIRE(math::abs(max) < 1e-2f);
        REQUIRE(math::abs(mean) < 1e-4f);
    } else {
        float diff = test::getDifference(expected.get(), output.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
    }
}

TEMPLATE_TEST_CASE("cuda::transform::rotate2D()", "[noa][cuda][transform]", float, cfloat_t) {
    // INTERP_NEAREST isn't there because it's very possible to have different results there.
    // Textures have only 1 bytes of precision on the fraction used for the interpolation, so it is possible
    // to select the "wrong" element simply because of this loss of precision...
    InterpMode interp = GENERATE(INTERP_LINEAR, INTERP_COSINE, INTERP_CUBIC_BSPLINE);
    BorderMode border = GENERATE(BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC, BORDER_MIRROR);
    if (interp != INTERP_LINEAR && (border == BORDER_PERIODIC || border == BORDER_MIRROR))
        return; // not supported
    INFO(interp);
    INFO(border);

    TestType value = test::RealRandomizer<TestType>(-3., 3.).get();
    float rotation = math::toRad(test::RealRandomizer<float>(-360., 360.).get());
    size3_t shape = test::getRandomShape(2U);
    size2_t shape_2d(shape.x, shape.y);
    size_t elements = getElements(shape);
    float2_t rotation_center(shape.x, shape.y);
    rotation_center /= test::RealRandomizer<float>(1, 4).get();

    // Get input.
    cpu::memory::PtrHost<TestType> input(elements);
    test::Randomizer<TestType> randomizer(-2., 2.);
    test::initDataRandom(input.get(), elements, randomizer);

    cuda::Stream stream;
    cuda::memory::PtrDevicePadded<TestType> d_input(shape);
    cuda::memory::copy(input.get(), shape.x, d_input.get(), d_input.pitch(), shape, stream);
    cuda::transform::rotate2D(d_input.get(), d_input.pitch(), d_input.get(), d_input.pitch(),
                              shape_2d, rotation, rotation_center, interp, border, stream);

    cpu::memory::PtrHost<TestType> output(elements);
    cpu::memory::PtrHost<TestType> output_cuda(elements);
    cuda::memory::copy(d_input.get(), d_input.pitch(), output_cuda.get(), shape.x, shape, stream);
    cpu::transform::rotate2D(input.get(), output.get(),
                             shape_2d, rotation, rotation_center, interp, border, value);
    stream.synchronize();

    cpu::math::subtractArray(output.get(), output_cuda.get(), input.get(), elements, 1);
    float min, max;
    if constexpr (noa::traits::is_complex_v<TestType>)
        cpu::math::minMax(reinterpret_cast<float*>(input.get()), &min, &max, elements * 2, 1);
    else
        cpu::math::minMax(input.get(), &min, &max, elements, 1);
    REQUIRE_THAT(min, test::isWithinAbs(TestType(0), 1e-1)); // usually around 0.01 to 0.05 for bspline
    REQUIRE_THAT(max, test::isWithinAbs(TestType(0), 1e-1));

    TestType diff = test::getAverageDifference(output.get(), output_cuda.get(), elements);
    REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 5e-2));
}

TEST_CASE("cuda::transform::rotate3D()", "[noa][cuda][transform]") {
    int test_number = GENERATE(0, 2, 4, 5, 6, 8, 10, 11);
    INFO(test_number);

    path_t filename_data;
    path_t filename_expected;
    float value;
    float3_t eulers;
    float3_t rotation_center;
    InterpMode interp;
    BorderMode border;
    test::assets::transform::getRotate3DParams(test_number, &filename_data, &filename_expected,
                                               &interp, &border, &value, &eulers, &rotation_center);

    // Get input.
    MRCFile file(filename_data, io::READ);
    size3_t shape = file.getShape();
    size_t elements = getElements(shape);
    cpu::memory::PtrHost<float> input(elements);
    file.readAll(input.get());

    // Get expected.
    cpu::memory::PtrHost<float> expected(elements);
    file.open(filename_expected, io::READ);
    file.readAll(expected.get());

    cuda::Stream stream;
    cpu::memory::PtrHost<float> output(elements);
    cuda::memory::PtrDevicePadded<float> d_input(shape);
    cuda::memory::copy(input.get(), shape.x, d_input.get(), d_input.pitch(), shape, stream);
    cuda::transform::rotate3D(d_input.get(), d_input.pitch(), d_input.get(), d_input.pitch(),
                              shape, eulers, rotation_center, interp, border, stream);
    cuda::memory::copy(d_input.get(), d_input.pitch(), output.get(), shape.x, shape, stream);
    stream.synchronize();

    if (interp == INTERP_LINEAR) {
        cpu::math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
        float min, max, mean;
        cpu::math::minMaxSumMean<float>(output.get(), &min, &max, nullptr, &mean, elements, 1);
        REQUIRE(math::abs(min) < 1e-2f); // don't know what to think about these values
        REQUIRE(math::abs(max) < 1e-2f);
        REQUIRE(math::abs(mean) < 1e-4f);
    } else {
        float diff = test::getDifference(expected.get(), output.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
    }
}

TEMPLATE_TEST_CASE("cuda::transform::rotate3D()", "[noa][cuda][transform]", float, cfloat_t) {
    InterpMode interp = GENERATE(INTERP_LINEAR, INTERP_COSINE, INTERP_CUBIC, INTERP_CUBIC_BSPLINE,
                                 INTERP_LINEAR_FAST, INTERP_COSINE_FAST, INTERP_CUBIC_BSPLINE_FAST);
    BorderMode border = GENERATE(BORDER_ZERO, BORDER_CLAMP, BORDER_PERIODIC, BORDER_MIRROR);
    if (interp != INTERP_LINEAR_FAST && (border == BORDER_PERIODIC || border == BORDER_MIRROR))
        return; // not supported
    INFO(interp);
    INFO(border);

    TestType value = test::RealRandomizer<TestType>(-3., 3.).get();
    test::RealRandomizer<float> angle_randomizer(-360., 360.);
    float3_t eulers(math::toRad(angle_randomizer.get()),
                    math::toRad(angle_randomizer.get()),
                    math::toRad(angle_randomizer.get()));
    size3_t shape = test::getRandomShape(3U);
    size_t elements = getElements(shape);
    float3_t rotation_center(shape);
    rotation_center /= test::RealRandomizer<float>(1, 4).get();

    // Get input.
    cpu::memory::PtrHost<TestType> input(elements);
    test::Randomizer<TestType> randomizer(-2., 2.);
    test::initDataRandom(input.get(), elements, randomizer);

    cuda::Stream stream;
    cuda::memory::PtrDevicePadded<TestType> d_input(shape);
    cuda::memory::PtrDevicePadded<TestType> d_output(shape);
    cuda::memory::copy(input.get(), shape.x, d_input.get(), d_input.pitch(), shape, stream);
    cuda::transform::rotate3D<true>(d_input.get(), d_input.pitch(), d_output.get(), d_output.pitch(),
                                    shape, eulers, rotation_center, interp, border, stream);

    cpu::memory::PtrHost<TestType> output(elements);
    cpu::memory::PtrHost<TestType> output_cuda(elements);
    cuda::memory::copy(d_output.get(), d_output.pitch(), output_cuda.get(), shape.x, shape, stream);
    cpu::transform::rotate3D(input.get(), output.get(),
                             shape, eulers, rotation_center, interp, border, value);
    stream.synchronize();

    cpu::math::subtractArray(output.get(), output_cuda.get(), input.get(), elements, 1);
    float min, max;
    if constexpr (noa::traits::is_complex_v<TestType>)
        cpu::math::minMax(reinterpret_cast<float*>(input.get()), &min, &max, elements * 2, 1);
    else
        cpu::math::minMax(input.get(), &min, &max, elements, 1);
    float err;
    if (interp == INTERP_CUBIC_BSPLINE)
        err = 0.2f; // usually it's lower than that but in some rare cases it goes close to ~0.15
    else
        err = 0.1f;
    REQUIRE_THAT(min, test::isWithinAbs(TestType(0), err));
    REQUIRE_THAT(max, test::isWithinAbs(TestType(0), err));

    TestType diff = test::getAverageDifference(output.get(), output_cuda.get(), elements);
    REQUIRE_THAT(diff, test::isWithinAbs(TestType(0), 0.02));
}
