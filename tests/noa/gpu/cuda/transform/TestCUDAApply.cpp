#include <noa/common/files/MRCFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/math/Arithmetics.h>
#include <noa/cpu/math/Reductions.h>
#include <noa/cpu/transform/Apply.h>

#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/transform/Apply.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cuda::transform::apply2D()", "[noa][cuda][transform]") {
    int test_number = GENERATE(0, 2, 4, 5, 6, 8, 10, 11);
    INFO(test_number);

    path_t filename_data;
    path_t filename_expected;
    path_t filename_matrix;
    float value;
    InterpMode interp;
    BorderMode border;
    test::assets::transform::getApply2DParams(test_number, &filename_data, &filename_expected,
                                              &interp, &border, &value, &filename_matrix);

    // Get input.
    MRCFile file(filename_data, io::READ);
    size3_t shape = file.getShape();
    size2_t shape_2d(shape.x, shape.y);
    size_t elements = getElements(shape);
    cpu::memory::PtrHost<float> input(elements);
    file.readAll(input.get());

    // Get expected.
    cpu::memory::PtrHost<float> expected(elements);
    file.open(filename_expected, io::READ);
    file.readAll(expected.get());

    std::array<float, 9> tmp{};
    file.open(filename_matrix, io::READ);
    file.readAll(tmp.data());
    float33_t matrix(tmp[0], tmp[1], tmp[2],
                     tmp[3], tmp[4], tmp[5],
                     tmp[6], tmp[7], tmp[8]);
    matrix = math::inverse(matrix);

    cuda::Stream stream;
    cpu::memory::PtrHost<float> output(elements);
    cuda::memory::PtrDevicePadded<float> d_input(shape);
    AND_THEN("3x3 matrix") {
        cuda::memory::copy(input.get(), shape.x, d_input.get(), d_input.pitch(), shape, stream);
        cuda::transform::apply2D(d_input.get(), d_input.pitch(), shape_2d,
                                 d_input.get(), d_input.pitch(), shape_2d,
                                 matrix, interp, border, stream);
        cuda::memory::copy(d_input.get(), d_input.pitch(), output.get(), shape.x, shape, stream);
        stream.synchronize();

        if (interp == INTERP_LINEAR) {
            cpu::math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
            float min, max, mean;
            cpu::math::minMaxSumMean<float>(output.get(), &min, &max, nullptr, &mean, elements, 1);
            REQUIRE(math::abs(min) < 1e-2f); // don't know what to think about these values
            REQUIRE(math::abs(max) < 1e-2f);
            REQUIRE(math::abs(mean) < 1e-5f);
        } else {
            float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
        }
    }

    AND_THEN("2x3 matrix") {
        cuda::memory::copy(input.get(), shape.x, d_input.get(), d_input.pitch(), shape, stream);
        cuda::transform::apply2D(d_input.get(), d_input.pitch(), shape_2d,
                                 d_input.get(), d_input.pitch(), shape_2d,
                                 float23_t(matrix), interp, border, stream);
        cuda::memory::copy(d_input.get(), d_input.pitch(), output.get(), shape.x, shape, stream);
        stream.synchronize();

        if (interp == INTERP_LINEAR) {
            cpu::math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
            float min, max, mean;
            cpu::math::minMaxSumMean<float>(output.get(), &min, &max, nullptr, &mean, elements, 1);
            REQUIRE(math::abs(min) < 1e-2f); // don't know what to think about these values
            REQUIRE(math::abs(max) < 1e-2f);
            REQUIRE(math::abs(mean) < 1e-5f);
        } else {
            float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
        }
    }
}

TEST_CASE("cuda::transform::apply3D()", "[noa][cuda][transform]") {
    int test_number = GENERATE(0, 2, 4, 5, 6, 8, 10, 11);
    INFO(test_number);

    path_t filename_data;
    path_t filename_expected;
    path_t filename_matrix;
    float value;
    InterpMode interp;
    BorderMode border;
    test::assets::transform::getApply3DParams(test_number, &filename_data, &filename_expected,
                                              &interp, &border, &value, &filename_matrix);

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

    std::array<float, 16> tmp{};
    file.open(filename_matrix, io::READ);
    file.readAll(tmp.data());
    float44_t matrix(tmp[0], tmp[1], tmp[2], tmp[3],
                     tmp[4], tmp[5], tmp[6], tmp[7],
                     tmp[8], tmp[9], tmp[10], tmp[11],
                     tmp[12], tmp[13], tmp[14], tmp[15]);
    matrix = math::inverse(matrix);

    cuda::Stream stream;
    cpu::memory::PtrHost<float> output(elements);
    cuda::memory::PtrDevicePadded<float> d_input(shape);
    AND_THEN("4x4 matrix") {
        cuda::memory::copy(input.get(), shape.x, d_input.get(), d_input.pitch(), shape, stream);
        cuda::transform::apply3D(d_input.get(), d_input.pitch(), shape,
                                 d_input.get(), d_input.pitch(), shape,
                                 matrix, interp, border, stream);
        cuda::memory::copy(d_input.get(), d_input.pitch(), output.get(), shape.x, shape, stream);
        stream.synchronize();

        if (interp == INTERP_LINEAR) {
            cpu::math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
            float min, max, mean;
            cpu::math::minMaxSumMean<float>(output.get(), &min, &max, nullptr, &mean, elements, 1);
            REQUIRE(math::abs(min) < 5e-2f); // don't know what to think about these values
            REQUIRE(math::abs(max) < 5e-2f);
            REQUIRE(math::abs(mean) < 1e-5f);
        } else {
            float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
        }
    }

    AND_THEN("3x4 matrix") {
        cuda::memory::copy(input.get(), shape.x, d_input.get(), d_input.pitch(), shape, stream);
        cuda::transform::apply3D(d_input.get(), d_input.pitch(), shape,
                                 d_input.get(), d_input.pitch(), shape,
                                 float34_t(matrix), interp, border, stream);
        cuda::memory::copy(d_input.get(), d_input.pitch(), output.get(), shape.x, shape, stream);
        stream.synchronize();

        if (interp == INTERP_LINEAR) {
            cpu::math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
            float min, max, mean;
            cpu::math::minMaxSumMean<float>(output.get(), &min, &max, nullptr, &mean, elements, 1);
            REQUIRE(math::abs(min) < 5e-2f); // don't know what to think about these values
            REQUIRE(math::abs(max) < 5e-2f);
            REQUIRE(math::abs(mean) < 1e-5f);
        } else {
            float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
        }
    }
}
