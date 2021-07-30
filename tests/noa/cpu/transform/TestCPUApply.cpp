#include <noa/common/files/MRCFile.h>
#include <noa/common/transform/Euler.h>
#include <noa/common/transform/Geometry.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/math/Arithmetics.h>
#include <noa/cpu/math/Reductions.h>
#include <noa/cpu/transform/Apply.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::transform::apply2D()", "[noa][cpu][transform]") {
    int test_number = GENERATE(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
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
    size_t elements = getElements(shape);
    memory::PtrHost<float> input(elements);
    file.readAll(input.get());

    // Get expected.
    memory::PtrHost<float> expected(elements);
    file.open(filename_expected, io::READ);
    file.readAll(expected.get());

    std::array<float, 9> tmp;
    file.open(filename_matrix, io::READ);
    file.readAll(tmp.data());
    float33_t matrix(tmp[0], tmp[1], tmp[2],
                     tmp[3], tmp[4], tmp[5],
                     tmp[6], tmp[7], tmp[8]);
    matrix = math::inverse(matrix);

    memory::PtrHost<float> output(elements);
    AND_THEN("3x3 matrix") {
        transform::apply2D(input.get(), size2_t(shape.x, shape.y), output.get(), size2_t(shape.x, shape.y),
                           matrix, interp, border, value);

        if (interp == INTERP_LINEAR) {
            math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
            float min, max, mean;
            math::minMaxSumMean<float>(output.get(), &min, &max, nullptr, &mean, elements, 1);
            REQUIRE(math::abs(min) < 1e-3f); // it seems that 1e-4f is fine as well
            REQUIRE(math::abs(max) < 1e-3f);
            REQUIRE(math::abs(mean) < 1e-6f);
        } else {
            float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
        }
    }

    AND_THEN("2x3 matrix") {
        transform::apply2D(input.get(), size2_t(shape.x, shape.y), output.get(), size2_t(shape.x, shape.y),
                           float23_t(matrix), interp, border, value);

        if (interp == INTERP_LINEAR) {
            math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
            float min, max, mean;
            math::minMaxSumMean<float>(output.get(), &min, &max, nullptr, &mean, elements, 1);
            REQUIRE(math::abs(min) < 1e-3f); // it seems that 1e-4f is fine as well
            REQUIRE(math::abs(max) < 1e-3f);
            REQUIRE(math::abs(mean) < 1e-6f);
        } else {
            float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
        }
    }
}

TEST_CASE("cpu::transform::apply2D(), cubic") {
    constexpr bool GENERATE_TEST_DATA = false;
    int test_number = GENERATE(0, 1, 2, 3, 4, 5);
    path_t filename_data;
    path_t filename_expected;
    path_t filename_matrix;
    InterpMode interp;
    BorderMode border;
    test::assets::transform::getCubic2D(test_number, &filename_data, &filename_expected,
                                        &filename_matrix, &interp, &border);

    // Get input:
    MRCFile file(filename_data, io::READ);
    size3_t shape = file.getShape();
    size2_t shape_2d(shape.x, shape.y);
    size_t elements = getElements(shape);
    memory::PtrHost<float> input(elements);
    file.readAll(input.get());

    if constexpr (GENERATE_TEST_DATA) {
        float rotation = -45.f;
        float2_t scale(1.2, 0.9);
        float2_t rotation_center((shape_2d - size_t(1)) / size_t(2));
        float33_t affine(transform::translate(rotation_center) *
                         float33_t(transform::rotate(math::toRad(rotation))) *
                         float33_t(transform::scale(scale)) *
                         transform::translate(-rotation_center));
        std::array<float, 9> tmp = toArray(affine);
        ImageFile::save(filename_matrix, tmp.data(), size3_t(3, 3, 1));

        memory::PtrHost<float> output(elements);
        transform::apply2D(input.get(), shape_2d, output.get(), shape_2d,
                           math::inverse(affine), interp, border, 1.f);
        ImageFile::save(filename_expected, output.get(), shape);
    } else {
        std::array<float, 9> tmp;
        file.open(filename_matrix, io::READ);
        file.readAll(tmp.data());
        float33_t affine(tmp[0], tmp[1], tmp[2],
                         tmp[3], tmp[4], tmp[5],
                         tmp[6], tmp[7], tmp[8]);
        affine = math::inverse(affine);

        memory::PtrHost<float> output(elements);
        transform::apply2D(input.get(), shape_2d, output.get(), shape_2d,
                           affine, interp, border, 1.f);

        file.open(filename_expected, io::READ);
        file.readAll(input.get());
        float diff = test::getAverageDifference(input.get(), output.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6f));
    }
}

TEST_CASE("cpu::transform::apply3D()", "[noa][cpu][transform]") {
    int test_number = GENERATE(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
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
    memory::PtrHost<float> input(elements);
    file.readAll(input.get());

    // Get expected.
    memory::PtrHost<float> expected(elements);
    file.open(filename_expected, io::READ);
    file.readAll(expected.get());

    std::array<float, 16> tmp;
    file.open(filename_matrix, io::READ);
    file.readAll(tmp.data());
    float44_t matrix(tmp[0], tmp[1], tmp[2], tmp[3],
                     tmp[4], tmp[5], tmp[6], tmp[7],
                     tmp[8], tmp[9], tmp[10], tmp[11],
                     tmp[12], tmp[13], tmp[14], tmp[15]);
    matrix = math::inverse(matrix);

    memory::PtrHost<float> output(elements);
    AND_THEN("4x4 matrix") {
        transform::apply3D(input.get(), shape, output.get(), shape,
                           matrix, interp, border, value);

        if (interp == INTERP_LINEAR) {
            math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
            float min, max, mean;
            math::minMaxSumMean<float>(output.get(), &min, &max, nullptr, &mean, elements, 1);
            REQUIRE(math::abs(min) < 1e-4f);
            REQUIRE(math::abs(max) < 1e-4f);
            REQUIRE(math::abs(mean) < 1e-6f);
        } else {
            float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
        }
    }

    AND_THEN("3x4 matrix") {
        transform::apply3D(input.get(), shape, output.get(), shape,
                           float34_t(matrix), interp, border, value);

        if (interp == INTERP_LINEAR) {
            math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
            float min, max, mean;
            math::minMaxSumMean<float>(output.get(), &min, &max, nullptr, &mean, elements, 1);
            REQUIRE(math::abs(min) < 1e-4f);
            REQUIRE(math::abs(max) < 1e-4f);
            REQUIRE(math::abs(mean) < 1e-6f);
        } else {
            float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
        }
    }
}

TEST_CASE("cpu::transform::apply3D(), cubic") {
    constexpr bool GENERATE_TEST_DATA = false;
    int test_number = GENERATE(0, 1, 2, 3, 4, 5);
    path_t filename_data;
    path_t filename_expected;
    path_t filename_matrix;
    InterpMode interp;
    BorderMode border;
    test::assets::transform::getCubic3D(test_number, &filename_data, &filename_expected,
                                        &filename_matrix, &interp, &border);

    // Get input:
    MRCFile file(filename_data, io::READ);
    size3_t shape = file.getShape();
    size_t elements = getElements(shape);
    memory::PtrHost<float> input(elements);
    file.readAll(input.get());

    if constexpr (GENERATE_TEST_DATA) {
        float3_t eulers(0, 30, 0);
        float3_t scale(0.8, 0.8, 0.8);
        float3_t rotation_center((shape - size_t(1)) / size_t(2));
        float44_t affine(transform::translate(rotation_center) *
                         float44_t(transform::toMatrix(math::toRad(eulers))) *
                         float44_t(transform::scale(scale)) *
                         transform::translate(-rotation_center));
        std::array<float, 16> tmp = toArray(affine);
        ImageFile::save(filename_matrix, tmp.data(), size3_t(4, 4, 1));

        memory::PtrHost<float> output(elements);
        transform::apply3D(input.get(), shape, output.get(), shape,
                           math::inverse(affine), interp, border, 1.f);
        ImageFile::save(filename_expected, output.get(), shape);
    } else {
        std::array<float, 16> tmp;
        file.open(filename_matrix, io::READ);
        file.readAll(tmp.data());
        float44_t affine(tmp[0], tmp[1], tmp[2], tmp[3],
                         tmp[4], tmp[5], tmp[6], tmp[7],
                         tmp[8], tmp[9], tmp[10], tmp[11],
                         tmp[12], tmp[13], tmp[14], tmp[15]);
        affine = math::inverse(affine);

        memory::PtrHost<float> output(elements);
        transform::apply3D(input.get(), shape, output.get(), shape,
                           affine, interp, border, 1.f);

        file.open(filename_expected, io::READ);
        file.readAll(input.get());
        float diff = test::getAverageDifference(input.get(), output.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6f));
    }
}
