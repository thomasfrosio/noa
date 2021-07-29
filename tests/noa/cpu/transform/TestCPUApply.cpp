#include <noa/common/files/MRCFile.h>
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
