#include <noa/common/files/MRCFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Resize.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

// Test against manually checked data.
static constexpr bool COMPUTE_TEST_DATA_INSTEAD = false;

TEST_CASE("cpu::memory::resize()", "[noa][cpu][memory]") {
    uint batches;
    size3_t i_shape;
    size3_t o_shape;
    int3_t border_left;
    int3_t border_right;
    path_t filename;
    BorderMode mode;
    float value;

    int test_number = GENERATE(0, 1, 2, 3, 4, 5,
                               10, 11, 12, 13, 14, 15,
                               20, 21, 22, 23, 24, 25, 26, 27, 28,
                               30, 31);
    test::assets::memory::getResizeParams(test_number, &filename, &batches, &i_shape, &o_shape,
                                          &border_left, &border_right, &mode, &value);
    INFO(test_number);

    size_t i_elements = getElements(i_shape);
    size_t o_elements = getElements(o_shape);
    cpu::memory::PtrHost<float> input(i_elements * batches);
    cpu::memory::PtrHost<float> output(o_elements * batches);
    test::assets::memory::initResizeInput(test_number, input.get(), i_shape, batches);
    if (test_number >= 30)
        test::assets::memory::initResizeOutput(output.get(), o_shape, batches);

    if (test_number <= 15 || test_number >= 30)
        cpu::memory::resize(input.get(), i_shape, border_left, border_right, output.get(), mode, value, batches);
    else
        cpu::memory::resize(input.get(), i_shape, output.get(), o_shape, mode, value, batches);

    if (COMPUTE_TEST_DATA_INSTEAD) {
        MRCFile file(filename, io::WRITE);
        file.setShape(batches > 1 ? size3_t{o_shape.x, o_shape.y, batches} : o_shape);
        file.writeAll(output.get());
        file.close();
        return;
    }

    cpu::memory::PtrHost<float> expected(o_elements * batches);
    MRCFile file(filename, io::READ);
    file.readAll(expected.get());
    float diff = test::getAverageNormalizedDifference(expected.get(), output.get(), o_elements * batches);
    REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
}

TEMPLATE_TEST_CASE("cpu::memory::resize() - edge cases", "[noa][cpu]",
                   int, uint, long long, unsigned long long, float, double) {
    uint ndim = GENERATE(2U, 3U);
    uint batches = test::IntRandomizer<uint>(1, 3).get();

    AND_THEN("copy") {
        size3_t shape = test::getRandomShape(ndim);
        size_t elements = getElements(shape) * batches;
        cpu::memory::PtrHost<TestType> input(elements);
        cpu::memory::PtrHost<TestType> output(elements);
        test::Randomizer<TestType> randomizer(0, 50);
        test::initDataRandom(input.get(), elements, randomizer);
        cpu::memory::resize(input.get(), shape, output.get(), shape, BORDER_VALUE, TestType{0}, batches);
        TestType diff = test::getDifference(input.get(), output.get(), elements);
        REQUIRE_THAT(diff, test::isWithinAbs(0, 1e-6));
    }
}
