#include <noa/cpu/memory/Resize.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/io/files/MRCFile.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

// Test against manually checked data.
static constexpr bool COMPUTE_TEST_DATA_INSTEAD = false;

TEST_CASE("Memory::resize()", "[noa][cpu]") {
    uint batches;
    size3_t i_shape;
    size3_t o_shape;
    int3_t border_left;
    int3_t border_right;
    path_t filename;
    BorderMode mode;
    float value;

    int test_number = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
    Test::Assets::Memory::getResizeParams(test_number, &filename, &batches, &i_shape, &o_shape,
                                          &border_left, &border_right, &mode, &value);
    INFO(test_number);

    size_t i_elements = getElements(i_shape);
    size_t o_elements = getElements(o_shape);
    Memory::PtrHost<float> input(i_elements * batches);
    Memory::PtrHost<float> output(o_elements * batches);
    Test::Assets::Memory::initResizeInput(test_number, input.get(), i_shape, batches);
    if (test_number >= 19)
        Test::Assets::Memory::initResizeOutput(output.get(), o_shape, batches);

    if (test_number < 11 || test_number >= 19)
        Memory::resize(input.get(), i_shape, border_left, border_right, output.get(), mode, value, batches);
    else
        Memory::resize(input.get(), i_shape, output.get(), o_shape, mode, value, batches);

    if (COMPUTE_TEST_DATA_INSTEAD) {
        MRCFile file(filename, IO::WRITE);
        file.setShape(batches > 1 ? size3_t{o_shape.x, o_shape.y, batches} : o_shape);
        file.writeAll(output.get());
        file.close();
        return;
    }

    Memory::PtrHost<float> expected(o_elements * batches);
    MRCFile file(filename, IO::READ);
    file.readAll(expected.get());
    float diff = Test::getAverageNormalizedDifference(expected.get(), output.get(), o_elements * batches);
    REQUIRE_THAT(diff, Test::isWithinAbs(0.f, 1e-6));
}

TEMPLATE_TEST_CASE("Memory::resize() - edge cases", "[noa][cpu]",
                   int, uint, long long, unsigned long long, float, double) {
    uint ndim = GENERATE(2U, 3U);
    uint batches = Test::IntRandomizer<uint>(1, 3).get();

    AND_THEN("copy") {
        size3_t shape = Test::getRandomShape(ndim);
        size_t elements = getElements(shape) * batches;
        Memory::PtrHost<TestType> input(elements);
        Memory::PtrHost<TestType> output(elements);
        Test::Randomizer<TestType> randomizer(0, 50);
        Test::initDataRandom(input.get(), elements, randomizer);
        Memory::resize(input.get(), shape, output.get(), shape, BORDER_VALUE, TestType{0}, batches);
        TestType diff = Test::getDifference(input.get(), output.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(0, 1e-6));
    }
}
