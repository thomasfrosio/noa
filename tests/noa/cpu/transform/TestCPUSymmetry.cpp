#include <noa/common/transform/Geometry.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/math/Arithmetics.h>
#include <noa/cpu/math/Reductions.h>
#include <noa/cpu/transform/Apply.h>
#include <noa/cpu/transform/Symmetry.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::transform::symmetrize2D()", "[noa][cpu][transform]") {
    test::Randomizer<float> randomizer(-100, 100);
    cpu::Stream stream;

    // Get input.
    size3_t shape = test::getRandomShape(2);
    size2_t shape_2d = {shape.x, shape.y};
    float2_t center(shape.x / 2, shape.y / 2);
    size_t elements = noa::elements(shape);
    cpu::memory::PtrHost<float> input(elements);
    test::randomize(input.get(), elements, randomizer);

    // Get expected (rely on apply2D) and output.
    cpu::memory::PtrHost<float> output(elements);
    cpu::memory::PtrHost<float> expected(elements);

    const char* symbol = GENERATE("c1", "C2", "C7", "D1", "D3");
    transform::Symmetry symmetry(symbol);

    cpu::transform::apply2D(input.get(), shape.x, expected.get(), shape.x, shape_2d,
                            {}, {}, symmetry, center, INTERP_LINEAR, true, stream);

    cpu::transform::symmetrize2D(input.get(), shape_2d, output.get(), shape_2d, shape_2d, 1,
                                 symmetry, center, INTERP_LINEAR, true, stream);

    cpu::math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
    float min, max, mean;
    cpu::math::minMaxSumMean<float>(output.get(), &min, &max, nullptr, &mean, elements, 1);
    REQUIRE(math::abs(min) < 5e-4f);
    REQUIRE(math::abs(max) < 5e-4f);
    REQUIRE(math::abs(mean) < 1e-6f);
}

TEST_CASE("cpu::transform::symmetrize3D()", "[noa][cpu][transform]") {
    test::Randomizer<float> randomizer(-100, 100);
    cpu::Stream stream;

    // Get input.
    size3_t shape = test::getRandomShape(3);
    size2_t shape_2d = {shape.x, shape.y};
    float3_t center(shape / size_t{2});
    size_t elements = noa::elements(shape);
    cpu::memory::PtrHost<float> input(elements);
    test::randomize(input.get(), elements, randomizer);

    // Get expected (rely on apply3D) and output.
    cpu::memory::PtrHost<float> output(elements);
    cpu::memory::PtrHost<float> expected(elements);

    const char* symbol = GENERATE("c1", "C2", "C7", "D1", "D3", "o", "i1", "I2");
    transform::Symmetry symmetry(symbol);

    cpu::transform::apply3D(input.get(), shape_2d, expected.get(), shape_2d, shape,
                            {}, {}, symmetry, center, INTERP_LINEAR, true, stream);

    cpu::transform::symmetrize3D(input.get(), shape, output.get(), shape, shape, 1,
                                 symmetry, center, INTERP_LINEAR, true, stream);

    cpu::math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
    float min, max, mean;
    cpu::math::minMaxSumMean<float>(output.get(), &min, &max, nullptr, &mean, elements, 1);
    REQUIRE(math::abs(min) < 5e-4f);
    REQUIRE(math::abs(max) < 5e-4f);
    REQUIRE(math::abs(mean) < 1e-6f);
}
