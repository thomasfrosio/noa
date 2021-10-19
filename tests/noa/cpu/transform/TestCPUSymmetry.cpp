#include <noa/common/transform/Geometry.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/math/Arithmetics.h>
#include <noa/cpu/math/Reductions.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/transform/Apply.h>
#include <noa/cpu/transform/Symmetry.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::transform::symmetrize2D()", "[noa][cpu][transform]") {
    test::RealRandomizer<float> randomizer(-100, 100);

    // Get input.
    size3_t shape = test::getRandomShape(2);
    float2_t center(shape.x / 2, shape.y / 2);
    size_t elements = getElements(shape);
    cpu::memory::PtrHost<float> input(elements);
    test::initDataRandom(input.get(), elements, randomizer);

    // Get expected (rely on apply2D) and output.
    cpu::memory::PtrHost<float> output(elements);
    cpu::memory::PtrHost<float> expected(elements);

    const char* symbol = GENERATE("c1", "C2", "C7", "D1", "D3");
    transform::Symmetry symmetry(symbol);

    cpu::transform::apply2D(input.get(), expected.get(), {shape.x, shape.y},
                            {}, {}, symmetry, center, INTERP_LINEAR);

    cpu::transform::symmetrize2D(input.get(), output.get(), {shape.x, shape.y}, 1,
                                 symmetry, center, INTERP_LINEAR);

    cpu::math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
    float min, max, mean;
    cpu::math::minMaxSumMean<float>(output.get(), &min, &max, nullptr, &mean, elements, 1);
    REQUIRE(math::abs(min) < 5e-4f);
    REQUIRE(math::abs(max) < 5e-4f);
    REQUIRE(math::abs(mean) < 1e-6f);
}

TEST_CASE("cpu::transform::symmetrize3D()", "[noa][cpu][transform]") {
    test::RealRandomizer<float> randomizer(-100, 100);

    // Get input.
    size3_t shape = test::getRandomShape(3);
    float3_t center(shape / size_t{2});
    size_t elements = getElements(shape);
    cpu::memory::PtrHost<float> input(elements);
    test::initDataRandom(input.get(), elements, randomizer);

    // Get expected (rely on apply3D) and output.
    cpu::memory::PtrHost<float> output(elements);
    cpu::memory::PtrHost<float> expected(elements);

    const char* symbol = GENERATE("c1", "C2", "C7", "D1", "D3", "o", "i1", "I2");
    transform::Symmetry symmetry(symbol);

    cpu::transform::apply3D(input.get(), expected.get(), shape,
                            {}, {}, symmetry, center, INTERP_LINEAR);

    cpu::transform::symmetrize3D(input.get(), output.get(), shape, 1,
                                 symmetry, center, INTERP_LINEAR);

    cpu::math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
    float min, max, mean;
    cpu::math::minMaxSumMean<float>(output.get(), &min, &max, nullptr, &mean, elements, 1);
    REQUIRE(math::abs(min) < 5e-4f);
    REQUIRE(math::abs(max) < 5e-4f);
    REQUIRE(math::abs(mean) < 1e-6f);
}
