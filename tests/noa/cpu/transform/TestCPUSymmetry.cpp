#include <noa/common/geometry/Transform.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/transform/Apply.h>
#include <noa/cpu/transform/Symmetry.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::geometry::symmetrize2D()", "[noa][cpu][geometry]") {
    test::Randomizer<float> randomizer(-100, 100);
    cpu::Stream stream(cpu::Stream::SERIAL);

    // Get input.
    const size4_t shape = test::getRandomShapeBatched(2);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();
    const float2_t center{shape[2] / 2, shape[3] / 2};
    cpu::memory::PtrHost<float> input(elements);
    test::randomize(input.get(), elements, randomizer);

    // Get expected (rely on transform2D) and output.
    cpu::memory::PtrHost<float> output(elements);
    cpu::memory::PtrHost<float> expected(elements);

    const char* symbol = GENERATE("  c1", "C2", " C7", "d1", "D3 ");
    geometry::Symmetry symmetry(symbol);

    cpu::geometry::transform2D(input.get(), stride, shape, expected.get(), stride, shape,
                               {}, {}, symmetry, center, INTERP_LINEAR, true, stream);

    cpu::geometry::symmetrize2D(input.get(), stride, output.get(), stride, shape,
                                symmetry, center, INTERP_LINEAR, true, stream);
    stream.synchronize();

    REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), elements, 5e-4f));
}

TEST_CASE("cpu::geometry::symmetrize3D()", "[noa][cpu][geometry]") {
    test::Randomizer<float> randomizer(-100, 100);
    cpu::Stream stream(cpu::Stream::SERIAL);

    // Get input.
    const size4_t shape = test::getRandomShapeBatched(3);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();
    const float3_t center{size3_t{shape.get() + 1} / 2};
    cpu::memory::PtrHost<float> input(elements);
    test::randomize(input.get(), elements, randomizer);

    // Get expected (rely on apply3D) and output.
    cpu::memory::PtrHost<float> output(elements);
    cpu::memory::PtrHost<float> expected(elements);

    const char* symbol = GENERATE("c1", "  C2", "C7 ", " D1", "D3", " o", "i1", "I2  ");
    geometry::Symmetry symmetry(symbol);

    cpu::geometry::transform3D(input.get(), stride, shape, expected.get(), stride, shape,
                               {}, {}, symmetry, center, INTERP_LINEAR, true, stream);

    cpu::geometry::symmetrize3D(input.get(), stride, output.get(), stride, shape,
                                symmetry, center, INTERP_LINEAR, true, stream);
    stream.synchronize();

    REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), elements, 5e-4f));
}
