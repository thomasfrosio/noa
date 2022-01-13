#include <noa/common/transform/Geometry.h>
#include <noa/cpu/memory/PtrHost.h>

#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/transform/Apply.h>
#include <noa/gpu/cuda/transform/Symmetry.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cuda::transform::symmetrize2D()", "[noa][cuda][transform]") {
    test::Randomizer<float> randomizer(-100, 100);

    // Get input.
    size3_t shape = test::getRandomShape(2);
    float2_t center(shape.x / 2, shape.y / 2);
    size_t elements = noa::elements(shape);
    cpu::memory::PtrHost<float> input(elements);
    test::randomize(input.get(), elements, randomizer);

    cuda::Stream stream;
    cuda::memory::PtrDevice<float> d_input(elements);
    cuda::memory::copy(input.get(), d_input.get(), elements, stream);

    // Get expected (rely on apply2D) and output.
    cpu::memory::PtrHost<float> output(elements);
    cpu::memory::PtrHost<float> expected(elements);
    cuda::memory::PtrDevice<float> d_output(elements);
    cuda::memory::PtrDevice<float> d_expected(elements);

    const char* symbol = GENERATE("c1", "C2", "C7", "D1", "D3");
    transform::Symmetry symmetry(symbol);

    cuda::transform::apply2D(d_input.get(), shape.x,
                             d_expected.get(), shape.x, {shape.x, shape.y},
                             {}, {}, symmetry, center, INTERP_LINEAR, stream);
    cuda::memory::copy(d_expected.get(), expected.get(), elements, stream);

    cuda::transform::symmetrize2D(d_input.get(), shape.x,
                                  d_output.get(), shape.x, {shape.x, shape.y}, 1,
                                  symmetry, center, INTERP_LINEAR, stream);
    cuda::memory::copy(d_output.get(), output.get(), elements, stream);
    stream.synchronize();

    REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), elements, 5e-4f));
}

TEST_CASE("cuda::transform::symmetrize3D()", "[noa][cuda][transform]") {
    test::Randomizer<float> randomizer(-100, 100);

    // Get input.
    size3_t shape = test::getRandomShape(3);
    float3_t center(shape / size_t{2});
    size_t elements = noa::elements(shape);
    cpu::memory::PtrHost<float> input(elements);
    test::randomize(input.get(), elements, randomizer);

    cuda::Stream stream;
    cuda::memory::PtrDevice<float> d_input(elements);
    cuda::memory::copy(input.get(), d_input.get(), elements, stream);

    // Get expected (rely on apply3D) and output.
    cpu::memory::PtrHost<float> output(elements);
    cpu::memory::PtrHost<float> expected(elements);
    cuda::memory::PtrDevice<float> d_output(elements);
    cuda::memory::PtrDevice<float> d_expected(elements);

    const char* symbol = GENERATE("c1", "C2", "C7", "D1", "D3", "o", "i1", "I2");
    transform::Symmetry symmetry(symbol);

    cuda::transform::apply3D(d_input.get(), shape.x,
                             d_expected.get(), shape.x, shape,
                             {}, {}, symmetry, center, INTERP_LINEAR, stream);
    cuda::memory::copy(d_expected.get(), expected.get(), elements, stream);

    cuda::transform::symmetrize3D(d_input.get(), shape.x,
                                  d_output.get(), shape.x, shape, 1,
                                  symmetry, center, INTERP_LINEAR, stream);
    cuda::memory::copy(d_output.get(), output.get(), elements, stream);
    stream.synchronize();

    REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), elements, 5e-4f));
}
