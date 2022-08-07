#include <noa/common/geometry/Transform.h>
#include <noa/cpu/memory/PtrHost.h>

#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/geometry/Transform.h>
#include <noa/gpu/cuda/geometry/Symmetry.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cuda::transform::symmetrize2D()", "[noa][cuda][transform]") {
    test::Randomizer<float> randomizer(-100, 100);

    // Get input.
    const size4_t shape = test::getRandomShapeBatched(2u);
    const size4_t stride = shape.strides();
    const float2_t center{shape.get() + 2};
    const size_t elements = shape.elements();
    cpu::memory::PtrHost<float> input(elements);
    test::randomize(input.get(), elements, randomizer);

    cuda::Stream stream;
    cuda::memory::PtrDevice<float> d_input(elements);
    cuda::memory::copy(input.get(), d_input.get(), elements, stream);

    // Get expected (rely on transform2D) and output.
    cpu::memory::PtrHost<float> output(elements);
    cpu::memory::PtrHost<float> expected(elements);
    cuda::memory::PtrDevice<float> d_output(elements);
    cuda::memory::PtrDevice<float> d_expected(elements);

    const char* symbol = GENERATE("c1", "C2 ", "C7", " D1", "d3");
    geometry::Symmetry symmetry(symbol);

    cuda::geometry::transform2D(d_input.share(), stride, shape,
                                d_expected.share(), stride, shape,
                                {}, {}, symmetry, center, INTERP_LINEAR, true, true, stream);
    cuda::memory::copy(d_expected.share(), expected.share(), elements, stream);

    cuda::geometry::symmetrize2D(d_input.share(), stride,
                                 d_output.share(), stride, shape,
                                 symmetry, center, INTERP_LINEAR, true, true, stream);
    cuda::memory::copy(d_output.share(), output.share(), elements, stream);
    stream.synchronize();

    REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), elements, 5e-4f));
}

TEST_CASE("cuda::geometry::symmetrize3D()", "[noa][cuda][geometry]") {
    test::Randomizer<float> randomizer(-100, 100);

    // Get input.
    const size4_t shape = test::getRandomShape(3u);
    const size4_t stride = shape.strides();
    const float3_t center{size3_t{shape.get() + 1} / 2};
    const size_t elements = shape.elements();
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

    const char* symbol = GENERATE("c1", "  c2", "C7", "D1", "D3", "o ", "i1", "  I2  ");
    geometry::Symmetry symmetry(symbol);

    INFO(symbol);
    INFO(shape);

    cuda::geometry::transform3D(d_input.share(), stride, shape,
                                d_expected.share(), stride, shape,
                                {}, {}, symmetry, center, INTERP_LINEAR, true, true, stream);
    cuda::memory::copy(d_expected.share(), expected.share(), elements, stream);
    stream.synchronize();

    cuda::geometry::symmetrize3D(d_input.share(), stride,
                                 d_output.share(), stride, shape,
                                 symmetry, center, INTERP_LINEAR, true, true, stream);
    cuda::memory::copy(d_output.share(), output.share(), elements, stream);
    stream.synchronize();

    REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), elements, 5e-4f));
}
