#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Resize.h>

#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Resize.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include <noa/io/files/MRCFile.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("CUDA::Memory::resize() -- against test data", "[noa][cuda][memory]", float) {
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
    Memory::PtrHost<float> expected(o_elements * batches);
    MRCFile file(filename, IO::READ);
    file.readAll(expected.get());

    Memory::PtrHost<float> h_input(i_elements * batches);
    Memory::PtrHost<float> h_output(o_elements * batches);
    Test::Assets::Memory::initResizeInput(test_number, h_input.get(), i_shape, batches);
    if (test_number >= 19)
        Test::Assets::Memory::initResizeOutput(h_output.get(), o_shape, batches);

    CUDA::Stream stream;

    AND_THEN("Contiguous") {
        CUDA::Memory::PtrDevice<float> d_input(i_elements * batches);
        CUDA::Memory::PtrDevice<float> d_output(o_elements * batches);
        CUDA::Memory::copy(h_input.get(), d_input.get(), d_input.elements() * sizeof(float), stream);
        if (test_number >= 19)
            CUDA::Memory::copy(h_output.get(), d_output.get(), d_output.elements() * sizeof(float), stream);

        if (test_number < 11 || test_number >= 19)
            CUDA::Memory::resize(d_input.get(), i_shape, d_output.get(), o_shape,
                                 border_left, border_right, mode, value, batches, stream);
        else
            CUDA::Memory::resize(d_input.get(), i_shape, d_output.get(), o_shape,
                                 mode, value, batches, stream);

        CUDA::Memory::copy(d_output.get(), h_output.get(), h_output.elements() * sizeof(float), stream);
        CUDA::Stream::synchronize(stream);
        float diff = Test::getAverageNormalizedDifference(expected.get(), h_output.get(), h_output.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(0.f, 1e-6));
    }

    Test::Assets::Memory::initResizeOutput(h_output.get(), o_shape, batches);
    AND_THEN("Padded") {
        CUDA::Memory::PtrDevicePadded<float> d_input({i_shape.x, i_shape.y * i_shape.z, batches});
        CUDA::Memory::PtrDevicePadded<float> d_output({o_shape.x, o_shape.y * o_shape.z, batches});
        CUDA::Memory::copy(h_input.get(), i_shape.x * sizeof(float),
                           d_input.get(), d_input.pitch(),
                           d_input.shape(), stream);
        if (test_number >= 19)
            CUDA::Memory::copy(h_output.get(), o_shape.x * sizeof(float),
                               d_output.get(), d_output.pitch(),
                               d_output.shape(), stream);

        if (test_number < 11 || test_number >= 19)
            CUDA::Memory::resize(d_input.get(), d_input.pitchElements(), i_shape,
                                 d_output.get(), d_output.pitchElements(), o_shape,
                                 border_left, border_right, mode, value, batches, stream);
        else
            CUDA::Memory::resize(d_input.get(), d_input.pitchElements(), i_shape,
                                 d_output.get(), d_output.pitchElements(), o_shape,
                                 mode, value, batches, stream);

        CUDA::Memory::copy(d_output.get(), d_output.pitch(),
                           h_output.get(), o_shape.x * sizeof(float), d_output.shape(), stream);
        CUDA::Stream::synchronize(stream);
        float diff = Test::getAverageNormalizedDifference(expected.get(), h_output.get(), h_output.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(0.f, 1e-6));
    }
}

TEMPLATE_TEST_CASE("CUDA::Memory::resize() - edge cases", "[noa][cpu]",
                   int, uint, long long, unsigned long long, float, double) {
    uint ndim = GENERATE(2U, 3U);
    uint batches = Test::IntRandomizer<uint>(1, 3).get();
    CUDA::Stream stream;

    AND_THEN("in-place is not allowed") {
        size3_t i_shape = Test::getRandomShape(ndim);
        size3_t o_shape = Test::getRandomShape(ndim);
        CUDA::Memory::PtrDevice<TestType> input;
        REQUIRE_THROWS_AS(CUDA::Memory::resize(input.get(), i_shape, input.get(), o_shape,
                                               BORDER_VALUE, TestType{0}, batches, stream),
                          Noa::Exception);
    }

    AND_THEN("output shape does not match") {
        size3_t i_shape = Test::getRandomShape(ndim);
        size3_t o_shape(i_shape + size_t{10});
        int3_t border_left(0);
        int3_t border_right(0);
        CUDA::Memory::PtrDevice<TestType> input;
        REQUIRE_THROWS_AS(CUDA::Memory::resize(input.get(), i_shape, input.get(), o_shape, border_left, border_right,
                                               BORDER_VALUE, TestType{0}, batches, stream),
                          Noa::Exception);
    }

    AND_THEN("copy") {
        size3_t shape = Test::getRandomShape(ndim);
        size_t elements = getElements(shape) * batches;
        Memory::PtrHost<TestType> input(elements);
        Test::Randomizer<TestType> randomizer(0, 50);
        Test::initDataRandom(input.get(), elements, randomizer);

        CUDA::Memory::PtrDevice<TestType> d_input(elements);
        CUDA::Memory::PtrDevice<TestType> d_output(elements);
        Memory::PtrHost<TestType> output(elements);

        CUDA::Memory::copy(input.get(), d_input.get(), elements * sizeof(TestType));
        CUDA::Memory::resize(d_input.get(), shape, d_output.get(), shape, BORDER_VALUE, TestType{0}, batches, stream);
        CUDA::Memory::copy(d_output.get(), output.get(), elements * sizeof(TestType));
        TestType diff = Test::getDifference(input.get(), output.get(), elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(0, 1e-6));
    }
}
