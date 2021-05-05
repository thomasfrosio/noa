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

    int test_number = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18);
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
    CUDA::Stream stream;

    AND_THEN("Contiguous") {
        CUDA::Memory::PtrDevice<float> d_input(i_elements * batches);
        CUDA::Memory::PtrDevice<float> d_output(o_elements * batches);
        CUDA::Memory::copy(h_input.get(), d_input.get(), d_input.elements() * sizeof(float), stream);

        if (test_number < 11)
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

    Test::initDataZero(h_output.get(), h_output.elements()); // this shouldn't be necessary but for safety...
    AND_THEN("Padded") {
        CUDA::Memory::PtrDevicePadded<float> d_input({i_shape.x, i_shape.y * i_shape.z, batches});
        CUDA::Memory::PtrDevicePadded<float> d_output({o_shape.x, o_shape.y * o_shape.z, batches});
        CUDA::Memory::copy(h_input.get(), i_shape.x * sizeof(float),
                           d_input.get(), d_input.pitch(),
                           d_input.shape(), stream);

        if (test_number < 11)
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
