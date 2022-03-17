#include <noa/cpu/memory/PtrHost.h>

#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrPinned.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/PtrArray.h>
#include <noa/gpu/cuda/Stream.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("cuda::memory::copy() - device memory", "[noa][cuda][memory]",
                   int32_t, uint32_t, float, cfloat_t, double, cdouble_t) {
    test::Randomizer<size_t> randomizer(2, 255);
    cuda::Stream stream(cuda::Stream::SERIAL);

    AND_THEN("host > device > host") {
        const size_t elements = randomizer.get();
        cpu::memory::PtrHost<TestType> host_in(elements);
        cpu::memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrDevice<TestType> device(elements);
        test::randomize(host_in.get(), elements, randomizer);
        test::memset(host_out.get(), elements, 0);
        cuda::memory::copy(host_in.get(), device.get(), elements, stream);
        cuda::memory::copy(device.get(), host_out.get(), elements, stream);
        stream.synchronize();
        const TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > device > pinned > host") {
        const size_t elements = randomizer.get();
        cpu::memory::PtrHost<TestType> host_in(elements);
        cpu::memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrPinned<TestType> pinned(elements);
        cuda::memory::PtrDevice<TestType> device(elements);
        test::randomize(host_in.get(), elements, randomizer);
        test::memset(host_out.get(), elements, 0);

        cuda::memory::copy(host_in.get(), pinned.get(), elements, stream);
        cuda::memory::copy(pinned.get(), device.get(), elements, stream);
        stream.synchronize();
        test::memset(pinned.get(), elements, 0); // Erase pinned to make sure the transfer works
        cuda::memory::copy(device.get(), pinned.get(), elements, stream);
        cuda::memory::copy(pinned.get(), host_out.get(), elements, stream);
        stream.synchronize();
        const TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > devicePadded > host") {
        const size4_t shape = test::getRandomShape(4);
        const size4_t stride = shape.stride();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<TestType> host_in(elements);
        cpu::memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrDevicePadded<TestType> device_ptr(shape);

        test::randomize(host_in.get(), elements, randomizer);
        test::memset(host_out.get(), elements, 0);

        cuda::memory::copy(host_in.get(), stride, device_ptr.get(), device_ptr.stride(), shape, stream);
        cuda::memory::copy(device_ptr.get(), device_ptr.stride(), host_out.get(), stride, shape, stream);
        stream.synchronize();
        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > devicePadded > pinned > host") {
        const size4_t shape = test::getRandomShape(4);
        const size4_t stride = shape.stride();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<TestType> host_in(elements);
        cpu::memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrPinned<TestType> pinned(elements);
        cuda::memory::PtrDevicePadded<TestType> device(shape);

        test::randomize(host_in.get(), elements, randomizer);
        test::memset(host_out.get(), elements, 0);
        cuda::memory::copy(host_in.get(), pinned.get(), elements, stream);
        cuda::memory::copy(pinned.get(), stride, device.get(), device.stride(), shape, stream);
        stream.synchronize();
        test::memset(pinned.get(), elements, 0); // Erase pinned to make sure the transfer works
        cuda::memory::copy(device.get(), device.stride(), pinned.get(), stride, shape, stream);
        cuda::memory::copy(pinned.get(), host_out.get(), elements, stream);
        stream.synchronize();
        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > device > devicePadded > device > host") {
        const size4_t shape = test::getRandomShape(4);
        const size4_t stride = shape.stride();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<TestType> host_in(elements);
        cpu::memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrDevice<TestType> device_in(elements);
        cuda::memory::PtrDevice<TestType> device_out(elements);
        cuda::memory::PtrDevicePadded<TestType> device_padded(shape);

        test::randomize(host_in.get(), elements, randomizer);
        test::memset(host_out.get(), elements, 0);
        cuda::memory::copy(host_in.get(), device_in.get(), elements, stream);
        cuda::memory::copy(device_in.get(), stride, device_padded.get(), device_padded.stride(), shape, stream);
        cuda::memory::copy(device_padded.get(), device_padded.stride(), device_out.get(), stride, shape, stream);
        cuda::memory::copy(device_out.get(), host_out.get(), elements, stream);
        stream.synchronize();
        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }
}

TEMPLATE_TEST_CASE("cuda::memory::copy() - CUDA arrays", "[noa][cuda][memory]",
                   int32_t, uint32_t, float, cfloat_t) {
    test::Randomizer<size_t> randomizer(2, 255);
    const uint ndim = GENERATE(1U, 2U, 3U);
    const size4_t shape = test::getRandomShape(ndim);
    const size4_t stride = shape.stride();
    const size3_t shape_3d{shape.get() + 1};
    const size_t pitch = shape[3];
    const size_t elements = shape.elements();
    cuda::Stream stream(cuda::Stream::SERIAL);

    AND_THEN("host > CUDA array > host") {
        cpu::memory::PtrHost<TestType> host_in(elements);
        cpu::memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrArray<TestType> array(shape_3d);
        test::randomize(host_in.get(), elements, randomizer);
        test::memset(host_out.get(), elements, 0);
        cuda::memory::copy(host_in.get(), pitch, array.get(), shape_3d, stream);
        cuda::memory::copy(array.get(), host_out.get(), pitch, shape_3d, stream);
        stream.synchronize();
        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > device > CUDA array > device > host") {
        cpu::memory::PtrHost<TestType> host_in(elements);
        cpu::memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrDevice<TestType> device(elements);
        cuda::memory::PtrArray<TestType> array(shape_3d);
        test::randomize(host_in.get(), elements, randomizer);
        test::memset(host_out.get(), elements, 0);
        cuda::memory::copy(host_in.get(), device.get(), elements, stream);
        cuda::memory::copy(device.get(), pitch, array.get(), shape_3d, stream);
        REQUIRE(cudaMemsetAsync(device.get(), 0, elements * sizeof(TestType), stream.get()) == cudaSuccess);
        cuda::memory::copy(array.get(), device.get(), pitch, shape_3d, stream);
        cuda::memory::copy(device.get(), host_out.get(), elements, stream);
        stream.synchronize();
        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > CUDA array > pinned > host") {
        cuda::memory::PtrPinned<TestType> host_in(elements);
        cuda::memory::PtrPinned<TestType> host_out(elements);
        cuda::memory::PtrArray<TestType> array(shape_3d);
        test::randomize(host_in.get(), elements, randomizer);
        test::memset(host_out.get(), elements, 0);
        cuda::memory::copy(host_in.get(), pitch, array.get(), shape_3d, stream);
        cuda::memory::copy(array.get(), host_out.get(), pitch, shape_3d, stream);
        stream.synchronize();
        TestType diff = test::getDifference(host_in.get(), host_out.get(), host_out.elements());
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > devicePadded > CUDA array > devicePadded > host") {
        cuda::memory::PtrPinned<TestType> host_in(elements);
        cuda::memory::PtrPinned<TestType> host_out(elements);
        cuda::memory::PtrDevicePadded<TestType> device_padded(shape);
        cuda::memory::PtrArray<TestType> array(shape_3d);
        const size_t total_bytes = device_padded.pitch().elements() * sizeof(TestType);

        test::randomize(host_in.get(), elements, randomizer);
        test::memset(host_out.get(), elements, 0);
        cuda::memory::copy(host_in.get(), stride, device_padded.get(), device_padded.stride(), shape, stream);
        cuda::memory::copy(device_padded.get(), device_padded.pitch()[2], array.get(), shape_3d, stream);
        REQUIRE(cudaMemsetAsync(device_padded.get(), 0, total_bytes, stream.get()) == cudaSuccess);
        cuda::memory::copy(array.get(), device_padded.get(), device_padded.pitch()[2], shape_3d, stream);
        cuda::memory::copy(device_padded.get(), device_padded.stride(), host_out.get(), stride, shape, stream);
        stream.synchronize();
        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }
}

TEMPLATE_TEST_CASE("cuda::memory::copy() - strided X", "[noa][cuda][memory]",
                   int32_t, uint32_t, float, cfloat_t) {
    test::Randomizer<size_t> randomizer(2, 255);
    const uint ndim = GENERATE(1U, 2U, 3U);
    const size4_t shape = test::getRandomShape(ndim);
    const size4_t stride = shape.stride() * 2;
    const size_t elements = stride[0] * shape[0];
    cuda::Stream stream(cuda::Stream::SERIAL);

    AND_THEN("host > pinned > device-strided > device > host") {
        cpu::memory::PtrHost<TestType> host_in(shape.elements());
        cpu::memory::PtrHost<TestType> host_out(host_in.elements());
        cuda::memory::PtrPinned<TestType> pinned(host_in.elements());

        cuda::memory::PtrDevice<TestType> device_strided(elements, stream);
        cuda::memory::PtrDevice<TestType> device_contiguous(host_in.elements(), stream);

        test::randomize(host_in.get(), host_in.elements(), randomizer);
        test::memset(host_in.get(), host_in.elements(), 10);
        test::memset(host_out.get(), host_out.elements(), 0);

        cuda::memory::copy(host_in.get(), pinned.get(), host_in.elements(), stream);
        cuda::memory::copy(pinned.get(), shape.stride(), device_strided.get(), stride, shape, stream);
        cuda::memory::copy(device_strided.get(), stride, device_contiguous.get(), shape.stride(), shape, stream);
        cuda::memory::copy(device_contiguous.get(), host_out.get(), host_in.elements(), stream);
        stream.synchronize();
        TestType diff = test::getDifference(host_in.get(), host_out.get(), host_in.elements());
        REQUIRE(diff == TestType{0});
    }
}

TEMPLATE_TEST_CASE("cuda::memory::copy() - strided Y", "[noa][cuda][memory]",
                   int32_t, uint32_t, float, cfloat_t) {
    test::Randomizer<size_t> randomizer(2, 255);
    const uint ndim = GENERATE(1U, 2U, 3U);
    const size4_t shape = test::getRandomShape(ndim);
    size4_t stride = shape.stride();
    stride[0] *= 2;
    stride[1] *= 2;
    stride[2] *= 2; // stride in Y
    const size_t elements = stride[0] * shape[0];
    cuda::Stream stream(cuda::Stream::SERIAL);

    AND_THEN("host > pinned > device-strided > device > host") {
        cpu::memory::PtrHost<TestType> host_in(shape.elements());
        cpu::memory::PtrHost<TestType> host_out(host_in.elements());
        cuda::memory::PtrPinned<TestType> pinned(host_in.elements());

        cuda::memory::PtrDevice<TestType> device_strided(elements, stream);
        cuda::memory::PtrDevice<TestType> device_contiguous(host_in.elements(), stream);

        test::randomize(host_in.get(), host_in.elements(), randomizer);
        test::memset(host_in.get(), host_in.elements(), 10);
        test::memset(host_out.get(), host_out.elements(), 0);

        cuda::memory::copy(host_in.get(), pinned.get(), host_in.elements(), stream);
        cuda::memory::copy(pinned.get(), shape.stride(), device_strided.get(), stride, shape, stream);
        cuda::memory::copy(device_strided.get(), stride, device_contiguous.get(), shape.stride(), shape, stream);
        cuda::memory::copy(device_contiguous.get(), host_out.get(), host_in.elements(), stream);
        stream.synchronize();
        TestType diff = test::getDifference(host_in.get(), host_out.get(), host_in.elements());
        REQUIRE(diff == TestType{0});
    }
}

TEMPLATE_TEST_CASE("cuda::memory::copy() - strided YX", "[noa][cuda][memory]",
                   int32_t, uint32_t, float, cfloat_t) {
    test::Randomizer<size_t> randomizer(2, 255);
    const uint ndim = GENERATE(1U, 2U, 3U);
    const size4_t shape = test::getRandomShape(ndim);
    size4_t stride = shape.stride();
    stride[0] *= 4;
    stride[1] *= 4;
    stride[2] *= 4; // stride in Y
    stride[3] *= 2; // stride in X
    const size_t elements = stride[0] * shape[0];
    cuda::Stream stream(cuda::Stream::SERIAL);

    AND_THEN("host > pinned > device-strided > device > host") {
        cpu::memory::PtrHost<TestType> host_in(shape.elements());
        cpu::memory::PtrHost<TestType> host_out(host_in.elements());
        cuda::memory::PtrPinned<TestType> pinned(host_in.elements());

        cuda::memory::PtrDevice<TestType> device_strided(elements, stream);
        cuda::memory::PtrDevice<TestType> device_contiguous(host_in.elements(), stream);

        test::randomize(host_in.get(), host_in.elements(), randomizer);
        test::memset(host_in.get(), host_in.elements(), 10);
        test::memset(host_out.get(), host_out.elements(), 0);

        cuda::memory::copy(host_in.get(), pinned.get(), host_in.elements(), stream);
        cuda::memory::copy(pinned.get(), shape.stride(), device_strided.get(), stride, shape, stream);
        cuda::memory::copy(device_strided.get(), stride, device_contiguous.get(), shape.stride(), shape, stream);
        cuda::memory::copy(device_contiguous.get(), host_out.get(), host_in.elements(), stream);
        stream.synchronize();
        TestType diff = test::getDifference(host_in.get(), host_out.get(), host_in.elements());
        REQUIRE(diff == TestType{0});
    }
}
