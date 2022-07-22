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
    cuda::Stream stream;

    AND_THEN("host > device > host") {
        const size_t elements = randomizer.get();
        cpu::memory::PtrHost<TestType> host_in(elements);
        cpu::memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrDevice<TestType> device(elements);
        test::randomize(host_in.get(), elements, randomizer);
        test::memset(host_out.get(), elements, 0);
        cuda::memory::copy<TestType>(host_in.share(), device.share(), elements, stream);
        cuda::memory::copy<TestType>(device.share(), host_out.share(), elements, stream);
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

        cuda::memory::copy<TestType>(host_in.share(), pinned.share(), elements, stream);
        cuda::memory::copy<TestType>(pinned.share(), device.share(), elements, stream);
        stream.synchronize();
        test::memset(pinned.get(), elements, 0); // Erase pinned to make sure the transfer works
        cuda::memory::copy<TestType>(device.share(), pinned.share(), elements, stream);
        cuda::memory::copy<TestType>(pinned.share(), host_out.share(), elements, stream);
        stream.synchronize();
        const TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > devicePadded > host") {
        const size4_t shape = test::getRandomShape(4);
        const size4_t stride = shape.strides();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<TestType> host_in(elements);
        cpu::memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrDevicePadded<TestType> device_ptr(shape);

        test::randomize(host_in.get(), elements, randomizer);
        test::memset(host_out.get(), elements, 0);

        cuda::memory::copy<TestType>(host_in.share(), stride, device_ptr.share(), device_ptr.strides(), shape, stream);
        cuda::memory::copy<TestType>(device_ptr.share(), device_ptr.strides(), host_out.share(), stride, shape, stream);
        stream.synchronize();
        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > devicePadded > pinned > host") {
        const size4_t shape = test::getRandomShape(4);
        const size4_t stride = shape.strides();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<TestType> host_in(elements);
        cpu::memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrPinned<TestType> pinned(elements);
        cuda::memory::PtrDevicePadded<TestType> device(shape);

        test::randomize(host_in.get(), elements, randomizer);
        test::memset(host_out.get(), elements, 0);
        cuda::memory::copy<TestType>(host_in.share(), pinned.share(), elements, stream);
        cuda::memory::copy<TestType>(pinned.share(), stride, device.share(), device.strides(), shape, stream);
        stream.synchronize();
        test::memset(pinned.get(), elements, 0); // Erase pinned to make sure the transfer works
        cuda::memory::copy<TestType>(device.share(), device.strides(), pinned.share(), stride, shape, stream);
        cuda::memory::copy<TestType>(pinned.share(), host_out.share(), elements, stream);
        stream.synchronize();
        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > device > devicePadded > device > host") {
        const size4_t shape = test::getRandomShape(4);
        const size4_t stride = shape.strides();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<TestType> host_in(elements);
        cpu::memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrDevice<TestType> device_in(elements);
        cuda::memory::PtrDevice<TestType> device_out(elements);
        cuda::memory::PtrDevicePadded<TestType> device_padded(shape);

        test::randomize(host_in.get(), elements, randomizer);
        test::memset(host_out.get(), elements, 0);
        cuda::memory::copy<TestType>(host_in.share(), device_in.share(), elements, stream);
        cuda::memory::copy<TestType>(device_in.share(), stride, device_padded.share(), device_padded.strides(), shape, stream);
        cuda::memory::copy<TestType>(device_padded.share(), device_padded.strides(), device_out.share(), stride, shape, stream);
        cuda::memory::copy<TestType>(device_out.share(), host_out.share(), elements, stream);
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
    const size4_t stride = shape.strides();
    const size3_t shape_3d{shape.get() + 1};
    const size_t pitch = shape[3];
    const size_t elements = shape.elements();
    cuda::Stream stream;

    AND_THEN("host > CUDA array > host") {
        cpu::memory::PtrHost<TestType> host_in(elements);
        cpu::memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrArray<TestType> array(shape_3d);
        test::randomize(host_in.get(), elements, randomizer);
        test::memset(host_out.get(), elements, 0);
        cuda::memory::copy<TestType>(host_in.share(), pitch, array.share(), shape_3d, stream);
        cuda::memory::copy<TestType>(array.share(), host_out.share(), pitch, shape_3d, stream);
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
        cuda::memory::copy<TestType>(host_in.share(), device.share(), elements, stream);
        cuda::memory::copy<TestType>(device.share(), pitch, array.share(), shape_3d, stream);
        REQUIRE(cudaMemsetAsync(device.get(), 0, elements * sizeof(TestType), stream.get()) == cudaSuccess);
        cuda::memory::copy<TestType>(array.share(), device.share(), pitch, shape_3d, stream);
        cuda::memory::copy<TestType>(device.share(), host_out.share(), elements, stream);
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
        cuda::memory::copy<TestType>(host_in.share(), pitch, array.share(), shape_3d, stream);
        cuda::memory::copy<TestType>(array.share(), host_out.share(), pitch, shape_3d, stream);
        stream.synchronize();
        TestType diff = test::getDifference(host_in.get(), host_out.get(), host_out.elements());
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > devicePadded > CUDA array > devicePadded > host") {
        cuda::memory::PtrPinned<TestType> host_in(elements);
        cuda::memory::PtrPinned<TestType> host_out(elements);
        cuda::memory::PtrDevicePadded<TestType> device_padded(shape);
        cuda::memory::PtrArray<TestType> array(shape_3d);
        const size_t total_bytes = device_padded.pitches().elements() * sizeof(TestType);

        test::randomize(host_in.get(), elements, randomizer);
        test::memset(host_out.get(), elements, 0);
        cuda::memory::copy<TestType>(host_in.share(), stride, device_padded.share(), device_padded.strides(), shape, stream);
        cuda::memory::copy<TestType>(device_padded.share(), device_padded.pitches()[2], array.share(), shape_3d, stream);
        REQUIRE(cudaMemsetAsync(device_padded.get(), 0, total_bytes, stream.get()) == cudaSuccess);
        cuda::memory::copy<TestType>(array.share(), device_padded.share(), device_padded.pitches()[2], shape_3d, stream);
        cuda::memory::copy<TestType>(device_padded.share(), device_padded.strides(), host_out.share(), stride, shape, stream);
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
    const size4_t stride = shape.strides() * 2;
    const size_t elements = stride[0] * shape[0];
    cuda::Stream stream;

    AND_THEN("host > pinned > device-strided > device > host") {
        cpu::memory::PtrHost<TestType> host_in(shape.elements());
        cpu::memory::PtrHost<TestType> host_out(host_in.elements());
        cuda::memory::PtrPinned<TestType> pinned(host_in.elements());

        cuda::memory::PtrDevice<TestType> device_strided(elements, stream);
        cuda::memory::PtrDevice<TestType> device_contiguous(host_in.elements(), stream);

        test::randomize(host_in.get(), host_in.elements(), randomizer);
        test::memset(host_in.get(), host_in.elements(), 10);
        test::memset(host_out.get(), host_out.elements(), 0);

        cuda::memory::copy<TestType>(host_in.share(), pinned.share(), host_in.elements(), stream);
        cuda::memory::copy<TestType>(pinned.share(), shape.strides(), device_strided.share(), stride, shape, stream);
        cuda::memory::copy<TestType>(device_strided.share(), stride, device_contiguous.share(), shape.strides(), shape, stream);
        cuda::memory::copy<TestType>(device_contiguous.share(), host_out.share(), host_in.elements(), stream);
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
    size4_t stride = shape.strides();
    stride[0] *= 2;
    stride[1] *= 2;
    stride[2] *= 2; // stride in Y
    const size_t elements = stride[0] * shape[0];
    cuda::Stream stream;

    AND_THEN("host > pinned > device-strided > device > host") {
        cpu::memory::PtrHost<TestType> host_in(shape.elements());
        cpu::memory::PtrHost<TestType> host_out(host_in.elements());
        cuda::memory::PtrPinned<TestType> pinned(host_in.elements());

        cuda::memory::PtrDevice<TestType> device_strided(elements, stream);
        cuda::memory::PtrDevice<TestType> device_contiguous(host_in.elements(), stream);

        test::randomize(host_in.get(), host_in.elements(), randomizer);
        test::memset(host_in.get(), host_in.elements(), 10);
        test::memset(host_out.get(), host_out.elements(), 0);

        cuda::memory::copy<TestType>(host_in.share(), pinned.share(), host_in.elements(), stream);
        cuda::memory::copy<TestType>(pinned.share(), shape.strides(), device_strided.share(), stride, shape, stream);
        cuda::memory::copy<TestType>(device_strided.share(), stride, device_contiguous.share(), shape.strides(), shape, stream);
        cuda::memory::copy<TestType>(device_contiguous.share(), host_out.share(), host_in.elements(), stream);
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
    size4_t stride = shape.strides();
    stride[0] *= 4;
    stride[1] *= 4;
    stride[2] *= 4; // stride in Y
    stride[3] *= 2; // stride in X
    const size_t elements = stride[0] * shape[0];
    cuda::Stream stream;

    AND_THEN("host > pinned > device-strided > device > host") {
        cpu::memory::PtrHost<TestType> host_in(shape.elements());
        cpu::memory::PtrHost<TestType> host_out(host_in.elements());
        cuda::memory::PtrPinned<TestType> pinned(host_in.elements());

        cuda::memory::PtrDevice<TestType> device_strided(elements, stream);
        cuda::memory::PtrDevice<TestType> device_contiguous(host_in.elements(), stream);

        test::randomize(host_in.get(), host_in.elements(), randomizer);
        test::memset(host_in.get(), host_in.elements(), 10);
        test::memset(host_out.get(), host_out.elements(), 0);

        cuda::memory::copy<TestType>(host_in.share(), pinned.share(), host_in.elements(), stream);
        cuda::memory::copy<TestType>(pinned.share(), shape.strides(), device_strided.share(), stride, shape, stream);
        cuda::memory::copy<TestType>(device_strided.share(), stride, device_contiguous.share(), shape.strides(), shape, stream);
        cuda::memory::copy<TestType>(device_contiguous.share(), host_out.share(), host_in.elements(), stream);
        stream.synchronize();
        TestType diff = test::getDifference(host_in.get(), host_out.get(), host_in.elements());
        REQUIRE(diff == TestType{0});
    }
}
