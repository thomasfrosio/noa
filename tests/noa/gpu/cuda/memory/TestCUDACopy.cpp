#include <noa/gpu/cuda/memory/Copy.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrPinned.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/PtrArray.h>
#include <noa/gpu/cuda/util/Stream.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("cuda::memory, synchronous transfers", "[noa][cuda]",
                   int32_t, uint32_t, float, cfloat_t, double, cdouble_t) {
    test::IntRandomizer<size_t> randomizer(2, 255);
    test::IntRandomizer<size_t> randomizer_small(2, 128);

    AND_THEN("host > device > host") {
        size_t elements = randomizer.get();
        memory::PtrHost<TestType> host_in(elements);
        memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrDevice<TestType> device(elements);
        test::initDataRandom(host_in.get(), elements, randomizer);
        test::initDataZero(host_out.get(), elements);
        cuda::memory::copy(host_in.get(), device.get(), elements);
        cuda::memory::copy(device.get(), host_out.get(), elements);
        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > device > pinned > host") {
        size_t elements = randomizer.get();
        memory::PtrHost<TestType> host_in(elements);
        memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrPinned<TestType> pinned(elements);
        cuda::memory::PtrDevice<TestType> device(elements);
        test::initDataRandom(host_in.get(), elements, randomizer);
        test::initDataZero(host_out.get(), elements);

        cuda::memory::copy(host_in.get(), pinned.get(), elements);
        cuda::memory::copy(pinned.get(), device.get(), elements);
        std::memset(static_cast<void*>(pinned.get()), 0, elements * sizeof(TestType)); // Erase pinned to make sure the transfer works
        cuda::memory::copy(device.get(), pinned.get(), elements);
        cuda::memory::copy(pinned.get(), host_out.get(), elements);

        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > devicePadded > host") {
        size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = getElements(shape);
        memory::PtrHost<TestType> host_in(elements);
        memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrDevicePadded<TestType> device(shape);

        test::initDataRandom(host_in.get(), elements, randomizer);
        test::initDataZero(host_out.get(), elements);

        cuda::memory::copy(host_in.get(), shape.x, device.get(), device.pitch(), shape);
        cuda::memory::copy(device.get(), device.pitch(), host_out.get(), shape.x, shape);

        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > devicePadded > pinned > host") {
        size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = getElements(shape);
        memory::PtrHost<TestType> host_in(elements);
        memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrPinned<TestType> pinned(elements);
        cuda::memory::PtrDevicePadded<TestType> device(shape);

        test::initDataRandom(host_in.get(), elements, randomizer);
        test::initDataZero(host_out.get(), elements);
        cuda::memory::copy(host_in.get(), pinned.get(), elements);
        cuda::memory::copy(pinned.get(), shape.x, device.get(), device.pitch(), shape);
        std::memset(static_cast<void*>(pinned.get()), 0, elements * sizeof(TestType)); // Erase pinned to make sure the transfer works
        cuda::memory::copy(device.get(), device.pitch(), pinned.get(), shape.x, shape);
        cuda::memory::copy(pinned.get(), host_out.get(), elements);
        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > device > devicePadded > device > host") {
        size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = getElements(shape);
        memory::PtrHost<TestType> host_in(elements);
        memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrDevice<TestType> device_in(elements);
        cuda::memory::PtrDevice<TestType> device_out(elements);
        cuda::memory::PtrDevicePadded<TestType> device_padded(shape);

        test::initDataRandom(host_in.get(), elements, randomizer);
        test::initDataZero(host_out.get(), elements);
        cuda::memory::copy(host_in.get(), device_in.get(), elements);
        cuda::memory::copy(device_in.get(), shape.x, device_padded.get(), device_padded.pitch(), shape);
        cuda::memory::copy(device_padded.get(), device_padded.pitch(), device_out.get(), shape.x, shape);
        cuda::memory::copy(device_out.get(), host_out.get(), elements);
        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }
}

TEMPLATE_TEST_CASE("cuda::memory, asynchronous transfers", "[noa][cuda]",
                   int32_t, uint32_t, float, cfloat_t, double, cdouble_t) {
    test::IntRandomizer<size_t> randomizer(2, 255);
    test::IntRandomizer<size_t> randomizer_small(2, 128);

    cuda::Stream stream(cuda::STREAM_CONCURRENT);

    AND_THEN("host > device > host") {
        size_t elements = randomizer.get();
        memory::PtrHost<TestType> host_in(elements);
        memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrDevice<TestType> device(elements);

        test::initDataRandom(host_in.get(), elements, randomizer);
        test::initDataZero(host_out.get(), elements);

        cuda::memory::copy(host_in.get(), device.get(), elements, stream);
        cuda::memory::copy(device.get(), host_out.get(), elements, stream);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > device > pinned > host") {
        size_t elements = randomizer.get();
        memory::PtrHost<TestType> host_in(elements);
        memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrPinned<TestType> pinned(elements);
        cuda::memory::PtrDevice<TestType> device(elements);
        test::initDataRandom(host_in.get(), elements, randomizer);
        test::initDataZero(host_out.get(), elements);

        cuda::memory::copy(host_in.get(), pinned.get(), elements, stream);
        cuda::memory::copy(pinned.get(), device.get(), elements, stream);
        cuda::Stream::synchronize(stream);
        std::memset(static_cast<void*>(pinned.get()), 0, elements * sizeof(TestType)); // Erase pinned to make sure the transfer works
        cuda::memory::copy(device.get(), pinned.get(), elements, stream);
        cuda::memory::copy(pinned.get(), host_out.get(), elements, stream);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > devicePadded > host") {
        size3_t shape{randomizer.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = getElements(shape);
        memory::PtrHost<TestType> host_in(elements);
        memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrDevicePadded<TestType> device(shape);

        test::initDataRandom(host_in.get(), elements, randomizer);
        test::initDataZero(host_out.get(), elements);

        cuda::memory::copy(host_in.get(), shape.x, device.get(), device.pitch(), shape, stream);
        cuda::memory::copy(device.get(), device.pitch(), host_out.get(), shape.x, shape, stream);
        cuda::Stream::synchronize(stream);

        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > devicePadded > pinned > host") {
        size3_t shape{randomizer.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = getElements(shape);
        memory::PtrHost<TestType> host_in(elements);
        memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrPinned<TestType> pinned(elements);
        cuda::memory::PtrDevicePadded<TestType> device(shape);

        test::initDataRandom(host_in.get(), elements, randomizer);
        test::initDataZero(host_out.get(), elements);
        cuda::memory::copy(host_in.get(), pinned.get(), elements, stream);
        cuda::memory::copy(pinned.get(), shape.x, device.get(), device.pitch(), shape, stream);
        cuda::Stream::synchronize(stream);
        std::memset(static_cast<void*>(pinned.get()), 0, elements * sizeof(TestType)); // Erase pinned to make sure the transfer works
        cuda::memory::copy(device.get(), device.pitch(), pinned.get(), shape.x, shape, stream);
        cuda::memory::copy(pinned.get(), host_out.get(), elements, stream);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > device > devicePadded > device > host") {
        size3_t shape{randomizer.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = getElements(shape);
        memory::PtrHost<TestType> host_in(elements);
        memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrDevice<TestType> device(elements);
        cuda::memory::PtrDevicePadded<TestType> device_padded(shape);

        test::initDataRandom(host_in.get(), elements, randomizer);
        test::initDataZero(host_out.get(), elements);
        cuda::memory::copy(host_in.get(), device.get(), elements, stream);
        cuda::memory::copy(device.get(), shape.x, device_padded.get(), device_padded.pitch(), shape, stream);
        REQUIRE(cudaMemsetAsync(device.get(), 0, elements * sizeof(TestType), stream.id()) == cudaSuccess);
        cuda::memory::copy(device_padded.get(), device_padded.pitch(), device.get(), shape.x, shape, stream);
        cuda::memory::copy(device.get(), host_out.get(), elements, stream);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }
}

TEMPLATE_TEST_CASE("cuda::memory, synchronous transfers - CUDA arrays", "[noa][cuda]",
                   int32_t, uint32_t, float, cfloat_t) {
    test::IntRandomizer<size_t> randomizer(2, 255);
    test::IntRandomizer<size_t> randomizer_small(2, 128);
    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size_t elements = getElements(shape);

    AND_THEN("host > CUDA array > host") {
        memory::PtrHost<TestType> host_in(elements);
        memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrArray<TestType> array(shape);
        test::initDataRandom(host_in.get(), elements, randomizer);
        test::initDataZero(host_out.get(), elements);
        cuda::memory::copy(host_in.get(), array.get(), shape);
        cuda::memory::copy(array.get(), host_out.get(), shape);
        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > device > CUDA array > device > host") {
        memory::PtrHost<TestType> host_in(elements);
        memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrDevice<TestType> device(elements);
        cuda::memory::PtrArray<TestType> array(shape);
        test::initDataRandom(host_in.get(), elements, randomizer);
        test::initDataZero(host_out.get(), elements);
        cuda::memory::copy(host_in.get(), device.get(), elements);
        cuda::memory::copy(device.get(), array.get(), shape);
        REQUIRE(cudaMemset(device.get(), 0, elements * sizeof(TestType)) == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        cuda::memory::copy(array.get(), device.get(), shape);
        cuda::memory::copy(device.get(), host_out.get(), elements);
        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > CUDA array > pinned > host") {
        cuda::memory::PtrPinned<TestType> host_in(elements);
        cuda::memory::PtrPinned<TestType> host_out(elements);
        cuda::memory::PtrArray<TestType> array(shape);
        test::initDataRandom(host_in.get(), elements, randomizer);
        test::initDataZero(host_out.get(), elements);
        cuda::memory::copy(host_in.get(), array.get(), shape);
        cuda::memory::copy(array.get(), host_out.get(), shape);
        TestType diff = test::getDifference(host_in.get(), host_out.get(), host_out.elements());
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > devicePadded > CUDA array > devicePadded > host") {
        cuda::memory::PtrPinned<TestType> host_in(elements);
        cuda::memory::PtrPinned<TestType> host_out(elements);
        cuda::memory::PtrDevicePadded<TestType> device_padded(shape);
        cuda::memory::PtrArray<TestType> array(shape);
        test::initDataRandom(host_in.get(), elements, randomizer);
        test::initDataZero(host_out.get(), elements);
        cuda::memory::copy(host_in.get(), shape.x, device_padded.get(), device_padded.pitch(), shape);
        cuda::memory::copy(device_padded.get(), device_padded.pitch(), array.get(), shape);
        REQUIRE(cudaMemset(device_padded.get(), 0, device_padded.bytesPadded()) == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        cuda::memory::copy(array.get(), device_padded.get(), device_padded.pitch(), shape);
        cuda::memory::copy(device_padded.get(), device_padded.pitch(), host_out.get(), shape.x, shape);
        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }
}

TEMPLATE_TEST_CASE("cuda::memory, asynchronous transfers - CUDA arrays", "[noa][cuda]",
                   int32_t, uint32_t, float, cfloat_t) {
    test::IntRandomizer<size_t> randomizer(2, 255);
    test::IntRandomizer<size_t> randomizer_small(2, 128);
    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = test::getRandomShape(ndim);
    size_t elements = getElements(shape);
    cuda::Stream stream(cuda::STREAM_SERIAL);

    AND_THEN("host > CUDA array > host") {
        memory::PtrHost<TestType> host_in(elements);
        memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrArray<TestType> array(shape);
        test::initDataRandom(host_in.get(), elements, randomizer);
        test::initDataZero(host_out.get(), elements);
        cuda::memory::copy(host_in.get(), array.get(), shape, stream);
        cuda::memory::copy(array.get(), host_out.get(), shape, stream);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > device > CUDA array > device > host") {
        memory::PtrHost<TestType> host_in(elements);
        memory::PtrHost<TestType> host_out(elements);
        cuda::memory::PtrDevice<TestType> device(elements);
        cuda::memory::PtrArray<TestType> array(shape);
        test::initDataRandom(host_in.get(), elements, randomizer);
        test::initDataZero(host_out.get(), elements);
        cuda::memory::copy(host_in.get(), device.get(), elements, stream);
        cuda::memory::copy(device.get(), array.get(), shape, stream);
        REQUIRE(cudaMemsetAsync(device.get(), 0, elements * sizeof(TestType), stream.get()) == cudaSuccess);
        cuda::memory::copy(array.get(), device.get(), shape, stream);
        cuda::memory::copy(device.get(), host_out.get(), elements, stream);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > CUDA array > pinned > host") {
        cuda::memory::PtrPinned<TestType> host_in(elements);
        cuda::memory::PtrPinned<TestType> host_out(elements);
        cuda::memory::PtrArray<TestType> array(shape);
        test::initDataRandom(host_in.get(), elements, randomizer);
        test::initDataZero(host_out.get(), elements);
        cuda::memory::copy(host_in.get(), array.get(), shape, stream);
        cuda::memory::copy(array.get(), host_out.get(), shape, stream);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getDifference(host_in.get(), host_out.get(), host_out.elements());
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > devicePadded > CUDA array > devicePadded > host") {
        cuda::memory::PtrPinned<TestType> host_in(elements);
        cuda::memory::PtrPinned<TestType> host_out(elements);
        cuda::memory::PtrDevicePadded<TestType> device_padded(shape);
        cuda::memory::PtrArray<TestType> array(shape);
        test::initDataRandom(host_in.get(), elements, randomizer);
        test::initDataZero(host_out.get(), elements);
        cuda::memory::copy(host_in.get(), shape.x, device_padded.get(), device_padded.pitch(), shape, stream);
        cuda::memory::copy(device_padded.get(), device_padded.pitch(), array.get(), shape, stream);
        REQUIRE(cudaMemsetAsync(device_padded.get(), 0, device_padded.bytesPadded(), stream.get()) == cudaSuccess);
        cuda::memory::copy(array.get(), device_padded.get(), device_padded.pitch(), shape, stream);
        cuda::memory::copy(device_padded.get(), device_padded.pitch(), host_out.get(), shape.x, shape, stream);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }
}
