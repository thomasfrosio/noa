#include <noa/gpu/cuda/memory/Copy.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrPinned.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/PtrArray.h>
#include <noa/gpu/cuda/util/Stream.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::Noa;

TEMPLATE_TEST_CASE("CUDA::Memory, synchronous transfers", "[noa][cuda]",
                   int32_t, uint32_t, float, cfloat_t, double, cdouble_t) {
    Test::IntRandomizer<size_t> randomizer(2, 255);
    Test::IntRandomizer<size_t> randomizer_small(2, 128);

    AND_THEN("host > device > host") {
        size_t elements = randomizer.get();
        Memory::PtrHost<TestType> host_in(elements);
        Memory::PtrHost<TestType> host_out(elements);
        CUDA::Memory::PtrDevice<TestType> device(elements);
        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        CUDA::Memory::copy(host_in.get(), device.get(), elements);
        CUDA::Memory::copy(device.get(), host_out.get(), elements);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > device > pinned > host") {
        size_t elements = randomizer.get();
        Memory::PtrHost<TestType> host_in(elements);
        Memory::PtrHost<TestType> host_out(elements);
        CUDA::Memory::PtrPinned<TestType> pinned(elements);
        CUDA::Memory::PtrDevice<TestType> device(elements);
        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);

        CUDA::Memory::copy(host_in.get(), pinned.get(), elements);
        CUDA::Memory::copy(pinned.get(), device.get(), elements);
        std::memset(static_cast<void*>(pinned.get()), 0, elements * sizeof(TestType)); // Erase pinned to make sure the transfer works
        CUDA::Memory::copy(device.get(), pinned.get(), elements);
        CUDA::Memory::copy(pinned.get(), host_out.get(), elements);

        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > devicePadded > host") {
        size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = getElements(shape);
        Memory::PtrHost<TestType> host_in(elements);
        Memory::PtrHost<TestType> host_out(elements);
        CUDA::Memory::PtrDevicePadded<TestType> device(shape);

        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);

        CUDA::Memory::copy(host_in.get(), shape.x, device.get(), device.pitch(), shape);
        CUDA::Memory::copy(device.get(), device.pitch(), host_out.get(), shape.x, shape);

        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > devicePadded > pinned > host") {
        size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = getElements(shape);
        Memory::PtrHost<TestType> host_in(elements);
        Memory::PtrHost<TestType> host_out(elements);
        CUDA::Memory::PtrPinned<TestType> pinned(elements);
        CUDA::Memory::PtrDevicePadded<TestType> device(shape);

        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        CUDA::Memory::copy(host_in.get(), pinned.get(), elements);
        CUDA::Memory::copy(pinned.get(), shape.x, device.get(), device.pitch(), shape);
        std::memset(static_cast<void*>(pinned.get()), 0, elements * sizeof(TestType)); // Erase pinned to make sure the transfer works
        CUDA::Memory::copy(device.get(), device.pitch(), pinned.get(), shape.x, shape);
        CUDA::Memory::copy(pinned.get(), host_out.get(), elements);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > device > devicePadded > device > host") {
        size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = getElements(shape);
        Memory::PtrHost<TestType> host_in(elements);
        Memory::PtrHost<TestType> host_out(elements);
        CUDA::Memory::PtrDevice<TestType> device_in(elements);
        CUDA::Memory::PtrDevice<TestType> device_out(elements);
        CUDA::Memory::PtrDevicePadded<TestType> device_padded(shape);

        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        CUDA::Memory::copy(host_in.get(), device_in.get(), elements);
        CUDA::Memory::copy(device_in.get(), shape.x, device_padded.get(), device_padded.pitch(), shape);
        CUDA::Memory::copy(device_padded.get(), device_padded.pitch(), device_out.get(), shape.x, shape);
        CUDA::Memory::copy(device_out.get(), host_out.get(), elements);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }
}

TEMPLATE_TEST_CASE("CUDA::Memory, asynchronous transfers", "[noa][cuda]",
                   int32_t, uint32_t, float, cfloat_t, double, cdouble_t) {
    Test::IntRandomizer<size_t> randomizer(2, 255);
    Test::IntRandomizer<size_t> randomizer_small(2, 128);

    CUDA::Stream stream(CUDA::Stream::CONCURRENT);

    AND_THEN("host > device > host") {
        size_t elements = randomizer.get();
        Memory::PtrHost<TestType> host_in(elements);
        Memory::PtrHost<TestType> host_out(elements);
        CUDA::Memory::PtrDevice<TestType> device(elements);

        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);

        CUDA::Memory::copy(host_in.get(), device.get(), elements, stream);
        CUDA::Memory::copy(device.get(), host_out.get(), elements, stream);
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > device > pinned > host") {
        size_t elements = randomizer.get();
        Memory::PtrHost<TestType> host_in(elements);
        Memory::PtrHost<TestType> host_out(elements);
        CUDA::Memory::PtrPinned<TestType> pinned(elements);
        CUDA::Memory::PtrDevice<TestType> device(elements);
        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);

        CUDA::Memory::copy(host_in.get(), pinned.get(), elements, stream);
        CUDA::Memory::copy(pinned.get(), device.get(), elements, stream);
        CUDA::Stream::synchronize(stream);
        std::memset(static_cast<void*>(pinned.get()), 0, elements * sizeof(TestType)); // Erase pinned to make sure the transfer works
        CUDA::Memory::copy(device.get(), pinned.get(), elements, stream);
        CUDA::Memory::copy(pinned.get(), host_out.get(), elements, stream);
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > devicePadded > host") {
        size3_t shape{randomizer.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = getElements(shape);
        Memory::PtrHost<TestType> host_in(elements);
        Memory::PtrHost<TestType> host_out(elements);
        CUDA::Memory::PtrDevicePadded<TestType> device(shape);

        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);

        CUDA::Memory::copy(host_in.get(), shape.x, device.get(), device.pitch(), shape, stream);
        CUDA::Memory::copy(device.get(), device.pitch(), host_out.get(), shape.x, shape, stream);
        CUDA::Stream::synchronize(stream);

        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > devicePadded > pinned > host") {
        size3_t shape{randomizer.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = getElements(shape);
        Memory::PtrHost<TestType> host_in(elements);
        Memory::PtrHost<TestType> host_out(elements);
        CUDA::Memory::PtrPinned<TestType> pinned(elements);
        CUDA::Memory::PtrDevicePadded<TestType> device(shape);

        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        CUDA::Memory::copy(host_in.get(), pinned.get(), elements, stream);
        CUDA::Memory::copy(pinned.get(), shape.x, device.get(), device.pitch(), shape, stream);
        CUDA::Stream::synchronize(stream);
        std::memset(static_cast<void*>(pinned.get()), 0, elements * sizeof(TestType)); // Erase pinned to make sure the transfer works
        CUDA::Memory::copy(device.get(), device.pitch(), pinned.get(), shape.x, shape, stream);
        CUDA::Memory::copy(pinned.get(), host_out.get(), elements, stream);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > device > devicePadded > device > host") {
        size3_t shape{randomizer.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = getElements(shape);
        Memory::PtrHost<TestType> host_in(elements);
        Memory::PtrHost<TestType> host_out(elements);
        CUDA::Memory::PtrDevice<TestType> device(elements);
        CUDA::Memory::PtrDevicePadded<TestType> device_padded(shape);

        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        CUDA::Memory::copy(host_in.get(), device.get(), elements, stream);
        CUDA::Memory::copy(device.get(), shape.x, device_padded.get(), device_padded.pitch(), shape, stream);
        REQUIRE(cudaMemsetAsync(device.get(), 0, elements * sizeof(TestType), stream.id()) == cudaSuccess);
        CUDA::Memory::copy(device_padded.get(), device_padded.pitch(), device.get(), shape.x, shape, stream);
        CUDA::Memory::copy(device.get(), host_out.get(), elements, stream);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }
}

TEMPLATE_TEST_CASE("CUDA::Memory, synchronous transfers - CUDA arrays", "[noa][cuda]",
                   int32_t, uint32_t, float, cfloat_t) {
    Test::IntRandomizer<size_t> randomizer(2, 255);
    Test::IntRandomizer<size_t> randomizer_small(2, 128);
    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = Test::getRandomShape(ndim);
    size_t elements = getElements(shape);

    AND_THEN("host > CUDA array > host") {
        Memory::PtrHost<TestType> host_in(elements);
        Memory::PtrHost<TestType> host_out(elements);
        CUDA::Memory::PtrArray<TestType> array(shape);
        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        CUDA::Memory::copy(host_in.get(), array.get(), shape);
        CUDA::Memory::copy(array.get(), host_out.get(), shape);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > device > CUDA array > device > host") {
        Memory::PtrHost<TestType> host_in(elements);
        Memory::PtrHost<TestType> host_out(elements);
        CUDA::Memory::PtrDevice<TestType> device(elements);
        CUDA::Memory::PtrArray<TestType> array(shape);
        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        CUDA::Memory::copy(host_in.get(), device.get(), elements);
        CUDA::Memory::copy(device.get(), array.get(), shape);
        REQUIRE(cudaMemset(device.get(), 0, elements * sizeof(TestType)) == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        CUDA::Memory::copy(array.get(), device.get(), shape);
        CUDA::Memory::copy(device.get(), host_out.get(), elements);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > CUDA array > pinned > host") {
        CUDA::Memory::PtrPinned<TestType> host_in(elements);
        CUDA::Memory::PtrPinned<TestType> host_out(elements);
        CUDA::Memory::PtrArray<TestType> array(shape);
        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        CUDA::Memory::copy(host_in.get(), array.get(), shape);
        CUDA::Memory::copy(array.get(), host_out.get(), shape);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), host_out.elements());
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > devicePadded > CUDA array > devicePadded > host") {
        CUDA::Memory::PtrPinned<TestType> host_in(elements);
        CUDA::Memory::PtrPinned<TestType> host_out(elements);
        CUDA::Memory::PtrDevicePadded<TestType> device_padded(shape);
        CUDA::Memory::PtrArray<TestType> array(shape);
        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        CUDA::Memory::copy(host_in.get(), shape.x, device_padded.get(), device_padded.pitch(), shape);
        CUDA::Memory::copy(device_padded.get(), device_padded.pitch(), array.get(), shape);
        REQUIRE(cudaMemset(device_padded.get(), 0, device_padded.bytesPadded()) == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        CUDA::Memory::copy(array.get(), device_padded.get(), device_padded.pitch(), shape);
        CUDA::Memory::copy(device_padded.get(), device_padded.pitch(), host_out.get(), shape.x, shape);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }
}

TEMPLATE_TEST_CASE("CUDA::Memory, asynchronous transfers - CUDA arrays", "[noa][cuda]",
                   int32_t, uint32_t, float, cfloat_t) {
    Test::IntRandomizer<size_t> randomizer(2, 255);
    Test::IntRandomizer<size_t> randomizer_small(2, 128);
    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = Test::getRandomShape(ndim);
    size_t elements = getElements(shape);
    CUDA::Stream stream(CUDA::Stream::SERIAL);

    AND_THEN("host > CUDA array > host") {
        Memory::PtrHost<TestType> host_in(elements);
        Memory::PtrHost<TestType> host_out(elements);
        CUDA::Memory::PtrArray<TestType> array(shape);
        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        CUDA::Memory::copy(host_in.get(), array.get(), shape, stream);
        CUDA::Memory::copy(array.get(), host_out.get(), shape, stream);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > device > CUDA array > device > host") {
        Memory::PtrHost<TestType> host_in(elements);
        Memory::PtrHost<TestType> host_out(elements);
        CUDA::Memory::PtrDevice<TestType> device(elements);
        CUDA::Memory::PtrArray<TestType> array(shape);
        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        CUDA::Memory::copy(host_in.get(), device.get(), elements, stream);
        CUDA::Memory::copy(device.get(), array.get(), shape, stream);
        REQUIRE(cudaMemsetAsync(device.get(), 0, elements * sizeof(TestType), stream.get()) == cudaSuccess);
        CUDA::Memory::copy(array.get(), device.get(), shape, stream);
        CUDA::Memory::copy(device.get(), host_out.get(), elements, stream);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > CUDA array > pinned > host") {
        CUDA::Memory::PtrPinned<TestType> host_in(elements);
        CUDA::Memory::PtrPinned<TestType> host_out(elements);
        CUDA::Memory::PtrArray<TestType> array(shape);
        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        CUDA::Memory::copy(host_in.get(), array.get(), shape, stream);
        CUDA::Memory::copy(array.get(), host_out.get(), shape, stream);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), host_out.elements());
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > devicePadded > CUDA array > devicePadded > host") {
        CUDA::Memory::PtrPinned<TestType> host_in(elements);
        CUDA::Memory::PtrPinned<TestType> host_out(elements);
        CUDA::Memory::PtrDevicePadded<TestType> device_padded(shape);
        CUDA::Memory::PtrArray<TestType> array(shape);
        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        CUDA::Memory::copy(host_in.get(), shape.x, device_padded.get(), device_padded.pitch(), shape, stream);
        CUDA::Memory::copy(device_padded.get(), device_padded.pitch(), array.get(), shape, stream);
        REQUIRE(cudaMemsetAsync(device_padded.get(), 0, device_padded.bytesPadded(), stream.get()) == cudaSuccess);
        CUDA::Memory::copy(array.get(), device_padded.get(), device_padded.pitch(), shape, stream);
        CUDA::Memory::copy(device_padded.get(), device_padded.pitch(), host_out.get(), shape.x, shape, stream);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }
}
