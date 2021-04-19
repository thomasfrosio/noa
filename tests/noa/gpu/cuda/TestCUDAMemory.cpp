#include <noa/gpu/cuda/Memory.h>

#include <noa/cpu/PtrHost.h>
#include <noa/gpu/cuda/PtrDevice.h>
#include <noa/gpu/cuda/PtrPinned.h>
#include <noa/gpu/cuda/PtrDevicePadded.h>
#include <noa/gpu/cuda/PtrArray.h>
#include <noa/gpu/cuda/util/Stream.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::Noa;

TEMPLATE_TEST_CASE("CUDA::Memory, synchronous transfers", "[noa][cuda]",
                   int32_t, uint32_t, float, cfloat_t, double, cdouble_t) {

    using namespace CUDA;
    Test::IntRandomizer<size_t> randomizer(2, 255);
    Test::IntRandomizer<size_t> randomizer_small(2, 128);

    AND_THEN("host > device > host") {
        size_t elements = randomizer.get();
        size_t bytes = elements * sizeof(TestType);
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrDevice<TestType> device(elements);
        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        Memory::copy(host_in.get(), device.get(), bytes);
        Memory::copy(device.get(), host_out.get(), bytes);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > device > pinned > host") {
        size_t elements = randomizer.get();
        size_t bytes = elements * sizeof(TestType);
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrPinned<TestType> pinned(elements);
        PtrDevice<TestType> device(elements);
        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);

        Memory::copy(host_in.get(), pinned.get(), bytes);
        Memory::copy(pinned.get(), device.get(), bytes);
        std::memset(static_cast<void*>(pinned.get()), 0, bytes); // Erase pinned to make sure the transfer works
        Memory::copy(device.get(), pinned.get(), bytes);
        Memory::copy(pinned.get(), host_out.get(), bytes);

        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > devicePadded > host") {
        size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = getElements(shape);
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrDevicePadded<TestType> device(shape);

        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);

        Memory::copy(host_in.get(), shape.x * sizeof(TestType), device.get(), device.pitch(), shape);
        Memory::copy(device.get(), device.pitch(), host_out.get(), shape.x * sizeof(TestType), shape);

        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > devicePadded > pinned > host") {
        size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = getElements(shape);
        size_t bytes = elements * sizeof(TestType);
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrPinned<TestType> pinned(elements);
        PtrDevicePadded<TestType> device(shape);

        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        Memory::copy(host_in.get(), pinned.get(), bytes);
        Memory::copy(pinned.get(), shape.x * sizeof(TestType), device.get(), device.pitch(), shape);
        std::memset(static_cast<void*>(pinned.get()), 0, bytes); // Erase pinned to make sure the transfer works
        Memory::copy(device.get(), device.pitch(), pinned.get(), shape.x * sizeof(TestType), shape);
        Memory::copy(pinned.get(), host_out.get(), bytes);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > device > devicePadded > device > host") {
        size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = getElements(shape);
        size_t bytes = elements * sizeof(TestType);
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrDevice<TestType> device_in(elements);
        PtrDevice<TestType> device_out(elements);
        PtrDevicePadded<TestType> device_padded(shape);

        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        Memory::copy(host_in.get(), device_in.get(), bytes);
        Memory::copy(device_in.get(), shape.x * sizeof(TestType), device_padded.get(), device_padded.pitch(), shape);
        Memory::copy(device_padded.get(), device_padded.pitch(), device_out.get(), shape.x * sizeof(TestType), shape);
        Memory::copy(device_out.get(), host_out.get(), bytes);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }
}

TEMPLATE_TEST_CASE("CUDA::Memory, asynchronous transfers", "[noa][cuda]",
                   int32_t, uint32_t, float, cfloat_t, double, cdouble_t) {

    using namespace CUDA;
    Test::IntRandomizer<size_t> randomizer(2, 255);
    Test::IntRandomizer<size_t> randomizer_small(2, 128);

    Stream stream(Stream::CONCURRENT);

    AND_THEN("host > device > host") {
        size_t elements = randomizer.get();
        size_t bytes = elements * sizeof(TestType);
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrDevice<TestType> device(elements);

        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);

        Memory::copy(host_in.get(), device.get(), bytes, stream);
        Memory::copy(device.get(), host_out.get(), bytes, stream);
        Stream::synchronize(stream);

        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > device > pinned > host") {
        size_t elements = randomizer.get();
        size_t bytes = elements * sizeof(TestType);
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrPinned<TestType> pinned(elements);
        PtrDevice<TestType> device(elements);
        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);

        Memory::copy(host_in.get(), pinned.get(), bytes, stream);
        Memory::copy(pinned.get(), device.get(), bytes, stream);
        Stream::synchronize(stream);
        std::memset(static_cast<void*>(pinned.get()), 0, bytes); // Erase pinned to make sure the transfer works
        Memory::copy(device.get(), pinned.get(), bytes, stream);
        Memory::copy(pinned.get(), host_out.get(), bytes, stream);
        Stream::synchronize(stream);

        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > devicePadded > host") {
        size3_t shape{randomizer.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = getElements(shape);
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrDevicePadded<TestType> device(shape);

        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);

        Memory::copy(host_in.get(), shape.x * sizeof(TestType), device.get(), device.pitch(), shape, stream);
        Memory::copy(device.get(), device.pitch(), host_out.get(), shape.x * sizeof(TestType), shape, stream);
        Stream::synchronize(stream);

        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > devicePadded > pinned > host") {
        size3_t shape{randomizer.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = getElements(shape);
        size_t bytes = elements * sizeof(TestType);
        size_t pitch = shape.x * sizeof(TestType);
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrPinned<TestType> pinned(elements);
        PtrDevicePadded<TestType> device(shape);

        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        Memory::copy(host_in.get(), pinned.get(), bytes, stream);
        Memory::copy(pinned.get(), pitch, device.get(), device.pitch(), shape, stream);
        Stream::synchronize(stream);
        std::memset(static_cast<void*>(pinned.get()), 0, bytes); // Erase pinned to make sure the transfer works
        Memory::copy(device.get(), device.pitch(), pinned.get(), pitch, shape, stream);
        Memory::copy(pinned.get(), host_out.get(), bytes, stream);
        Stream::synchronize(stream);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > device > devicePadded > device > host") {
        size3_t shape{randomizer.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = getElements(shape);
        size_t bytes = elements * sizeof(TestType);
        size_t pitch = shape.x * sizeof(TestType);
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrDevice<TestType> device(elements);
        PtrDevicePadded<TestType> device_padded(shape);

        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        Memory::copy(host_in.get(), device.get(), bytes, stream);
        Memory::copy(device.get(), pitch, device_padded.get(), device_padded.pitch(), shape, stream);
        REQUIRE(cudaMemsetAsync(device.get(), 0, bytes, stream.id()) == cudaSuccess);
        Memory::copy(device_padded.get(), device_padded.pitch(), device.get(), pitch, shape, stream);
        Memory::copy(device.get(), host_out.get(), bytes, stream);
        Stream::synchronize(stream);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }
}

TEMPLATE_TEST_CASE("CUDA::Memory, synchronous transfers - CUDA arrays", "[noa][cuda]",
                   int32_t, uint32_t, float, cfloat_t) {
    using namespace CUDA;
    Test::IntRandomizer<size_t> randomizer(2, 255);
    Test::IntRandomizer<size_t> randomizer_small(2, 128);
    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = Test::getRandomShape(ndim);
    size_t elements = getElements(shape);
    size_t bytes = elements * sizeof(TestType);
    size_t pitch = shape.x * sizeof(TestType);

    AND_THEN("host > CUDA array > host") {
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrArray<TestType> array(shape);
        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        Memory::copy(host_in.get(), array.get(), shape);
        Memory::copy(array.get(), host_out.get(), shape);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > device > CUDA array > device > host") {
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrDevice<TestType> device(elements);
        PtrArray<TestType> array(shape);
        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        Memory::copy(host_in.get(), device.get(), bytes);
        Memory::copy(device.get(), array.get(), shape);
        REQUIRE(cudaMemset(device.get(), 0, bytes) == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        Memory::copy(array.get(), device.get(), shape);
        Memory::copy(device.get(), host_out.get(), bytes);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > CUDA array > pinned > host") {
        PtrPinned<TestType> host_in(elements);
        PtrPinned<TestType> host_out(elements);
        PtrArray<TestType> array(shape);
        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        Memory::copy(host_in.get(), array.get(), shape);
        Memory::copy(array.get(), host_out.get(), shape);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), host_out.elements());
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > devicePadded > CUDA array > devicePadded > host") {
        PtrPinned<TestType> host_in(elements);
        PtrPinned<TestType> host_out(elements);
        PtrDevicePadded<TestType> device_padded(shape);
        PtrArray<TestType> array(shape);
        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        Memory::copy(host_in.get(), pitch, device_padded.get(), device_padded.pitch(), shape);
        Memory::copy(device_padded.get(), device_padded.pitch(), array.get(), shape);
        REQUIRE(cudaMemset(device_padded.get(), 0, device_padded.bytesPadded()) == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        Memory::copy(array.get(), device_padded.get(), device_padded.pitch(), shape);
        Memory::copy(device_padded.get(), device_padded.pitch(), host_out.get(), pitch, shape);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }
}

TEMPLATE_TEST_CASE("CUDA::Memory, asynchronous transfers - CUDA arrays", "[noa][cuda]",
                   int32_t, uint32_t, float, cfloat_t) {
    using namespace CUDA;
    Test::IntRandomizer<size_t> randomizer(2, 255);
    Test::IntRandomizer<size_t> randomizer_small(2, 128);
    uint ndim = GENERATE(1U, 2U, 3U);
    size3_t shape = Test::getRandomShape(ndim);
    size_t elements = getElements(shape);
    size_t bytes = elements * sizeof(TestType);
    size_t pitch = shape.x * sizeof(TestType);

    Stream stream(Stream::SERIAL);

    AND_THEN("host > CUDA array > host") {
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrArray<TestType> array(shape);
        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        Memory::copy(host_in.get(), array.get(), shape, stream);
        Memory::copy(array.get(), host_out.get(), shape, stream);
        Stream::synchronize(stream);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > device > CUDA array > device > host") {
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrDevice<TestType> device(elements);
        PtrArray<TestType> array(shape);
        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        Memory::copy(host_in.get(), device.get(), bytes, stream);
        Memory::copy(device.get(), array.get(), shape, stream);
        REQUIRE(cudaMemsetAsync(device.get(), 0, bytes, stream.get()) == cudaSuccess);
        Memory::copy(array.get(), device.get(), shape, stream);
        Memory::copy(device.get(), host_out.get(), bytes, stream);
        Stream::synchronize(stream);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > CUDA array > pinned > host") {
        PtrPinned<TestType> host_in(elements);
        PtrPinned<TestType> host_out(elements);
        PtrArray<TestType> array(shape);
        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        Memory::copy(host_in.get(), array.get(), shape, stream);
        Memory::copy(array.get(), host_out.get(), shape, stream);
        Stream::synchronize(stream);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), host_out.elements());
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > devicePadded > CUDA array > devicePadded > host") {
        PtrPinned<TestType> host_in(elements);
        PtrPinned<TestType> host_out(elements);
        PtrDevicePadded<TestType> device_padded(shape);
        PtrArray<TestType> array(shape);
        Test::initDataRandom(host_in.get(), elements, randomizer);
        Test::initDataZero(host_out.get(), elements);
        Memory::copy(host_in.get(), pitch, device_padded.get(), device_padded.pitch(), shape, stream);
        Memory::copy(device_padded.get(), device_padded.pitch(), array.get(), shape, stream);
        REQUIRE(cudaMemsetAsync(device_padded.get(), 0, device_padded.bytesPadded(), stream.get()) == cudaSuccess);
        Memory::copy(array.get(), device_padded.get(), device_padded.pitch(), shape, stream);
        Memory::copy(device_padded.get(), device_padded.pitch(), host_out.get(), pitch, shape, stream);
        Stream::synchronize(stream);
        TestType diff = Test::getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }
}
