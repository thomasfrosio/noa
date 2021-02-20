#include <noa/gpu/cuda/Memory.h>

#include <noa/cpu/PtrHost.h>
#include <noa/gpu/cuda/PtrDevice.h>
#include <noa/gpu/cuda/PtrPinned.h>
#include <noa/gpu/cuda/PtrDevicePadded.h>
#include <noa/gpu/cuda/PtrArray.h>
#include <noa/gpu/cuda/Stream.h>

#include <catch2/catch.hpp>
#include "../../../Helpers.h"

using namespace ::Noa;

template<typename T>
void initHostDataRandom(T* data, size_t elements, Test::IntRandomizer<size_t>& randomizer) {
    if constexpr (std::is_same_v<T, cfloat_t>) {
        for (size_t idx{0}; idx < elements; ++idx) {
            data[idx] = T{static_cast<float>(randomizer.get())};
        }
    } else if constexpr (std::is_same_v<T, cdouble_t>) {
        for (size_t idx{0}; idx < elements; ++idx) {
            data[idx] = T{static_cast<double>(randomizer.get())};
        }
    } else {
        for (size_t idx{0}; idx < elements; ++idx)
            data[idx] = static_cast<T>(randomizer.get());
    }
}

template<typename T>
void initHostDataZero(T* data, size_t elements) {
    for (size_t idx{0}; idx < elements; ++idx)
        data[idx] = 0;
}

template<typename T>
T getDifference(const T* in, const T* out, size_t elements) {
    T diff{0};
    for (size_t idx{0}; idx < elements; ++idx)
        diff += out[idx] - in[idx];
    return diff;
}

TEMPLATE_TEST_CASE("CUDA::Memory, synchronous transfers", "[noa][cuda]",
                   int32_t, uint32_t, float, cfloat_t, double, cdouble_t) {

    using namespace CUDA;
    Test::IntRandomizer<size_t> randomizer(1, 255);
    Test::IntRandomizer<size_t> randomizer_small(1, 128);

    AND_THEN("host > device > host") {
        size_t elements = randomizer.get();
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrDevice<TestType> device(elements);
        initHostDataRandom(host_in.get(), elements, randomizer);
        initHostDataZero(host_out.get(), elements);
        Memory::copy(device.get(), host_in.get(), host_in.bytes());
        Memory::copy(host_out.get(), device.get(), device.bytes());
        TestType diff = getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > device > pinned > host") {
        size_t elements = randomizer.get();
        size_t bytes = elements * sizeof(TestType);
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrPinned<TestType> pinned(elements);
        PtrDevice<TestType> device(elements);
        initHostDataRandom(host_in.get(), elements, randomizer);
        initHostDataZero(host_out.get(), elements);

        Memory::copy(pinned.get(), host_in.get(), bytes);
        Memory::copy(device.get(), pinned.get(), bytes);
        std::memset(static_cast<void*>(pinned.get()), 0, bytes); // Erase pinned to make sure the transfer works
        Memory::copy(pinned.get(), device.get(), bytes);
        Memory::copy(host_out.get(), pinned.get(), bytes);

        TestType diff = getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > devicePadded > host") {
        size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = Math::elements(shape);
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrDevicePadded<TestType> device(shape);

        initHostDataRandom(host_in.get(), elements, randomizer);
        initHostDataZero(host_out.get(), elements);

        Memory::copy(&device, host_in.get());
        Memory::copy(host_out.get(), &device);

        TestType diff = getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > devicePadded > pinned > host") {
        size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = Math::elements(shape);
        size_t bytes = elements * sizeof(TestType);
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrPinned<TestType> pinned(elements);
        PtrDevicePadded<TestType> device(shape);

        initHostDataRandom(host_in.get(), elements, randomizer);
        initHostDataZero(host_out.get(), elements);
        Memory::copy(pinned.get(), host_in.get(), bytes);
        Memory::copy(&device, pinned.get());
        std::memset(static_cast<void*>(pinned.get()), 0, bytes); // Erase pinned to make sure the transfer works
        Memory::copy(pinned.get(), &device);
        Memory::copy(host_out.get(), pinned.get(), bytes);
        TestType diff = getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > device > devicePadded > device > host") {
        size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = Math::elements(shape);
        size_t bytes = elements * sizeof(TestType);
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrDevice<TestType> device_in(elements);
        PtrDevice<TestType> device_out(elements);
        PtrDevicePadded<TestType> device_padded(shape);

        initHostDataRandom(host_in.get(), elements, randomizer);
        initHostDataZero(host_out.get(), elements);
        Memory::copy(device_in.get(), host_in.get(), bytes);
        Memory::copy(&device_padded, device_in.get());
        Memory::copy(device_out.get(), &device_padded);
        Memory::copy(host_out.get(), device_out.get(), bytes);
        TestType diff = getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }
}

TEMPLATE_TEST_CASE("CUDA::Memory, asynchronous transfers", "[noa][cuda]",
                   int32_t, uint32_t, float, cfloat_t, double, cdouble_t) {

    using namespace CUDA;
    Test::IntRandomizer<size_t> randomizer(1, 255);
    Test::IntRandomizer<size_t> randomizer_small(1, 128);

    Stream stream(Stream::concurrent);

    AND_THEN("host > device > host") {
        size_t elements = randomizer.get();
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrDevice<TestType> device(elements);

        initHostDataRandom(host_in.get(), elements, randomizer);
        initHostDataZero(host_out.get(), elements);

        Memory::copy(device.get(), host_in.get(), host_in.bytes(), stream);
        Memory::copy(host_out.get(), device.get(), device.bytes(), stream);
        Stream::synchronize(stream);

        TestType diff = getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > device > pinned > host") {
        size_t elements = randomizer.get();
        size_t bytes = elements * sizeof(TestType);
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrPinned<TestType> pinned(elements);
        PtrDevice<TestType> device(elements);

        initHostDataRandom(host_in.get(), elements, randomizer);
        initHostDataZero(host_out.get(), elements);

        Memory::copy(pinned.get(), host_in.get(), bytes, stream);
        Memory::copy(device.get(), pinned.get(), bytes, stream);
        Stream::synchronize(stream);
        std::memset(static_cast<void*>(pinned.get()), 0, bytes); // Erase pinned to make sure the transfer works
        Memory::copy(pinned.get(), device.get(), bytes, stream);
        Memory::copy(host_out.get(), pinned.get(), bytes, stream);
        Stream::synchronize(stream);
        TestType diff = getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > devicePadded > host") {
        size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = Math::elements(shape);
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrDevicePadded<TestType> device(shape);

        initHostDataRandom(host_in.get(), elements, randomizer);
        initHostDataZero(host_out.get(), elements);

        Memory::copy(&device, host_in.get(), stream);
        Memory::copy(host_out.get(), &device, stream);
        Stream::synchronize(stream);

        TestType diff = getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > pinned > devicePadded > pinned > host") {
        size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = Math::elements(shape);
        size_t bytes = elements * sizeof(TestType);
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrPinned<TestType> pinned(elements);
        PtrDevicePadded<TestType> device(shape);

        initHostDataRandom(host_in.get(), elements, randomizer);
        initHostDataZero(host_out.get(), elements);
        Memory::copy(pinned.get(), host_in.get(), bytes, stream);
        Memory::copy(&device, pinned.get(), stream);
        Stream::synchronize(stream);
        std::memset(static_cast<void*>(pinned.get()), 0, bytes); // Erase pinned to make sure the transfer works
        Memory::copy(pinned.get(), &device, stream);
        Memory::copy(host_out.get(), pinned.get(), bytes, stream);
        Stream::synchronize(stream);
        TestType diff = getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("host > device > devicePadded > device > host") {
        size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
        size_t elements = Math::elements(shape);
        size_t bytes = elements * sizeof(TestType);
        PtrHost<TestType> host_in(elements);
        PtrHost<TestType> host_out(elements);
        PtrDevice<TestType> device(elements);
        PtrDevicePadded<TestType> device_padded(shape);

        initHostDataRandom(host_in.get(), elements, randomizer);
        initHostDataZero(host_out.get(), elements);
        Memory::copy(device.get(), host_in.get(), bytes, stream);
        Memory::copy(&device_padded, device.get(), stream);
        REQUIRE(cudaMemsetAsync(device.get(), 0, bytes, stream.id()) == cudaSuccess);
        Memory::copy(device.get(), &device_padded, stream);
        Memory::copy(host_out.get(), device.get(), bytes, stream);
        Stream::synchronize(stream);
        TestType diff = getDifference(host_in.get(), host_out.get(), elements);
        REQUIRE(diff == TestType{0});
    }
}

// There's probably a better way to generate all combinations of types and ndim...
TEMPLATE_TEST_CASE("CUDA::Memory, synchronous transfers - CUDA arrays", "[noa][cuda]",
                   int32_t, uint32_t, float, cfloat_t) {
    using namespace CUDA;
    Test::IntRandomizer<size_t> randomizer(1, 255);
    Test::IntRandomizer<size_t> randomizer_small(1, 128);

    AND_THEN("host > CUDA array > host") {
        #define NOA_TEST_PERFORM_COPY()                             \
        initHostDataRandom(host_in.get(), elements, randomizer);    \
        initHostDataZero(host_out.get(), elements);                 \
        Memory::copy(&array, host_in.get());                        \
        Memory::copy(host_out.get(), &array)

        AND_THEN("1D") {
            size_t elements = randomizer.get();
            PtrHost<TestType> host_in(elements);
            PtrHost<TestType> host_out(elements);
            PtrArray<TestType, 1> array(elements);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), elements);
            REQUIRE(diff == TestType{0});
        }

        AND_THEN("2D") {
            size2_t shape{randomizer.get(), randomizer.get()};
            size_t elements = Math::elements(shape);
            PtrHost<TestType> host_in(elements);
            PtrHost<TestType> host_out(elements);
            PtrArray<TestType, 2> array(shape);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), elements);
            REQUIRE(diff == TestType{0});
        }

        AND_THEN("3D") {
            size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
            size_t elements = Math::elements(shape);
            PtrHost<TestType> host_in(elements);
            PtrHost<TestType> host_out(elements);
            PtrArray<TestType, 3> array(shape);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), elements);
            REQUIRE(diff == TestType{0});
        }
        #undef NOA_TEST_PERFORM_COPY
    }

    AND_THEN("host > device > CUDA array > device > host") {
        #define NOA_TEST_PERFORM_COPY()                             \
        initHostDataRandom(host_in.get(), elements, randomizer);    \
        initHostDataZero(host_out.get(), elements);                 \
        Memory::copy(device.get(), host_in.get(), bytes);           \
        Memory::copy(&array, device.get());                         \
        REQUIRE(cudaMemset(device.get(), 0, bytes) == cudaSuccess); \
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);            \
        Memory::copy(device.get(), &array);                         \
        Memory::copy(host_out.get(), device.get(), bytes)

        AND_THEN("1D") {
            size_t elements = randomizer.get();
            size_t bytes = elements * sizeof(TestType);
            PtrHost<TestType> host_in(elements);
            PtrHost<TestType> host_out(elements);
            PtrDevice<TestType> device(elements);
            PtrArray<TestType, 1> array(elements);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), elements);
            REQUIRE(diff == TestType{0});
        }

        AND_THEN("2D") {
            size2_t shape{randomizer.get(), randomizer.get()};
            size_t elements = Math::elements(shape);
            size_t bytes = elements * sizeof(TestType);
            PtrHost<TestType> host_in(elements);
            PtrHost<TestType> host_out(elements);
            PtrDevice<TestType> device(elements);
            PtrArray<TestType, 2> array(shape);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), elements);
            REQUIRE(diff == TestType{0});
        }

        AND_THEN("3D") {
            size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
            size_t elements = Math::elements(shape);
            size_t bytes = elements * sizeof(TestType);
            PtrHost<TestType> host_in(elements);
            PtrHost<TestType> host_out(elements);
            PtrDevice<TestType> device(elements);
            PtrArray<TestType, 3> array(shape);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), elements);
            REQUIRE(diff == TestType{0});
        }
        #undef NOA_TEST_PERFORM_COPY
    }

    AND_THEN("host > pinned > CUDA array > pinned > host") {
        #define NOA_TEST_PERFORM_COPY()                             \
        initHostDataRandom(host_in.get(), elements, randomizer);    \
        initHostDataZero(host_out.get(), elements);                 \
        Memory::copy(&array, host_in.get());                        \
        Memory::copy(host_out.get(), &array)

        AND_THEN("1D") {
            size_t elements = randomizer.get();
            PtrPinned<TestType> host_in(elements);
            PtrPinned<TestType> host_out(elements);
            PtrArray<TestType, 1> array(elements);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), host_out.elements());
            REQUIRE(diff == TestType{0});
        }

        AND_THEN("2D") {
            size2_t shape{randomizer.get(), randomizer.get()};
            size_t elements = Math::elements(shape);
            PtrPinned<TestType> host_in(elements);
            PtrPinned<TestType> host_out(elements);
            PtrArray<TestType, 2> array(shape);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), host_out.elements());
            REQUIRE(diff == TestType{0});
        }

        AND_THEN("3D") {
            size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
            size_t elements = Math::elements(shape);
            PtrPinned<TestType> host_in(elements);
            PtrPinned<TestType> host_out(elements);
            PtrArray<TestType, 3> array(shape);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), host_out.elements());
            REQUIRE(diff == TestType{0});
        }
        #undef NOA_TEST_PERFORM_COPY
    }

    AND_THEN("host > devicePadded > CUDA array > devicePadded > host") {
        #define NOA_TEST_PERFORM_COPY()                                             \
        initHostDataRandom(host_in.get(), elements, randomizer);                    \
        initHostDataZero(host_out.get(), elements);                                 \
        Memory::copy(&device, host_in.get());                                       \
        Memory::copy(&array, &device);                                              \
        REQUIRE(cudaMemset(device.get(), 0, device.bytesPadded()) == cudaSuccess);  \
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);                            \
        Memory::copy(&device, &array);                                              \
        Memory::copy(host_out.get(), &device)

        AND_THEN("1D") {
            size_t elements = randomizer.get();
            PtrPinned<TestType> host_in(elements);
            PtrPinned<TestType> host_out(elements);
            PtrDevicePadded<TestType> device(size3_t{elements, 1, 1});
            PtrArray<TestType, 1> array(elements);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), host_out.elements());
            REQUIRE(diff == TestType{0});
        }

        AND_THEN("2D") {
            size2_t shape{randomizer.get(), randomizer.get()};
            size_t elements = Math::elements(shape);
            PtrPinned<TestType> host_in(elements);
            PtrPinned<TestType> host_out(elements);
            PtrDevicePadded<TestType> device(size3_t{shape.x, shape.y, 1});
            PtrArray<TestType, 2> array(shape);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), host_out.elements());
            REQUIRE(diff == TestType{0});
        }

        AND_THEN("3D") {
            size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
            size_t elements = Math::elements(shape);
            PtrPinned<TestType> host_in(elements);
            PtrPinned<TestType> host_out(elements);
            PtrDevicePadded<TestType> device(shape);
            PtrArray<TestType, 3> array(shape);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), host_out.elements());
            REQUIRE(diff == TestType{0});
        }
        #undef NOA_TEST_PERFORM_COPY
    }
}

// There's probably a better way to generate all combinations of types and ndim...
TEMPLATE_TEST_CASE("CUDA::Memory, asynchronous transfers - CUDA arrays", "[noa][cuda]",
                   int32_t, uint32_t, float, cfloat_t) {
    using namespace CUDA;
    Test::IntRandomizer<size_t> randomizer(1, 255);
    Test::IntRandomizer<size_t> randomizer_small(1, 128);
    Stream stream(Stream::concurrent);

    AND_THEN("host > CUDA array > host") {
        #define NOA_TEST_PERFORM_COPY()                             \
        initHostDataRandom(host_in.get(), elements, randomizer);    \
        initHostDataZero(host_out.get(), elements);                 \
        Memory::copy(&array, host_in.get(), stream);                \
        Memory::copy(host_out.get(), &array, stream);               \
        Stream::synchronize(stream)

        AND_THEN("1D") {
            size_t elements = randomizer.get();
            PtrHost<TestType> host_in(elements);
            PtrHost<TestType> host_out(elements);
            PtrArray<TestType, 1> array(elements);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), elements);
            REQUIRE(diff == TestType{0});
        }

        AND_THEN("2D") {
            size2_t shape{randomizer.get(), randomizer.get()};
            size_t elements = Math::elements(shape);
            PtrHost<TestType> host_in(elements);
            PtrHost<TestType> host_out(elements);
            PtrArray<TestType, 2> array(shape);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), elements);
            REQUIRE(diff == TestType{0});
        }

        AND_THEN("3D") {
            size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
            size_t elements = Math::elements(shape);
            PtrHost<TestType> host_in(elements);
            PtrHost<TestType> host_out(elements);
            PtrArray<TestType, 3> array(shape);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), elements);
            REQUIRE(diff == TestType{0});
        }
        #undef NOA_TEST_PERFORM_COPY
    }

    AND_THEN("host > device > CUDA array > device > host") {
        #define NOA_TEST_PERFORM_COPY()                                                 \
        initHostDataRandom(host_in.get(), elements, randomizer);                        \
        initHostDataZero(host_out.get(), elements);                                     \
        Memory::copy(device.get(), host_in.get(), bytes, stream);                       \
        Memory::copy(&array, device.get(), stream);                                     \
        REQUIRE(cudaMemsetAsync(device.get(), 0, bytes, stream.id()) == cudaSuccess);   \
        Memory::copy(device.get(), &array, stream);                                     \
        Memory::copy(host_out.get(), device.get(), bytes, stream);                      \
        Stream::synchronize(stream)

        AND_THEN("1D") {
            size_t elements = randomizer.get();
            size_t bytes = elements * sizeof(TestType);
            PtrHost<TestType> host_in(elements);
            PtrHost<TestType> host_out(elements);
            PtrDevice<TestType> device(elements);
            PtrArray<TestType, 1> array(elements);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), elements);
            REQUIRE(diff == TestType{0});
        }

        AND_THEN("2D") {
            size2_t shape{randomizer.get(), randomizer.get()};
            size_t elements = Math::elements(shape);
            size_t bytes = elements * sizeof(TestType);
            PtrHost<TestType> host_in(elements);
            PtrHost<TestType> host_out(elements);
            PtrDevice<TestType> device(elements);
            PtrArray<TestType, 2> array(shape);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), elements);
            REQUIRE(diff == TestType{0});
        }

        AND_THEN("3D") {
            size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
            size_t elements = Math::elements(shape);
            size_t bytes = elements * sizeof(TestType);
            PtrHost<TestType> host_in(elements);
            PtrHost<TestType> host_out(elements);
            PtrDevice<TestType> device(elements);
            PtrArray<TestType, 3> array(shape);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), elements);
            REQUIRE(diff == TestType{0});
        }
        #undef NOA_TEST_PERFORM_COPY
    }

    AND_THEN("host > pinned > CUDA array > pinned > host") {
        #define NOA_TEST_PERFORM_COPY()                             \
        initHostDataRandom(host_in.get(), elements, randomizer);    \
        initHostDataZero(host_out.get(), elements);                 \
        Memory::copy(&array, host_in.get(), stream);                \
        Memory::copy(host_out.get(), &array, stream);               \
        Stream::synchronize(stream)

        AND_THEN("1D") {
            size_t elements = randomizer.get();
            PtrPinned<TestType> host_in(elements);
            PtrPinned<TestType> host_out(elements);
            PtrArray<TestType, 1> array(elements);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), host_out.elements());
            REQUIRE(diff == TestType{0});
        }

        AND_THEN("2D") {
            size2_t shape{randomizer.get(), randomizer.get()};
            size_t elements = Math::elements(shape);
            PtrPinned<TestType> host_in(elements);
            PtrPinned<TestType> host_out(elements);
            PtrArray<TestType, 2> array(shape);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), host_out.elements());
            REQUIRE(diff == TestType{0});
        }

        AND_THEN("3D") {
            size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
            size_t elements = Math::elements(shape);
            PtrPinned<TestType> host_in(elements);
            PtrPinned<TestType> host_out(elements);
            PtrArray<TestType, 3> array(shape);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), host_out.elements());
            REQUIRE(diff == TestType{0});
        }
        #undef NOA_TEST_PERFORM_COPY
    }

    AND_THEN("host > devicePadded > CUDA array > devicePadded > host") {
        #define NOA_TEST_PERFORM_COPY()                                                                 \
        initHostDataRandom(host_in.get(), elements, randomizer);                                        \
        initHostDataZero(host_out.get(), elements);                                                     \
        Memory::copy(&device, host_in.get(), stream);                                                   \
        Memory::copy(&array, &device, stream);                                                          \
        REQUIRE(cudaMemsetAsync(device.get(), 0, device.bytesPadded(), stream.id()) == cudaSuccess);    \
        Memory::copy(&device, &array, stream);                                                          \
        Memory::copy(host_out.get(), &device, stream);                                                  \
        Stream::synchronize(stream)

        AND_THEN("1D") {
            size_t elements = randomizer.get();
            PtrPinned<TestType> host_in(elements);
            PtrPinned<TestType> host_out(elements);
            PtrDevicePadded<TestType> device(size3_t{elements, 1, 1});
            PtrArray<TestType, 1> array(elements);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), host_out.elements());
            REQUIRE(diff == TestType{0});
        }

        AND_THEN("2D") {
            size2_t shape{randomizer.get(), randomizer.get()};
            size_t elements = Math::elements(shape);
            PtrPinned<TestType> host_in(elements);
            PtrPinned<TestType> host_out(elements);
            PtrDevicePadded<TestType> device(size3_t{shape.x, shape.y, 1});
            PtrArray<TestType, 2> array(shape);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), host_out.elements());
            REQUIRE(diff == TestType{0});
        }

        AND_THEN("3D") {
            size3_t shape{randomizer_small.get(), randomizer_small.get(), randomizer_small.get()};
            size_t elements = Math::elements(shape);
            PtrPinned<TestType> host_in(elements);
            PtrPinned<TestType> host_out(elements);
            PtrDevicePadded<TestType> device(shape);
            PtrArray<TestType, 3> array(shape);
            NOA_TEST_PERFORM_COPY();
            TestType diff = getDifference(host_in.get(), host_out.get(), host_out.elements());
            REQUIRE(diff == TestType{0});
        }
        #undef NOA_TEST_PERFORM_COPY
    }
}
