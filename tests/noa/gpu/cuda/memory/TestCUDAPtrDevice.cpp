#include <cuda_runtime_api.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

// These are very simple tests. PtrDevice will be tested extensively since
// it is a dependency for many parts of the CUDA backend.
TEMPLATE_TEST_CASE("cuda::memory::PtrDevice", "[noa][cuda][memory]",
                   int32_t, uint32_t, int64_t, uint64_t, float, double, cfloat_t, cdouble_t) {
    cuda::memory::PtrDevice<TestType> ptr;
    test::Randomizer<size_t> randomizer(1, 255);

    AND_THEN("copy data to device and back to host") {
        const size_t elements = randomizer.get();
        const size_t bytes = elements * sizeof(TestType);

        // transfer: h_in -> d_inter -> h_out.
        cpu::memory::PtrHost<TestType> h_in(elements);
        cuda::memory::PtrDevice<TestType> d_inter(elements);
        cpu::memory::PtrHost<TestType> h_out(elements);

        test::randomize(h_in.get(), h_in.elements(), randomizer);
        test::memset(h_out.get(), h_out.elements(), 0);

        REQUIRE(cudaMemcpy(d_inter.get(), h_in.get(), bytes, cudaMemcpyDefault) == cudaSuccess);
        REQUIRE(cudaMemcpy(h_out.get(), d_inter.get(), bytes, cudaMemcpyDefault) == cudaSuccess);

        const TestType diff = test::getDifference(h_in.get(), h_out.get(), h_in.elements());
        REQUIRE(diff == TestType{0});
    }

    // test allocation and free
    AND_THEN("allocation, free, ownership") {
        cuda::memory::PtrDevice<TestType> ptr1;
        REQUIRE_FALSE(ptr1);
        const size_t elements = randomizer.get();
        {
            cuda::memory::PtrDevice<TestType> ptr2(elements);
            REQUIRE(ptr2);
            REQUIRE(ptr2.get());
            REQUIRE_FALSE(ptr2.empty());
            REQUIRE(ptr2.elements() == elements);
            REQUIRE(ptr2.bytes() == elements * sizeof(TestType));
            ptr1.reset(ptr2.release(), elements); // transfer ownership.
            REQUIRE_FALSE(ptr2);
            REQUIRE_FALSE(ptr2.get());
            REQUIRE(ptr2.empty());
            REQUIRE_FALSE(ptr2.elements());
            REQUIRE_FALSE(ptr2.bytes());
        }
        REQUIRE(ptr1);
        REQUIRE(ptr1.get());
        REQUIRE_FALSE(ptr1.empty());
        REQUIRE(ptr1.elements() == elements);
        REQUIRE(ptr1.bytes() == elements * sizeof(TestType));
    }

    AND_THEN("empty states") {
        cuda::memory::PtrDevice<TestType> ptr1(randomizer.get());
        ptr1.reset(randomizer.get());
        ptr1.dispose();
        ptr1.dispose(); // no double delete.
        ptr1.reset(0); // allocate but 0 elements...
        REQUIRE(ptr1.empty());
        REQUIRE_FALSE(ptr1);
        ptr1.reset();
    }
}

TEMPLATE_TEST_CASE("cuda::memory::PtrDevice - async", "[noa][cuda][memory]",
                   int32_t, uint32_t, int64_t, uint64_t, float, double, cfloat_t, cdouble_t) {
    cuda::memory::PtrDevice<TestType> ptr;
    test::Randomizer<size_t> randomizer(1, 255);
    cuda::Stream stream;

    AND_THEN("copy data to device and back to host") {
        const size_t elements = randomizer.get();
        const size_t bytes = elements * sizeof(TestType);

        // transfer: h_in -> d_inter -> h_out.
        cpu::memory::PtrHost<TestType> h_in(elements);
        cuda::memory::PtrDevice<TestType> d_inter(elements, stream);
        cpu::memory::PtrHost<TestType> h_out(elements);
        REQUIRE(d_inter.stream()->id() == stream.id());

        test::randomize(h_in.get(), h_in.elements(), randomizer);
        test::memset(h_out.get(), h_out.elements(), 0);

        REQUIRE(cudaMemcpyAsync(d_inter.get(), h_in.get(), bytes, cudaMemcpyDefault, d_inter.stream()->id()) == cudaSuccess);
        REQUIRE(cudaMemcpyAsync(h_out.get(), d_inter.get(), bytes, cudaMemcpyDefault, d_inter.stream()->id()) == cudaSuccess);
        stream.synchronize();
        const TestType diff = test::getDifference(h_in.get(), h_out.get(), h_in.elements());
        REQUIRE(diff == TestType{0});
    }
}
