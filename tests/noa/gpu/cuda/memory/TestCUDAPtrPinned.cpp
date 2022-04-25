#include <cuda_runtime_api.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrPinned.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEMPLATE_TEST_CASE("cuda::memory::PtrPinned", "[noa][cuda][memory]",
                   int32_t, uint32_t, int64_t, uint64_t, float, double, cfloat_t, cdouble_t) {
    test::Randomizer<size_t> randomizer(1, 2550);

    AND_THEN("copy data to device and back to host") {
        const size_t elements = randomizer.get();
        const size_t bytes = elements * sizeof(TestType);

        // transfer: p_in -> d_inter -> h_out.
        cpu::memory::PtrHost<TestType> h_out(elements);
        cuda::memory::PtrPinned<TestType> p_in(elements);

        // device array
        TestType* d_inter{};
        void* tmp;
        REQUIRE(cudaMalloc(&tmp, bytes) == cudaSuccess);
        d_inter = static_cast<TestType*>(tmp);

        test::randomize(p_in.get(), p_in.elements(), randomizer);
        for (auto& e: h_out)
            e = 0;

        REQUIRE(cudaMemcpy(d_inter, p_in.get(), bytes, cudaMemcpyDefault) == cudaSuccess);
        REQUIRE(cudaMemcpy(h_out.get(), d_inter, bytes, cudaMemcpyDefault) == cudaSuccess);

        const TestType diff = test::getDifference(h_out.get(), p_in.get(), h_out.elements());
        REQUIRE(diff == TestType{0});

        REQUIRE(cudaFree(d_inter) == cudaSuccess);
    }

    AND_THEN("allocation, free, ownership") {
        cuda::memory::PtrPinned<TestType> ptr1;
        REQUIRE_FALSE(ptr1);
        const size_t elements = randomizer.get();
        {
            cuda::memory::PtrPinned<TestType> ptr2(elements);
            REQUIRE(ptr2);
            REQUIRE(ptr2.get());
            REQUIRE_FALSE(ptr2.empty());
            REQUIRE(ptr2.elements() == elements);
            REQUIRE(ptr2.bytes() == elements * sizeof(TestType));
            ptr1 = std::move(ptr2);
        }
        REQUIRE(ptr1);
        REQUIRE(ptr1.get());
        REQUIRE_FALSE(ptr1.empty());
        REQUIRE(ptr1.elements() == elements);
        REQUIRE(ptr1.bytes() == elements * sizeof(TestType));
    }

    AND_THEN("empty states") {
        cuda::memory::PtrPinned<TestType> ptr1(randomizer.get());
        ptr1 = cuda::memory::PtrPinned<TestType>(randomizer.get());
        ptr1 = nullptr;
        ptr1 = nullptr; // no double delete.
        ptr1 = cuda::memory::PtrPinned<TestType>(size_t{0}); // allocate but 0 elements...
        REQUIRE(ptr1.empty());
        REQUIRE_FALSE(ptr1);
        ptr1.release();
    }
}
