#include <cuda_runtime.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

// These are very simple tests. PtrDevice will be tested extensively since
// it is a dependency for many parts of the CUDA backend.
TEMPLATE_TEST_CASE("cuda::memory::PtrDevicePadded", "[noa][cuda][memory]",
                   int32_t, uint32_t, int64_t, uint64_t, float, double, cfloat_t, cdouble_t) {

    AND_THEN("copy 2D data to device and back to host") {
        const size4_t shape = test::getRandomShape(2);
        const size_t elements = shape.elements();
        const size_t bytes_per_elements = sizeof(TestType);

        // transfer: h_in -> d_inter -> h_out.
        cpu::memory::PtrHost<TestType> h_in(elements);
        cuda::memory::PtrDevicePadded<TestType> d_inter(shape);
        cpu::memory::PtrHost<TestType> h_out(elements);

        test::Randomizer<TestType> randomizer(0, 500);
        test::randomize(h_in.get(), h_in.elements(), randomizer);
        test::memset(h_out.get(), h_out.elements(), 0);

        cudaError_t err;
        err = cudaMemcpy2D(d_inter.get(), d_inter.pitches()[2] * bytes_per_elements,
                           h_in.get(), shape[3] * bytes_per_elements,
                           shape[3] * bytes_per_elements, shape[2], cudaMemcpyDefault);
        REQUIRE(err == cudaSuccess);
        err = cudaMemcpy2D(h_out.get(), shape[3] * bytes_per_elements,
                           d_inter.get(), d_inter.pitches()[2] * bytes_per_elements,
                           shape[3] * bytes_per_elements, shape[2], cudaMemcpyDefault);
        REQUIRE(err == cudaSuccess);

        const TestType diff = test::getDifference(h_in.get(), h_out.get(), h_in.elements());
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("copy 3D data to device and back to host") {
        const size4_t shape = test::getRandomShape(3);
        const size_t elements = shape.elements();
        const size_t bytes_per_elements = sizeof(TestType);

        // transfer: h_in -> d_inter -> h_out.
        cpu::memory::PtrHost<TestType> h_in(elements);
        cuda::memory::PtrDevicePadded<TestType> d_inter(shape);
        cpu::memory::PtrHost<TestType> h_out(elements);

        test::Randomizer<TestType> randomizer(0, 500);
        test::randomize(h_in.get(), h_in.elements(), randomizer);
        test::memset(h_out.get(), h_out.elements(), 0);

        cudaError_t err;
        cudaMemcpy3DParms params{};
        params.extent.width = shape[3] * sizeof(TestType);
        params.extent.height = shape[2];
        params.extent.depth = shape[1];
        params.kind = cudaMemcpyDefault;

        params.srcPtr = make_cudaPitchedPtr(h_in.get(), shape[3] * bytes_per_elements, shape[3], shape[2]);
        params.dstPtr = make_cudaPitchedPtr(d_inter.get(), d_inter.pitches()[2] * bytes_per_elements, shape[3], shape[2]);
        err = cudaMemcpy3D(&params);
        REQUIRE(err == cudaSuccess);

        params.srcPtr = make_cudaPitchedPtr(d_inter.get(), d_inter.pitches()[2] * bytes_per_elements, shape[3], shape[2]);
        params.dstPtr = make_cudaPitchedPtr(h_out.get(), shape[3] * sizeof(TestType), shape[3], shape[2]);
        err = cudaMemcpy3D(&params);
        REQUIRE(err == cudaSuccess);

        const TestType diff = test::getDifference(h_in.get(), h_out.get(), h_in.elements());
        REQUIRE(diff == TestType{0});
    }

        // test allocation and free
    AND_THEN("allocation, free, ownership") {
        cuda::memory::PtrDevicePadded<TestType> ptr1;
        const size4_t shape = test::getRandomShape(3);
        REQUIRE_FALSE(ptr1);
        {
            cuda::memory::PtrDevicePadded<TestType> ptr2(shape);
            REQUIRE(ptr2);
            REQUIRE(ptr2.get());
            REQUIRE_FALSE(ptr2.empty());
            REQUIRE(ptr2.shape().elements() == shape.elements());
            REQUIRE(ptr2.pitches()[2] >= shape[3]);
            REQUIRE(all(ptr2.pitches() == size3_t{shape[1], shape[2], ptr2.pitches()[2]}));
            ptr1 = std::move(ptr2);
        }
        REQUIRE(ptr1);
        REQUIRE(ptr1.get());
        REQUIRE_FALSE(ptr1.empty());
        REQUIRE(ptr1.shape().elements() == shape.elements());
    }

    AND_THEN("empty states") {
        const size4_t shape = test::getRandomShape(3);
        cuda::memory::PtrDevicePadded<TestType> ptr1(shape);
        ptr1 = cuda::memory::PtrDevicePadded<TestType>(shape);
        ptr1 = nullptr;
        ptr1 = nullptr; // no double delete.
        ptr1 = cuda::memory::PtrDevicePadded<TestType>({0,0,0,0}); // allocate but 0 elements...
        REQUIRE(ptr1.empty());
        REQUIRE_FALSE(ptr1);
        ptr1.release();
    }
}
