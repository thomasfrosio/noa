#include <cuda_runtime.h>

#include <noa/cpu/PtrHost.h>
#include <noa/gpu/cuda/PtrDevicePadded.h>

#include "../../../Helpers.h"
#include <catch2/catch.hpp>

using namespace ::Noa;

// These are very simple tests. PtrDevice will be tested extensively since
// it is a dependency for many parts of the CUDA backend.
TEMPLATE_TEST_CASE("PtrDevicePadded", "[noa][cuda]",
                   int32_t, uint32_t, int64_t, uint64_t, float, double, cfloat_t, cdouble_t) {
    Test::IntRandomizer<size_t> randomizer_large(1, 512);
    Test::IntRandomizer<size_t> randomizer_small(1, 128);

    AND_THEN("copy 2D data to device and back to host") {
        size3_t shape(randomizer_large.get(), randomizer_large.get(), 1);
        size_t elements = getElements(shape);

        // transfer: h_in -> d_inter -> h_out.
        Noa::PtrHost<TestType> h_in(elements);
        Noa::CUDA::PtrDevicePadded<TestType> d_inter(shape);
        Noa::PtrHost<TestType> h_out(elements);

        Test::initDataRandom(h_in.get(), h_in.elements(), randomizer_small);
        Test::initDataZero(h_out.get(), h_out.elements());

        cudaError_t err;
        err = cudaMemcpy2D(d_inter.get(), d_inter.pitch(),
                           h_in.get(), shape.x * sizeof(TestType),
                           shape.x * sizeof(TestType), shape.y, cudaMemcpyDefault);
        REQUIRE(err == cudaSuccess);
        err = cudaMemcpy2D(h_out.get(), shape.x * sizeof(TestType),
                           d_inter.get(), d_inter.pitch(),
                           shape.x * sizeof(TestType), shape.y, cudaMemcpyDefault);
        REQUIRE(err == cudaSuccess);

        TestType diff = Test::getDifference(h_in.get(), h_out.get(), h_in.elements());
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("copy 3D data to device and back to host") {
        size3_t shape(randomizer_small.get(), randomizer_small.get(), randomizer_small.get());
        size_t elements = getElements(shape);

        // transfer: h_in -> d_inter -> h_out.
        Noa::PtrHost<TestType> h_in(elements);
        Noa::CUDA::PtrDevicePadded<TestType> d_inter(shape);
        Noa::PtrHost<TestType> h_out(elements);

        Test::initDataRandom(h_in.get(), h_in.elements(), randomizer_small);
        Test::initDataZero(h_out.get(), h_out.elements());

        cudaError_t err;
        cudaMemcpy3DParms params{};
        params.extent.width = shape.x * sizeof(TestType);
        params.extent.height = shape.y;
        params.extent.depth = shape.z;
        params.kind = cudaMemcpyDefault;

        params.srcPtr = make_cudaPitchedPtr(h_in.get(), shape.x * sizeof(TestType), shape.x, shape.y);
        params.dstPtr = make_cudaPitchedPtr(d_inter.get(), d_inter.pitch(), shape.x, shape.y);
        err = cudaMemcpy3D(&params);
        REQUIRE(err == cudaSuccess);

        params.srcPtr = make_cudaPitchedPtr(d_inter.get(), d_inter.pitch(), shape.x, shape.y);
        params.dstPtr = make_cudaPitchedPtr(h_out.get(), shape.x * sizeof(TestType), shape.x, shape.y);
        err = cudaMemcpy3D(&params);
        REQUIRE(err == cudaSuccess);

        TestType diff = Test::getDifference(h_in.get(), h_out.get(), h_in.elements());
        REQUIRE(diff == TestType{0});
    }

        // test allocation and free
    AND_THEN("allocation, free, ownership") {
        Noa::CUDA::PtrDevicePadded<TestType> ptr1;
        size3_t shape(randomizer_small.get(), randomizer_small.get(), randomizer_small.get());
        REQUIRE_FALSE(ptr1);
        {
            Noa::CUDA::PtrDevicePadded<TestType> ptr2(shape);
            REQUIRE(ptr2);
            REQUIRE(ptr2.get());
            REQUIRE_FALSE(ptr2.empty());
            REQUIRE(ptr2.elements() == getElements(shape));
            REQUIRE(ptr2.bytes() == getElements(shape) * sizeof(TestType));
            REQUIRE(ptr2.bytesPadded() >= ptr2.bytes());
            REQUIRE(ptr2.pitch() >= shape.x * sizeof(TestType));
            size_t pitch = ptr2.pitch();
            ptr1.reset(ptr2.release(), pitch, shape); // transfer ownership.
            REQUIRE_FALSE(ptr2);
            REQUIRE_FALSE(ptr2.get());
            REQUIRE(ptr2.empty());
            REQUIRE_FALSE(ptr2.elements());
            REQUIRE_FALSE(ptr2.bytes());
        }
        REQUIRE(ptr1);
        REQUIRE(ptr1.get());
        REQUIRE_FALSE(ptr1.empty());
        REQUIRE(ptr1.elements() == getElements(shape));
        REQUIRE(ptr1.bytes() == getElements(shape) * sizeof(TestType));
    }

    AND_THEN("empty states") {
        size3_t shape(randomizer_small.get(), randomizer_small.get(), randomizer_small.get());
        Noa::CUDA::PtrDevicePadded<TestType> ptr1(shape);
        ptr1.reset(shape);
        ptr1.dispose();
        ptr1.dispose(); // no double delete.
        ptr1.reset({0,0,0}); // allocate but 0 elements...
        REQUIRE(ptr1.empty());
        REQUIRE_FALSE(ptr1);
        ptr1.reset();
    }
}
