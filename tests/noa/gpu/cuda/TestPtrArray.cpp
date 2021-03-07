#include <cuda_runtime_api.h>

#include <noa/cpu/PtrHost.h>
#include <noa/gpu/cuda/PtrArray.h>

#include "../../../Helpers.h"
#include <catch2/catch.hpp>

using namespace ::Noa;

TEMPLATE_TEST_CASE("PtrArray 1D: base", "[noa][cuda]", int32_t, uint32_t, float, cfloat_t) {
    Test::IntRandomizer<size_t> randomizer_small(1, 128);

    // test allocation and free
    AND_THEN("allocation, free, ownership") {
        Noa::CUDA::PtrArray<TestType, 1> ptr1;
        size_t elements = randomizer_small.get();
        size3_t shape{elements, 1, 1};
        REQUIRE_FALSE(ptr1);
        {
            Noa::CUDA::PtrArray<TestType, 1> ptr2(elements);
            REQUIRE(ptr2);
            REQUIRE(ptr2.get());
            REQUIRE_FALSE(ptr2.empty());
            REQUIRE(ptr2.elements() == elements);
            REQUIRE(ptr2.shape() == shape);
            ptr1.reset(ptr2.release(), elements); // transfer ownership.
            REQUIRE_FALSE(ptr2);
            REQUIRE_FALSE(ptr2.get());
            REQUIRE(ptr2.empty());
            REQUIRE_FALSE(ptr2.elements());
            REQUIRE(ptr2.shape() == size3_t{0, 1, 1});
        }
        REQUIRE(ptr1);
        REQUIRE(ptr1.get());
        REQUIRE_FALSE(ptr1.empty());
        REQUIRE(ptr1.elements() == elements);
    }

    AND_THEN("empty states") {
        size_t elements = randomizer_small.get();
        Noa::CUDA::PtrArray<TestType, 1> ptr1(elements);
        ptr1.reset(elements);
        ptr1.dispose();
        ptr1.dispose(); // no double delete.
        REQUIRE_THROWS_AS(ptr1.reset(0), Noa::Exception);
        REQUIRE(ptr1.empty());
        REQUIRE_FALSE(ptr1);
        ptr1.reset();
    }
}

// These are very simple tests. PtrArray will be tested extensively since
// it is a dependency for many parts of the CUDA backend.
TEMPLATE_TEST_CASE("PtrArray: memcpy", "[noa][cuda]", int32_t, uint32_t, float, cfloat_t) {
    Test::IntRandomizer<size_t> randomizer_large(1, 512);
    Test::IntRandomizer<size_t> randomizer_small(1, 128);

    AND_THEN("copy 1D data to device and back to host") {
        size_t elements = randomizer_large.get();

        // transfer: h_in -> d_inter -> h_out.
        Noa::PtrHost<TestType> h_in(elements);
        Noa::CUDA::PtrArray<TestType, 1> d_inter(elements);
        Noa::PtrHost<TestType> h_out(elements);

        Test::initDataRandom(h_in.get(), h_in.elements(), randomizer_small);
        Test::initDataZero(h_out.get(), h_out.elements());

        cudaError_t err;
        cudaMemcpy3DParms params{};
        params.extent = make_cudaExtent(elements, 1, 1);
        params.kind = cudaMemcpyDefault;

        size_t src_pitch = elements * sizeof(TestType);
        params.srcPtr = make_cudaPitchedPtr(h_in.get(), src_pitch, elements, 1);
        params.dstArray = d_inter.get();
        err = cudaMemcpy3D(&params);
        REQUIRE(err == cudaSuccess);

        params.srcPtr = {};
        params.dstArray = {};

        params.srcArray = d_inter.get();
        params.dstPtr = make_cudaPitchedPtr(h_out.get(), src_pitch, elements, 1);
        err = cudaMemcpy3D(&params);
        REQUIRE(err == cudaSuccess);

        TestType diff = Test::getDifference(h_in.get(), h_out.get(), h_in.elements());
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("copy 2D data to device and back to host") {
        size2_t shape(randomizer_small.get(), randomizer_small.get());
        size_t elements = getElements(shape);

        // transfer: h_in -> d_inter -> h_out.
        Noa::PtrHost<TestType> h_in(elements);
        Noa::CUDA::PtrArray<TestType, 2> d_inter(shape);
        Noa::PtrHost<TestType> h_out(elements);

        Test::initDataRandom(h_in.get(), h_in.elements(), randomizer_small);
        Test::initDataZero(h_out.get(), h_out.elements());

        cudaError_t err;
        cudaMemcpy3DParms params{};
        params.extent = make_cudaExtent(shape.x, shape.y, 1);
        params.kind = cudaMemcpyDefault;

        size_t src_pitch = shape.x * sizeof(TestType);
        params.srcPtr = make_cudaPitchedPtr(h_in.get(), src_pitch, shape.x, shape.y);
        params.dstArray = d_inter.get();
        err = cudaMemcpy3D(&params);
        REQUIRE(err == cudaSuccess);

        params.srcPtr = {};
        params.dstArray = {};

        params.srcArray = d_inter.get();
        params.dstPtr = make_cudaPitchedPtr(h_out.get(), src_pitch, shape.x, shape.y);
        err = cudaMemcpy3D(&params);
        REQUIRE(err == cudaSuccess);

        TestType diff = Test::getDifference(h_in.get(), h_out.get(), h_in.elements());
        REQUIRE(diff == TestType{0});
    }

    AND_THEN("copy 3D data to device and back to host") {
        size3_t shape(randomizer_small.get(), randomizer_small.get(), randomizer_small.get());
        size_t elements = getElements(shape);

        // transfer: h_in -> d_inter -> h_out.
        Noa::PtrHost<TestType> h_in(elements);
        Noa::CUDA::PtrArray<TestType, 3> d_inter(shape);
        Noa::PtrHost<TestType> h_out(elements);

        Test::initDataRandom(h_in.get(), h_in.elements(), randomizer_small);
        Test::initDataZero(h_out.get(), h_out.elements());

        cudaError_t err;
        cudaMemcpy3DParms params{};
        params.extent = make_cudaExtent(shape.x, shape.y, shape.z);
        params.kind = cudaMemcpyDefault;

        size_t src_pitch = shape.x * sizeof(TestType);
        params.srcPtr = make_cudaPitchedPtr(h_in.get(), src_pitch, shape.x, shape.y);
        params.dstArray = d_inter.get();
        err = cudaMemcpy3D(&params);
        REQUIRE(err == cudaSuccess);

        params.srcPtr = {};
        params.dstArray = {};

        params.srcArray = d_inter.get();
        params.dstPtr = make_cudaPitchedPtr(h_out.get(), src_pitch, shape.x, shape.y);
        err = cudaMemcpy3D(&params);
        REQUIRE(err == cudaSuccess);

        TestType diff = Test::getDifference(h_in.get(), h_out.get(), h_in.elements());
        REQUIRE(diff == TestType{0});
    }
}
