#include <noa/gpu/cuda/memory/Remap.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include <noa/cpu/memory/Remap.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEMPLATE_TEST_CASE("CUDA::Memory::extract(), insert()", "[noa][cuda][memory]",
                   int, long, float) {
    uint ndim = GENERATE(2U, 3U);
    BorderMode border_mode = GENERATE(BORDER_ZERO, BORDER_NOTHING, BORDER_VALUE);
    TestType border_value = 5;
    Test::IntRandomizer<size_t> input_shape_randomizer(128, 256);
    size3_t input_shape(input_shape_randomizer.get(), input_shape_randomizer.get(), ndim == 3 ? 128 : 1);
    size_t input_elements = getElements(input_shape);

    Test::IntRandomizer<size_t> size_randomizer(1, 64);
    size3_t subregion_shape(size_randomizer.get(), size_randomizer.get(), ndim == 3 ? size_randomizer.get() : 1);
    size_t subregion_elements = getElements(subregion_shape);

    uint subregion_count = Test::IntRandomizer<uint>(1, 10).get();
    Memory::PtrHost<size3_t> h_subregion_centers(subregion_count);
    Test::IntRandomizer<size_t> center_randomizer(0, 300);
    for (uint idx = 0; idx < subregion_count; ++idx) {
        h_subregion_centers[idx] = {center_randomizer.get(),
                                    center_randomizer.get(),
                                    ndim == 3 ? center_randomizer.get() : 0};
    }

    // CPU backend
    Memory::PtrHost<TestType> h_input(input_elements);
    Memory::PtrHost<TestType> h_subregions(subregion_elements * subregion_count);
    Test::Randomizer<TestType> randomizer(0, 120);
    Test::initDataRandom(h_input.get(), h_input.elements(), randomizer);
    if (BORDER_NOTHING == border_mode)
        Test::initDataZero(h_subregions.get(), h_subregions.elements());
    Memory::extract(h_input.get(), input_shape,
                    h_subregions.get(), subregion_shape, h_subregion_centers.get(), subregion_count,
                    border_mode, border_value);

    Memory::PtrHost<TestType> h_insert_back(input_elements);
    Test::initDataZero(h_insert_back.get(), h_insert_back.elements());
    for (uint i = 0; i < subregion_count; ++i) {
        TestType* subregion = h_subregions.get() + i * subregion_elements;
        Memory::insert(subregion, subregion_shape, h_subregion_centers[i], h_insert_back.get(), input_shape);
    }

    CUDA::Stream stream(CUDA::Stream::SERIAL);
    AND_THEN("contiguous") {
        CUDA::Memory::PtrDevice<TestType> d_input(input_elements);
        CUDA::Memory::PtrDevice<TestType> d_subregions(subregion_elements * subregion_count);
        CUDA::Memory::PtrDevice<size3_t> d_centers(subregion_count);
        Memory::PtrHost<TestType> h_subregions_cuda(d_subregions.elements());
        if (BORDER_NOTHING == border_mode) {
            Test::initDataZero(h_subregions_cuda.get(), h_subregions_cuda.elements());
            CUDA::Memory::copy(h_subregions_cuda.get(), d_subregions.get(), d_subregions.size(), stream);
        }

        // Extract
        CUDA::Memory::copy(h_input.get(), d_input.get(), d_input.size(), stream);
        CUDA::Memory::copy(h_subregion_centers.get(), d_centers.get(), d_centers.size(), stream);
        CUDA::Memory::extract(d_input.get(), input_shape,
                              d_subregions.get(), subregion_shape, d_centers.get(), subregion_count,
                              border_mode, border_value, stream);
        CUDA::Memory::copy(d_subregions.get(), h_subregions_cuda.get(), d_subregions.size(), stream);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getDifference(h_subregions.get(), h_subregions_cuda.get(), h_subregions.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(0, 1e-6));

        // Insert
        CUDA::Memory::PtrDevice<TestType> d_insert_back(input_elements);
        Memory::PtrHost<TestType> h_insert_back_cuda(input_elements);
        Test::initDataZero(h_insert_back_cuda.get(), h_insert_back_cuda.elements());
        CUDA::Memory::copy(h_insert_back_cuda.get(), d_insert_back.get(), d_insert_back.size(), stream);
        for (uint i = 0; i < subregion_count; ++i) {
            TestType* subregion = d_subregions.get() + i * subregion_elements;
            CUDA::Memory::insert(subregion, subregion_shape, h_subregion_centers[i],
                                 d_insert_back.get(), input_shape, stream);
        }
        CUDA::Memory::copy(d_insert_back.get(), h_insert_back_cuda.get(), d_insert_back.size(), stream);
        CUDA::Stream::synchronize(stream);
        diff = Test::getDifference(h_insert_back.get(), h_insert_back_cuda.get(), input_elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(0, 1e-6));
    }

    AND_THEN("contiguous - extract single subregion") {
        // This is a different code path... so just to make sure.
        CUDA::Memory::PtrDevice<TestType> d_input(input_elements);
        CUDA::Memory::PtrDevice<TestType> d_subregion(subregion_elements);
        Memory::PtrHost<TestType> h_subregion_cuda(d_subregion.elements());
        if (BORDER_NOTHING == border_mode) {
            Test::initDataZero(h_subregion_cuda.get(), h_subregion_cuda.elements());
            CUDA::Memory::copy(h_subregion_cuda.get(), d_subregion.get(), d_subregion.size(), stream);
        }

        // Extract
        CUDA::Memory::copy(h_input.get(), d_input.get(), d_input.size(), stream);
        CUDA::Memory::extract(d_input.get(), input_shape,
                              d_subregion.get(), subregion_shape, h_subregion_centers[0],
                              border_mode, border_value, stream);
        CUDA::Memory::copy(d_subregion.get(), h_subregion_cuda.get(), d_subregion.size(), stream);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getDifference(h_subregions.get(), h_subregion_cuda.get(), h_subregion_cuda.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(0, 1e-6));
    }

    AND_THEN("padded") {
        using PtrDevicePadded = CUDA::Memory::PtrDevicePadded<TestType>;
        PtrDevicePadded d_input(input_shape);
        PtrDevicePadded d_subregions({subregion_shape.x, getRows(subregion_shape), subregion_count});
        CUDA::Memory::PtrDevice<size3_t> d_centers(subregion_count);
        Memory::PtrHost<TestType> h_subregions_cuda(d_subregions.elements());
        if (BORDER_NOTHING == border_mode) {
            Test::initDataZero(h_subregions_cuda.get(), h_subregions_cuda.elements());
            CUDA::Memory::copy(h_subregions_cuda.get(), subregion_shape.x,
                               d_subregions.get(), d_subregions.pitch(), d_subregions.shape(), stream);
        }

        // Extract
        CUDA::Memory::copy(h_input.get(), input_shape.x,
                           d_input.get(), d_input.pitch(), input_shape, stream);
        CUDA::Memory::copy(h_subregion_centers.get(), d_centers.get(), d_centers.size(), stream);
        CUDA::Memory::extract(d_input.get(), d_input.pitch(), input_shape,
                              d_subregions.get(), d_subregions.pitch(), subregion_shape,
                              d_centers.get(), subregion_count,
                              border_mode, border_value, stream);
        CUDA::Memory::copy(d_subregions.get(), d_subregions.pitch(),
                           h_subregions_cuda.get(), subregion_shape.x,
                           d_subregions.shape(), stream);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getDifference(h_subregions.get(), h_subregions_cuda.get(), h_subregions.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(0, 1e-6));

        // Insert
        CUDA::Memory::PtrDevicePadded<TestType> d_insert_back(input_shape);
        Memory::PtrHost<TestType> h_insert_back_cuda(input_elements);
        Test::initDataZero(h_insert_back_cuda.get(), h_insert_back_cuda.elements());
        CUDA::Memory::copy(h_insert_back_cuda.get(), input_shape.x,
                           d_insert_back.get(), d_insert_back.pitch(), input_shape, stream);
        for (uint i = 0; i < subregion_count; ++i) {
            TestType* subregion = d_subregions.get() + i * (d_subregions.pitch() * getRows(subregion_shape));
            CUDA::Memory::insert(subregion, d_subregions.pitch(), subregion_shape, h_subregion_centers[i],
                                 d_insert_back.get(), d_insert_back.pitch(), input_shape, stream);
        }
        CUDA::Memory::copy(d_insert_back.get(), d_insert_back.pitch(),
                           h_insert_back_cuda.get(), input_shape.x,
                           input_shape, stream);
        CUDA::Stream::synchronize(stream);
        diff = Test::getDifference(h_insert_back.get(), h_insert_back_cuda.get(), input_elements);
        REQUIRE_THAT(diff, Test::isWithinAbs(0, 1e-6));
    }

    AND_THEN("padded - extract single subregion") {
        using PtrDevicePadded = CUDA::Memory::PtrDevicePadded<TestType>;
        PtrDevicePadded d_input(input_shape);
        PtrDevicePadded d_subregion(subregion_shape);
        Memory::PtrHost<TestType> h_subregion_cuda(d_subregion.elements());
        if (BORDER_NOTHING == border_mode) {
            Test::initDataZero(h_subregion_cuda.get(), h_subregion_cuda.elements());
            CUDA::Memory::copy(h_subregion_cuda.get(), subregion_shape.x,
                               d_subregion.get(), d_subregion.pitch(), d_subregion.shape(), stream);
        }

        // Extract
        CUDA::Memory::copy(h_input.get(), input_shape.x,
                           d_input.get(), d_input.pitch(), input_shape, stream);
        CUDA::Memory::extract(d_input.get(), d_input.pitch(), input_shape,
                              d_subregion.get(), d_subregion.pitch(), subregion_shape,
                              h_subregion_centers[0], border_mode, border_value, stream);
        CUDA::Memory::copy(d_subregion.get(), d_subregion.pitch(),
                           h_subregion_cuda.get(), subregion_shape.x,
                           d_subregion.shape(), stream);
        CUDA::Stream::synchronize(stream);
        TestType diff = Test::getDifference(h_subregions.get(), h_subregion_cuda.get(), h_subregion_cuda.elements());
        REQUIRE_THAT(diff, Test::isWithinAbs(0, 1e-6));
    }
}

TEMPLATE_TEST_CASE("CUDA::Memory::getMap(), extract(), insert()", "[noa][cuda][memory]", float, int, long) {
    size3_t shape = Test::getRandomShape(3);
    size_t elements = getElements(shape);
    Test::IntRandomizer<size_t> index_randomizer(size_t{0}, elements - 1);

    // Init data
    Test::Randomizer<TestType> data_randomizer(1., 100.);
    Memory::PtrHost<TestType> h_sparse(elements);
    Test::initDataRandom(h_sparse.get(), h_sparse.elements(), data_randomizer);

    // CPU backend
    TestType threshold = 1;
    Test::Randomizer<TestType> mask_randomizer(0, 4);
    Memory::PtrHost<TestType> h_mask(elements);
    Test::initDataRandom(h_mask.get(), elements, mask_randomizer);
    auto[h_tmp_map, h_elements_mapped] = Memory::getMap(h_mask.get(), elements, threshold);
    Memory::PtrHost<size_t> h_map(h_tmp_map, h_elements_mapped);

    Memory::PtrHost<TestType> h_dense(h_map.elements());
    Memory::extract(h_sparse.get(), h_sparse.elements(), h_dense.get(), h_dense.elements(), h_map.get(), 1);

    Memory::PtrHost<TestType> h_inserted_back(elements);
    Test::initDataZero(h_inserted_back.get(), elements);
    Memory::insert(h_dense.get(), h_dense.elements(), h_inserted_back.get(), elements, h_map.get(), 1);

    CUDA::Stream stream(CUDA::Stream::SERIAL);
    CUDA::Memory::PtrDevice<size_t> d_map;

    THEN("getMap() - contiguous") {
        CUDA::Memory::PtrDevice<TestType> d_mask(elements);
        CUDA::Memory::copy(h_mask.get(), d_mask.get(), d_mask.size(), stream);
        auto[d_tmp_map, d_elements_mapped] = CUDA::Memory::getMap(d_mask.get(), elements, threshold, stream);
        d_map.reset(d_tmp_map, d_elements_mapped);
        Memory::PtrHost<size_t> h_map_cuda(d_elements_mapped);
        CUDA::Memory::copy(d_map.get(), h_map_cuda.get(), d_map.size(), stream);
        CUDA::Stream::synchronize(stream);

        REQUIRE(h_elements_mapped == d_elements_mapped);
        size_t diff = Test::getDifference(h_map.get(), h_map_cuda.get(), d_elements_mapped);
        REQUIRE(diff == 0);

        THEN("extract(), insert()") {
            CUDA::Memory::PtrDevice<TestType> d_sparse(h_sparse.elements());
            CUDA::Memory::copy(h_sparse.get(), d_sparse.get(), h_sparse.size(), stream);
            CUDA::Memory::PtrDevice<TestType> d_dense(h_map.elements());
            Memory::PtrHost<TestType> h_dense_cuda(h_map.elements());
            CUDA::Memory::extract(d_sparse.get(), d_sparse.elements(),
                                  d_dense.get(), d_dense.elements(), d_map.get(), 1, stream);
            CUDA::Memory::copy(d_dense.get(), h_dense_cuda.get(), d_dense.size(), stream);
            CUDA::Stream::synchronize(stream);
            TestType diff2 = Test::getDifference(h_dense.get(), h_dense_cuda.get(), h_dense.elements());
            REQUIRE(diff2 == 0);

            Memory::PtrHost<TestType> h_inserted_cuda(elements);
            Test::initDataZero(h_inserted_cuda.get(), elements);
            CUDA::Memory::PtrDevice<TestType> d_inserted(elements);
            CUDA::Memory::copy(h_inserted_cuda.get(), d_inserted.get(), d_inserted.size(), stream);
            CUDA::Memory::insert(d_dense.get(), d_dense.elements(),
                                 d_inserted.get(), elements, d_map.get(), 1, stream);
            CUDA::Memory::copy(d_inserted.get(), h_inserted_cuda.get(), d_inserted.size(), stream);
            CUDA::Stream::synchronize(stream);
            diff2 = Test::getDifference(h_inserted_back.get(), h_inserted_cuda.get(), h_inserted_back.elements());
            REQUIRE(diff2 == 0);
        }
    }

    THEN("getMap() - padded") {
        CUDA::Memory::PtrDevicePadded<TestType> d_mask(shape);
        CUDA::Memory::copy(h_mask.get(), shape.x, d_mask.get(), d_mask.pitch(), shape, stream);
        auto[tmp_map, d_elements_mapped] = CUDA::Memory::getMap(d_mask.get(), d_mask.pitch(), shape,
                                                                threshold, stream);
        CUDA::Memory::PtrDevice<size_t> d_tmp_map(tmp_map, d_elements_mapped);
        Memory::PtrHost<size_t> h_map_cuda(d_elements_mapped);
        CUDA::Memory::copy(d_tmp_map.get(), h_map_cuda.get(), d_tmp_map.size(), stream);
        CUDA::Stream::synchronize(stream);

        REQUIRE(h_elements_mapped == d_elements_mapped);
        size_t diff = Test::getDifference(h_map.get(), h_map_cuda.get(), d_elements_mapped);
        REQUIRE(diff == 0);
    }
}
