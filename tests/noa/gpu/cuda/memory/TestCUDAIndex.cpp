#include <noa/gpu/cuda/memory/Index.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/memory/Set.h>

#include <noa/cpu/memory/Index.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::memory::extract(), insert()", "[noa][cuda][memory]",
                   int, long, float) {
    uint ndim = GENERATE(2U, 3U);
    BorderMode border_mode = GENERATE(BORDER_ZERO, BORDER_NOTHING, BORDER_VALUE,
                                      BORDER_CLAMP, BORDER_MIRROR, BORDER_REFLECT);
    INFO(ndim);
    INFO(border_mode);
    TestType border_value = 5;
    test::IntRandomizer<size_t> input_shape_randomizer(128, 256);
    size3_t input_shape(input_shape_randomizer.get(), input_shape_randomizer.get(), ndim == 3 ? 128 : 1);
    size_t input_elements = noa::elements(input_shape);

    test::IntRandomizer<size_t> size_randomizer(1, 64);
    size3_t subregion_shape(size_randomizer.get(), size_randomizer.get(), ndim == 3 ? size_randomizer.get() : 1);
    size_t subregion_elements = noa::elements(subregion_shape);

    uint subregion_count = test::IntRandomizer<uint>(1, 10).get();
    cpu::memory::PtrHost<size3_t> h_subregion_centers(subregion_count);
    test::IntRandomizer<size_t> center_randomizer(0, 300);
    for (uint idx = 0; idx < subregion_count; ++idx) {
        h_subregion_centers[idx] = {center_randomizer.get(),
                                    center_randomizer.get(),
                                    ndim == 3 ? center_randomizer.get() : 0};
    }

    // CPU backend
    cpu::memory::PtrHost<TestType> h_input(input_elements);
    cpu::memory::PtrHost<TestType> h_subregions(subregion_elements * subregion_count);
    test::Randomizer<TestType> randomizer(0, 120);
    test::initDataRandom(h_input.get(), h_input.elements(), randomizer);
    if (BORDER_NOTHING == border_mode)
        test::initDataZero(h_subregions.get(), h_subregions.elements());
    cpu::memory::extract(h_input.get(), input_shape,
                         h_subregions.get(), subregion_shape, h_subregion_centers.get(), subregion_count,
                         border_mode, border_value);

    cpu::memory::PtrHost<TestType> h_insert_back(input_elements);
    test::initDataZero(h_insert_back.get(), h_insert_back.elements());
    for (uint i = 0; i < subregion_count; ++i) {
        TestType* subregion = h_subregions.get() + i * subregion_elements;
        cpu::memory::insert(subregion, subregion_shape, h_subregion_centers[i], h_insert_back.get(), input_shape);
    }

    cuda::Stream stream(cuda::Stream::SERIAL);
    AND_THEN("contiguous") {
        cuda::memory::PtrDevice<TestType> d_input(input_elements);
        cuda::memory::PtrDevice<TestType> d_subregions(subregion_elements * subregion_count);
        cuda::memory::PtrDevice<size3_t> d_centers(subregion_count);
        cpu::memory::PtrHost<TestType> h_subregions_cuda(d_subregions.elements());
        if (BORDER_NOTHING == border_mode) {
            test::initDataZero(h_subregions_cuda.get(), h_subregions_cuda.elements());
            cuda::memory::copy(h_subregions_cuda.get(), d_subregions.get(), d_subregions.size(), stream);
        }

        // Extract
        cuda::memory::copy(h_input.get(), d_input.get(), d_input.size(), stream);
        cuda::memory::copy(h_subregion_centers.get(), d_centers.get(), d_centers.size(), stream);
        cuda::memory::extract(d_input.get(), input_shape.x, input_shape,
                              d_subregions.get(), subregion_shape.x, subregion_shape, d_centers.get(), subregion_count,
                              border_mode, border_value, stream);
        cuda::memory::copy(d_subregions.get(), h_subregions_cuda.get(), d_subregions.size(), stream);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getDifference(h_subregions.get(), h_subregions_cuda.get(), h_subregions.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(0, 1e-6));

        // Insert
        cuda::memory::PtrDevice<TestType> d_insert_back(input_elements);
        cpu::memory::PtrHost<TestType> h_insert_back_cuda(input_elements);
        test::initDataZero(h_insert_back_cuda.get(), h_insert_back_cuda.elements());
        cuda::memory::copy(h_insert_back_cuda.get(), d_insert_back.get(), d_insert_back.size(), stream);
        for (uint i = 0; i < subregion_count; ++i) {
            TestType* subregion = d_subregions.get() + i * subregion_elements;
            cuda::memory::insert(subregion, subregion_shape.x, subregion_shape, h_subregion_centers[i],
                                 d_insert_back.get(), input_shape.x, input_shape, stream);
        }
        cuda::memory::copy(d_insert_back.get(), h_insert_back_cuda.get(), d_insert_back.size(), stream);
        cuda::Stream::synchronize(stream);
        diff = test::getDifference(h_insert_back.get(), h_insert_back_cuda.get(), input_elements);
        REQUIRE_THAT(diff, test::isWithinAbs(0, 1e-6));
    }

    AND_THEN("contiguous - extract single subregion") {
        // This is a different code path... so just to make sure.
        cuda::memory::PtrDevice<TestType> d_input(input_elements);
        cuda::memory::PtrDevice<TestType> d_subregion(subregion_elements);
        cpu::memory::PtrHost<TestType> h_subregion_cuda(d_subregion.elements());
        if (BORDER_NOTHING == border_mode) {
            test::initDataZero(h_subregion_cuda.get(), h_subregion_cuda.elements());
            cuda::memory::copy(h_subregion_cuda.get(), d_subregion.get(), d_subregion.size(), stream);
        }

        // Extract
        cuda::memory::copy(h_input.get(), d_input.get(), d_input.size(), stream);
        cuda::memory::extract(d_input.get(), input_shape.x, input_shape,
                              d_subregion.get(), subregion_shape.x, subregion_shape, h_subregion_centers[0],
                              border_mode, border_value, stream);
        cuda::memory::copy(d_subregion.get(), h_subregion_cuda.get(), d_subregion.size(), stream);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getDifference(h_subregions.get(), h_subregion_cuda.get(), h_subregion_cuda.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(0, 1e-6));
    }

    AND_THEN("padded") {
        using PtrDevicePadded = cuda::memory::PtrDevicePadded<TestType>;
        PtrDevicePadded d_input(input_shape);
        PtrDevicePadded d_subregions({subregion_shape.x, rows(subregion_shape), subregion_count});
        cuda::memory::PtrDevice<size3_t> d_centers(subregion_count);
        cpu::memory::PtrHost<TestType> h_subregions_cuda(d_subregions.elements());
        if (BORDER_NOTHING == border_mode) {
            test::initDataZero(h_subregions_cuda.get(), h_subregions_cuda.elements());
            cuda::memory::copy(h_subregions_cuda.get(), subregion_shape.x,
                               d_subregions.get(), d_subregions.pitch(), d_subregions.shape(), stream);
        }

        // Extract
        cuda::memory::copy(h_input.get(), input_shape.x,
                           d_input.get(), d_input.pitch(), input_shape, stream);
        cuda::memory::copy(h_subregion_centers.get(), d_centers.get(), d_centers.size(), stream);
        cuda::memory::extract(d_input.get(), d_input.pitch(), input_shape,
                              d_subregions.get(), d_subregions.pitch(), subregion_shape,
                              d_centers.get(), subregion_count,
                              border_mode, border_value, stream);
        cuda::memory::copy(d_subregions.get(), d_subregions.pitch(),
                           h_subregions_cuda.get(), subregion_shape.x,
                           d_subregions.shape(), stream);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getDifference(h_subregions.get(), h_subregions_cuda.get(), h_subregions.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(0, 1e-6));

        // Insert
        cuda::memory::PtrDevicePadded<TestType> d_insert_back(input_shape);
        cpu::memory::PtrHost<TestType> h_insert_back_cuda(input_elements);
        test::initDataZero(h_insert_back_cuda.get(), h_insert_back_cuda.elements());
        cuda::memory::copy(h_insert_back_cuda.get(), input_shape.x,
                           d_insert_back.get(), d_insert_back.pitch(), input_shape, stream);
        for (uint i = 0; i < subregion_count; ++i) {
            TestType* subregion = d_subregions.get() + i * (d_subregions.pitch() * rows(subregion_shape));
            cuda::memory::insert(subregion, d_subregions.pitch(), subregion_shape, h_subregion_centers[i],
                                 d_insert_back.get(), d_insert_back.pitch(), input_shape, stream);
        }
        cuda::memory::copy(d_insert_back.get(), d_insert_back.pitch(),
                           h_insert_back_cuda.get(), input_shape.x,
                           input_shape, stream);
        cuda::Stream::synchronize(stream);
        diff = test::getDifference(h_insert_back.get(), h_insert_back_cuda.get(), input_elements);
        REQUIRE_THAT(diff, test::isWithinAbs(0, 1e-6));
    }

    AND_THEN("padded - extract single subregion") {
        using PtrDevicePadded = cuda::memory::PtrDevicePadded<TestType>;
        PtrDevicePadded d_input(input_shape);
        PtrDevicePadded d_subregion(subregion_shape);
        cpu::memory::PtrHost<TestType> h_subregion_cuda(d_subregion.elements());
        if (BORDER_NOTHING == border_mode) {
            test::initDataZero(h_subregion_cuda.get(), h_subregion_cuda.elements());
            cuda::memory::copy(h_subregion_cuda.get(), subregion_shape.x,
                               d_subregion.get(), d_subregion.pitch(), d_subregion.shape(), stream);
        }

        // Extract
        cuda::memory::copy(h_input.get(), input_shape.x,
                           d_input.get(), d_input.pitch(), input_shape, stream);
        cuda::memory::extract(d_input.get(), d_input.pitch(), input_shape,
                              d_subregion.get(), d_subregion.pitch(), subregion_shape,
                              h_subregion_centers[0], border_mode, border_value, stream);
        cuda::memory::copy(d_subregion.get(), d_subregion.pitch(),
                           h_subregion_cuda.get(), subregion_shape.x,
                           d_subregion.shape(), stream);
        cuda::Stream::synchronize(stream);
        TestType diff = test::getDifference(h_subregions.get(), h_subregion_cuda.get(), h_subregion_cuda.elements());
        REQUIRE_THAT(diff, test::isWithinAbs(0, 1e-6));
    }
}

TEMPLATE_TEST_CASE("cuda::memory::where(), extract(), insert()", "[noa][cuda][memory]", float, int, long) {
    size3_t shape = test::getRandomShape(3);
    size_t elements = noa::elements(shape);
    test::IntRandomizer<size_t> index_randomizer(size_t{0}, elements - 1);

    // Init data
    test::Randomizer<TestType> data_randomizer(1., 100.);
    cpu::memory::PtrHost<TestType> h_sparse(elements);
    test::initDataRandom(h_sparse.get(), h_sparse.elements(), data_randomizer);

    // CPU backend
    TestType threshold = 1;
    test::Randomizer<TestType> mask_randomizer(0, 4);
    cpu::memory::PtrHost<TestType> h_mask(elements);
    test::initDataRandom(h_mask.get(), elements, mask_randomizer);
    auto[h_tmp_map, h_elements_mapped] = cpu::memory::where(h_mask.get(), elements, threshold);
    cpu::memory::PtrHost<size_t> h_map(h_tmp_map, h_elements_mapped);

    cpu::memory::PtrHost<TestType> h_dense(h_map.elements());
    cpu::memory::extract(h_sparse.get(), h_sparse.elements(), h_dense.get(), h_dense.elements(), h_map.get(), 1);

    cpu::memory::PtrHost<TestType> h_inserted_back(elements);
    test::initDataZero(h_inserted_back.get(), elements);
    cpu::memory::insert(h_dense.get(), h_dense.elements(), h_inserted_back.get(), elements, h_map.get(), 1);

    cuda::Stream stream(cuda::Stream::SERIAL);
    cuda::memory::PtrDevice<size_t> d_map;

    THEN("where() - contiguous") {
        cuda::memory::PtrDevice<TestType> d_mask(elements);
        cuda::memory::copy(h_mask.get(), d_mask.get(), d_mask.size(), stream);
        auto[d_tmp_map, d_elements_mapped] = cuda::memory::where(d_mask.get(), elements, threshold, stream);
        d_map.reset(d_tmp_map, d_elements_mapped);
        cpu::memory::PtrHost<size_t> h_map_cuda(d_elements_mapped);
        cuda::memory::copy(d_map.get(), h_map_cuda.get(), d_map.size(), stream);
        cuda::Stream::synchronize(stream);

        REQUIRE(h_elements_mapped == d_elements_mapped);
        size_t diff = test::getDifference(h_map.get(), h_map_cuda.get(), d_elements_mapped);
        REQUIRE(diff == 0);

        THEN("extract(), insert()") {
            cuda::memory::PtrDevice<TestType> d_sparse(h_sparse.elements());
            cuda::memory::copy(h_sparse.get(), d_sparse.get(), h_sparse.size(), stream);
            cuda::memory::PtrDevice<TestType> d_dense(h_map.elements());
            cpu::memory::PtrHost<TestType> h_dense_cuda(h_map.elements());
            cuda::memory::extract(d_sparse.get(), d_sparse.elements(),
                                  d_dense.get(), d_dense.elements(), d_map.get(), 1, stream);
            cuda::memory::copy(d_dense.get(), h_dense_cuda.get(), d_dense.size(), stream);
            cuda::Stream::synchronize(stream);
            TestType diff2 = test::getDifference(h_dense.get(), h_dense_cuda.get(), h_dense.elements());
            REQUIRE(diff2 == 0);

            cpu::memory::PtrHost<TestType> h_inserted_cuda(elements);
            test::initDataZero(h_inserted_cuda.get(), elements);
            cuda::memory::PtrDevice<TestType> d_inserted(elements);
            cuda::memory::copy(h_inserted_cuda.get(), d_inserted.get(), d_inserted.size(), stream);
            cuda::memory::insert(d_dense.get(), d_dense.elements(),
                                 d_inserted.get(), elements, d_map.get(), 1, stream);
            cuda::memory::copy(d_inserted.get(), h_inserted_cuda.get(), d_inserted.size(), stream);
            cuda::Stream::synchronize(stream);
            diff2 = test::getDifference(h_inserted_back.get(), h_inserted_cuda.get(), h_inserted_back.elements());
            REQUIRE(diff2 == 0);
        }
    }

    THEN("where() - padded") {
        cuda::memory::PtrDevicePadded<TestType> d_mask(shape);
        cpu::memory::PtrHost<TestType> h_mask1(d_mask.elementsPadded());
        test::initDataRandom(h_mask1.get(), d_mask.elementsPadded(), mask_randomizer);
        cuda::memory::copy(h_mask1.get(), d_mask.pitch(), d_mask.get(), d_mask.pitch(), shape, stream);
        auto[tmp_map, d_elements_mapped] = cuda::memory::where(d_mask.get(), d_mask.pitch(), shape, 1,
                                                               threshold, stream);
        cuda::memory::PtrDevice<size_t> d_tmp_map(tmp_map, d_elements_mapped);
        cpu::memory::PtrHost<size_t> h_map_cuda(d_elements_mapped);
        cuda::memory::copy(d_tmp_map.get(), h_map_cuda.get(), d_tmp_map.size(), stream);

        // Update the map since it's not the same physical size.
        auto[h_tmp_map1, h_elements_mapped1] = cpu::memory::where(h_mask1.get(), d_mask.pitch(), shape, 1, threshold);
        cuda::Stream::synchronize(stream);

        REQUIRE(h_elements_mapped1 == d_elements_mapped);
        size_t diff = test::getDifference(h_tmp_map1, h_map_cuda.get(), d_elements_mapped);
        REQUIRE(diff == 0);
    }
}

TEMPLATE_TEST_CASE("cuda::memory::atlasLayout(), insert()", "[noa][cuda][memory]", float, int) {
    uint ndim = GENERATE(2U, 3U);
    test::IntRandomizer<uint> dim_randomizer(40, 60);
    size3_t subregion_shape(dim_randomizer.get(), dim_randomizer.get(), ndim == 3 ? dim_randomizer.get() : 1);
    uint subregion_count = test::IntRandomizer<uint>(1, 40).get();
    size_t nb = rows(subregion_shape);
    cuda::memory::PtrDevicePadded<TestType> d_subregions({subregion_shape.x, nb, subregion_count});
    size_t subregion_physical_elements = d_subregions.pitch() * nb;

    cuda::Stream stream(cuda::Stream::SERIAL);
    for (uint idx = 0; idx < subregion_count; ++idx)
        cuda::memory::set(d_subregions.get() + idx * subregion_physical_elements,
                          subregion_physical_elements, static_cast<TestType>(idx), stream);

    // Copy to host for assertion
    cpu::memory::PtrHost<TestType> h_subregions(d_subregions.elements());
    cuda::memory::copy(d_subregions.get(), d_subregions.pitch(),
                       h_subregions.get(), subregion_shape.x,
                       d_subregions.shape(), stream);

    // Insert atlas
    cpu::memory::PtrHost<size3_t> h_centers(subregion_count);
    size3_t atlas_shape = cpu::memory::atlasLayout(subregion_shape, subregion_count, h_centers.get());
    cuda::memory::PtrDevice<size3_t> d_centers(subregion_count);
    cuda::memory::copy(h_centers.get(), d_centers.get(), h_centers.elements(), stream);

    cuda::memory::PtrDevicePadded<TestType> atlas(atlas_shape);
    cuda::memory::insert(d_subregions.get(), d_subregions.pitch(), subregion_shape, d_centers.get(), subregion_count,
                         atlas.get(), atlas.pitch(), atlas_shape, stream);

    // Extract atlas
    cuda::memory::PtrDevicePadded<TestType> o_subregions(d_subregions.shape());
    cuda::memory::extract(atlas.get(), atlas.pitch(), atlas_shape,
                          o_subregions.get(), o_subregions.pitch(), subregion_shape, d_centers.get(), subregion_count,
                          BORDER_ZERO, TestType{0}, stream);

    // Copy to host for assertion
    cpu::memory::PtrHost<TestType> h_o_subregions(d_subregions.elements());
    cuda::memory::copy(o_subregions.get(), o_subregions.pitch(),
                       h_o_subregions.get(), subregion_shape.x, o_subregions.shape(), stream);

    cuda::Stream::synchronize(stream);
    TestType diff = test::getDifference(h_subregions.get(), h_o_subregions.get(), d_subregions.elements());
    REQUIRE(diff == 0);
}
