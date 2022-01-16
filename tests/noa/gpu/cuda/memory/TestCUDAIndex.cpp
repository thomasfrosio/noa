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

// Turn this test off for now. When CUDA backend is updated, update this test and turn it on.

//TEMPLATE_TEST_CASE("cuda::memory::extract(), insert()", "[noa][cuda][memory]",
//                   int, long, float) {
//    const uint ndim = GENERATE(2U, 3U);
//    const BorderMode border_mode = GENERATE(BORDER_ZERO, BORDER_NOTHING, BORDER_VALUE,
//                                            BORDER_CLAMP, BORDER_MIRROR, BORDER_REFLECT);
//    const TestType border_value = 5;
//    test::Randomizer<size_t> input_shape_randomizer(128, 256);
//    const size3_t input_shape(input_shape_randomizer.get(), input_shape_randomizer.get(), ndim == 3 ? 128 : 1);
//    const size_t input_elements = noa::elements(input_shape);
//
//    test::Randomizer<size_t> size_randomizer(1, 64);
//    const size3_t subregion_shape(size_randomizer.get(), size_randomizer.get(), ndim == 3 ? size_randomizer.get() : 1);
//    const size_t subregion_elements = noa::elements(subregion_shape);
//
//    const size_t subregion_count = test::Randomizer<uint>(1, 10).get();
//    cpu::memory::PtrHost<size3_t> h_subregion_centers(subregion_count);
//    test::Randomizer<size_t> center_randomizer(0, 300);
//    for (size_t idx = 0; idx < subregion_count; ++idx) {
//        h_subregion_centers[idx] = {center_randomizer.get(),
//                                    center_randomizer.get(),
//                                    ndim == 3 ? center_randomizer.get() : 0};
//    }
//
//    // CPU backend
//    cpu::memory::PtrHost<TestType> h_input(input_elements);
//    cpu::memory::PtrHost<TestType> h_subregions(subregion_elements * subregion_count);
//    test::Randomizer<TestType> randomizer(0, 120);
//    test::randomize(h_input.get(), h_input.elements(), randomizer);
//    if (BORDER_NOTHING == border_mode)
//        test::memset(h_subregions.get(), h_subregions.elements(), 0);
//    cpu::memory::extract(h_input.get(), input_shape,
//                         h_subregions.get(), subregion_shape, h_subregion_centers.get(), subregion_count,
//                         border_mode, border_value);
//
//    cpu::memory::PtrHost<TestType> h_insert_back(input_elements);
//    test::memset(h_insert_back.get(), h_insert_back.elements(), 0);
//    for (uint i = 0; i < subregion_count; ++i) {
//        TestType* subregion = h_subregions.get() + i * subregion_elements;
//        cpu::memory::insert(subregion, subregion_shape, h_subregion_centers[i], h_insert_back.get(), input_shape);
//    }
//
//    cuda::Stream stream(cuda::Stream::SERIAL);
//    AND_THEN("contiguous") {
//        cuda::memory::PtrDevice<TestType> d_input(input_elements);
//        cuda::memory::PtrDevice<TestType> d_subregions(subregion_elements * subregion_count);
//        cuda::memory::PtrDevice<size3_t> d_centers(subregion_count);
//        cpu::memory::PtrHost<TestType> h_subregions_cuda(d_subregions.elements());
//        if (BORDER_NOTHING == border_mode) {
//            test::memset(h_subregions_cuda.get(), h_subregions_cuda.elements(), 0);
//            cuda::memory::copy(h_subregions_cuda.get(), d_subregions.get(), d_subregions.size(), stream);
//        }
//
//        // Extract
//        cuda::memory::copy(h_input.get(), d_input.get(), d_input.size(), stream);
//        cuda::memory::copy(h_subregion_centers.get(), d_centers.get(), d_centers.size(), stream);
//        cuda::memory::extract(d_input.get(), input_shape.x, input_shape,
//                              d_subregions.get(), subregion_shape.x, subregion_shape, d_centers.get(), subregion_count,
//                              border_mode, border_value, stream);
//        cuda::memory::copy(d_subregions.get(), h_subregions_cuda.get(), d_subregions.size(), stream);
//        cuda::Stream::synchronize(stream);
//        REQUIRE(test::Matcher(test::MATCH_ABS, h_subregions.get(), h_subregions_cuda.get(), h_subregions.size(), 1e-6));
//
//        // Insert
//        cuda::memory::PtrDevice<TestType> d_insert_back(input_elements);
//        cpu::memory::PtrHost<TestType> h_insert_back_cuda(input_elements);
//        test::memset(h_insert_back_cuda.get(), h_insert_back_cuda.elements(), 0);
//        cuda::memory::copy(h_insert_back_cuda.get(), d_insert_back.get(), d_insert_back.size(), stream);
//        for (uint i = 0; i < subregion_count; ++i) {
//            TestType* subregion = d_subregions.get() + i * subregion_elements;
//            cuda::memory::insert(subregion, subregion_shape.x, subregion_shape, h_subregion_centers[i],
//                                 d_insert_back.get(), input_shape.x, input_shape, stream);
//        }
//        cuda::memory::copy(d_insert_back.get(), h_insert_back_cuda.get(), d_insert_back.size(), stream);
//        cuda::Stream::synchronize(stream);
//        REQUIRE(test::Matcher(test::MATCH_ABS, h_insert_back.get(), h_insert_back_cuda.get(), input_elements, 1e-6));
//    }
//
//    AND_THEN("contiguous - extract single subregion") {
//        // This is a different code path... so just to make sure.
//        cuda::memory::PtrDevice<TestType> d_input(input_elements);
//        cuda::memory::PtrDevice<TestType> d_subregion(subregion_elements);
//        cpu::memory::PtrHost<TestType> h_subregion_cuda(d_subregion.elements());
//        if (BORDER_NOTHING == border_mode) {
//            test::memset(h_subregion_cuda.get(), h_subregion_cuda.elements(), 0);
//            cuda::memory::copy(h_subregion_cuda.get(), d_subregion.get(), d_subregion.size(), stream);
//        }
//
//        // Extract
//        cuda::memory::copy(h_input.get(), d_input.get(), d_input.size(), stream);
//        cuda::memory::extract(d_input.get(), input_shape.x, input_shape,
//                              d_subregion.get(), subregion_shape.x, subregion_shape, h_subregion_centers[0],
//                              border_mode, border_value, stream);
//        cuda::memory::copy(d_subregion.get(), h_subregion_cuda.get(), d_subregion.size(), stream);
//        cuda::Stream::synchronize(stream);
//        REQUIRE(test::Matcher(test::MATCH_ABS,
//                              h_subregions.get(), h_subregion_cuda.get(), h_subregion_cuda.size(), 1e-6));
//    }
//
//    AND_THEN("padded") {
//        using PtrDevicePadded = cuda::memory::PtrDevicePadded<TestType>;
//        PtrDevicePadded d_input(input_shape);
//        PtrDevicePadded d_subregions({subregion_shape.x, rows(subregion_shape), subregion_count});
//        cuda::memory::PtrDevice<size3_t> d_centers(subregion_count);
//        cpu::memory::PtrHost<TestType> h_subregions_cuda(d_subregions.elements());
//        if (BORDER_NOTHING == border_mode) {
//            test::memset(h_subregions_cuda.get(), h_subregions_cuda.elements(), 0);
//            cuda::memory::copy(h_subregions_cuda.get(), subregion_shape.x,
//                               d_subregions.get(), d_subregions.pitch(), d_subregions.shape(), stream);
//        }
//
//        // Extract
//        cuda::memory::copy(h_input.get(), input_shape.x,
//                           d_input.get(), d_input.pitch(), input_shape, stream);
//        cuda::memory::copy(h_subregion_centers.get(), d_centers.get(), d_centers.size(), stream);
//        cuda::memory::extract(d_input.get(), d_input.pitch(), input_shape,
//                              d_subregions.get(), d_subregions.pitch(), subregion_shape,
//                              d_centers.get(), subregion_count,
//                              border_mode, border_value, stream);
//        cuda::memory::copy(d_subregions.get(), d_subregions.pitch(),
//                           h_subregions_cuda.get(), subregion_shape.x,
//                           d_subregions.shape(), stream);
//        cuda::Stream::synchronize(stream);
//        REQUIRE(test::Matcher(test::MATCH_ABS, h_subregions.get(), h_subregions_cuda.get(), h_subregions.size(), 1e-6));
//
//        // Insert
//        cuda::memory::PtrDevicePadded<TestType> d_insert_back(input_shape);
//        cpu::memory::PtrHost<TestType> h_insert_back_cuda(input_elements);
//        test::memset(h_insert_back_cuda.get(), h_insert_back_cuda.elements(), 0);
//        cuda::memory::copy(h_insert_back_cuda.get(), input_shape.x,
//                           d_insert_back.get(), d_insert_back.pitch(), input_shape, stream);
//        for (uint i = 0; i < subregion_count; ++i) {
//            TestType* subregion = d_subregions.get() + i * (d_subregions.pitch() * rows(subregion_shape));
//            cuda::memory::insert(subregion, d_subregions.pitch(), subregion_shape, h_subregion_centers[i],
//                                 d_insert_back.get(), d_insert_back.pitch(), input_shape, stream);
//        }
//        cuda::memory::copy(d_insert_back.get(), d_insert_back.pitch(),
//                           h_insert_back_cuda.get(), input_shape.x,
//                           input_shape, stream);
//        cuda::Stream::synchronize(stream);
//        REQUIRE(test::Matcher(test::MATCH_ABS, h_insert_back.get(), h_insert_back_cuda.get(), input_elements, 1e-6));
//    }
//
//    AND_THEN("padded - extract single subregion") {
//        using PtrDevicePadded = cuda::memory::PtrDevicePadded<TestType>;
//        PtrDevicePadded d_input(input_shape);
//        PtrDevicePadded d_subregion(subregion_shape);
//        cpu::memory::PtrHost<TestType> h_subregion_cuda(d_subregion.elements());
//        if (BORDER_NOTHING == border_mode) {
//            test::memset(h_subregion_cuda.get(), h_subregion_cuda.elements(), 0);
//            cuda::memory::copy(h_subregion_cuda.get(), subregion_shape.x,
//                               d_subregion.get(), d_subregion.pitch(), d_subregion.shape(), stream);
//        }
//
//        // Extract
//        cuda::memory::copy(h_input.get(), input_shape.x,
//                           d_input.get(), d_input.pitch(), input_shape, stream);
//        cuda::memory::extract(d_input.get(), d_input.pitch(), input_shape,
//                              d_subregion.get(), d_subregion.pitch(), subregion_shape,
//                              h_subregion_centers[0], border_mode, border_value, stream);
//        cuda::memory::copy(d_subregion.get(), d_subregion.pitch(),
//                           h_subregion_cuda.get(), subregion_shape.x,
//                           d_subregion.shape(), stream);
//        cuda::Stream::synchronize(stream);
//        REQUIRE(test::Matcher(test::MATCH_ABS,
//                              h_subregions.get(), h_subregion_cuda.get(), h_subregion_cuda.size(), 1e-6));
//    }
//}

TEMPLATE_TEST_CASE("cuda::memory::where(), extract(), insert()", "[noa][cuda][memory]", float, int, long) {
    const size3_t shape = test::getRandomShape(3);
    const size_t elements = noa::elements(shape);

    // Init data
    test::Randomizer<TestType> data_randomizer(1., 100.);
    cpu::memory::PtrHost<TestType> h_sparse(elements);
    test::randomize(h_sparse.get(), h_sparse.elements(), data_randomizer);

    cpu::Stream cpu_stream;
    cuda::Stream stream(cuda::Stream::SERIAL);

    THEN("contiguous") {
        // Mask.
        test::Randomizer<TestType> mask_randomizer(0, 4);
        cpu::memory::PtrHost<TestType> h_mask(elements);
        test::randomize(h_mask.get(), elements, mask_randomizer);
        const TestType threshold = 1; // extract elements > threshold

        // CPU backend
        auto[h_values_, h_indexes_, h_extracted] = cpu::memory::extract<true, size_t>(
                h_sparse.get(), shape, h_mask.get(), shape, shape, 1,
                [](TestType, TestType m) { return m > 1; }, cpu_stream);
        cpu::memory::PtrHost<TestType> h_sequence_values(h_values_, h_extracted);
        cpu::memory::PtrHost<size_t> h_sequence_indexes(h_indexes_, h_extracted);
        cpu::memory::PtrHost<TestType> h_insert(elements);
        test::memset(h_insert.get(), elements, 0);
        cpu::memory::insert(h_sequence_values.get(), h_sequence_indexes.get(), h_extracted, h_insert.get(), cpu_stream);

        cuda::memory::PtrDevice<TestType> d_mask(elements);
        cuda::memory::copy(h_mask.get(), d_mask.get(), d_mask.size(), stream);
        auto[d_indexes_, d_extracted] = cuda::memory::where(d_mask.get(), elements, threshold, stream);
        cuda::memory::PtrDevice<size_t> d_sequence_indexes(d_indexes_, d_extracted);
        cpu::memory::PtrHost<size_t> h_cuda_sequence_indexes(d_extracted);
        cuda::memory::copy(d_sequence_indexes.get(), h_cuda_sequence_indexes.get(), d_extracted, stream);
        cuda::Stream::synchronize(stream);

        REQUIRE(h_extracted == d_extracted);
        size_t diff = test::getDifference(h_sequence_indexes.get(), h_cuda_sequence_indexes.get(), d_extracted);
        REQUIRE(diff == 0);

        cuda::memory::PtrDevice<TestType> d_sparse(h_sparse.elements());
        cuda::memory::copy(h_sparse.get(), d_sparse.get(), h_sparse.size(), stream);
        cuda::memory::PtrDevice<TestType> d_dense(h_sequence_indexes.elements());
        cpu::memory::PtrHost<TestType> h_dense_cuda(h_sequence_indexes.elements());
        cuda::memory::extract(d_sparse.get(), d_sparse.elements(),
                              d_dense.get(), d_dense.elements(), d_sequence_indexes.get(), 1, stream);
        cuda::memory::copy(d_dense.get(), h_dense_cuda.get(), d_dense.size(), stream);
        cuda::Stream::synchronize(stream);
        TestType diff2 = test::getDifference(h_sequence_values.get(), h_dense_cuda.get(), h_extracted);
        REQUIRE(diff2 == 0);

        cpu::memory::PtrHost<TestType> h_inserted_cuda(elements);
        test::memset(h_inserted_cuda.get(), elements, 0);
        cuda::memory::PtrDevice<TestType> d_inserted(elements);
        cuda::memory::copy(h_inserted_cuda.get(), d_inserted.get(), d_inserted.size(), stream);
        cuda::memory::insert(d_dense.get(), d_dense.elements(),
                             d_inserted.get(), elements, d_sequence_indexes.get(), 1, stream);
        cuda::memory::copy(d_inserted.get(), h_inserted_cuda.get(), d_inserted.size(), stream);
        cuda::Stream::synchronize(stream);
        diff2 = test::getDifference(h_insert.get(), h_inserted_cuda.get(), h_insert.elements());
        REQUIRE(diff2 == 0);
    }

    THEN("padded") {
        // Generate mask:
        test::Randomizer<TestType> mask_randomizer(0, 4);
        cuda::memory::PtrDevicePadded<TestType> d_mask(shape);
        const size3_t pitch{d_mask.pitch(), shape.y, shape.z};
        cpu::memory::PtrHost<TestType> h_mask(d_mask.elementsPadded());
        test::randomize(h_mask.get(), h_mask.size(), mask_randomizer);
        cuda::memory::copy(h_mask.get(), d_mask.pitch(), d_mask.get(), d_mask.pitch(), shape, stream);
        const TestType threshold = 1;

        // CUDA: extract >threshold
        auto[d_indexes_, d_extracted] = cuda::memory::where(d_mask.get(), d_mask.pitch(), shape, 1,
                                                               threshold, stream);
        cuda::memory::PtrDevice<size_t> d_indexes(d_indexes_, d_extracted);
        cpu::memory::PtrHost<size_t> h_cuda_indexes(d_extracted);
        cuda::memory::copy(d_indexes.get(), h_cuda_indexes.get(), d_indexes.size(), stream);

        // CPU: extract >threshold
        auto[_, h_indexes_, h_extracted] = cpu::memory::extract<false, size_t>(
                h_mask.get(), pitch, shape, 1, [=](TestType m) { return m > threshold; }, cpu_stream);
        cpu::memory::PtrHost<size_t> h_indexes(h_indexes_, h_extracted);

        stream.synchronize();

        REQUIRE(h_extracted == d_extracted);
        size_t diff = test::getDifference(h_indexes.get(), h_cuda_indexes.get(), d_extracted);
        REQUIRE(diff == 0);
    }
}

TEMPLATE_TEST_CASE("cuda::memory::atlasLayout(), insert()", "[noa][cuda][memory]", float, int) {
    uint ndim = GENERATE(2U, 3U);
    test::Randomizer<uint> dim_randomizer(40, 60);
    size3_t subregion_shape(dim_randomizer.get(), dim_randomizer.get(), ndim == 3 ? dim_randomizer.get() : 1);
    uint subregion_count = test::Randomizer<uint>(1, 40).get();
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
    size3_t atlas_shape = cuda::memory::atlasLayout(subregion_shape, subregion_count, h_centers.get());
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
