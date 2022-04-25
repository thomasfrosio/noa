#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/Index.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Set.h>
#include <noa/gpu/cuda/memory/Index.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/memory/Set.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cuda::memory::extract(), insert() - subregions", "[assets][noa][cuda][memory]") {
    const path_t path_base = test::NOA_DATA_PATH / "memory";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["index"];
    io::ImageFile file;
    cuda::Stream stream;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto shape = test["shape"].as<size4_t>();
        const auto subregion_shape = test["sub_shape"].as<size4_t>();
        auto subregion_origins = test["sub_origins"].as<std::vector<int4_t>>();
        const auto border_mode = test["border"].as<BorderMode>();
        const auto border_value = test["border_value"].as<float>();
        const size4_t stride = shape.stride();
        const size4_t subregion_stride = subregion_shape.stride();

        cpu::memory::PtrHost<float> input(shape.elements());
        cpu::memory::PtrHost<float> subregions(subregion_shape.elements());
        cpu::memory::PtrHost<float> h_cuda_subregions(subregion_shape.elements());
        const shared_t<int4_t[]> origins = input.attach(subregion_origins.data());
        test::memset(subregions.get(), subregions.elements(), 4.f);
        test::arange(input.get(), input.elements());

        cuda::memory::PtrDevice<float> d_input(shape.elements(), stream);
        cuda::memory::PtrDevice<float> d_subregions(subregion_shape.elements(), stream);
        cuda::memory::copy<float>(input.share(), stride, d_input.share(), stride, shape, stream);
        cuda::memory::copy<float>(subregions.share(), subregion_stride,
                                  d_subregions.share(), subregion_stride, subregion_shape, stream);

        // Extract:
        cuda::memory::extract<float>(d_input.share(), stride, shape,
                                     d_subregions.share(), subregion_stride, subregion_shape,
                                     origins, border_mode, border_value, stream);
        cuda::memory::copy<float>(d_subregions.share(), subregion_stride,
                                  h_cuda_subregions.share(), subregion_stride, subregion_shape, stream);

        const auto expected_subregion_filenames = test["expected_extract"].as<std::vector<path_t>>();
        cpu::memory::PtrHost<float> expected_subregions(subregions.elements());
        for (size_t i = 0; i < subregion_shape[0]; ++i) {
            float* expected_subregion = expected_subregions.get() + i * subregion_stride[0];
            file.open(path_base / expected_subregion_filenames[i], io::READ);
            file.readAll(expected_subregion);
        }
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected_subregions.get(),
                              h_cuda_subregions.get(), subregions.size(), 1e-7));

        // Insert:
        cuda::memory::set<float>(d_input.share(), d_input.elements(), 4.f, stream);
        cuda::memory::insert<float>(d_subregions.share(), subregion_stride, subregion_shape,
                                    d_input.share(), shape.stride(), shape,
                                    origins, stream);
        cuda::memory::copy<float>(d_input.share(), stride, input.share(), stride, shape, stream);

        const path_t expected_insert_filename = path_base / test["expected_insert"][0].as<path_t>();
        cpu::memory::PtrHost<float> expected_insert_back(input.elements());
        file.open(expected_insert_filename, io::READ);
        file.readAll(expected_insert_back.get());
        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, expected_insert_back.get(),
                              input.get(), input.elements(), 1e-7));
    }
}

TEMPLATE_TEST_CASE("cuda::memory::extract(), insert() - sequences", "[noa][cuda][memory]", float, int) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();
    cpu::Stream cpu_stream(cpu::Stream::DEFAULT);
    cuda::Stream gpu_stream;

    // Initialize data.
    test::Randomizer<TestType> data_randomizer(-100., 100.);
    cpu::memory::PtrHost<TestType> data(elements);
    test::randomize(data.get(), data.size(), data_randomizer);
    cuda::memory::PtrDevice<TestType> d_data(elements, gpu_stream);
    cuda::memory::copy<TestType>(data.share(), stride, d_data.share(), stride, shape, gpu_stream);

    THEN("contiguous") {
        const auto h_extracted = cpu::memory::extract<TestType, uint32_t>(
                data.share(), stride, data.share(), stride, TestType(0),
                shape, noa::math::greater_t{}, true, true, cpu_stream);

        const auto d_extracted = cuda::memory::extract<TestType, uint32_t>(
                d_data.share(), stride, d_data.share(), stride, TestType(0),
                shape, noa::math::greater_t{}, true, true, gpu_stream);

        REQUIRE(h_extracted.count == d_extracted.count);
        const size_t count = h_extracted.count;
        cpu::memory::PtrHost<TestType> h_cuda_seq_values(count);
        cpu::memory::PtrHost<uint32_t> h_cuda_seq_indexes(count);
        cuda::memory::copy<TestType>(d_extracted.values, h_cuda_seq_values.share(), count, gpu_stream);
        cuda::memory::copy<uint32_t>(d_extracted.indexes, h_cuda_seq_indexes.share(), count, gpu_stream);
        gpu_stream.synchronize();

        REQUIRE(test::Matcher(test::MATCH_ABS, h_extracted.indexes.get(),
                              h_cuda_seq_indexes.get(), count, 1e-6));
        REQUIRE(test::Matcher(test::MATCH_ABS, h_extracted.values.get(),
                              h_cuda_seq_values.get(), count, 1e-6));

        cpu::memory::PtrHost<TestType> reinsert(elements);
        cuda::memory::PtrDevice<TestType> d_reinsert(elements, gpu_stream);
        cpu::memory::PtrHost<TestType> h_cuda_reinsert(elements);
        cpu::memory::set<TestType>(reinsert.share(), elements, TestType(0), cpu_stream);
        cuda::memory::set<TestType>(d_reinsert.share(), count, TestType(0), gpu_stream);
        cpu::memory::insert<TestType>(h_extracted, reinsert.share(), cpu_stream);
        cuda::memory::insert<TestType>(d_extracted, d_reinsert.share(), gpu_stream);
        cuda::memory::copy<TestType>(d_reinsert.share(), h_cuda_reinsert.share(), elements, gpu_stream);

        gpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, h_cuda_reinsert.get(), reinsert.get(), elements, 1e-7));
    }

    THEN("padded") {
        cuda::memory::PtrDevicePadded<TestType> padded(shape);
        cuda::memory::set(padded.share(), padded.stride(), padded.shape(), TestType(2), gpu_stream);

        const auto d_extracted = cuda::memory::extract<TestType, uint64_t>(
                padded.share(), padded.stride(), padded.share(), padded.stride(), TestType(1), shape,
                noa::math::greater_equal_t{}, false, true, gpu_stream);
        cpu::memory::PtrHost<uint64_t> h_seq_indexes(d_extracted.count);
        cuda::memory::copy<uint64_t>(d_extracted.indexes, h_seq_indexes.share(), d_extracted.count, gpu_stream);
        gpu_stream.synchronize();

        REQUIRE(d_extracted.count == elements); // elements in pitch should not be selected
        const size_t last = indexing::at(shape - 1, padded.stride());
        REQUIRE(h_seq_indexes[d_extracted.count - 1] == last); // indexes should follow the physical layout of the input
    }
}

TEMPLATE_TEST_CASE("cuda::memory::atlasLayout(), insert()", "[noa][cuda][memory]", float, int) {
    const uint ndim = GENERATE(2u, 3u);
    test::Randomizer<uint> dim_randomizer(40, 60);
    const size4_t subregion_shape(test::Randomizer<uint>(1, 40).get(),
                                  ndim == 3 ? dim_randomizer.get() : 1,
                                  dim_randomizer.get(),
                                  dim_randomizer.get());
    cuda::memory::PtrDevicePadded<TestType> d_subregions(subregion_shape);
    const size4_t subregion_stride = subregion_shape.stride();
    INFO(subregion_shape);

    cuda::Stream stream;
    for (uint idx = 0; idx < subregion_shape[0]; ++idx)
        cuda::memory::set(d_subregions.attach(d_subregions.get() + idx * d_subregions.stride()[0]),
                          d_subregions.stride()[0], static_cast<TestType>(idx), stream);

    // Copy to host for assertion
    cpu::memory::PtrHost<TestType> h_subregions(subregion_shape.elements());
    cuda::memory::copy<TestType>(d_subregions.share(), d_subregions.stride(),
                                 h_subregions.share(), subregion_stride, subregion_shape, stream);

    // Insert atlas
    cpu::memory::PtrHost<int4_t> origins(subregion_shape[0]);
    const size4_t atlas_shape = cuda::memory::atlasLayout(subregion_shape, origins.get());
    INFO(atlas_shape);

    cuda::memory::PtrDevicePadded<TestType> atlas(atlas_shape);
    cuda::memory::insert<TestType>(d_subregions.share(), d_subregions.stride(), subregion_shape,
                                   atlas.share(), atlas.stride(), atlas_shape,
                                   origins.share(), stream);

    // Extract atlas
    cuda::memory::PtrDevicePadded<TestType> o_subregions(subregion_shape);
    cuda::memory::extract<TestType>(atlas.share(), atlas.stride(), atlas_shape,
                                    o_subregions.share(), o_subregions.stride(), subregion_shape,
                                    origins.share(), BORDER_VALUE, TestType{0}, stream);

    // Copy to host for assertion
    cpu::memory::PtrHost<TestType> h_o_subregions(subregion_shape.elements());
    cuda::memory::copy<TestType>(o_subregions.share(), o_subregions.stride(),
                                 h_o_subregions.share(), subregion_stride, subregion_shape, stream);

    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, h_subregions.get(), h_o_subregions.get(), h_o_subregions.elements(), 1e-7));
}
