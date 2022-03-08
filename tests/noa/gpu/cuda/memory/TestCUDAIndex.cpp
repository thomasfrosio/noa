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
    cuda::Stream stream(cuda::Stream::SERIAL);

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto shape = test["shape"].as<size4_t>();
        const auto subregion_shape = test["sub_shape"].as<size4_t>();
        const auto subregion_origins = test["sub_origins"].as<std::vector<int4_t>>();
        const auto border_mode = test["border"].as<BorderMode>();
        const auto border_value = test["border_value"].as<float>();
        const size4_t stride = shape.strides();
        const size4_t subregion_stride = subregion_shape.strides();

        cpu::memory::PtrHost<float> input(shape.elements());
        cpu::memory::PtrHost<float> subregions(subregion_shape.elements());
        cpu::memory::PtrHost<float> h_cuda_subregions(subregion_shape.elements());
        test::memset(subregions.get(), subregions.elements(), 4.f);
        test::arange(input.get(), input.elements());

        cuda::memory::PtrDevice<float> d_input(shape.elements(), stream);
        cuda::memory::PtrDevice<float> d_subregions(subregion_shape.elements(), stream);
        cuda::memory::copy(input.get(), stride, d_input.get(), stride, shape, stream);
        cuda::memory::copy(subregions.get(), subregion_stride,
                           d_subregions.get(), subregion_stride, subregion_shape, stream);

        // Extract:
        cuda::memory::extract(d_input.get(), stride, shape,
                              d_subregions.get(), subregion_stride, subregion_shape,
                              subregion_origins.data(), border_mode, border_value, stream);
        cuda::memory::copy(d_subregions.get(), subregion_stride,
                           h_cuda_subregions.get(), subregion_stride, subregion_shape, stream);

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
        cuda::memory::set(d_input.get(), d_input.elements(), 4.f, stream);
        cuda::memory::insert(d_subregions.get(), subregion_stride, subregion_shape,
                             d_input.get(), shape.strides(), shape,
                             subregion_origins.data(), stream);
        cuda::memory::copy(d_input.get(), stride, input.get(), stride, shape, stream);

        path_t expected_insert_filename = path_base / test["expected_insert"][0].as<path_t>();
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
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();
    cpu::Stream cpu_stream(cpu::Stream::SERIAL);
    cuda::Stream gpu_stream(cuda::Stream::SERIAL);

    // Initialize data.
    test::Randomizer<TestType> data_randomizer(-100., 100.);
    cpu::memory::PtrHost<TestType> data(elements);
    test::randomize(data.get(), data.size(), data_randomizer);
    cuda::memory::PtrDevice<TestType> d_data(elements, gpu_stream);
    cuda::memory::copy(data.get(), stride, d_data.get(), stride, shape, gpu_stream);

    THEN("contiguous") {
        auto[h_values_, h_indexes_, h_extracted] = cpu::memory::extract<TestType, uint32_t>(
                data.get(), stride, shape, TestType(0), noa::math::greater_t{}, cpu_stream);
        cpu::memory::PtrHost<TestType> h_seq_values(h_values_, h_extracted);
        cpu::memory::PtrHost<uint32_t> h_seq_indexes(h_indexes_, h_extracted);

        auto[d_values_, d_indexes_, d_extracted] = cuda::memory::extract<TestType, uint32_t>(
                d_data.get(), stride, shape, TestType(0), noa::math::greater_t{}, gpu_stream);
        cuda::memory::PtrDevice<TestType> d_seq_values(d_values_, d_extracted, gpu_stream);
        cuda::memory::PtrDevice<uint32_t> d_seq_indexes(d_indexes_, d_extracted, gpu_stream);

        REQUIRE(h_extracted == d_extracted);
        cpu::memory::PtrHost<TestType> h_cuda_seq_values(h_extracted);
        cpu::memory::PtrHost<uint32_t> h_cuda_seq_indexes(h_extracted);
        cuda::memory::copy(d_seq_values.get(), h_cuda_seq_values.get(), h_extracted, gpu_stream);
        cuda::memory::copy(d_seq_indexes.get(), h_cuda_seq_indexes.get(), h_extracted, gpu_stream);
        gpu_stream.synchronize();

        REQUIRE(test::Matcher(test::MATCH_ABS, h_seq_indexes.get(),
                              h_cuda_seq_indexes.get(), h_extracted, 1e-6));
        REQUIRE(test::Matcher(test::MATCH_ABS, h_seq_values.get(),
                              h_cuda_seq_values.get(), h_extracted, 1e-6));

        cpu::memory::PtrHost<TestType> reinsert(elements);
        cuda::memory::PtrDevice<TestType> d_reinsert(elements, gpu_stream);
        cpu::memory::PtrHost<TestType> h_cuda_reinsert(elements);
        cpu::memory::set(reinsert.get(), elements, TestType(0), cpu_stream);
        cuda::memory::set(d_reinsert.get(), h_extracted, TestType(0), gpu_stream);
        cpu::memory::insert(h_seq_values.get(), h_seq_indexes.get(), h_extracted, reinsert.get(), cpu_stream);
        cuda::memory::insert(d_seq_values.get(), d_seq_indexes.get(), d_extracted, d_reinsert.get(), gpu_stream);
        cuda::memory::copy(d_reinsert.get(), h_cuda_reinsert.get(), elements, gpu_stream);

        cpu_stream.synchronize();
        gpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, h_cuda_reinsert.get(), reinsert.get(), elements, 1e-7));
    }

    THEN("padded") {
        cuda::memory::PtrDevicePadded<TestType> padded(shape);
        cuda::memory::set(padded.get(), padded.strides(), padded.shape(), TestType(2), gpu_stream);

        auto[_, indexes_, extracted] = cuda::memory::extract<void, uint64_t>(
                padded.get(), padded.strides(), shape, TestType(1), noa::math::greater_equal_t{}, gpu_stream);
        cuda::memory::PtrDevice<uint64_t> seq_indexes(indexes_, extracted);
        cpu::memory::PtrHost<uint64_t> h_seq_indexes(extracted);
        cuda::memory::copy(seq_indexes.get(), h_seq_indexes.get(), extracted, gpu_stream);
        gpu_stream.synchronize();

        REQUIRE(extracted == elements); // elements in pitch should not be selected
        const size_t last = at(shape - 1, padded.strides());
        REQUIRE(h_seq_indexes[extracted - 1] == last); // indexes should follow the physical layout of the input
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
    const size4_t subregion_stride = subregion_shape.strides();
    INFO(subregion_shape);

    cuda::Stream stream(cuda::Stream::SERIAL);
    for (uint idx = 0; idx < subregion_shape[0]; ++idx)
        cuda::memory::set(d_subregions.get() + idx * d_subregions.strides()[0],
                          d_subregions.strides()[0], static_cast<TestType>(idx), stream);

    // Copy to host for assertion
    cpu::memory::PtrHost<TestType> h_subregions(subregion_shape.elements());
    cuda::memory::copy(d_subregions.get(), d_subregions.strides(),
                       h_subregions.get(), subregion_stride, subregion_shape, stream);

    // Insert atlas
    cpu::memory::PtrHost<int4_t> origins(subregion_shape[0]);
    const size4_t atlas_shape = cuda::memory::atlasLayout(subregion_shape, origins.get());
    INFO(atlas_shape);

    cuda::memory::PtrDevicePadded<TestType> atlas(atlas_shape);
    cuda::memory::insert(d_subregions.get(), d_subregions.strides(), subregion_shape,
                         atlas.get(), atlas.strides(), atlas_shape,
                         origins.get(), stream);

    // Extract atlas
    cuda::memory::PtrDevicePadded<TestType> o_subregions(subregion_shape);
    cuda::memory::extract(atlas.get(), atlas.strides(), atlas_shape,
                          o_subregions.get(), o_subregions.strides(), subregion_shape,
                          origins.get(), BORDER_VALUE, TestType{0}, stream);

    // Copy to host for assertion
    cpu::memory::PtrHost<TestType> h_o_subregions(subregion_shape.elements());
    cuda::memory::copy(o_subregions.get(), o_subregions.strides(),
                       h_o_subregions.get(), subregion_stride, subregion_shape, stream);

    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, h_subregions.get(), h_o_subregions.get(), h_o_subregions.elements(), 1e-7));
}
