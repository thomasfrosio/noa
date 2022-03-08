#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/Index.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Set.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cpu::memory::extract(), insert() - subregions", "[assets][noa][cpu][memory]") {
    constexpr bool COMPUTE_ASSETS = false;
    const path_t path_base = test::NOA_DATA_PATH / "memory";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["index"];
    io::ImageFile file;
    cpu::Stream stream;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto shape = test["shape"].as<size4_t>();
        const auto subregion_shape = test["sub_shape"].as<size4_t>();
        const auto subregion_origins = test["sub_origins"].as<std::vector<int4_t>>();
        const auto border_mode = test["border"].as<BorderMode>();
        const auto border_value = test["border_value"].as<float>();
        const size4_t subregion_stride = subregion_shape.strides();

        cpu::memory::PtrHost<float> input(shape.elements());
        cpu::memory::PtrHost<float> subregions(subregion_shape.elements());
        cpu::memory::set(subregions.begin(), subregions.end(), 4.f);
        test::arange(input.get(), input.elements());

        // Extract:
        cpu::memory::extract(input.get(), shape.strides(), shape,
                             subregions.get(), subregion_stride, subregion_shape,
                             subregion_origins.data(), border_mode, border_value, stream);

        const auto expected_subregion_filenames = test["expected_extract"].as<std::vector<path_t>>();
        if constexpr (COMPUTE_ASSETS) {
            for (size_t i = 0; i < subregion_shape[0]; ++i) {
                float* subregion = subregions.get() + i * subregion_stride[0];
                file.open(path_base / expected_subregion_filenames[i], io::WRITE);
                file.shape(subregion_shape);
                file.readAll(subregion);
            }
        } else {
            cpu::memory::PtrHost<float> expected_subregions(subregions.elements());
            for (size_t i = 0; i < subregion_shape[0]; ++i) {
                float* expected_subregion = expected_subregions.get() + i * subregion_stride[0];
                file.open(path_base / expected_subregion_filenames[i], io::READ);
                file.readAll(expected_subregion);
            }
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE,
                                  expected_subregions.get(), subregions.get(), subregions.size(), 1e-7));
        }

        // Insert:
        cpu::memory::set(input.begin(), input.end(), 4.f);
        cpu::memory::insert(subregions.get(), subregion_stride, subregion_shape,
                            input.get(), shape.strides(), shape,
                            subregion_origins.data(), stream);

        path_t expected_insert_filename = path_base / test["expected_insert"][0].as<path_t>();
        if constexpr(COMPUTE_ASSETS) {
            file.open(expected_insert_filename, io::WRITE);
            file.shape(shape);
            file.readAll(input.get());
        } else {
            cpu::memory::PtrHost<float> expected_insert_back(input.elements());
            file.open(expected_insert_filename, io::READ);
            file.readAll(expected_insert_back.get());
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE,
                                  expected_insert_back.get(), input.get(), input.size(), 1e-7));
        }
    }
}

TEMPLATE_TEST_CASE("cpu::memory::extract(), insert() - sequences", "[noa][cpu][memory]", float, int) {
    const size4_t shape = test::getRandomShapeBatched(3);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();
    cpu::Stream stream;

    // Initialize data.
    test::Randomizer<TestType> data_randomizer(1., 100.);
    cpu::memory::PtrHost<TestType> data(elements);
    test::randomize(data.get(), data.size(), data_randomizer);

    // Prepare expected data.
    test::Randomizer<int> mask_randomizer(0, 4);
    const size4_t mask_shape{1, shape[1], shape[2], shape[3]};
    size4_t mask_stride = mask_shape.strides();
    mask_stride[0] = 0; // mask is not batched
    cpu::memory::PtrHost<int> mask(mask_shape.elements());
    test::randomize(mask.get(), mask.elements(), mask_randomizer);

    // Extract elements from data only if mask isn't 0.
    std::vector<size_t> expected_indexes;
    std::vector<TestType> expected_values;
    cpu::memory::PtrHost<TestType> expected_data_reinsert(data.size());
    test::memset(expected_data_reinsert.get(), data.size(), 0);
    for (size_t batch = 0; batch < shape[0]; ++batch) {
        for (size_t i = 0; i < stride[0]; ++i) {
            if (mask[i] == 0)
                continue;
            const size_t index = batch * stride[0] + i;
            expected_indexes.emplace_back(index);
            expected_values.emplace_back(data[index]);
            expected_data_reinsert[index] = data[index];
        }
    }

    THEN("contiguous") {
        auto[values_, indexes_, extracted] = cpu::memory::extract<TestType, size_t>(
                data.get(), stride, mask.get(), mask_stride, shape, [](TestType, int m) { return m; }, stream);

        cpu::memory::PtrHost<TestType> sequence_values(values_, extracted);
        cpu::memory::PtrHost<size_t> sequence_indexes(indexes_, extracted);
        REQUIRE(extracted == expected_indexes.size());
        REQUIRE(test::Matcher(test::MATCH_ABS, expected_indexes.data(), sequence_indexes.get(), extracted, 0));
        REQUIRE(test::Matcher(test::MATCH_ABS, expected_values.data(), sequence_values.get(), extracted, 0));

        cpu::memory::PtrHost<TestType> reinsert(elements);
        test::memset(reinsert.get(), elements, 0);
        cpu::memory::insert(sequence_values.get(), sequence_indexes.get(), extracted, reinsert.get(), stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, expected_data_reinsert.data(), reinsert.get(), elements, 0));
    }

    THEN("padded") {
        const size4_t pitch = shape + test::Randomizer<size_t>(5, 10).get() * size4_t{shape != 1};
        cpu::memory::PtrHost<TestType> padded(pitch.elements());
        test::memset(padded.get(), padded.elements(), 2);
        auto[_, indexes_, extracted] = cpu::memory::extract<void, size_t>(
                padded.get(), pitch.strides(), shape, [](TestType v) { return v > 1; }, stream);
        cpu::memory::PtrHost<size_t> sequence_indexes(indexes_, extracted);

        REQUIRE(extracted == elements); // elements in pitch should not be selected
        const size_t last = at(shape - 1, pitch.strides());
        REQUIRE(sequence_indexes[extracted - 1] == last); // indexes should follow the physical layout of the input
    }
}

TEMPLATE_TEST_CASE("cpu::memory::atlasLayout(), insert()", "[noa][cpu][memory]", float, int) {
    const uint ndim = GENERATE(2U, 3U);
    test::Randomizer<uint> dim_randomizer(40, 60);
    const size4_t subregion_shape{test::Randomizer<size_t>(1, 40).get(), // subregion count
                                  ndim == 3 ? dim_randomizer.get() : 1, dim_randomizer.get(), dim_randomizer.get()};
    const size4_t subregion_stride = subregion_shape.strides();
    cpu::Stream stream;

    // Prepare subregions.
    cpu::memory::PtrHost<TestType> subregions(subregion_shape.elements());
    for (uint idx = 0; idx < subregion_shape[0]; ++idx)
        cpu::memory::set(subregions.get() + idx * subregion_stride[0],
                         subregion_stride[0], static_cast<TestType>(idx));

    // Insert into atlas.
    cpu::memory::PtrHost<int4_t> atlas_origins(subregion_shape[0]);
    const size4_t atlas_shape = cpu::memory::atlasLayout(subregion_shape, atlas_origins.get());
    for (uint idx = 0; idx < subregion_shape[0]; ++idx)
        REQUIRE((atlas_origins[idx][0] == 0 && atlas_origins[idx][1] == 0));

    cpu::memory::PtrHost<TestType> atlas(atlas_shape.elements());
    cpu::memory::insert(subregions.get(), subregion_stride, subregion_shape,
                        atlas.get(), atlas_shape.strides(), atlas_shape,
                        atlas_origins.get(), stream);

    // Extract from atlas
    cpu::memory::PtrHost<TestType> o_subregions(subregions.elements());
    cpu::memory::extract(atlas.get(), atlas_shape.strides(), atlas_shape,
                         o_subregions.get(), subregion_stride, subregion_shape,
                         atlas_origins.get(), BORDER_ZERO, TestType{0}, stream);

    REQUIRE(test::Matcher(test::MATCH_ABS, subregions.get(), o_subregions.get(), subregions.elements(), 0));
}
