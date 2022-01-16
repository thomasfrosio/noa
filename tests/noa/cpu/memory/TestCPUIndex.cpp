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
    path_t path_base = test::PATH_NOA_DATA / "memory";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["index"];
    io::ImageFile file;
    cpu::Stream stream;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        auto shape = test["shape"].as<size3_t>();
        auto subregion_shape = test["sub_shape"].as<size3_t>();
        auto subregion_origins = test["sub_origins"].as<std::vector<int3_t>>();
        auto border_mode = test["border"].as<BorderMode>();
        auto border_value = test["border_value"].as<float>();

        cpu::memory::PtrHost<float> input(noa::elements(shape));
        size_t subregion_elements = noa::elements(subregion_shape);
        size_t subregion_count = subregion_origins.size();
        cpu::memory::PtrHost<float> subregions(subregion_elements * subregion_count);
        cpu::memory::set(subregions.begin(), subregions.end(), 4.f);
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = float(i);

        // Extract:
        cpu::memory::extract(input.get(), {shape.x, shape.y, 0}, shape,
                             subregions.get(), subregion_shape, subregion_shape,
                             subregion_origins.data(), subregion_count,
                             border_mode, border_value, stream);

        auto expected_subregion_filenames = test["expected_extract"].as<std::vector<path_t>>();
        if constexpr (COMPUTE_ASSETS) {
            for (uint i = 0; i < subregion_count; ++i) {
                float* subregion = subregions.get() + i * subregion_elements;
                file.open(path_base / expected_subregion_filenames[i], io::WRITE);
                file.shape(subregion_shape);
                file.readAll(subregion);
            }
        } else {
            cpu::memory::PtrHost<float> expected_subregions(subregions.size());
            for (uint i = 0; i < subregion_count; ++i) {
                float* expected_subregion = expected_subregions.get() + i * subregion_elements;
                file.open(path_base / expected_subregion_filenames[i], io::READ);
                file.readAll(expected_subregion);
            }
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE,
                                  expected_subregions.get(), subregions.get(), subregions.size(), 1e-7));
        }

        // Insert:
        cpu::memory::set(input.begin(), input.end(), 4.f);
        cpu::memory::insert(subregions.get(), subregion_shape, subregion_shape,
                            input.get(), {shape.x, shape.y, 0}, shape,
                            subregion_origins.data(), subregion_count, stream);

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

TEMPLATE_TEST_CASE("cpu::memory::extract(), insert() - elements", "[noa][cpu][memory]", float, int) {
    const size3_t shape = test::getRandomShape(3);
    const size_t elements = noa::elements(shape);
    const size_t batches = test::Randomizer<size_t>(1, 4).get();
    cpu::Stream stream;

    // Initialize data.
    test::Randomizer<TestType> data_randomizer(1., 100.);
    cpu::memory::PtrHost<TestType> i_sparse(elements * batches);
    test::randomize(i_sparse.get(), i_sparse.size(), data_randomizer);

    // Prepare expected data.
    test::Randomizer<int> mask_randomizer(0, 4);
    cpu::memory::PtrHost<int> mask(elements); // not batched
    test::randomize(mask.get(), elements, mask_randomizer);

    // Extract elements from data only if mask isn't 0.
    std::vector<size_t> expected_indexes;
    std::vector<TestType> expected_values;
    cpu::memory::PtrHost<TestType> expected_insert(i_sparse.size());
    test::memset(expected_insert.get(), i_sparse.size(), 0);
    for (size_t batch = 0; batch < batches; ++batch) {
        for (size_t i = 0; i < elements; ++i) {
            if (mask[i] == 0)
                continue;
            const size_t index = batch * elements + i;
            expected_indexes.emplace_back(index);
            expected_values.emplace_back(i_sparse[index]);
            expected_insert[index] = i_sparse[index];
        }
    }

    THEN("contiguous") {
        auto[values_, indexes_, extracted] = cpu::memory::extract<true, size_t>(
                i_sparse.get(), shape, mask.get(), {shape.x, shape.y, 0}, shape, batches,
                [](TestType, int m) { return m; }, stream);

        cpu::memory::PtrHost<TestType> sequence_values(values_, extracted);
        cpu::memory::PtrHost<size_t> sequence_indexes(indexes_, extracted);
        REQUIRE(extracted == expected_indexes.size());
        REQUIRE(test::Matcher(test::MATCH_ABS, expected_indexes.data(), sequence_indexes.get(), extracted, 0));
        REQUIRE(test::Matcher(test::MATCH_ABS, expected_values.data(), sequence_values.get(), extracted, 0));

        cpu::memory::PtrHost<TestType> insert(elements * batches);
        test::memset(insert.get(), elements, 0);
        cpu::memory::insert(sequence_values.get(), sequence_indexes.get(), extracted, insert.get(), stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, expected_insert.data(), insert.get(), elements, 0));
    }

    THEN("padded") {
        size3_t pitch = shape + test::Randomizer<size_t>(5, 10).get();
        cpu::memory::PtrHost<TestType> padded(noa::elements(pitch) * batches);
        test::memset(padded.get(), padded.size(), 2);
        auto[_, indexes_, extracted] = cpu::memory::extract<false, size_t>(
                padded.get(), pitch, shape, batches, [](TestType v) { return v > 1; }, stream);
        cpu::memory::PtrHost<size_t> sequence_indexes(indexes_, extracted);

        REQUIRE(extracted == noa::elements(shape) * batches); // elements in pitch should not be selected
        const size_t last = index(shape - 1, pitch) + (batches - 1) * noa::elements(pitch);
        INFO(shape);
        INFO(batches);
        INFO(pitch);
        REQUIRE(sequence_indexes[extracted - 1] == last); // indexes should follow the physical layout of the input
    }
}

TEMPLATE_TEST_CASE("cpu::memory::atlasLayout(), insert()", "[noa][cpu][memory]", float, int) {
    const uint ndim = GENERATE(2U, 3U);
    test::Randomizer<uint> dim_randomizer(40, 60);
    const size3_t subregion_shape{dim_randomizer.get(), dim_randomizer.get(), ndim == 3 ? dim_randomizer.get() : 1};
    const size_t subregion_count = test::Randomizer<size_t>(1, 40).get();
    const size_t elements = noa::elements(subregion_shape);
    cpu::Stream stream;

    // Prepare subregions.
    cpu::memory::PtrHost<TestType> subregions(elements * subregion_count);
    for (uint idx = 0; idx < subregion_count; ++idx)
        cpu::memory::set(subregions.get() + idx * elements, elements, static_cast<TestType>(idx));

    // Insert into atlas.
    cpu::memory::PtrHost<int3_t> atlas_origins(subregion_count);
    const size3_t atlas_shape = cpu::memory::atlasLayout(subregion_shape, subregion_count, atlas_origins.get());
    cpu::memory::PtrHost<TestType> atlas(noa::elements(atlas_shape));
    cpu::memory::insert(subregions.get(), subregion_shape, subregion_shape,
                        atlas.get(), {atlas_shape.x, atlas_shape.y, 0}, atlas_shape,
                        atlas_origins.get(), subregion_count, stream);

    // Extract from atlas
    cpu::memory::PtrHost<TestType> o_subregions(subregions.size());
    cpu::memory::extract(atlas.get(), {atlas_shape.x, atlas_shape.y, 0}, atlas_shape,
                         o_subregions.get(), subregion_shape, subregion_shape,
                         atlas_origins.get(), subregion_count, BORDER_ZERO, TestType{0}, stream);

    REQUIRE(test::Matcher(test::MATCH_ABS, subregions.get(), o_subregions.get(), subregions.elements(), 0));
}
