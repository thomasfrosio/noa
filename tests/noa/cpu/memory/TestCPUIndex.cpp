#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/Index.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Set.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cpu::memory::extract(), insert()", "[assets][noa][cpu][memory]") {
    constexpr bool COMPUTE_ASSETS = false;
    path_t path_base = test::PATH_NOA_DATA / "memory";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["remap"];
    io::ImageFile file;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        auto shape = test["shape"].as<size3_t>();
        auto subregion_shape = test["sub_shape"].as<size3_t>();
        auto subregion_centers = test["sub_centers"].as<std::vector<size3_t>>();
        auto border_mode = test["border"].as<BorderMode>();
        auto border_value = test["border_value"].as<float>();

        cpu::memory::PtrHost<float> input(noa::elements(shape));
        size_t subregion_elements = noa::elements(subregion_shape);
        size_t subregion_count = subregion_centers.size();
        cpu::memory::PtrHost<float> subregions(subregion_elements * subregion_count);
        cpu::memory::set(subregions.begin(), subregions.end(), 4.f);
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = float(i);

        // Extract:
        cpu::memory::extract(input.get(), shape,
                             subregions.get(), subregion_shape, subregion_centers.data(), uint(subregion_count),
                             border_mode, border_value);

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
        cpu::memory::insert(subregions.get(), subregion_shape, subregion_centers.data(), uint(subregion_count),
                            input.get(), shape);

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

TEMPLATE_TEST_CASE("cpu::memory::where(), extract(), insert()", "[noa][cpu][memory]", float, int) {
    size_t elements = test::Randomizer<size_t>(4000, 500000).get();
    test::Randomizer<size_t> index_randomizer(size_t{0}, elements - 1);

    // Init data
    test::Randomizer<TestType> data_randomizer(1., 100.);
    cpu::memory::PtrHost<TestType> i_sparse(elements);
    test::randomize(i_sparse.get(), i_sparse.elements(), data_randomizer);

    // Prepare expected data
    test::Randomizer<int> mask_randomizer(0, 4);
    cpu::memory::PtrHost<int> mask(elements);
    test::randomize(mask.get(), elements, mask_randomizer);

    std::vector<size_t> expected_map;
    std::vector<TestType> expected_dense;
    cpu::memory::PtrHost<TestType> expected_insert(elements);
    test::memset(expected_insert.get(), elements, 0);
    for (size_t i = 0; i < elements; ++i) {
        if (mask[i] == 0)
            continue;
        expected_map.emplace_back(i);
        expected_dense.emplace_back(i_sparse[i]);
        expected_insert[i] = i_sparse[i];
    }

    THEN("where") {
        auto[tmp_map, elements_mapped] = cpu::memory::where(mask.get(), elements, 0);
        cpu::memory::PtrHost<size_t> map(tmp_map, elements_mapped);
        REQUIRE(elements_mapped == expected_map.size());
        size_t size = test::getDifference(expected_map.data(), map.get(), elements_mapped);
        REQUIRE(size == 0);

        THEN("extract, insert") {
            cpu::memory::PtrHost<TestType> dense(expected_map.size());
            cpu::memory::extract(i_sparse.get(), i_sparse.elements(), dense.get(), dense.elements(),
                                 expected_map.data(), 1);
            TestType diff = test::getDifference(expected_dense.data(), dense.get(), dense.elements());
            REQUIRE(diff == 0);

            cpu::memory::PtrHost<TestType> insert(elements);
            test::memset(insert.get(), elements, 0);
            cpu::memory::insert(dense.get(), dense.elements(), insert.get(), insert.elements(), expected_map.data(), 1);
            diff = test::getDifference(expected_insert.get(), insert.get(), insert.elements());
            REQUIRE(diff == 0);
        }
    }

    THEN("where, padded") {
        size3_t shape = test::getRandomShape(3U);
        size_t pitch = shape.x + test::Randomizer<size_t>(10, 100).get();
        size_t p_elements = pitch * rows(shape);
        cpu::memory::PtrHost<TestType> padded(p_elements);
        for (auto& e: padded)
            e = static_cast<TestType>(2);
        auto[tmp_map, elements_mapped] = cpu::memory::where<uint>(padded.get(), pitch, shape, 1,
                                                                  static_cast<TestType>(1));
        cpu::memory::PtrHost<uint> map(tmp_map, elements_mapped);
        REQUIRE(elements_mapped == noa::elements(shape)); // elements in pitch should not be selected
        uint index_last = static_cast<uint>(p_elements - (pitch - shape.x) - 1); // index of the last valid element
        INFO(pitch);
        REQUIRE(map[elements_mapped - 1] == index_last); // indexes should follow the physical layout of the input
    }
}

TEMPLATE_TEST_CASE("cpu::memory::atlasLayout(), insert()", "[noa][cpu][memory]", float, int) {
    uint ndim = GENERATE(2U, 3U);
    test::Randomizer<uint> dim_randomizer(40, 60);
    size3_t subregion_shape(dim_randomizer.get(), dim_randomizer.get(), ndim == 3 ? dim_randomizer.get() : 1);
    uint subregion_count = test::Randomizer<uint>(1, 40).get();
    size_t elements = noa::elements(subregion_shape);
    cpu::memory::PtrHost<TestType> subregions(elements * subregion_count);

    for (uint idx = 0; idx < subregion_count; ++idx)
        cpu::memory::set(subregions.get() + idx * elements, elements, static_cast<TestType>(idx));

    // Insert atlas
    cpu::memory::PtrHost<size3_t> atlas_centers(subregion_count);
    size3_t atlas_shape = cpu::memory::atlasLayout(subregion_shape, subregion_count, atlas_centers.get());
    cpu::memory::PtrHost<TestType> atlas(noa::elements(atlas_shape));
    cpu::memory::insert(subregions.get(), subregion_shape, atlas_centers.get(),
                        subregion_count, atlas.get(), atlas_shape);

    // Extract atlas
    cpu::memory::PtrHost<TestType> o_subregions(elements * subregion_count);
    cpu::memory::extract(atlas.get(), atlas_shape, o_subregions.get(), subregion_shape,
                         atlas_centers.get(), subregion_count, BORDER_ZERO, TestType{0});

    TestType diff = test::getDifference(subregions.get(), o_subregions.get(), subregions.elements());
    REQUIRE(diff == 0);
}
