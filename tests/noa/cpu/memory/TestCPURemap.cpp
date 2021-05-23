#include <noa/cpu/memory/Remap.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Set.h>
#include <noa/io/files/ImageFile.h>
#include <noa/io/files/MRCFile.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

// Test against manually checked data.
static constexpr bool COMPUTE_TEST_DATA_INSTEAD = false;

TEST_CASE("Memory::extract(), insert()", "[noa][memory]") {
    size3_t input_shape;
    size3_t subregion_shape;
    uint subregion_count;
    size3_t subregion_centers[5]; // 5 subregions at most.
    BorderMode border_mode;
    float border_value;

    int test_number = GENERATE(1, 2, 3);
    Test::Assets::Memory::getExtractParams(test_number, &input_shape,
                                           &subregion_shape, subregion_centers, &subregion_count,
                                           &border_mode, &border_value);

    size_t input_elements = getElements(input_shape);
    size_t subregion_elements = getElements(subregion_shape);
    Memory::PtrHost<float> input(input_elements);
    Memory::PtrHost<float> subregions(subregion_elements * subregion_count);
    Test::Assets::Memory::initExtractInput(input.get(), input_elements);
    Test::Assets::Memory::initInsertOutput(subregions.get(), subregion_elements * subregion_count);

    Memory::extract(input.get(), input_shape,
                    subregions.get(), subregion_shape, subregion_centers, subregion_count,
                    border_mode, border_value);

    if constexpr (COMPUTE_TEST_DATA_INSTEAD) {
        Test::Assets::Memory::initInsertOutput(input.get(), input.elements());
        for (uint i = 0; i < subregion_count; ++i) {
            path_t tmp = Test::Assets::Memory::getExtractFilename(test_number, i);
            float* subregion = subregions.get() + i * subregion_elements;
            ImageFile::save(tmp, subregion, subregion_shape);
            Memory::insert(subregion, subregion_shape, subregion_centers[i], input.get(), input_shape);
        }
        path_t tmp = Test::Assets::Memory::getInsertFilename(test_number);
        ImageFile::save(tmp, input.get(), input_shape);
    }

    MRCFile file;
    Memory::PtrHost<float> expected_subregions(subregion_elements * subregion_count);
    for (uint i = 0; i < subregion_count; ++i) {
        path_t tmp = Test::Assets::Memory::getExtractFilename(test_number, i);
        float* expected_subregion = expected_subregions.get() + i * subregion_elements;
        file.open(tmp, IO::READ);
        file.readAll(expected_subregion);
    }
    float diff = Test::getDifference(expected_subregions.get(), subregions.get(), subregions.elements());
    REQUIRE_THAT(diff, Test::isWithinAbs(0.f, 1e-7));

    // Insert these subregions back to the input frame.
    // Note: We could have used the copy input, insert and make sure this is equal to input...
    Memory::PtrHost<float> insert_back(input.elements());
    Memory::PtrHost<float> expected_insert_back(input.elements());
    path_t tmp = Test::Assets::Memory::getInsertFilename(test_number);
    file.open(tmp, IO::READ);
    file.readAll(expected_insert_back.get());

    Test::Assets::Memory::initInsertOutput(insert_back.get(), input.elements());
    for (uint i = 0; i < subregion_count; ++i) {
        float* subregion = subregions.get() + i * subregion_elements;
        Memory::insert(subregion, subregion_shape, subregion_centers[i], insert_back.get(), input_shape);
    }
    diff = Test::getDifference(expected_insert_back.get(), insert_back.get(), insert_back.elements());
    REQUIRE_THAT(diff, Test::isWithinAbs(0.f, 1e-7));
}

TEMPLATE_TEST_CASE("Memory::getMap(), extract(), insert()", "[noa][memory]", float, int) {
    size_t elements = Test::IntRandomizer<size_t>(4000, 500000).get();
    Test::IntRandomizer<size_t> index_randomizer(size_t{0}, elements - 1);

    // Init data
    Test::Randomizer<TestType> data_randomizer(1., 100.);
    Memory::PtrHost<TestType> i_sparse(elements);
    Test::initDataRandom(i_sparse.get(), i_sparse.elements(), data_randomizer);

    // Prepare expected data
    Test::IntRandomizer<int> mask_randomizer(0, 4);
    Memory::PtrHost<int> mask(elements);
    Test::initDataRandom(mask.get(), elements, mask_randomizer);

    std::vector<size_t> expected_map;
    std::vector<TestType> expected_dense;
    Memory::PtrHost<TestType> expected_insert(elements);
    Test::initDataZero(expected_insert.get(), elements);
    for (size_t i = 0; i < elements; ++i) {
        if (mask[i] == 0)
            continue;
        expected_map.emplace_back(i);
        expected_dense.emplace_back(i_sparse[i]);
        expected_insert[i] = i_sparse[i];
    }

    THEN("getMap") {
        auto[tmp_map, elements_mapped] = Memory::getMap(mask.get(), elements, 0);
        Memory::PtrHost<size_t> map(tmp_map, elements_mapped);
        REQUIRE(elements_mapped == expected_map.size());
        size_t size = Test::getDifference(expected_map.data(), map.get(), elements_mapped);
        REQUIRE(size == 0);

        THEN("extract, insert") {
            Memory::PtrHost<TestType> dense(expected_map.size());
            Memory::extract(i_sparse.get(), i_sparse.elements(), dense.get(), dense.elements(), expected_map.data(), 1);
            TestType diff = Test::getDifference(expected_dense.data(), dense.get(), dense.elements());
            REQUIRE(diff == 0);

            Memory::PtrHost<TestType> insert(elements);
            Test::initDataZero(insert.get(), elements);
            Memory::insert(dense.get(), dense.elements(), insert.get(), insert.elements(), expected_map.data(), 1);
            diff = Test::getDifference(expected_insert.get(), insert.get(), insert.elements());
            REQUIRE(diff == 0);
        }
    }
}

TEMPLATE_TEST_CASE("Memory::getAtlasLayout(), insert()", "[noa][cpu]", float, int) {
    uint ndim = GENERATE(2U, 3U);
    Test::IntRandomizer<uint> dim_randomizer(40, 60);
    size3_t subregion_shape(dim_randomizer.get(), dim_randomizer.get(), ndim == 3 ? dim_randomizer.get() : 1);
    uint subregion_count = Test::IntRandomizer<uint>(1, 40).get();
    size_t elements = getElements(subregion_shape);
    Memory::PtrHost<TestType> subregions(elements * subregion_count);

    for (uint idx = 0; idx < subregion_count; ++idx)
        Memory::set(subregions.get() + idx * elements, elements, static_cast<TestType>(idx));

    // Insert atlas
    Memory::PtrHost<size3_t> atlas_centers(subregion_count);
    size3_t atlas_shape = Memory::getAtlasLayout(subregion_shape, subregion_count, atlas_centers.get());
    Memory::PtrHost<TestType> atlas(getElements(atlas_shape));
    Memory::insert(subregions.get(), subregion_shape, atlas_centers.get(), subregion_count, atlas.get(), atlas_shape);

    // Extract atlas
    Memory::PtrHost<TestType> o_subregions(elements * subregion_count);
    Memory::extract(atlas.get(), atlas_shape, o_subregions.get(), subregion_shape, atlas_centers.get(), subregion_count,
                    BORDER_ZERO, TestType{0});

    TestType diff = Test::getDifference(subregions.get(), o_subregions.get(), subregions.elements());
    REQUIRE(diff == 0);
}
