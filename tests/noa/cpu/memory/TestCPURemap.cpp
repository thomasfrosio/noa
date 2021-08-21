#include <noa/common/files/ImageFile.h>
#include <noa/common/files/MRCFile.h>
#include <noa/cpu/memory/Remap.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Set.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

// Test against manually checked data.
static constexpr bool COMPUTE_TEST_DATA_INSTEAD = false;

TEST_CASE("cpu::memory::extract(), insert()", "[noa][cpu][memory]") {
    size3_t input_shape;
    size3_t subregion_shape;
    uint subregion_count;
    size3_t subregion_centers[5]; // 5 subregions at most.
    BorderMode border_mode;
    float border_value;

    int test_number = GENERATE(1, 2, 3);
    test::assets::memory::getExtractParams(test_number, &input_shape,
                                           &subregion_shape, subregion_centers, &subregion_count,
                                           &border_mode, &border_value);

    size_t input_elements = getElements(input_shape);
    size_t subregion_elements = getElements(subregion_shape);
    cpu::memory::PtrHost<float> input(input_elements);
    cpu::memory::PtrHost<float> subregions(subregion_elements * subregion_count);
    test::assets::memory::initExtractInput(input.get(), input_elements);
    test::assets::memory::initInsertOutput(subregions.get(), subregion_elements * subregion_count);

    cpu::memory::extract(input.get(), input_shape,
                    subregions.get(), subregion_shape, subregion_centers, subregion_count,
                    border_mode, border_value);

    if constexpr (COMPUTE_TEST_DATA_INSTEAD) {
        test::assets::memory::initInsertOutput(input.get(), input.elements());
        for (uint i = 0; i < subregion_count; ++i) {
            path_t tmp = test::assets::memory::getExtractFilename(test_number, i);
            float* subregion = subregions.get() + i * subregion_elements;
            ImageFile::save(tmp, subregion, subregion_shape);
            cpu::memory::insert(subregion, subregion_shape, subregion_centers[i], input.get(), input_shape);
        }
        path_t tmp = test::assets::memory::getInsertFilename(test_number);
        ImageFile::save(tmp, input.get(), input_shape);
    }

    MRCFile file;
    cpu::memory::PtrHost<float> expected_subregions(subregion_elements * subregion_count);
    for (uint i = 0; i < subregion_count; ++i) {
        path_t tmp = test::assets::memory::getExtractFilename(test_number, i);
        float* expected_subregion = expected_subregions.get() + i * subregion_elements;
        file.open(tmp, io::READ);
        file.readAll(expected_subregion);
    }
    float diff = test::getDifference(expected_subregions.get(), subregions.get(), subregions.elements());
    REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-7));

    // Insert these subregions back to the input frame.
    // Note: We could have used the copy input, insert and make sure this is equal to input...
    cpu::memory::PtrHost<float> insert_back(input.elements());
    cpu::memory::PtrHost<float> expected_insert_back(input.elements());
    path_t tmp = test::assets::memory::getInsertFilename(test_number);
    file.open(tmp, io::READ);
    file.readAll(expected_insert_back.get());

    test::assets::memory::initInsertOutput(insert_back.get(), input.elements());
    for (uint i = 0; i < subregion_count; ++i) {
        float* subregion = subregions.get() + i * subregion_elements;
        cpu::memory::insert(subregion, subregion_shape, subregion_centers[i], insert_back.get(), input_shape);
    }
    diff = test::getDifference(expected_insert_back.get(), insert_back.get(), insert_back.elements());
    REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-7));
}

TEMPLATE_TEST_CASE("memory::getMap(), extract(), insert()", "[noa][memory]", float, int) {
    size_t elements = test::IntRandomizer<size_t>(4000, 500000).get();
    test::IntRandomizer<size_t> index_randomizer(size_t{0}, elements - 1);

    // Init data
    test::Randomizer<TestType> data_randomizer(1., 100.);
    cpu::memory::PtrHost<TestType> i_sparse(elements);
    test::initDataRandom(i_sparse.get(), i_sparse.elements(), data_randomizer);

    // Prepare expected data
    test::IntRandomizer<int> mask_randomizer(0, 4);
    cpu::memory::PtrHost<int> mask(elements);
    test::initDataRandom(mask.get(), elements, mask_randomizer);

    std::vector<size_t> expected_map;
    std::vector<TestType> expected_dense;
    cpu::memory::PtrHost<TestType> expected_insert(elements);
    test::initDataZero(expected_insert.get(), elements);
    for (size_t i = 0; i < elements; ++i) {
        if (mask[i] == 0)
            continue;
        expected_map.emplace_back(i);
        expected_dense.emplace_back(i_sparse[i]);
        expected_insert[i] = i_sparse[i];
    }

    THEN("getMap") {
        auto[tmp_map, elements_mapped] = cpu::memory::getMap(mask.get(), elements, 0);
        cpu::memory::PtrHost<size_t> map(tmp_map, elements_mapped);
        REQUIRE(elements_mapped == expected_map.size());
        size_t size = test::getDifference(expected_map.data(), map.get(), elements_mapped);
        REQUIRE(size == 0);

        THEN("extract, insert") {
            cpu::memory::PtrHost<TestType> dense(expected_map.size());
            cpu::memory::extract(i_sparse.get(), i_sparse.elements(), dense.get(), dense.elements(), expected_map.data(), 1);
            TestType diff = test::getDifference(expected_dense.data(), dense.get(), dense.elements());
            REQUIRE(diff == 0);

            cpu::memory::PtrHost<TestType> insert(elements);
            test::initDataZero(insert.get(), elements);
            cpu::memory::insert(dense.get(), dense.elements(), insert.get(), insert.elements(), expected_map.data(), 1);
            diff = test::getDifference(expected_insert.get(), insert.get(), insert.elements());
            REQUIRE(diff == 0);
        }
    }

    THEN("getMap, padded") {
        size3_t shape = test::getRandomShape(3U);
        size_t pitch = shape.x + test::IntRandomizer<size_t>(10, 100).get();
        size_t p_elements = pitch * getRows(shape);
        cpu::memory::PtrHost<TestType> padded(p_elements);
        for (auto& e: padded)
            e = static_cast<TestType>(2);
        auto[tmp_map, elements_mapped] = cpu::memory::getMap<uint>(padded.get(), pitch, shape, static_cast<TestType>(1));
        cpu::memory::PtrHost<uint> map(tmp_map, elements_mapped);
        REQUIRE(elements_mapped == getElements(shape)); // elements in pitch should not be selected
        uint index_last = static_cast<uint>(p_elements - (pitch - shape.x) - 1); // index of the last valid element
        INFO(pitch);
        REQUIRE(map[elements_mapped - 1] == index_last); // the indexes should follow the physical layout of the input array
    }
}

TEMPLATE_TEST_CASE("memory::getAtlasLayout(), insert()", "[noa][cpu]", float, int) {
    uint ndim = GENERATE(2U, 3U);
    test::IntRandomizer<uint> dim_randomizer(40, 60);
    size3_t subregion_shape(dim_randomizer.get(), dim_randomizer.get(), ndim == 3 ? dim_randomizer.get() : 1);
    uint subregion_count = test::IntRandomizer<uint>(1, 40).get();
    size_t elements = getElements(subregion_shape);
    cpu::memory::PtrHost<TestType> subregions(elements * subregion_count);

    for (uint idx = 0; idx < subregion_count; ++idx)
        cpu::memory::set(subregions.get() + idx * elements, elements, static_cast<TestType>(idx));

    // Insert atlas
    cpu::memory::PtrHost<size3_t> atlas_centers(subregion_count);
    size3_t atlas_shape = cpu::memory::getAtlasLayout(subregion_shape, subregion_count, atlas_centers.get());
    cpu::memory::PtrHost<TestType> atlas(getElements(atlas_shape));
    cpu::memory::insert(subregions.get(), subregion_shape, atlas_centers.get(),
                        subregion_count, atlas.get(), atlas_shape);

    // Extract atlas
    cpu::memory::PtrHost<TestType> o_subregions(elements * subregion_count);
    cpu::memory::extract(atlas.get(), atlas_shape, o_subregions.get(), subregion_shape,
                         atlas_centers.get(), subregion_count, BORDER_ZERO, TestType{0});

    TestType diff = test::getDifference(subregions.get(), o_subregions.get(), subregions.elements());
    REQUIRE(diff == 0);
}
