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
    cpu::Stream stream(cpu::Stream::DEFAULT);

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto shape = test["shape"].as<size4_t>();
        const auto subregion_shape = test["sub_shape"].as<size4_t>();
        auto subregion_origins = test["sub_origins"].as<std::vector<int4_t>>();
        const auto border_mode = test["border"].as<BorderMode>();
        const auto border_value = test["border_value"].as<float>();
        const size4_t subregion_stride = subregion_shape.strides();

        cpu::memory::PtrHost<float> input(shape.elements());
        cpu::memory::PtrHost<float> subregions(subregion_shape.elements());
        cpu::memory::set(subregions.begin(), subregions.end(), 4.f);
        test::arange(input.get(), input.elements());

        const shared_t<int4_t[]> origins = input.attach(subregion_origins.data());

        // Extract:
        cpu::memory::extract<float>(input.share(), shape.strides(), shape,
                                    subregions.share(), subregion_stride, subregion_shape,
                                    origins, border_mode, border_value, stream);

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
        cpu::memory::insert<float>(subregions.share(), subregion_stride, subregion_shape,
                                   input.share(), shape.strides(), shape,
                                   origins, stream);

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
    cpu::Stream stream(cpu::Stream::DEFAULT);

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
        cpu::memory::Extracted<TestType, size_t> extracted =
                cpu::memory::extract<TestType, size_t>(data.share(), stride, mask.share(), mask_stride, shape,
                                                       [](int m) { return m; }, true, true, stream);

        REQUIRE(extracted.count == expected_indexes.size());
        REQUIRE(test::Matcher(test::MATCH_ABS, expected_indexes.data(), extracted.offsets.get(), extracted.count, 0));
        REQUIRE(test::Matcher(test::MATCH_ABS, expected_values.data(), extracted.values.get(), extracted.count, 0));

        cpu::memory::PtrHost<TestType> reinsert(elements);
        test::memset(reinsert.get(), elements, 0);
        cpu::memory::insert(extracted, reinsert.share(), stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, expected_data_reinsert.data(), reinsert.get(), elements, 0));
    }

    THEN("padded") {
        const size4_t pitch = shape + test::Randomizer<size_t>(5, 10).get() * size4_t{shape != 1};
        cpu::memory::PtrHost<TestType> padded(pitch.elements());
        test::memset(padded.get(), padded.elements(), 2);
        cpu::memory::Extracted<TestType, size_t> extracted = cpu::memory::extract<TestType, size_t>(
                padded.share(), pitch.strides(), padded.share(), pitch.strides(), shape,
                [](TestType v) { return v > 1; }, false, true, stream);

        REQUIRE(extracted.count == elements); // elements in pitch should not be selected
        const size_t last = indexing::at(shape - 1, pitch.strides());
        REQUIRE(extracted.offsets.get()[extracted.count - 1] == last); // indexes should follow the physical layout
    }
}

TEMPLATE_TEST_CASE("cpu::memory::atlasLayout(), insert()", "[noa][cpu][memory]", float, int) {
    const uint ndim = GENERATE(2U, 3U);
    test::Randomizer<uint> dim_randomizer(40, 60);
    const size4_t subregion_shape{test::Randomizer<size_t>(1, 40).get(), // subregion count
                                  ndim == 3 ? dim_randomizer.get() : 1, dim_randomizer.get(), dim_randomizer.get()};
    const size4_t subregion_stride = subregion_shape.strides();
    cpu::Stream stream(cpu::Stream::DEFAULT);

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
    cpu::memory::insert<TestType>(subregions.share(), subregion_stride, subregion_shape,
                                  atlas.share(), atlas_shape.strides(), atlas_shape,
                                  atlas_origins.share(), stream);

    // Extract from atlas
    cpu::memory::PtrHost<TestType> o_subregions(subregions.elements());
    cpu::memory::extract<TestType>(atlas.share(), atlas_shape.strides(), atlas_shape,
                                   o_subregions.share(), subregion_stride, subregion_shape,
                                   atlas_origins.share(), BORDER_ZERO, TestType{0}, stream);

    REQUIRE(test::Matcher(test::MATCH_ABS, subregions.get(), o_subregions.get(), subregions.elements(), 0));
}
