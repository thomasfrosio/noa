#include <fstream>
#include <noa/common/io/IO.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEMPLATE_TEST_CASE("io::(de)serialize - real types", "[noa][common][io]",
                   uint8_t, short, int, uint, half_t, float, double) {
    path_t test_dir = "testIO";
    path_t test_file = test_dir / "test.bin";
    fs::create_directory(test_dir);

    auto elements = test::Randomizer<size_t>(0, 10'000).get();
    io::DataType dtype = GENERATE(io::DataType::INT8, io::DataType::UINT8,
                                  io::DataType::INT16, io::DataType::UINT16,
                                  io::DataType::INT32, io::DataType::UINT32,
                                  io::DataType::INT64, io::DataType::UINT64,
                                  io::DataType::FLOAT16, io::DataType::FLOAT32, io::DataType::FLOAT64);
    bool clamp = GENERATE(true, false);
    bool swap = GENERATE(true, false);
    INFO("size: " << elements << ", dtype: " << dtype << ", clamp:" << clamp << ", swap: " << swap);

    std::unique_ptr<TestType[]> data = std::make_unique<TestType[]>(elements);
    std::unique_ptr<TestType[]> read_data = std::make_unique<TestType[]>(elements);

    // Randomize data. No decimals, otherwise it makes everything more complicated.
    int64_t range_min, range_max;
    if (clamp) {
        range_min = dtype == io::DataType::FLOAT16 ? -2048 : -30000;
        range_max = dtype == io::DataType::FLOAT16 ? 2048 : 30000;
    } else {
        range_min = 0;
        range_max = 127;
    }
    test::Randomizer<int64_t> randomizer(range_min, range_max);
    for (size_t i = 0; i < elements; ++i)
        data[i] = clamp_cast<TestType>(randomizer.get());

    // Serialize:
    std::fstream file(test_file, std::ios::out | std::ios::trunc);
    io::serialize(data.get(), file, dtype, elements, clamp, swap);
    file.close();
    REQUIRE(fs::file_size(test_file) == io::getSerializedSize(dtype, elements));

    // Deserialize:
    file.open(test_file, std::ios::in);
    io::deserialize(file, dtype, read_data.get(), elements, clamp, swap);

    if (clamp) {
        // Serialized data was clamped to fit the data type, so clamp input data as well.
        TestType min{}, max{};
        io::getDataTypeMinMax<TestType>(dtype, &min, &max);
        for (size_t i = 0; i < elements; ++i)
            data[i] = math::clamp(data[i], min, max);
    }

    REQUIRE(test::Matcher<TestType>(test::MATCH_ABS, data.get(), read_data.get(), elements, 1e-6));
    fs::remove_all("testIO");
}

TEMPLATE_TEST_CASE("io::(de)serialize - uint4", "[noa][common][io]", uint8_t, short, int, uint, half_t, float) {
    path_t test_dir = "testIO";
    path_t test_file = test_dir / "test.bin";
    fs::create_directory(test_dir);

    bool clamp = GENERATE(true, false);
    bool swap = GENERATE(true, false);
    auto rows = test::Randomizer<size_t>(0, 500).get();
    auto elements_per_row = test::Randomizer<size_t>(0, 500).get();
    auto elements = elements_per_row * rows;
    io::DataType dtype = io::DataType::UINT4;
    INFO("size: " << elements << ", clamp:" << clamp << ", swap: " << swap);

    std::unique_ptr<TestType[]> data = std::make_unique<TestType[]>(elements);
    std::unique_ptr<TestType[]> read_data = std::make_unique<TestType[]>(elements);

    // Randomize data. No decimals, otherwise it makes everything more complicated.
    test::Randomizer<int64_t> randomizer(0, clamp ? 30 : 15);
    for (size_t i = 0; i < elements; ++i)
        data[i] = clamp_cast<TestType>(randomizer.get());

    // Serialize:
    std::fstream file(test_file, std::ios::out | std::ios::trunc);
    io::serialize(data.get(), file, dtype, elements, clamp, swap, elements_per_row);
    file.close();
    REQUIRE(fs::file_size(test_file) == io::getSerializedSize(dtype, elements, elements_per_row));

    // Deserialize:
    file.open(test_file, std::ios::in);
    io::deserialize(file, dtype, read_data.get(), elements, clamp, swap, elements_per_row);

    if (clamp) {
        // Serialized data was clamped to fit the data type, so clamp input data as well.
        TestType min, max;
        io::getDataTypeMinMax<TestType>(dtype, &min, &max);
        for (size_t i = 0; i < elements; ++i)
            data[i] = math::clamp(data[i], min, max);
    }

    REQUIRE(test::Matcher<TestType>(test::MATCH_ABS, data.get(), read_data.get(), elements, 1e-6));
    fs::remove_all("testIO");
}

TEMPLATE_TEST_CASE("io::(de)serialize - complex", "[noa][common][io]", chalf_t, cfloat_t, cdouble_t) {
    path_t test_dir = "testIO";
    path_t test_file = test_dir / "test.bin";
    fs::create_directory(test_dir);

    bool clamp = GENERATE(true, false);
    bool swap = GENERATE(true, false);
    io::DataType dtype = GENERATE(io::DataType::CFLOAT16, io::DataType::CFLOAT32, io::DataType::CFLOAT64);
    auto elements = test::Randomizer<size_t>(0, 1000).get();
    INFO("size: " << elements << ", clamp:" << clamp << ", swap: " << swap);

    std::unique_ptr<TestType[]> data = std::make_unique<TestType[]>(elements);
    std::unique_ptr<TestType[]> read_data = std::make_unique<TestType[]>(elements);

    // Randomize data.
    test::Randomizer<float> randomizer(-10000, 10000);
    for (size_t i = 0; i < elements; ++i)
        data[i] = static_cast<TestType>(randomizer.get());

    // Serialize:
    std::fstream file(test_file, std::ios::out | std::ios::trunc);
    io::serialize(data.get(), file, dtype, elements, clamp, swap);
    file.close();
    REQUIRE(fs::file_size(test_file) == io::getSerializedSize(dtype, elements));

    // Deserialize:
    file.open(test_file, std::ios::in);
    io::deserialize(file, dtype, read_data.get(), elements, clamp, swap);

    if (dtype == io::DataType::CFLOAT16) {
        for (size_t i = 0; i < elements; ++i)
            data[i] = static_cast<TestType>(static_cast<chalf_t>(data[i])); // for half, mimic conversion on raw data
    }
    REQUIRE(test::Matcher<TestType>(test::MATCH_ABS, data.get(), read_data.get(), elements, 1e-6));
    fs::remove_all("testIO");
}

TEST_CASE("io::(de)serialize - many elements", "[noa][common][io]") {
    path_t test_dir = "testIO";
    path_t test_file = test_dir / "test.bin";
    fs::create_directory(test_dir);

    bool clamp = false;
    bool swap = true;
    io::DataType dtype = io::DataType::INT16;
    size_t elements = (1 << 26) + test::Randomizer<size_t>(0, 1000).get(); // should be 3 batches

    std::unique_ptr<float[]> data = std::make_unique<float[]>(elements);
    std::unique_ptr<float[]> read_data = std::make_unique<float[]>(elements);

    // Randomize data.
    test::Randomizer<float> randomizer(-10000, 10000);
    for (size_t i = 0; i < elements; ++i)
        data[i] = randomizer.get();

    // Serialize:
    std::fstream file(test_file, std::ios::out | std::ios::trunc);
    io::serialize(data.get(), file, dtype, elements, clamp, swap);
    file.close();
    REQUIRE(fs::file_size(test_file) == io::getSerializedSize(dtype, elements));

    // Deserialize:
    file.open(test_file, std::ios::in);
    io::deserialize(file, dtype, read_data.get(), elements, clamp, swap);

    for (size_t i = 0; i < elements; ++i)
        data[i] = std::trunc(data[i]); // float/double -> int16 -> float/double
    float diff = test::getDifference(data.get(), read_data.get(), elements);
    REQUIRE_THAT(diff, Catch::WithinULP(0.f, 2));

    fs::remove_all("testIO");
}

TEST_CASE("io::swapEndian()", "[noa][common][io]") {
    std::unique_ptr<float[]> data1 = std::make_unique<float[]>(100);
    std::unique_ptr<float[]> data2 = std::make_unique<float[]>(100);
    for (size_t i{0}; i < 100; ++i) {
        auto t = static_cast<float>(test::Randomizer<int>(-1234434, 94321458).get());
        data1[i] = t;
        data2[i] = t;
    }
    io::swapEndian(data1.get(), 100);
    io::swapEndian(data1.get(), 100);
    float diff{0};
    for (size_t i{0}; i < 100; ++i)
        diff += data1[i] - data2[i];
    REQUIRE_THAT(diff, Catch::WithinULP(0.f, 2));
}
