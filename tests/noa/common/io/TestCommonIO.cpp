#include <fstream>
#include <noa/common/io/IO.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEMPLATE_TEST_CASE("io::(de)serialize - real types", "[noa][common][io]",
                   uint8_t, short, int, uint, half_t, float, double) {
    const path_t test_dir = "testIO";
    const path_t test_file = test_dir / "test.bin";
    fs::create_directory(test_dir);

    const size4_t shape = test::getRandomShapeBatched(1);
    const size_t elements = shape.elements();
    const io::DataType dtype =
            GENERATE(io::DataType::INT8, io::DataType::UINT8,
                     io::DataType::INT16, io::DataType::UINT16,
                     io::DataType::INT32, io::DataType::UINT32,
                     io::DataType::INT64, io::DataType::UINT64,
                     io::DataType::FLOAT16, io::DataType::FLOAT32, io::DataType::FLOAT64);
    const bool clamp = GENERATE(true, false);
    const bool swap = GENERATE(true, false);

    INFO("shape: " << shape << ", dtype: " << dtype << ", clamp:" << clamp << ", swap: " << swap);

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
    io::serialize(data.get(), shape.strides(), shape, file, dtype, clamp, swap);
    file.close();
    REQUIRE(fs::file_size(test_file) == io::serializedSize(dtype, elements));

    // Deserialize:
    file.open(test_file, std::ios::in);
    io::deserialize(file, dtype, read_data.get(), shape.strides(), shape, clamp, swap);

    if (clamp) {
        // Serialized data was clamped to fit the data type, so clamp input data as well.
        auto[min, max] = io::typeMinMax<TestType>(dtype);
        for (size_t i = 0; i < elements; ++i)
            data[i] = math::clamp(data[i], min, max);
    }

    REQUIRE(test::Matcher<TestType>(test::MATCH_ABS, data.get(), read_data.get(), elements, 1e-6));
    std::error_code er;
    fs::remove_all(test_dir, er); // silence error
}

TEMPLATE_TEST_CASE("io::(de)serialize - uint4", "[noa][common][io]", uint8_t, short, int, uint, half_t, float) {
    const path_t test_dir = "testIO";
    const path_t test_file = test_dir / "test.bin";
    fs::create_directory(test_dir);

    const bool clamp = GENERATE(true, false);
    const bool swap = GENERATE(true, false);
    const size4_t shape = test::getRandomShapeBatched(1);
    const size_t elements = shape.elements();

    const io::DataType dtype = io::DataType::UINT4;
    INFO("size: " << elements << ", clamp:" << clamp << ", swap: " << swap);

    std::unique_ptr<TestType[]> data = std::make_unique<TestType[]>(elements);
    std::unique_ptr<TestType[]> read_data = std::make_unique<TestType[]>(elements);

    // Randomize data. No decimals, otherwise it makes everything more complicated.
    test::Randomizer<int64_t> randomizer(0, clamp ? 30 : 15);
    for (size_t i = 0; i < elements; ++i)
        data[i] = clamp_cast<TestType>(randomizer.get());

    // Serialize:
    std::fstream file(test_file, std::ios::out | std::ios::trunc);
    io::serialize(data.get(), shape.strides(), shape, file, dtype, clamp, swap);
    file.close();
    REQUIRE(fs::file_size(test_file) == io::serializedSize(dtype, elements, shape[3]));

    // Deserialize:
    file.open(test_file, std::ios::in);
    io::deserialize(file, dtype, read_data.get(), shape.strides(), shape, clamp, swap);

    if (clamp) {
        // Serialized data was clamped to fit the data type, so clamp input data as well.
        auto[min, max] = io::typeMinMax<TestType>(dtype);
        for (size_t i = 0; i < elements; ++i)
            data[i] = math::clamp(data[i], min, max);
    }

    REQUIRE(test::Matcher<TestType>(test::MATCH_ABS, data.get(), read_data.get(), elements, 1e-6));
    std::error_code er;
    fs::remove_all(test_dir, er); // silence error
}

TEMPLATE_TEST_CASE("io::(de)serialize - complex", "[noa][common][io]", chalf_t, cfloat_t, cdouble_t) {
    const path_t test_dir = "testIO";
    const path_t test_file = test_dir / "test.bin";
    fs::create_directory(test_dir);

    const bool clamp = GENERATE(true, false);
    const bool swap = GENERATE(true, false);
    const io::DataType dtype = GENERATE(io::DataType::CFLOAT16, io::DataType::CFLOAT32, io::DataType::CFLOAT64);
    const size4_t shape = test::getRandomShapeBatched(1);
    const auto elements = shape.elements();
    INFO("size: " << elements << ", clamp:" << clamp << ", swap: " << swap);

    std::unique_ptr<TestType[]> data = std::make_unique<TestType[]>(elements);
    std::unique_ptr<TestType[]> read_data = std::make_unique<TestType[]>(elements);

    // Randomize data.
    test::Randomizer<float> randomizer(-10000, 10000);
    for (size_t i = 0; i < elements; ++i)
        data[i] = static_cast<TestType>(randomizer.get());

    // Serialize:
    std::fstream file(test_file, std::ios::out | std::ios::trunc);
    io::serialize(data.get(), shape.strides(), shape, file, dtype, clamp, swap);
    file.close();
    REQUIRE(fs::file_size(test_file) == io::serializedSize(dtype, elements));

    // Deserialize:
    file.open(test_file, std::ios::in);
    io::deserialize(file, dtype, read_data.get(), shape.strides(), shape, clamp, swap);

    if (dtype == io::DataType::CFLOAT16) {
        for (size_t i = 0; i < elements; ++i)
            data[i] = static_cast<TestType>(static_cast<chalf_t>(data[i])); // for half, mimic conversion on raw data
    }
    REQUIRE(test::Matcher<TestType>(test::MATCH_ABS, data.get(), read_data.get(), elements, 1e-6));
    std::error_code er;
    fs::remove_all(test_dir, er); // silence error
}

TEST_CASE("io::(de)serialize - many elements", "[noa][common][io]") {
    const path_t test_dir = "testIO";
    const path_t test_file = test_dir / "test.bin";
    fs::create_directory(test_dir);

    const bool clamp = false;
    const bool swap = true;
    const io::DataType dtype = io::DataType::INT16;
    const size4_t shape{2, 256, 256, 256};
    const size_t elements = shape.elements();

    std::unique_ptr<float[]> data = std::make_unique<float[]>(elements);
    std::unique_ptr<float[]> read_data = std::make_unique<float[]>(elements);

    // Randomize data.
    test::Randomizer<float> randomizer(-10000, 10000);
    for (size_t i = 0; i < elements; ++i)
        data[i] = randomizer.get();

    // Serialize:
    std::fstream file(test_file, std::ios::out | std::ios::trunc);
    io::serialize(data.get(), shape.strides(), shape, file, dtype, clamp, swap);
    file.close();
    REQUIRE(fs::file_size(test_file) == io::serializedSize(dtype, elements));

    // Deserialize:
    file.open(test_file, std::ios::in);
    io::deserialize(file, dtype, read_data.get(), shape.strides(), shape, clamp, swap);

    for (size_t i = 0; i < elements; ++i)
        data[i] = std::trunc(data[i]); // float/double -> int16 -> float/double
    const float diff = test::getDifference(data.get(), read_data.get(), elements);
    REQUIRE_THAT(diff, Catch::WithinULP(0.f, 2));

    std::error_code er;
    fs::remove_all(test_dir, er); // silence error
}

TEST_CASE("io::swapEndian()", "[noa][common][io]") {
    constexpr size_t N = 100;
    std::unique_ptr<float[]> data1 = std::make_unique<float[]>(N);
    std::unique_ptr<float[]> data2 = std::make_unique<float[]>(N);
    for (size_t i{0}; i < N; ++i) {
        auto t = static_cast<float>(test::Randomizer<int>(-1234434, 94321458).get());
        data1[i] = t;
        data2[i] = t;
    }
    io::swapEndian(data1.get(), N);
    io::swapEndian(data1.get(), N);
    float diff{0};
    for (size_t i{0}; i < N; ++i)
        diff += data1[i] - data2[i];
    REQUIRE_THAT(diff, Catch::WithinULP(0.f, 2));
}
