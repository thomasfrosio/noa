#include <fstream>
#include <noa/core/io/IO.hpp>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEMPLATE_TEST_CASE("core::io::(de)serialize - real types", "[noa][core]",
                   u8, i16, i32, u32, f16, f32, f64) {
    const Path test_dir = "testIO";
    const Path test_file = test_dir / "test.bin";
    fs::create_directory(test_dir);

    const auto shape = test::get_random_shape4_batched(1).as<i64>();
    const auto ssize = shape.elements();
    const auto size = static_cast<size_t>(ssize);
    const io::DataType dtype =
            GENERATE(io::DataType::I8, io::DataType::U8,
                     io::DataType::I16, io::DataType::U16,
                     io::DataType::I32, io::DataType::U32,
                     io::DataType::I64, io::DataType::U64,
                     io::DataType::F16, io::DataType::F32, io::DataType::F64);
    const bool clamp = GENERATE(true, false);
    const bool swap = GENERATE(true, false);

    INFO("shape: " << shape << ", dtype: " << dtype << ", clamp:" << clamp << ", swap: " << swap);

    const auto data = std::make_unique<TestType[]>(size);
    const auto read_data = std::make_unique<TestType[]>(size);

    // Randomize data. No decimals, otherwise it makes everything more complicated.
    i64 range_min{}, range_max{};
    if (clamp) {
        range_min = dtype == io::DataType::F16 ? -2048 : -30000;
        range_max = dtype == io::DataType::F16 ? 2048 : 30000;
    } else {
        range_min = 0;
        range_max = 127;
    }
    test::Randomizer<i64> randomizer(range_min, range_max);
    for (size_t i = 0; i < size; ++i)
        data[i] = clamp_cast<TestType>(randomizer.get());

    // Serialize:
    std::fstream file(test_file, std::ios::out | std::ios::trunc);
    io::serialize(data.get(), shape.strides(), shape, file, dtype, clamp, swap);
    file.close();
    REQUIRE(static_cast<i64>(fs::file_size(test_file)) == io::serialized_size(dtype, ssize));

    // Deserialize:
    file.open(test_file, std::ios::in);
    io::deserialize(file, dtype, read_data.get(), shape.strides(), shape, clamp, swap);

    if (clamp) {
        // Serialized data was clamped to fit the data type, so clamp input data as well.
        auto[min, max] = io::type_range<TestType>(dtype);
        for (size_t i = 0; i < size; ++i)
            data[i] = math::clamp(data[i], min, max);
    }

    REQUIRE(test::Matcher<TestType>(test::MATCH_ABS, data.get(), read_data.get(), size, 1e-6));
    std::error_code er;
    fs::remove_all(test_dir, er); // silence error
}

TEMPLATE_TEST_CASE("core::io::(de)serialize - uint4", "[noa][core]", u8, i16, i32, u32, f16, f32) {
    const Path test_dir = "testIO";
    const Path test_file = test_dir / "test.bin";
    fs::create_directory(test_dir);

    const bool clamp = GENERATE(true, false);
    const bool swap = GENERATE(true, false);
    const auto shape = test::get_random_shape4_batched(1).as<i64>();
    const auto ssize = shape.elements();
    const auto size = static_cast<size_t>(ssize);

    const io::DataType dtype = io::DataType::U4;
    INFO("size: " << size << ", clamp:" << clamp << ", swap: " << swap);

    const auto data = std::make_unique<TestType[]>(size);
    const auto read_data = std::make_unique<TestType[]>(size);

    // Randomize data. No decimals, otherwise it makes everything more complicated.
    test::Randomizer<int64_t> randomizer(0, clamp ? 30 : 15);
    for (size_t i = 0; i < size; ++i)
        data[i] = clamp_cast<TestType>(randomizer.get());

    // Serialize:
    std::fstream file(test_file, std::ios::out | std::ios::trunc);
    io::serialize(data.get(), shape.strides(), shape, file, dtype, clamp, swap);
    file.close();
    REQUIRE(static_cast<i64>(fs::file_size(test_file)) == io::serialized_size(dtype, ssize, shape[3]));

    // Deserialize:
    file.open(test_file, std::ios::in);
    io::deserialize(file, dtype, read_data.get(), shape.strides(), shape, clamp, swap);

    if (clamp) {
        // Serialized data was clamped to fit the data type, so clamp input data as well.
        auto[min, max] = io::type_range<TestType>(dtype);
        for (size_t i = 0; i < size; ++i)
            data[i] = math::clamp(data[i], min, max);
    }

    REQUIRE(test::Matcher<TestType>(test::MATCH_ABS, data.get(), read_data.get(), size, 1e-6));
    std::error_code er;
    fs::remove_all(test_dir, er); // silence error
}

TEMPLATE_TEST_CASE("core::io::(de)serialize - complex", "[noa][core]", c16, c32, c64) {
    const Path test_dir = "testIO";
    const Path test_file = test_dir / "test.bin";
    fs::create_directory(test_dir);

    const bool clamp = GENERATE(true, false);
    const bool swap = GENERATE(true, false);
    const io::DataType dtype = GENERATE(io::DataType::C16, io::DataType::C32, io::DataType::C64);
    const auto shape = test::get_random_shape4_batched(1).as<i64>();
    const auto ssize = shape.elements();
    const auto size = static_cast<size_t>(ssize);
    INFO("size: " << size << ", clamp:" << clamp << ", swap: " << swap);

    const auto data = std::make_unique<TestType[]>(size);
    const auto read_data = std::make_unique<TestType[]>(size);

    // Randomize data.
    test::Randomizer<float> randomizer(-10000, 10000);
    for (size_t i = 0; i < size; ++i)
        data[i] = static_cast<TestType>(randomizer.get());

    // Serialize:
    std::fstream file(test_file, std::ios::out | std::ios::trunc);
    io::serialize(data.get(), shape.strides(), shape, file, dtype, clamp, swap);
    file.close();
    REQUIRE(static_cast<i64>(fs::file_size(test_file)) == io::serialized_size(dtype, ssize));

    // Deserialize:
    file.open(test_file, std::ios::in);
    io::deserialize(file, dtype, read_data.get(), shape.strides(), shape, clamp, swap);

    if (dtype == io::DataType::C16) {
        for (size_t i = 0; i < size; ++i)
            data[i] = static_cast<TestType>(static_cast<c16>(data[i])); // for half, mimic conversion on raw data
    }
    REQUIRE(test::Matcher<TestType>(test::MATCH_ABS, data.get(), read_data.get(), size, 1e-6));
    std::error_code er;
    fs::remove_all(test_dir, er); // silence error
}

TEST_CASE("core::io::(de)serialize - many elements", "[noa][core]") {
    const Path test_dir = "testIO";
    const Path test_file = test_dir / "test.bin";
    fs::create_directory(test_dir);

    const bool clamp = false;
    const bool swap = true;
    const io::DataType dtype = io::DataType::I16;
    const Shape4<i64> shape{2, 256, 256, 256};
    const auto ssize = shape.elements();
    const auto size = static_cast<size_t>(ssize);

    const auto data = std::make_unique<float[]>(size);
    const auto read_data = std::make_unique<float[]>(size);

    // Randomize data.
    test::Randomizer<float> randomizer(-10000, 10000);
    for (size_t i = 0; i < size; ++i)
        data[i] = randomizer.get();

    // Serialize:
    std::fstream file(test_file, std::ios::out | std::ios::trunc);
    io::serialize(data.get(), shape.strides(), shape, file, dtype, clamp, swap);
    file.close();
    REQUIRE(static_cast<i64>(fs::file_size(test_file)) == io::serialized_size(dtype, ssize));

    // Deserialize:
    file.open(test_file, std::ios::in);
    io::deserialize(file, dtype, read_data.get(), shape.strides(), shape, clamp, swap);

    for (size_t i = 0; i < size; ++i)
        data[i] = std::trunc(data[i]); // float/double -> int16 -> float/double
    const float diff = test::get_difference(data.get(), read_data.get(), size);
    REQUIRE_THAT(diff, Catch::WithinULP(0.f, 2));

    std::error_code er;
    fs::remove_all(test_dir, er); // silence error
}

TEST_CASE("core::io::swapEndian()", "[noa][core]") {
    constexpr size_t N = 100;
    const auto data1 = std::make_unique<float[]>(N);
    const auto data2 = std::make_unique<float[]>(N);
    for (size_t i{0}; i < N; ++i) {
        auto t = static_cast<float>(test::Randomizer<int>(-1234434, 94321458).get());
        data1[i] = t;
        data2[i] = t;
    }
    io::swap_endian(data1.get(), N);
    io::swap_endian(data1.get(), N);
    float diff{0};
    for (size_t i{0}; i < N; ++i)
        diff += data1[i] - data2[i];
    REQUIRE_THAT(diff, Catch::WithinULP(0.f, 2));
}
