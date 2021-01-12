#include <catch2/catch.hpp>
#include "../../Helpers.h"
#include "noa/util/IO.h"

using namespace ::Noa;

TEST_CASE("IO: read and write", "[noa][IO]") {
    fs::path test_dir = "testIO";
    fs::path test_file = test_dir / "test_write_float1.txt";
    fs::create_directory(test_dir);

    auto random_size = GENERATE(Test::random(134, 4000),
                                Test::random(134, 4000),
                                Test::random(134, 4000));
    auto elements = static_cast<size_t>(random_size);
    DataType dtype = GENERATE(DataType::byte, DataType::ubyte,
                              DataType::int16, DataType::uint16,
                              DataType::int32, DataType::uint32,
                              DataType::float32);
    size_t bytes_per_elements = IO::bytesPerElement(dtype);
    bool batch = GENERATE(true, false);
    bool swap = GENERATE(true, false);

    INFO("size: " << elements << ", dtype: " << IO::toString(dtype) <<
                  ", batch: " << batch << ", swap: " << swap);

    AND_WHEN("write and read") {
        // write an array to stream
        std::unique_ptr<float[]> data = std::make_unique<float[]>(elements);
        for (size_t i{0}; i < elements; ++i)
            data[i] = static_cast<float>(Test::random(0, 127));

        std::fstream file(test_file, std::ios::out | std::ios::trunc);
        Flag <Errno> err = IO::writeFloat<2048>(data.get(), file, elements, dtype, batch, swap);
        REQUIRE_ERRNO_GOOD(err);
        file.close();

        REQUIRE(fs::file_size(test_file) == elements * bytes_per_elements);

        // read back the array
        std::unique_ptr<float[]> read_data = std::make_unique<float[]>(elements);
        file.open(test_file, std::ios::in);
        err = IO::readFloat<2048>(file, read_data.get(), elements, dtype, batch, swap);
        REQUIRE_ERRNO_GOOD(err);

        float diff{0};
        for (size_t i{0}; i < elements; ++i)
            diff += read_data[i] - data[i];
        REQUIRE_THAT(diff, Catch::WithinULP(0.f, 2));
    }
    fs::remove_all("testIO");
}

TEST_CASE("IO: swapEndian", "[noa][IO]") {
    std::unique_ptr<float[]> data1 = std::make_unique<float[]>(100);
    std::unique_ptr<float[]> data2 = std::make_unique<float[]>(100);
    for (size_t i{0}; i < 100; ++i) {
        auto t = static_cast<float>(Test::random(-1234434, 94321458));
        data1[i] = t;
        data2[i] = t;
    }
    size_t dtype = IO::bytesPerElement(DataType::float32);
    Flag <Errno> err = IO::swapEndian(reinterpret_cast<char*>(data1.get()), 100, dtype);
    REQUIRE_ERRNO_GOOD(err);
    err = IO::swapEndian(reinterpret_cast<char*>(data1.get()), 100, dtype);
    REQUIRE_ERRNO_GOOD(err);
    float diff{0};
    for (size_t i{0}; i < 100; ++i)
        diff += data1[i] - data2[i];
    REQUIRE_THAT(diff, Catch::WithinULP(0.f, 2));
}
