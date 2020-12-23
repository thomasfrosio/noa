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
    IO::DataType dtype = GENERATE(IO::DataType::byte, IO::DataType::ubyte,
                                  IO::DataType::int16, IO::DataType::uint16,
                                  IO::DataType::int32, IO::DataType::uint32,
                                  IO::DataType::float32);
    size_t bytes_per_elements = IO::bytesPerElement(dtype);
    bool batch = GENERATE(true, false);
    bool swap = GENERATE(true, false);

    INFO("size: " << elements << ", dtype: " << IO::toString(dtype) <<
                  ", batch: " << batch << ", swap: " << swap);

    AND_WHEN("write and read") {
        // write an array to stream
        auto* data = new float[elements];
        for (size_t i{0}; i < elements; ++i)
            data[i] = static_cast<float>(Test::random(0, 127));

        std::fstream file(test_file, std::ios::out | std::ios::trunc);
        IO::writeFloat<2048>(data, file, elements, dtype, batch, swap);
        file.close();

        REQUIRE(fs::file_size(test_file) == elements * bytes_per_elements);

        // read back the array
        auto* read_data = new float[elements];
        file.open(test_file, std::ios::in);
        IO::readFloat<2048>(file, read_data, elements, dtype, batch, swap);

        for (size_t i{0}; i < elements; ++i) {
            REQUIRE(read_data[i] == data[i]);
        }
    }
    fs::remove_all("testIO");
}


TEST_CASE("IO: swapEndian", "[noa][IO]") {
    auto* data1 = new float[100];
    auto* data2 = new float[100];
    for (size_t i{0}; i < 100; ++i) {
        auto t = static_cast<float>(Test::random(-1234434, 94321458));
        data1[i] = t;
        data2[i] = t;
    }
    IO::swapEndian(reinterpret_cast<char*>(data1), 100, 4);
    IO::swapEndian(reinterpret_cast<char*>(data1), 100, 4);
    for (size_t i{0}; i < 100; ++i)
        REQUIRE(data1[i] == data2[i]);
}

