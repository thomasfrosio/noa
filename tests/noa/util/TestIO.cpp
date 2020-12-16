#include <catch2/catch.hpp>
#include "../../Helpers.h"
#include "noa/util/IO.h"

using namespace ::Noa;


TEST_CASE("IO: read and write", "[noa][IO]") {
    fs::path test_dir = "testIO";
    fs::path test_file = test_dir / "test_write_float1.txt";
    fs::create_directory(test_dir);

    size_t elements = GENERATE(151U, 2048U, 3000U, 4003U);
    iolayout_t layout = GENERATE(IO::Layout::byte, IO::Layout::ubyte,
                                 IO::Layout::int16, IO::Layout::uint16, IO::Layout::int32,
                                 IO::Layout::uint32, IO::Layout::float32);
    size_t bytes_per_elements = IO::Layout::bytesPerElement(layout);
    bool use_buffer = GENERATE(true, false);

    INFO("size: " << elements << ", layout: " << layout << ", use_buffer: " << use_buffer);

    AND_WHEN("write and read") {
        // write an array to stream
        auto* data = new float[elements];
        for (size_t i{0}; i < elements; ++i)
            data[i] = static_cast<float>(Test::random(0, 127));

        std::fstream file(test_file, std::ios::out | std::ios::trunc);
        IO::writeFloat<2048>(file, data, elements, layout, use_buffer);
        file.close();

        REQUIRE(fs::file_size(test_file) == elements * bytes_per_elements);

        // read back the array
        auto* read_data = new float[elements];
        file.open(test_file, std::ios::in);
        IO::readFloat<2048>(file, read_data, elements, layout, use_buffer);

        for (size_t i{0}; i < elements; ++i) {
            REQUIRE(read_data[i] == data[i]);
        }
    }

    AND_WHEN("swap endianness") {
        // write an array to stream
        auto test_size = static_cast<size_t>(Test::random(134, 1931));
        auto* data = new float[test_size];
        for (size_t i{0}; i < test_size; ++i)
            data[i] = static_cast<float>(Test::random(0, 127));

        std::fstream file(test_file, std::ios::out | std::ios::trunc);
        IO::writeFloat<2048>(file, data, test_size, layout);
        file.close();

        // Swap 1
        auto* read_data = new float[test_size];
        file.open(test_file, std::ios::in | std::ios::out);
        IO::readFloat<2048>(file, read_data, test_size, layout, use_buffer, true);
        file.seekp(0);
        IO::writeFloat<2048>(file, read_data, test_size, layout);
        file.close();
        for (size_t i{0}; i < test_size; ++i)
            read_data[i] = 0;

        // Swap 2
        file.open(test_file, std::ios::in);
        IO::readFloat<2048>(file, read_data, test_size, layout, use_buffer, true);
        for (size_t i{0}; i < test_size; ++i) {
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

