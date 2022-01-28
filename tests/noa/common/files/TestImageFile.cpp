#include <noa/common/OS.h>
#include "noa/common/io/IO.h"
#include <noa/common/io/ImageFile.h>

#include <iostream>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("ImageFile: MRC, real dtype", "[noa][common][io]") {
    auto data_file = test::PATH_NOA_DATA / "io" / "files" / "example_MRCFile.mrc";
    const std::string fixture_expected_header = "Format: MRC File\n"
                                                "Shape (batches, sections, rows, columns): (1,11,576,410)\n"
                                                "Pixel size (sections, rows, columns): (2.100, 21.000,21.000)\n"
                                                "Data type: INT16\n"
                                                "Labels: 9\n"
                                                "Extended headers: 0 bytes";
    const path_t test_dir = fs::current_path() / "test_MRCFile";
    fs::remove_all(test_dir);

    AND_THEN("write and read") {
        // create a MRC file...
        const path_t file1 = test_dir / "file1.mrc";
        io::ImageFile file(file1, io::WRITE);
        REQUIRE(file);

        const io::DataType dtype = GENERATE(io::DataType::INT16, io::DataType::UINT16,
                                            io::DataType::UINT8, io::DataType::INT8,
                                            io::DataType::FLOAT16, io::DataType::FLOAT32);

        // initialize data to put into the file...
        const size4_t shape = {1, 64, 64, 64};
        const float3_t pixel_size = {1.23f, 1.23f, 1.23f};
        const Stats<float> stats{-1.f, 1.f, 100.f, 0.f, 100.f, 0.5f};
        std::unique_ptr<float[]> to_write = std::make_unique<float[]>(shape.elements());
        for (size_t i{0}; i < shape.elements(); ++i)
            to_write[i] = static_cast<float>(test::Randomizer<int>(0, 127).get());

        // write to file...
        file.dataType(dtype);
        file.shape(shape);
        file.pixelSize(pixel_size);
        file.stats(stats);
        file.writeAll(to_write.get());
        file.close();

        // reading the file and check that it matches...
        io::ImageFile file_to_read(file1, io::READ);
        REQUIRE(all(file_to_read.shape() == shape));
        REQUIRE(all(file_to_read.pixelSize() == pixel_size));
        Stats<float> file_stats = file_to_read.stats();
        REQUIRE(stats.min == file_stats.min);
        REQUIRE(stats.max == file_stats.max);
        REQUIRE(stats.mean == file_stats.mean);
        REQUIRE(stats.stddev == file_stats.stddev);

        std::unique_ptr<float[]> to_read = std::make_unique<float[]>(shape.elements());
        file_to_read.readAll(to_read.get());
        float diff = test::getDifference(to_write.get(), to_read.get(), shape.elements());
        REQUIRE_THAT(diff, Catch::WithinULP(0.f, 4));
    }

    AND_THEN("reading files") {
        const path_t fixture_copy = test_dir / "file2.mrc";
        os::mkdir(test_dir);
        REQUIRE(os::copyFile(data_file, fixture_copy));

        // Writing permissions should not be necessary.
        fs::permissions(fixture_copy, fs::perms::owner_write | fs::perms::group_write |
                                      fs::perms::others_write, fs::perm_options::remove);
        io::ImageFile file;
        REQUIRE_THROWS_AS(file.open(fixture_copy, io::READ | io::WRITE), noa::Exception);
        os::remove(fixture_copy.string() + "~"); // Remove backup copy from this attempt.
        REQUIRE_FALSE(file.isOpen());

        // There should be no backup since it is read only.
        file.open(fixture_copy, io::READ);
        REQUIRE(file.isOpen());
        REQUIRE_FALSE(os::existsFile(fixture_copy.string() + "~"));

        // Any writing operation should fail.
        const size_t elements_per_slice = file.shape()[2] * file.shape()[3];
        std::unique_ptr<float[]> ptr = std::make_unique<float[]>(elements_per_slice);
        REQUIRE_THROWS_AS(file.writeSlice(ptr.get(), 0, 1), noa::Exception);

        std::string str = file.info(false);
        REQUIRE(str == fixture_expected_header);
    }

    AND_THEN("writing to an existing file") {
        const path_t fixture_copy = test_dir / "file2.mrc";
        os::mkdir(test_dir);
        REQUIRE(os::copyFile(data_file, fixture_copy));

        io::ImageFile image_file(fixture_copy, io::READ | io::WRITE);
        REQUIRE(image_file.isOpen());

        // Check backup copy.
        REQUIRE(os::existsFile(fixture_copy.string() + "~"));
        REQUIRE(image_file.info(false) == fixture_expected_header);

        const size_t elements_per_slice = image_file.shape()[2] * image_file.shape()[3];
        std::unique_ptr<float[]> to_write = std::make_unique<float[]>(elements_per_slice);
        test::Randomizer<float> randomizer(-1000, 1000);
        for (size_t idx{0}; idx < elements_per_slice; ++idx)
            to_write[idx] = randomizer.get();
        image_file.writeSlice(to_write.get(), 5, 6);
        image_file.close();

        image_file.open(fixture_copy, io::READ);
        std::unique_ptr<float[]> to_read = std::make_unique<float[]>(elements_per_slice);
        image_file.readSlice(to_read.get(), 5, 6);
        float diff{0};
        // cast to int16_t is necessary: it happens during writeSlice() since the file is in mode=int16.
        for (size_t i{0}; i < elements_per_slice; ++i)
            diff += static_cast<float>(static_cast<int16_t>(to_write[i])) - to_read[i];
        REQUIRE_THAT(diff, Catch::WithinULP(0.f, 4));
    }
    fs::remove_all(test_dir);
}
