#include <noa/common/OS.h>
#include "noa/common/io/IO.h"
#include <noa/common/io/ImageFile.h>

#include <iostream>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("io::Stats", "[noa][common][io]") {
    io::stats_t out;
    REQUIRE_FALSE(out.hasMin());
    REQUIRE_FALSE(out.hasMax());
    REQUIRE_FALSE(out.hasSum());
    REQUIRE_FALSE(out.hasMean());
    REQUIRE_FALSE(out.hasVar());
    REQUIRE_FALSE(out.hasStd());

    out.min(1);
    REQUIRE(out.hasMin());
    REQUIRE(out.min() == 1);
    out.max(2);
    REQUIRE(out.hasMax());
    REQUIRE(out.max() == 2);
    out.sum(3);
    REQUIRE(out.hasSum());
    REQUIRE(out.sum() == 3);
    out.mean(4);
    REQUIRE(out.hasMean());
    REQUIRE(out.mean() == 4);
    out.var(5);
    REQUIRE(out.hasVar());
    REQUIRE(out.var() == 5);
    out.std(6);
    REQUIRE(out.hasStd());
    REQUIRE(out.std() == 6);
}

TEST_CASE("io::ImageFile: MRC, real dtype", "[noa][common][io]") {
    auto data_file = test::NOA_DATA_PATH / "common" / "io" / "files" / "example_MRCFile.mrc";
    const std::string fixture_expected_header = "Format: MRC File\n"
                                                "Shape (batches, depth, height, width): (11,1,576,410)\n"
                                                "Pixel size (depth, height, width): (2.100,21.000,21.000)\n"
                                                "Data type: INT16\n"
                                                "Labels: 9\n"
                                                "Extended header: 0 bytes";
    const path_t test_dir = fs::current_path() / "test_MRCFile";
    fs::remove_all(test_dir);

    AND_THEN("write and read to a volume") {
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
        io::stats_t stats;
        stats.min(-1);
        stats.max(1);
        stats.mean(0.5);
        std::unique_ptr<float[]> to_write = std::make_unique<float[]>(shape.elements());
        test::Randomizer<int> randomizer(0, 127);
        for (size_t i{0}; i < shape.elements(); ++i)
            to_write[i] = static_cast<float>(randomizer.get());

        // write to file...
        file.dtype(dtype);
        file.shape(shape);
        file.pixelSize(pixel_size);
        file.stats(stats);
        file.writeAll(to_write.get());
        file.close();

        // reading the file and check that it matches...
        io::ImageFile file_to_read(file1, io::READ);
        REQUIRE(all(file_to_read.shape() == shape));
        REQUIRE(all(file_to_read.pixelSize() == pixel_size));
        const io::stats_t file_stats = file_to_read.stats();
        REQUIRE_FALSE(stats.hasSum());
        REQUIRE_FALSE(stats.hasVar());
        REQUIRE_FALSE(stats.hasStd());
        REQUIRE(stats.hasMin());
        REQUIRE(stats.hasMax());
        REQUIRE(stats.hasMean());
        REQUIRE(stats.min() == file_stats.min());
        REQUIRE(stats.max() == file_stats.max());
        REQUIRE(stats.mean() == file_stats.mean());
        REQUIRE(stats.std() == file_stats.std());

        std::unique_ptr<float[]> to_read = std::make_unique<float[]>(shape.elements());
        file_to_read.readAll(to_read.get());
        float diff = test::getDifference(to_write.get(), to_read.get(), shape.elements());
        REQUIRE_THAT(diff, Catch::WithinULP(0.f, 4));
    }

    AND_THEN("write and read to a stack of volumes") {
        // create a MRC file...
        const path_t file1 = test_dir / "file1.mrc";
        io::ImageFile file(file1, io::WRITE);
        REQUIRE(file);

        const io::DataType dtype = GENERATE(io::DataType::INT16, io::DataType::UINT16,
                                            io::DataType::UINT8, io::DataType::INT8,
                                            io::DataType::FLOAT16, io::DataType::FLOAT32);

        // initialize data to put into the file...
        const size4_t shape = {5, 64, 64, 64};
        const float3_t pixel_size = {1.23f, 1.23f, 1.23f};
        const io::stats_t stats;
        std::unique_ptr<float[]> to_write = std::make_unique<float[]>(shape.elements());
        test::Randomizer<int> randomizer(0, 127);
        for (size_t i{0}; i < shape.elements(); ++i)
            to_write[i] = static_cast<float>(randomizer.get());

        // write to file...
        file.dtype(dtype);
        file.shape(shape);
        file.pixelSize(pixel_size);
        file.stats(stats);
        file.writeAll(to_write.get());
        file.close();

        // reading the file and check that it matches...
        io::ImageFile file_to_read(file1, io::READ);
        REQUIRE(all(file_to_read.shape() == shape));
        REQUIRE(all(file_to_read.pixelSize() == pixel_size));
        const io::stats_t file_stats = file_to_read.stats();
        REQUIRE_FALSE(file_stats.hasMin());
        REQUIRE_FALSE(file_stats.hasMax());
        REQUIRE_FALSE(file_stats.hasSum());
        REQUIRE_FALSE(file_stats.hasMean());
        REQUIRE_FALSE(file_stats.hasVar());
        REQUIRE_FALSE(file_stats.hasStd());

        std::unique_ptr<float[]> to_read = std::make_unique<float[]>(shape.elements());
        file_to_read.readAll(to_read.get());
        float diff = test::getDifference(to_write.get(), to_read.get(), shape.elements());
        REQUIRE_THAT(diff, Catch::WithinULP(0.f, 4));
    }

    AND_THEN("write and read a stack of 2D images") {
        // create a MRC file...
        const path_t file1 = test_dir / "file1.mrc";
        io::ImageFile file(file1, io::WRITE);
        REQUIRE(file);

        const io::DataType dtype = GENERATE(io::DataType::INT16, io::DataType::UINT16,
                                            io::DataType::UINT8, io::DataType::INT8,
                                            io::DataType::FLOAT16, io::DataType::FLOAT32);

        // initialize data to put into the file...
        const size4_t shape = {41, 1, 64, 64};
        const float3_t pixel_size = {1, 1.23f, 1.23f};
        std::unique_ptr<float[]> to_write = std::make_unique<float[]>(shape.elements());
        test::Randomizer<int> randomizer(0, 127);
        for (size_t i{0}; i < shape.elements(); ++i)
            to_write[i] = static_cast<float>(randomizer.get());

        // write to file...
        file.dtype(dtype);
        file.shape(shape);
        file.pixelSize(pixel_size);
        file.writeAll(to_write.get());
        file.close();

        // reading the file and check that it matches...
        io::ImageFile file_to_read(file1, io::READ);
        REQUIRE(all(file_to_read.shape() == shape));
        REQUIRE(all(file_to_read.pixelSize() == pixel_size));

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
