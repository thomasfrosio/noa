#include <noa/common/files/MRCFile.h>
#include <noa/common/OS.h>

#include <iostream>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("MRCFile, real dtype", "[noa][common][files]") {
    auto data_file = test::PATH_TEST_DATA / "io" / "files" / "example_MRCFile.mrc";
    std::string fixture_expected_header = "Format: MRC File\n"
                                          "Shape (columns, rows, sections): (410,576,11)\n"
                                          "Pixel size (columns, rows, sections): (21.000,21.000,2.100)\n"
                                          "Data type: INT16\n"
                                          "Labels: 9\n"
                                          "Extended headers: 0 bytes";
    path_t test_dir = fs::current_path() / "test_MRCFile";
    fs::remove_all(test_dir);

    AND_THEN("write and read") {
        // create a MRC file...
        path_t file1 = test_dir / "file1.mrc";
        MRCFile file(file1, io::WRITE);
        REQUIRE(file);

        io::DataType dtype = GENERATE(io::DataType::INT16, io::DataType::UINT16,
                                      io::DataType::UBYTE, io::DataType::BYTE,
                                      io::DataType::FLOAT32);

        // initialize data to put into the file...
        size3_t shape = {64, 64, 64};
        float3_t pixel_size = {1.23f, 1.23f, 1.23f};
        Stats<float> stats{-1.f, 1.f, 100.f, 0.f, 100.f, 0.5f};
        std::unique_ptr<float[]> to_write = std::make_unique<float[]>(getElements(shape));
        for (size_t i{0}; i < getElements(shape); ++i)
            to_write[i] = static_cast<float>(test::pseudoRandom(0, 127));

        // write to file...
        file.setDataType(dtype);
        file.setShape(shape);
        file.setPixelSize(pixel_size);
        file.setStatistics(stats);
        file.writeAll(to_write.get());
        file.close();

        // reading the file and check that it matches...
        MRCFile file_to_read(file1, io::READ);
        REQUIRE(all(file_to_read.getShape() == shape));
        REQUIRE(all(file_to_read.getPixelSize() == pixel_size));
        Stats<float> file_stats = file_to_read.getStatistics();
        REQUIRE(stats.min == file_stats.min);
        REQUIRE(stats.max == file_stats.max);
        REQUIRE(stats.mean == file_stats.mean);
        REQUIRE(stats.stddev == file_stats.stddev);

        std::unique_ptr<float[]> to_read = std::make_unique<float[]>(getElements(shape));
        file_to_read.readAll(to_read.get());
        float diff = test::getDifference(to_write.get(), to_read.get(), getElements(shape));
        REQUIRE_THAT(diff, Catch::WithinULP(0.f, 4));
    }

    AND_THEN("reading files") {
        path_t fixture_copy = test_dir / "file2.mrc";
        os::mkdir(test_dir);
        REQUIRE(os::copyFile(data_file, fixture_copy));

        // Writing permissions should not be necessary.
        fs::permissions(fixture_copy, fs::perms::owner_write | fs::perms::group_write |
                                      fs::perms::others_write, fs::perm_options::remove);
        MRCFile file(fixture_copy);
        REQUIRE_THROWS_AS(file.open(io::READ | io::WRITE), noa::Exception);
        os::remove(fixture_copy.string() + "~"); // Remove backup copy from this attempt.
        REQUIRE_FALSE(file.isOpen());
        file.clear();

        // There should be no backup since it is read only.
        file.open(io::READ);
        REQUIRE(file.isOpen());
        REQUIRE_FALSE(os::existsFile(fixture_copy.string() + "~"));

        // Any writing operation should fail.
        std::unique_ptr<float[]> ptr = std::make_unique<float[]>(getElementsSlice(file.getShape()));
        REQUIRE_THROWS_AS(file.writeSlice(ptr.get(), 0, 1), noa::Exception);
        file.clear();

        std::string str = file.describe(false);
        REQUIRE(str == fixture_expected_header);
    }

    AND_THEN("writing to an existing file") {
        path_t fixture_copy = test_dir / "file2.mrc";
        os::mkdir(test_dir);
        REQUIRE(os::copyFile(data_file, fixture_copy));

        // Using the runtime deduction.
        std::unique_ptr<ImageFile> image_file = ImageFile::get(fixture_copy.extension());
        REQUIRE(image_file);
        image_file->open(fixture_copy, io::READ | io::WRITE);
        REQUIRE(image_file->isOpen());

        // Check backup copy.
        REQUIRE(os::existsFile(fixture_copy.string() + "~"));
        REQUIRE(image_file->describe(false) == fixture_expected_header);

        // Changing the dataset is not supported in in|out mode.
        REQUIRE_THROWS_AS(image_file->setDataType(io::DataType::FLOAT32), noa::Exception);
        image_file->clear();

        size_t slice_size = getElementsSlice(image_file->getShape());
        std::unique_ptr<float[]> to_write = std::make_unique<float[]>(slice_size);
        test::RealRandomizer<float> randomizer(-1000, 1000);
        for (size_t idx{0}; idx < slice_size; ++idx)
            to_write[idx] = randomizer.get();
        image_file->writeSlice(to_write.get(), 5, 1);
        image_file->close();

        image_file->open(io::READ);
        std::unique_ptr<float[]> to_read = std::make_unique<float[]>(slice_size);
        image_file->readSlice(to_read.get(), 5, 1);
        float diff{0};
        // cast to int16_t is necessary: it happens during writeSlice() since the file is in mode=int16.
        for (size_t i{0}; i < slice_size; ++i) {
            diff += static_cast<float>(static_cast<int16_t>(to_write[i])) - to_read[i];
        }
        REQUIRE_THAT(diff, Catch::WithinULP(0.f, 4));
    }
    fs::remove_all(test_dir);
}

TEST_CASE("MRCFile, complex dtype", "[noa][common][files]") {
    path_t test_dir = fs::current_path() / "test_MRCFile";
    fs::remove_all(test_dir);

    AND_THEN("write and read") {
        // create a MRC file...
        path_t file1 = test_dir / "file1.mrc";
        MRCFile file(file1);
        file.open(io::WRITE);
        REQUIRE(file.isOpen());

        io::DataType dtype = GENERATE(io::DataType::CINT16, io::DataType::CFLOAT32);
        INFO("dtype: " << dtype);
        // initialize data to put into the file...
        size3_t shape = {64, 64, 64};
        float3_t pixel_size = {1.23f, 1.23f, 1.23f};
        Stats<float> stats{-1.f, 1.f, 100.f, 0.f, 100.f, 0.5f};
        std::unique_ptr<cfloat_t[]> to_write = std::make_unique<cfloat_t[]>(getElements(shape));
        test::IntRandomizer<int> randomizer(0, 127);
        auto* to_write_tmp = reinterpret_cast<float*>(to_write.get());
        for (size_t i{0}; i < getElements(shape) * 2; ++i)
            to_write_tmp[i] = static_cast<float>(randomizer.get());

        // write to file...
        file.setDataType(dtype);
        file.setShape(shape);
        file.setPixelSize(pixel_size);
        file.setStatistics(stats);

        AND_THEN("entire file") {
            file.writeAll(to_write.get());
            file.close();

            // reading the file and check that it matches...
            file.open(io::READ);
            REQUIRE(all(file.getShape() == shape));
            REQUIRE(all(file.getPixelSize() == pixel_size));
            Stats<float> file_stats = file.getStatistics();
            REQUIRE(stats.min == file_stats.min);
            REQUIRE(stats.max == file_stats.max);
            REQUIRE(stats.mean == file_stats.mean);
            REQUIRE(stats.stddev == file_stats.stddev);

            std::unique_ptr<cfloat_t[]> to_read = std::make_unique<cfloat_t[]>(getElements(shape));
            file.readAll(to_read.get());

            float diff = test::getDifference(reinterpret_cast<float*>(to_write.get()),
                                             reinterpret_cast<float*>(to_read.get()),
                                             getElements(shape) * 2);
            REQUIRE_THAT(diff, Catch::WithinULP(0.f, 4));
        }

        AND_THEN("slices") {
            size_t slices = 10;
            file.writeSlice(to_write.get(), 5, slices);
            file.close();

            // reading the file and check that it matches...
            file.open(io::READ);
            REQUIRE(all(file.getShape() == shape));
            REQUIRE(all(file.getPixelSize() == pixel_size));
            Stats<float> file_stats = file.getStatistics();
            REQUIRE(stats.min == file_stats.min);
            REQUIRE(stats.max == file_stats.max);
            REQUIRE(stats.mean == file_stats.mean);
            REQUIRE(stats.stddev == file_stats.stddev);

            std::unique_ptr<cfloat_t[]> to_read = std::make_unique<cfloat_t[]>(getElementsSlice(shape) * slices);
            file.readSlice(to_read.get(), 5, slices);

            float diff = test::getDifference(reinterpret_cast<float*>(to_write.get()),
                                             reinterpret_cast<float*>(to_read.get()),
                                             getElementsSlice(shape) * slices * 2);
            REQUIRE_THAT(diff, Catch::WithinULP(0.f, 4));
        }
    }
    fs::remove_all(test_dir);
}