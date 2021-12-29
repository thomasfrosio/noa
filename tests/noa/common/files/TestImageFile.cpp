#include <noa/common/OS.h>
#include "noa/common/io/IO.h"
#include <noa/common/io/ImageFile.h>

#include <iostream>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("ImageFile: MRC, real dtype", "[noa][common][io]") {
    auto data_file = test::PATH_NOA_DATA / "io" / "files" / "example_MRCFile.mrc";
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
        io::ImageFile file(file1, io::WRITE);
        REQUIRE(file);

        io::DataType dtype = GENERATE(io::DataType::INT16, io::DataType::UINT16,
                                      io::DataType::UINT8, io::DataType::INT8,
                                      io::DataType::FLOAT16, io::DataType::FLOAT32);

        // initialize data to put into the file...
        size3_t shape = {64, 64, 64};
        float3_t pixel_size = {1.23f, 1.23f, 1.23f};
        Stats<float> stats{-1.f, 1.f, 100.f, 0.f, 100.f, 0.5f};
        std::unique_ptr<float[]> to_write = std::make_unique<float[]>(noa::elements(shape));
        for (size_t i{0}; i < noa::elements(shape); ++i)
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

        std::unique_ptr<float[]> to_read = std::make_unique<float[]>(noa::elements(shape));
        file_to_read.readAll(to_read.get());
        float diff = test::getDifference(to_write.get(), to_read.get(), noa::elements(shape));
        REQUIRE_THAT(diff, Catch::WithinULP(0.f, 4));
    }

    AND_THEN("reading files") {
        path_t fixture_copy = test_dir / "file2.mrc";
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
        std::unique_ptr<float[]> ptr = std::make_unique<float[]>(noa::elementsSlice(file.shape()));
        REQUIRE_THROWS_AS(file.writeSlice(ptr.get(), 0, 1), noa::Exception);

        std::string str = file.info(false);
        REQUIRE(str == fixture_expected_header);
    }

    AND_THEN("writing to an existing file") {
        path_t fixture_copy = test_dir / "file2.mrc";
        os::mkdir(test_dir);
        REQUIRE(os::copyFile(data_file, fixture_copy));

        io::ImageFile image_file(fixture_copy, io::READ | io::WRITE);
        REQUIRE(image_file.isOpen());

        // Check backup copy.
        REQUIRE(os::existsFile(fixture_copy.string() + "~"));
        REQUIRE(image_file.info(false) == fixture_expected_header);

        size_t slice_size = noa::elementsSlice(image_file.shape());
        std::unique_ptr<float[]> to_write = std::make_unique<float[]>(slice_size);
        test::Randomizer<float> randomizer(-1000, 1000);
        for (size_t idx{0}; idx < slice_size; ++idx)
            to_write[idx] = randomizer.get();
        image_file.writeSlice(to_write.get(), 5, 6);
        image_file.close();

        image_file.open(fixture_copy, io::READ);
        std::unique_ptr<float[]> to_read = std::make_unique<float[]>(slice_size);
        image_file.readSlice(to_read.get(), 5, 6);
        float diff{0};
        // cast to int16_t is necessary: it happens during writeSlice() since the file is in mode=int16.
        for (size_t i{0}; i < slice_size; ++i)
            diff += static_cast<float>(static_cast<int16_t>(to_write[i])) - to_read[i];
        REQUIRE_THAT(diff, Catch::WithinULP(0.f, 4));
    }
    fs::remove_all(test_dir);
}

//TEST_CASE("ImageFile: MRC, complex dtype", "[noa][common][files]") {
//    path_t test_dir = fs::current_path() / "test_MRCFile";
//    fs::remove_all(test_dir);
//
//    AND_THEN("write and read") {
//        // create a MRC file...
//        path_t file1 = test_dir / "file1.mrc";
//        io::ImageFile file(file1, io::WRITE);
//        REQUIRE(file.isOpen());
//
//        io::DataType dtype = GENERATE(io::DataType::CINT16, io::DataType::CFLOAT32);
//        INFO("dtype: " << dtype);
//        // initialize data to put into the file...
//        size3_t shape = {64, 64, 64};
//        float3_t pixel_size = {1.23f, 1.23f, 1.23f};
//        Stats<float> stats{-1.f, 1.f, 100.f, 0.f, 100.f, 0.5f};
//        std::unique_ptr<cfloat_t[]> to_write = std::make_unique<cfloat_t[]>(noa::elements(shape));
//        test::IntRandomizer<int> randomizer(0, 127);
//        auto* to_write_tmp = reinterpret_cast<float*>(to_write.get());
//        for (size_t i{0}; i < noa::elements(shape) * 2; ++i)
//            to_write_tmp[i] = static_cast<float>(randomizer.get());
//
//        // write to file...
//        file.dataType(dtype);
//        file.shape(shape);
//        file.pixelSize(pixel_size);
//        file.stats(stats);
//
//        AND_THEN("entire file") {
//            file.writeAll(to_write.get());
//            file.close();
//
//            // reading the file and check that it matches...
//            file.open(file1, io::READ);
//            REQUIRE(all(file.shape() == shape));
//            REQUIRE(all(file.pixelSize() == pixel_size));
//            Stats<float> file_stats = file.stats();
//            REQUIRE(stats.min == file_stats.min);
//            REQUIRE(stats.max == file_stats.max);
//            REQUIRE(stats.mean == file_stats.mean);
//            REQUIRE(stats.stddev == file_stats.stddev);
//
//            std::unique_ptr<cfloat_t[]> to_read = std::make_unique<cfloat_t[]>(noa::elements(shape));
//            file.readAll(to_read.get());
//
//            float diff = test::getDifference(reinterpret_cast<float*>(to_write.get()),
//                                             reinterpret_cast<float*>(to_read.get()),
//                                             noa::elements(shape) * 2);
//            REQUIRE_THAT(diff, Catch::WithinULP(0.f, 4));
//        }
//
//        AND_THEN("slices") {
//            size_t slices = 10;
//            file.writeSlice(to_write.get(), 5, slices);
//            file.close();
//
//            // reading the file and check that it matches...
//            file.open(file1, io::READ);
//            REQUIRE(all(file.shape() == shape));
//            REQUIRE(all(file.pixelSize() == pixel_size));
//            Stats<float> file_stats = file.stats();
//            REQUIRE(stats.min == file_stats.min);
//            REQUIRE(stats.max == file_stats.max);
//            REQUIRE(stats.mean == file_stats.mean);
//            REQUIRE(stats.stddev == file_stats.stddev);
//
//            std::unique_ptr<cfloat_t[]> to_read = std::make_unique<cfloat_t[]>(noa::elementsSlice(shape) * slices);
//            file.readSlice(to_read.get(), 5, slices);
//
//            float diff = test::getDifference(reinterpret_cast<float*>(to_write.get()),
//                                             reinterpret_cast<float*>(to_read.get()),
//                                             noa::elementsSlice(shape) * slices * 2);
//            REQUIRE_THAT(diff, Catch::WithinULP(0.f, 4));
//        }
//    }
//    fs::remove_all(test_dir);
//}
