#include <noa/util/files/MRCFile.h>
#include <noa/util/OS.h>

#include "../../../Helpers.h"

#include <catch2/catch.hpp>

using namespace ::Noa;

TEST_CASE("MRCFile: real dtype", "[noa][files]") {
    path_t fixture_dir = NOA_TESTS_FIXTURE;
    path_t fixture = fixture_dir / "TestImageFile_mrcfile.mrc";
    std::string fixture_expected_header = "Format: MRC File\n"
                                          "Shape (columns, rows, sections): (410,576,11)\n"
                                          "Pixel size (columns, rows, sections): (21.000,21.000,2.100)\n"
                                          "Data type: int16\n"
                                          "Labels: 9\n"
                                          "Extended headers: 0 bytes";
    path_t test_dir = fs::current_path() / "test_MRCFile";
    fs::remove_all(test_dir);

    Errno err;
    AND_THEN("write and read") {
        // create a MRC file...
        path_t file1 = test_dir / "file1.mrc";
        MRCFile file(file1);
        REQUIRE_ERRNO_GOOD(file.open(std::ios::out, false));

        IO::DataType dtype = GENERATE(IO::DataType::int16, IO::DataType::uint16,
                                      IO::DataType::ubyte, IO::DataType::byte,
                                      IO::DataType::float32);

        // initialize data to put into the file...
        size3_t shape = {64, 64, 64};
        float3_t pixel_size = {1.23f, 1.23f, 1.23f};
        float min{-1}, max{1}, mean{0}, rms{0.5};
        std::unique_ptr<float[]> to_write = std::make_unique<float[]>(Math::elements(shape));
        for (size_t i{0}; i < Math::elements(shape); ++i)
            to_write[i] = static_cast<float>(Test::pseudoRandom(0, 127));

        // write to file...
        REQUIRE_ERRNO_GOOD(file.setDataType(dtype));
        REQUIRE_ERRNO_GOOD(file.setShape(shape));
        REQUIRE_ERRNO_GOOD(file.setPixelSize(pixel_size));
        file.setStatistics(min, max, mean, rms);
        REQUIRE_ERRNO_GOOD(file.writeAll(to_write.get()));
        REQUIRE_ERRNO_GOOD(file.close());

        // reading the file and check that it matches...
        REQUIRE_ERRNO_GOOD(file.open(std::ios::in, false));
        REQUIRE(file.getShape() == shape);
        REQUIRE(file.getPixelSize() == pixel_size);
        REQUIRE(file.getStatistics() == std::tuple<float, float, float, float>(min, max, mean, rms));

        std::unique_ptr<float[]> to_read = std::make_unique<float[]>(Math::elements(shape));
        REQUIRE_ERRNO_GOOD(file.readAll(to_read.get()));
        float diff = Test::getDifference(to_write.get(), to_read.get(), Math::elements(shape));
        REQUIRE_THAT(diff, Catch::WithinULP(0.f, 4));
    }

    AND_THEN("reading files") {
        path_t fixture_copy = test_dir / "file2.mrc";
        REQUIRE_ERRNO_GOOD(OS::mkdir(test_dir));
        REQUIRE_ERRNO_GOOD(OS::copyFile(fixture, fixture_copy));

        // Writing permissions should not be necessary.
        fs::permissions(fixture_copy, fs::perms::owner_write | fs::perms::group_write |
                                      fs::perms::others_write, fs::perm_options::remove);
        MRCFile file(fixture_copy);
        REQUIRE(file.open(std::ios::in | std::ios::out, false) == Errno::fail_open);
        OS::remove(fixture_copy.string() + "~"); // Remove backup copy from this attempt.
        REQUIRE_FALSE(file.isOpen());

        // There should be no backup since it is read only.
        REQUIRE_ERRNO_GOOD(file.open(std::ios::in, false));
        REQUIRE_FALSE(OS::existsFile(fixture_copy.string() + "~", err));
        REQUIRE_ERRNO_GOOD(err);

        // Any writing operation should fail.
        std::unique_ptr<float[]> ptr = std::make_unique<float[]>(Math::elementsSlice(file.getShape()));
        REQUIRE(file.writeSlice(ptr.get(), 0, 1) == Errno::fail_write);

        std::string str = file.toString(false);
        REQUIRE(str == fixture_expected_header);
    }

    AND_THEN("writing to an existing file") {
        path_t fixture_copy = test_dir / "file2.mrc";
        REQUIRE_ERRNO_GOOD(OS::mkdir(test_dir));
        REQUIRE_ERRNO_GOOD(OS::copyFile(fixture, fixture_copy));

        // Using the runtime deduction.
        std::unique_ptr<ImageFile> image_file = ImageFile::get(fixture_copy.extension());
        REQUIRE(image_file);
        REQUIRE_ERRNO_GOOD(image_file->open(fixture_copy, std::ios::in | std::ios::out, false));

        // Check backup copy.
        REQUIRE(OS::existsFile(fixture_copy.string() + "~", err));
        REQUIRE(image_file->toString(false) == fixture_expected_header);
        REQUIRE_ERRNO_GOOD(err);

        // Changing the dataset is not supported in in|out mode.
        REQUIRE(image_file->setDataType(IO::DataType::float32) == Errno::not_supported);

        size_t slice_size = Math::elementsSlice(image_file->getShape());
        std::unique_ptr<float[]> to_write = std::make_unique<float[]>(slice_size);
        for (size_t idx{0}; idx < slice_size; ++idx)
            to_write[idx] = static_cast<float>(idx);
        REQUIRE_ERRNO_GOOD(image_file->writeSlice(to_write.get(), 5, 1));
        image_file->close();

        image_file->open(std::ios::in, false);
        std::unique_ptr<float[]> to_read = std::make_unique<float[]>(slice_size);
        REQUIRE_ERRNO_GOOD(image_file->readSlice(to_read.get(), 5, 1));
        float diff{0};
        // cast to int16_t is necessary: it happens during writeSlice() since mode=1.
        for (size_t i{0}; i < slice_size; ++i)
            diff += static_cast<float>(static_cast<int16_t>(to_write[i])) - to_read[i];
        REQUIRE_THAT(diff, Catch::WithinULP(0.f, 4));
    }
    fs::remove_all(test_dir);
}

TEST_CASE("MRCFile: complex dtype", "[noa][files]") {
    path_t test_dir = fs::current_path() / "test_MRCFile";
    fs::remove_all(test_dir);

    Errno err;
    AND_THEN("write and read") {
        // create a MRC file...
        path_t file1 = test_dir / "file1.mrc";
        MRCFile file(file1);
        err = file.open(std::ios::out, false);
        REQUIRE_ERRNO_GOOD(err);
        REQUIRE(file.isOpen());

        IO::DataType dtype = GENERATE(IO::DataType::cint16, IO::DataType::cfloat32);
        INFO("dtype: " << IO::toString(dtype));
        // initialize data to put into the file...
        size3_t shape = {64, 64, 64};
        float3_t pixel_size = {1.23f, 1.23f, 1.23f};
        float min{-1}, max{1}, mean{0}, rms{0.5};
        std::unique_ptr<cfloat_t[]> to_write = std::make_unique<cfloat_t[]>(Math::elements(shape));
        Test::IntRandomizer<int> randomizer(0, 127);
        auto* to_write_tmp = reinterpret_cast<float*>(to_write.get());
        for (size_t i{0}; i < Math::elements(shape) * 2; ++i)
            to_write_tmp[i] = static_cast<float>(randomizer.get());

        // write to file...
        REQUIRE_ERRNO_GOOD(file.setDataType(dtype));
        REQUIRE_ERRNO_GOOD(file.setShape(shape));
        REQUIRE_ERRNO_GOOD(file.setPixelSize(pixel_size));
        file.setStatistics(min, max, mean, rms);

        AND_THEN("entire file") {
            REQUIRE_ERRNO_GOOD(file.writeAll(to_write.get()));
            REQUIRE_ERRNO_GOOD(file.close());

            // reading the file and check that it matches...
            REQUIRE_ERRNO_GOOD(file.open(std::ios::in, false));
            REQUIRE(file.getShape() == shape);
            REQUIRE(file.getPixelSize() == pixel_size);
            REQUIRE(file.getStatistics() == std::tuple<float, float, float, float>(min, max, mean, rms));

            std::unique_ptr<cfloat_t[]> to_read = std::make_unique<cfloat_t[]>(Math::elements(shape));
            REQUIRE_ERRNO_GOOD(file.readAll(to_read.get()));

            float diff = Test::getDifference(reinterpret_cast<float*>(to_write.get()),
                                             reinterpret_cast<float*>(to_read.get()),
                                             Math::elements(shape) * 2);
            REQUIRE_THAT(diff, Catch::WithinULP(0.f, 4));
        }

        AND_THEN("slices") {
            size_t slices = 10;
            REQUIRE_ERRNO_GOOD(file.writeSlice(to_write.get(), 5, slices));
            REQUIRE_ERRNO_GOOD(file.close());

            // reading the file and check that it matches...
            REQUIRE_ERRNO_GOOD(file.open(std::ios::in, false));
            REQUIRE(file.getShape() == shape);
            REQUIRE(file.getPixelSize() == pixel_size);
            REQUIRE(file.getStatistics() == std::tuple<float, float, float, float>(min, max, mean, rms));

            std::unique_ptr<cfloat_t[]> to_read = std::make_unique<cfloat_t[]>(Math::elementsSlice(shape) * slices);
            REQUIRE_ERRNO_GOOD(file.readSlice(to_read.get(), 5, slices));

            float diff = Test::getDifference(reinterpret_cast<float*>(to_write.get()),
                                             reinterpret_cast<float*>(to_read.get()),
                                             Math::elementsSlice(shape) * slices * 2);
            REQUIRE_THAT(diff, Catch::WithinULP(0.f, 4));
        }
    }
    fs::remove_all(test_dir);
}
