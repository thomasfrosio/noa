#include <catch2/catch.hpp>
#include "../../../Helpers.h"

#include "noa/util/files/MRCFile.h"

using namespace ::Noa;

TEST_CASE("MRCFile", "[noa][files]") {
    fs::path fixture_dir = NOA_TESTS_FIXTURE;
    fs::path fixture = fixture_dir / "TestImageFile_mrcfile.mrc";
    std::string fixture_expected_header = "Format: MRC File\n"
                                          "Shape (columns, rows, sections): (410, 576, 11)\n"
                                          "Pixel size (columns, rows, sections): (21.000, 21.000, 2.100)\n"
                                          "Data type: int16\n"
                                          "Labels: 9\n"
                                          "Extended headers: 0 bytes";
    fs::path test_dir = fs::current_path() / "test_MRCFile";
    fs::remove_all(test_dir);

    AND_THEN("write and read") {
        // create a MRC file...
        fs::path file1 = test_dir / "file1.mrc";
        MRCFile file(file1, std::ios::out, false);

        DataType dtype = GENERATE(DataType::int16, DataType::uint16,
                                  DataType::ubyte, DataType::byte,
                                  DataType::float32);

        // initialize data to put into the file...
        Int3<size_t> shape = {64, 64, 64};
        Float3 pixel_size = {1.23f, 1.23f, 1.23f};
        float min{-1}, max{1}, mean{0}, rms{0.5};
        std::unique_ptr<float[]> to_write = std::make_unique<float[]>(shape.prod());
        for (size_t i{0}; i < shape.prod(); ++i)
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

        std::unique_ptr<float[]> to_read = std::make_unique<float[]>(shape.prod());
        REQUIRE_ERRNO_GOOD(file.readAll(to_read.get()));
        float diff{0};
        for (size_t i{0}; i < shape.prod(); ++i)
            diff += to_write[i] - to_read[i];
        REQUIRE_THAT(diff, Catch::WithinULP(0.f, 4));
    }

    AND_THEN("reading files") {
        fs::path fixture_copy = test_dir / "file2.mrc";
        REQUIRE_ERRNO_GOOD(OS::mkdir(test_dir));
        REQUIRE_ERRNO_GOOD(OS::copyFile(fixture, fixture_copy));

        // Writing permissions should not be necessary.
        fs::permissions(fixture_copy, fs::perms::owner_write | fs::perms::group_write | fs::perms::others_write,
                        fs::perm_options::remove);
        MRCFile file(fixture_copy, std::ios::in | std::ios::out, false);
        OS::remove(fixture_copy.string() + "~"); // Remove backup copy from this attempt.
        REQUIRE_FALSE(file.isOpen());

        // There should be no backup since it is read only.
        REQUIRE_ERRNO_GOOD(file.open(std::ios::in, false));
        Errno err;
        REQUIRE_FALSE(OS::existsFile(fixture_copy.string() + "~", err));
        REQUIRE_ERRNO_GOOD(err);

        // Any writing operation should fail.
        std::unique_ptr<float[]> ptr = std::make_unique<float[]>(file.getShape().prodSlice());
        REQUIRE(file.writeSlice(ptr.get(), 0, 1) == Errno::fail_write);

        std::string str = file.toString(false);
        REQUIRE(str == fixture_expected_header);
    }

    AND_THEN("writing to an existing file") {
        fs::path fixture_copy = test_dir / "file2.mrc";
        REQUIRE_ERRNO_GOOD(OS::mkdir(test_dir));
        REQUIRE_ERRNO_GOOD(OS::copyFile(fixture, fixture_copy));

        // Using the runtime deduction.
        std::unique_ptr<ImageFile> image_file = ImageFile::get(fixture_copy.extension());
        REQUIRE(image_file);
        REQUIRE_ERRNO_GOOD(image_file->open(fixture_copy, std::ios::in | std::ios::out, false));

        // Check backup copy.
        Errno err;
        REQUIRE(OS::existsFile(fixture_copy.string() + "~", err));
        REQUIRE(image_file->toString(false) == fixture_expected_header);
        REQUIRE_ERRNO_GOOD(err);

        // Changing the dataset is not supported in in|out mode.
        REQUIRE(image_file->setDataType(DataType::float32) == Errno::not_supported);

        size_t slice_size = image_file->getShape().prodSlice();
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
