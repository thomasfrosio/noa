#include <catch2/catch.hpp>
#include "../../Helpers.h"

#include "noa/util/files/MRCFile.h"

using namespace ::Noa;

TEST_CASE("MRCFile", "[noa][files]") {
    fs::path fixture1 = NOA_TESTS_FIXTURE;
    fs::path test_dir = fs::current_path() / "test_MRCFile";
    fs::remove_all(test_dir);

    AND_THEN("write and read") {
        // create a MRC file...
        fs::path file1 = test_dir / "file1.mrc";
        MRCFile file(file1, std::ios::out);

        DataType dtype = GENERATE(DataType::int16, DataType::uint16, DataType::ubyte, DataType::byte);

        // initialize data to put into the file...
        Int3<size_t> shape = {12, 12, 12};
        Float3 pixel_size = {1.23f, 1.23f, 1.23f};
        float min{-1}, max{1}, mean{0}, rms{0.5};
        std::unique_ptr<float[]> to_write = std::make_unique<float[]>(shape.prod());
        for (size_t i{0}; i < shape.prod(); ++i)
            to_write[i] = static_cast<float>(Test::random(0, 127));

        // write to file...
        Noa::Flag<Errno> err;
        err.update(file.setDataType(dtype));
        err.update(file.setShape(shape));
        err.update(file.setPixelSize(pixel_size));
        file.setStatistics(min, max, mean, rms);
        err.update(file.writeAll(to_write.get()));
        err.update(file.close());
        REQUIRE_ERRNO_GOOD(err);

        // reading the file and check that it matches...
        err.update(file.open(file1, std::ios::in, false));
        REQUIRE(file.getShape() == shape);
        REQUIRE(file.getPixelSize() == pixel_size);

        std::unique_ptr<float[]> to_read = std::make_unique<float[]>(shape.prod());
        err.update(file.readAll(to_read.get()));
        float diff{0};
        for (size_t i{0}; i < shape.prod(); ++i)
            diff += to_write[i] - to_read[i];
        REQUIRE_THAT(diff, Catch::WithinULP(0.f, 2));
        REQUIRE_ERRNO_GOOD(err);
    }

    AND_THEN("backing up files when using open()") {


    }

    AND_THEN("getting MRCFile at runtime") {
        fs::path file = test_dir / "TestImageFile_mrcfile.mrc";
        std::unique_ptr<ImageFile> image_file = ImageFile::get(file.extension().string());
        REQUIRE(image_file);
        image_file->open(file, std::ios::in, false);
    }
    fs::remove_all(test_dir);
}
