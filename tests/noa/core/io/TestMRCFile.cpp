#include <noa/core/io/IO.hpp>
#include <noa/core/io/OS.hpp>
#include <noa/core/io/MRCFile.hpp>

#include <iostream>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("core::io::Stats", "[noa][core]") {
    io::Stats<f32> out;
    REQUIRE_FALSE(out.has_min());
    REQUIRE_FALSE(out.has_max());
    REQUIRE_FALSE(out.has_sum());
    REQUIRE_FALSE(out.has_mean());
    REQUIRE_FALSE(out.has_var());
    REQUIRE_FALSE(out.has_std());

    out.set_min(1);
    REQUIRE(out.has_min());
    REQUIRE(out.min() == 1);
    out.set_max(2);
    REQUIRE(out.has_max());
    REQUIRE(out.max() == 2);
    out.set_sum(3);
    REQUIRE(out.has_sum());
    REQUIRE(out.sum() == 3);
    out.set_mean(4);
    REQUIRE(out.has_mean());
    REQUIRE(out.mean() == 4);
    out.set_var(5);
    REQUIRE(out.has_var());
    REQUIRE(out.var() == 5);
    out.set_std(6);
    REQUIRE(out.has_std());
    REQUIRE(out.std() == 6);
}

TEST_CASE("core::io::MRCFile: real dtype", "[noa][core]") {
    const auto data_file = test::NOA_DATA_PATH / "common" / "io" / "files" / "example_MRCFile.mrc";
    const std::string fixture_expected_header =
            "Format: MRC File\n"
            "Shape (batches, depth, height, width): [11, 1, 576, 410]\n"
            "Pixel size (depth, height, width): [2.100, 21.000, 21.000]\n"
            "Data type: DataType::I16\n"
            "Labels: 9\n"
            "Extended header: 0 bytes";
    const Path test_dir = fs::current_path() / "test_MRCFile";
    fs::remove_all(test_dir);

    AND_THEN("write and read to a volume") {
        // create a MRC file...
        const Path file1 = test_dir / "file1.mrc";
        io::MRCFile file(file1, io::WRITE);
        REQUIRE(file);

        const io::DataType dtype =
                GENERATE(io::DataType::I16, io::DataType::U16,
                         io::DataType::U8, io::DataType::I8,
                         io::DataType::F16, io::DataType::F32);

        // initialize data to put into the file...
        const auto shape = Shape4<i64>{1, 64, 64, 64};
        const auto size = static_cast<size_t>(shape.elements());
        const auto pixel_size = Vec3<f32>{1.23f, 1.23f, 1.23f};
        io::Stats<f32> stats;
        stats.set_min(-1);
        stats.set_max(1);
        stats.set_mean(0.5);
        const auto to_write = std::make_unique<f32[]>(size);
        test::Randomizer<int> randomizer(0, 127);
        for (size_t i{0}; i < size; ++i)
            to_write[i] = static_cast<f32>(randomizer.get());

        // write to file...
        file.set_dtype(dtype);
        file.set_shape(shape);
        file.set_pixel_size(pixel_size);
        file.set_stats(stats);
        file.write_all(to_write.get());
        file.close();

        // reading the file and check that it matches...
        io::MRCFile file_to_read(file1, io::READ);
        REQUIRE(all(file_to_read.shape() == shape));
        REQUIRE(all(file_to_read.pixel_size() == pixel_size));
        const io::Stats<f32> file_stats = file_to_read.stats();
        REQUIRE_FALSE(file_stats.has_sum());
        REQUIRE_FALSE(file_stats.has_var());
        REQUIRE_FALSE(file_stats.has_std());
        REQUIRE(file_stats.has_min());
        REQUIRE(file_stats.has_max());
        REQUIRE(file_stats.has_mean());
        REQUIRE(stats.min() == file_stats.min());
        REQUIRE(stats.max() == file_stats.max());
        REQUIRE(stats.mean() == file_stats.mean());
        REQUIRE(stats.std() == file_stats.std());

        const auto to_read = std::make_unique<f32[]>(size);
        file_to_read.read_all(to_read.get());
        const f32 diff = test::get_difference(to_write.get(), to_read.get(), size);
        REQUIRE_THAT(diff, Catch::WithinULP(0.f, 4));
    }

    AND_THEN("write and read to a stack of volumes") {
        // create a MRC file...
        const Path file1 = test_dir / "file1.mrc";
        io::MRCFile file(file1, io::WRITE);
        REQUIRE(file);

        const io::DataType dtype = GENERATE(
                io::DataType::I16, io::DataType::U16,
                io::DataType::U8, io::DataType::I8,
                io::DataType::F16, io::DataType::F32);

        // initialize data to put into the file...
        const auto shape = Shape4<i64>{5, 64, 64, 64};
        const auto size = static_cast<size_t>(shape.elements());
        const auto pixel_size = Vec3<f32>{1.23f, 1.23f, 1.23f};
        const io::Stats<f32> stats;
        const auto to_write = std::make_unique<f32[]>(size);
        test::Randomizer<int> randomizer(0, 127);
        for (size_t i{0}; i < size; ++i)
            to_write[i] = static_cast<f32>(randomizer.get());

        // write to file...
        file.set_dtype(dtype);
        file.set_shape(shape);
        file.set_pixel_size(pixel_size);
        file.set_stats(stats);
        file.write_all(to_write.get());
        file.close();

        // reading the file and check that it matches...
        io::MRCFile file_to_read(file1, io::READ);
        REQUIRE(all(file_to_read.shape() == shape));
        REQUIRE(all(file_to_read.pixel_size() == pixel_size));
        const io::Stats<f32> file_stats = file_to_read.stats();
        REQUIRE_FALSE(file_stats.has_min());
        REQUIRE_FALSE(file_stats.has_max());
        REQUIRE_FALSE(file_stats.has_sum());
        REQUIRE_FALSE(file_stats.has_mean());
        REQUIRE_FALSE(file_stats.has_var());
        REQUIRE_FALSE(file_stats.has_std());

        const auto to_read = std::make_unique<f32[]>(size);
        file_to_read.read_all(to_read.get());
        const f32 diff = test::get_difference(to_write.get(), to_read.get(), size);
        REQUIRE_THAT(diff, Catch::WithinULP(0.f, 4));
    }

    AND_THEN("write and read a stack of 2D images") {
        // create a MRC file...
        const Path file1 = test_dir / "file1.mrc";
        io::MRCFile file(file1, io::WRITE);
        REQUIRE(file);

        const io::DataType dtype = GENERATE(
                io::DataType::I16, io::DataType::U16,
                io::DataType::U8, io::DataType::I8,
                io::DataType::F16, io::DataType::F32);

        // initialize data to put into the file...
        const auto shape = Shape4<i64>{41, 1, 64, 64};
        const auto size = static_cast<size_t>(shape.elements());
        const auto pixel_size = Vec3<f32>{1, 1.23f, 1.23f};
        const auto to_write = std::make_unique<f32[]>(size);
        test::Randomizer<int> randomizer(0, 127);
        for (size_t i{0}; i < size; ++i)
            to_write[i] = static_cast<f32>(randomizer.get());

        // write to file...
        file.set_dtype(dtype);
        file.set_shape(shape);
        file.set_pixel_size(pixel_size);
        file.write_all(to_write.get());
        file.close();

        // reading the file and check that it matches...
        io::MRCFile file_to_read(file1, io::READ);
        REQUIRE(all(file_to_read.shape() == shape));
        REQUIRE(all(file_to_read.pixel_size() == pixel_size));

        const auto to_read = std::make_unique<f32[]>(size);
        file_to_read.read_all(to_read.get());
        const f32 diff = test::get_difference(to_write.get(), to_read.get(), size);
        REQUIRE_THAT(diff, Catch::WithinULP(0.f, 4));
    }

    AND_THEN("reading files") {
        const Path fixture_copy = test_dir / "file2.mrc";
        io::mkdir(test_dir);
        REQUIRE(io::copy_file(data_file, fixture_copy));

        // Writing permissions should not be necessary.
        fs::permissions(fixture_copy, fs::perms::owner_write | fs::perms::group_write |
                                      fs::perms::others_write, fs::perm_options::remove);
        io::MRCFile file;
        REQUIRE_THROWS_AS(file.open(fixture_copy, io::READ | io::WRITE), noa::Exception);
        io::remove(fixture_copy.string() + "~"); // Remove backup copy from this attempt.
        REQUIRE_FALSE(file.is_open());

        // There should be no backup since it is read only.
        file.open(fixture_copy, io::READ);
        REQUIRE(file.is_open());
        REQUIRE_FALSE(io::is_file(fixture_copy.string() + "~"));

        // Any writing operation should fail.
        const auto elements_per_slice = static_cast<size_t>(file.shape()[2] * file.shape()[3]);
        const auto ptr = std::make_unique<f32[]>(elements_per_slice);
        REQUIRE_THROWS_AS(file.write_slice(ptr.get(), 0, 1), noa::Exception);

        const std::string str = file.info_string(false);
        REQUIRE(str == fixture_expected_header);
    }

    AND_THEN("writing to an existing file") {
        const Path fixture_copy = test_dir / "file2.mrc";
        io::mkdir(test_dir);
        REQUIRE(io::copy_file(data_file, fixture_copy));

        io::MRCFile image_file(fixture_copy, io::READ | io::WRITE);
        REQUIRE(image_file.is_open());

        // Check backup copy.
        REQUIRE(io::is_file(fixture_copy.string() + "~"));
        REQUIRE(image_file.info_string(false) == fixture_expected_header);

        const auto elements_per_slice = static_cast<size_t>(image_file.shape()[2] * image_file.shape()[3]);
        const auto to_write = std::make_unique<f32[]>(elements_per_slice);
        test::Randomizer<f32> randomizer(-1000, 1000);
        for (size_t idx{0}; idx < elements_per_slice; ++idx)
            to_write[idx] = randomizer.get();
        image_file.write_slice(to_write.get(), 5, 6);
        image_file.close();

        image_file.open(fixture_copy, io::READ);
        const auto to_read = std::make_unique<f32[]>(elements_per_slice);
        image_file.read_slice(to_read.get(), 5, 6);
        f32 diff{0};
        // cast to int16_t is necessary: it happens during write_slice() since the file is in mode=int16.
        for (size_t i{0}; i < elements_per_slice; ++i)
            diff += static_cast<f32>(static_cast<int16_t>(to_write[i])) - to_read[i];
        REQUIRE_THAT(diff, Catch::WithinULP(0.f, 4));
    }
    std::error_code er;
    fs::remove_all(test_dir, er); // silence error
}
