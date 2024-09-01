#include <noa/core/io/IO.hpp>
#include <noa/core/io/OS.hpp>
#include <noa/core/io/MRCFile.hpp>

#include <iostream>

#include "Utils.hpp"
#include <catch2/catch.hpp>

using namespace ::noa::types;
namespace nio = ::noa::io;
namespace fs = std::filesystem;

TEST_CASE("core::io::Stats", "[noa][core]") {
    nio::Stats<f32> out;
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

TEST_CASE("core::io::MrcFile: real dtype", "[noa][core]") {
    const auto data_file = test::NOA_DATA_PATH / "common" / "io" / "files" / "example_MRCFile.mrc";
    const std::string fixture_expected_header =
            "Format: MRC File\n"
            "Shape (batch, depth, height, width): [11, 1, 576, 410]\n"
            "Pixel size (depth, height, width): [2.100, 21.000, 21.000]\n"
            "Data type: Encoding::I16\n"
            "Labels: 9\n"
            "Extended header: 0 bytes";
    const Path test_dir = fs::current_path() / "test_MrcFile";
    fs::remove_all(test_dir);

    AND_THEN("write and read to a volume") {
        // create a MRC file...
        const Path file1 = test_dir / "file1.mrc";
        auto file = nio::MrcFile(file1, {.write=true});
        REQUIRE(file.is_open());

        const nio::Encoding::Format dtype =
                GENERATE(nio::Encoding::I16, nio::Encoding::U16,
                         nio::Encoding::U8, nio::Encoding::I8,
                         nio::Encoding::F16, nio::Encoding::F32);

        // initialize data to put into the file...
        const auto shape = Shape4<i64>{1, 64, 64, 64};
        const auto size = static_cast<size_t>(shape.n_elements());
        const auto pixel_size = Vec{1.23, 1.23, 1.23};
        nio::Stats<f64> stats;
        stats.set_min(-1);
        stats.set_max(1);
        stats.set_mean(0.5);

        const auto to_write = std::make_unique<f32[]>(size);
        const auto s0 = Span(to_write.get(), shape);
        test::Randomizer<i32> randomizer(0, 127);
        for (auto& e: s0.as_contiguous_1d())
            e = static_cast<f32>(randomizer.get());

        // write to file...
        file.set_encoding_format(dtype);
        file.set_shape(shape);
        file.set_pixel_size(pixel_size);
        file.set_stats(stats);
        file.write_all(s0.as_const(), true);
        file.close();

        // reading the file and check that it matches...
        auto file_to_read = nio::MrcFile(file1, {.read=true});
        REQUIRE(all(file_to_read.shape() == shape));
        REQUIRE(all(file_to_read.pixel_size().as<f32>() == pixel_size.as<f32>()));
        const nio::Stats file_stats = file_to_read.stats();
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
        const auto s1 = Span(to_read.get(), shape);
        file_to_read.read_all(s1, true);
        REQUIRE(test::allclose_rel(to_write.get(), to_read.get(), size, 1e-8));
    }

    AND_THEN("write and read to a stack of volumes") {
        // create a MRC file...
        const Path file1 = test_dir / "file1.mrc";
        auto file = nio::MrcFile(file1, {.write=true});
        REQUIRE(file.is_open());

        const nio::Encoding::Format encoding_format = GENERATE(
                nio::Encoding::I16, nio::Encoding::U16,
                nio::Encoding::U8, nio::Encoding::I8,
                nio::Encoding::F16, nio::Encoding::F32);

        // initialize data to put into the file...
        const auto shape = Shape4<i64>{5, 64, 64, 64};
        const auto size = static_cast<size_t>(shape.n_elements());
        const auto pixel_size = Vec{1.23, 1.23, 1.23};
        const auto to_write = std::make_unique<f32[]>(size);
        const auto s0 = Span(to_write.get(), shape);
        test::Randomizer<i32> randomizer(0, 127);
        for (auto& e: s0.as_1d())
            e = static_cast<f32>(randomizer.get());

        // write to file...
        file.set_encoding_format(encoding_format);
        file.set_shape(shape);
        file.set_pixel_size(pixel_size);
        file.write_all(s0.as_const(), true);
        file.close();

        // reading the file and check that it matches...
        nio::MrcFile file_to_read(file1, {.read=true});
        REQUIRE(all(file_to_read.shape() == shape));
        REQUIRE(all(file_to_read.pixel_size().as<f32>() == pixel_size.as<f32>()));
        const nio::Stats file_stats = file_to_read.stats();
        REQUIRE_FALSE(file_stats.has_min());
        REQUIRE_FALSE(file_stats.has_max());
        REQUIRE_FALSE(file_stats.has_sum());
        REQUIRE_FALSE(file_stats.has_mean());
        REQUIRE_FALSE(file_stats.has_var());
        REQUIRE_FALSE(file_stats.has_std());

        const auto to_read = std::make_unique<f32[]>(size);
        const auto s1 = Span(to_read.get(), shape);
        file_to_read.read_all(s1, true);
        REQUIRE(test::allclose_abs(s0, s1, 1e-8));
    }

    AND_THEN("write and read a stack of 2D images") {
        // create a MRC file...
        const Path file1 = test_dir / "file1.mrc";
        auto file = nio::MrcFile(file1, {.write=true});
        REQUIRE(file.is_open());

        const nio::Encoding::Format encoding_format = GENERATE(
                nio::Encoding::I16, nio::Encoding::U16,
                nio::Encoding::U8, nio::Encoding::I8,
                nio::Encoding::F16, nio::Encoding::F32);

        // initialize data to put into the file...
        const auto shape = Shape4<i64>{41, 1, 64, 64};
        const auto size = static_cast<size_t>(shape.n_elements());
        const auto pixel_size = Vec{1., 1.23, 1.23};
        const auto to_write = std::make_unique<f32[]>(size);
        const auto s0 = Span(to_write.get(), shape);
        test::Randomizer<i32> randomizer(0, 127);
        for (auto& e: s0.as_1d())
            e = static_cast<f32>(randomizer.get());

        // write to file...
        file.set_encoding_format(encoding_format);
        file.set_shape(shape);
        file.set_pixel_size(pixel_size);
        file.write_all(s0.as_const(), true);
        file.close();

        // reading the file and check that it matches...
        auto file_to_read = nio::MrcFile(file1, {.read=true});
        REQUIRE(all(file_to_read.shape() == shape));
        REQUIRE(all(file_to_read.pixel_size().as<f32>() == pixel_size.as<f32>()));

        const auto to_read = std::make_unique<f32[]>(size);
        const auto s1 = Span(to_read.get(), shape);
        file_to_read.read_all(s1, true);
        REQUIRE(test::allclose_abs(s0, s1, 1e-8));
    }

    AND_THEN("reading files") {
        const Path fixture_copy = test_dir / "file2.mrc";
        nio::mkdir(test_dir);
        REQUIRE(nio::copy_file(data_file, fixture_copy));

        // Writing permissions should not be necessary.
        fs::permissions(fixture_copy, fs::perms::owner_write | fs::perms::group_write |
                                      fs::perms::others_write, fs::perm_options::remove);
        nio::MrcFile file;
        REQUIRE_THROWS_AS(file.open(fixture_copy, {.read=true, .write=true}), noa::Exception);
        nio::remove(fixture_copy.string() + "~"); // Remove backup copy from this attempt.
        REQUIRE_FALSE(file.is_open());

        // There should be no backup since it is read only.
        file.open(fixture_copy, {.read=true});
        REQUIRE(file.is_open());
        REQUIRE_FALSE(nio::is_file(fixture_copy.string() + "~"));

        // Any writing operation should fail.
        const auto n_elements_per_slice = file.shape()[2] * file.shape()[3];
        const auto ptr = std::make_unique<f32[]>(static_cast<size_t>(n_elements_per_slice));
        const auto s0 = Span(ptr.get(), n_elements_per_slice);
        REQUIRE_THROWS_AS(file.write_slice(s0.as_const().as_4d(), 0, true), noa::Exception);

        const std::string str = file.info_string(false);
        REQUIRE(str == fixture_expected_header);
    }

    AND_THEN("writing to an existing file") {
        const Path fixture_copy = test_dir / "file2.mrc";
        nio::mkdir(test_dir);
        REQUIRE(copy_file(data_file, fixture_copy));

        auto image_file = nio::MrcFile(fixture_copy, {.read=true, .write=true});
        REQUIRE(image_file.is_open());

        // Check backup copy.
        REQUIRE(nio::is_file(fixture_copy.string() + "~"));
        REQUIRE(image_file.info_string(false) == fixture_expected_header);

        const auto slice_shape = image_file.shape().pop_front<2>().push_front<2>(1); // ensure one slice
        const auto n_elements = static_cast<size_t>(slice_shape.n_elements());
        const auto to_write = std::make_unique<f32[]>(n_elements);
        const auto s0 = Span(to_write.get(), slice_shape);
        test::Randomizer<f32> randomizer(-1000, 1000);
        for (auto& e: s0.as_1d())
            e = randomizer.get();
        image_file.write_slice(s0.as_const(), 5, false);
        image_file.close();

        image_file.open(fixture_copy, {.read=true});
        const auto to_read = std::make_unique<f32[]>(n_elements);
        const auto s1 = Span(to_read.get(), slice_shape);
        image_file.read_slice(s1, 5, false);

        f32 diff{};
        // cast to int16_t is necessary: it happens during write_slice() since the file is in mode=int16.
        for (size_t i{}; i < s1.size(); ++i)
            diff += static_cast<f32>(static_cast<int16_t>(to_write[i])) - to_read[i];
        REQUIRE_THAT(diff, Catch::WithinULP(0.f, 4));
    }
    std::error_code er;
    fs::remove_all(test_dir, er); // silence error
}
