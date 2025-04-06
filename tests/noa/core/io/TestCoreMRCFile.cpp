#include <noa/core/io/IO.hpp>
#include <noa/core/io/ImageFile.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;
namespace nio = ::noa::io;
namespace fs = std::filesystem;

TEST_CASE("core::io::BasicImageFile<EncoderMrc>: real dtype", "[asset]") {
    const auto data_file = test::NOA_DATA_PATH / "common" / "io" / "files" / "example_MRCFile.mrc";
    const Path test_dir = fs::current_path() / "test_MRCFile";
    fs::remove_all(test_dir);

    const Path file1 = test_dir / "file1.mrc";
    const Path file2 = test_dir / "file2.mrc";
    using MrcFile = nio::BasicImageFile<nio::EncoderMrc>;

    const auto dtype = GENERATE(
        nio::Encoding::I16, nio::Encoding::U16,
        nio::Encoding::U8, nio::Encoding::I8,
        nio::Encoding::F16, nio::Encoding::F32
    );

    AND_THEN("write and read to a volume") {
        // Output shape and spacing.
        constexpr auto shape = Shape4<i64>{1, 64, 64, 64};
        constexpr auto spacing = Vec{1.23, 1.23, 1.23};

        // Create an MRC file...
        auto file = MrcFile(file1, {.write = true}, {
            .shape = shape,
            .spacing = spacing,
            .dtype = dtype
        });
        REQUIRE(file.is_open());

        // Initialize data to put into the file.
        const auto size = static_cast<size_t>(shape.n_elements());
        const auto to_write = std::make_unique<f32[]>(size);
        const auto s0 = Span(to_write.get(), shape);
        auto randomizer = test::Randomizer<i32>(0, 127);
        for (auto& e: s0.as_1d())
            e = static_cast<f32>(randomizer.get());

        // Write to file.
        file.write_all(s0.as_const(), {.clamp = true});
        file.close();

        // Reading the file and check that it matches.
        auto file_to_read = MrcFile(file1, {.read = true});
        REQUIRE(noa::all(file_to_read.shape() == shape));
        REQUIRE(noa::all(file_to_read.spacing().as<f32>() == spacing.as<f32>()));

        const auto to_read = std::make_unique<f32[]>(size);
        const auto s1 = Span(to_read.get(), shape);
        file_to_read.read_all(s1, {.clamp = true});
        REQUIRE(test::allclose_rel(s0, s1, 1e-8));
    }

    AND_THEN("write and read to a stack of volumes") {
        // Output shape and spacing.
        constexpr auto shape = Shape4<i64>{5, 64, 64, 64};
        constexpr auto spacing = Vec{1.23, 1.23, 1.23};

        // Create an MRC file.
        auto file = MrcFile(file1, {.write = true}, {
            .shape = shape,
            .spacing = spacing,
            .dtype = dtype
        });
        REQUIRE(file.is_open());

        const auto size = static_cast<size_t>(shape.n_elements());
        const auto to_write = std::make_unique<f32[]>(size);
        const auto s0 = Span(to_write.get(), shape);
        auto randomizer = test::Randomizer<i32>(0, 127);
        for (auto& e: s0.as_1d())
            e = static_cast<f32>(randomizer.get());

        // Write to file.
        file.write_all(s0.as_const(), {.clamp = true});
        file.close();

        // Read the file and check that it matches.
        auto file_to_read = MrcFile(file1, {.read = true});
        REQUIRE(noa::all(file_to_read.shape() == shape));
        REQUIRE(noa::all(file_to_read.spacing().as<f32>() == spacing.as<f32>()));

        const auto to_read = std::make_unique<f32[]>(size);
        const auto s1 = Span(to_read.get(), shape);
        file_to_read.read_all(s1, {.clamp = true});
        REQUIRE(test::allclose_abs(s0, s1, 1e-8));
    }

    AND_THEN("write and read a stack of 2d images") {
        constexpr auto shape = Shape4<i64>{41, 1, 64, 64};
        constexpr auto spacing = Vec{1., 1.23, 1.23};

        // Create an MRC file...
        auto file = MrcFile(file1, {.write = true}, {
            .shape = shape,
            .spacing = spacing,
            .dtype = dtype
        });
        REQUIRE(file.is_open());

        // initialize data to put into the file...
        const auto size = static_cast<size_t>(shape.n_elements());
        const auto to_write = std::make_unique<f32[]>(size);
        const auto s0 = Span(to_write.get(), shape);
        auto randomizer = test::Randomizer<i32>(0, 127);
        for (auto& e: s0.as_1d())
            e = static_cast<f32>(randomizer.get());

        // Write to file.
        file.write_all(s0.as_const(), {.clamp = true});
        file.close();

        // Read the file and check that it matches.
        auto file_to_read = MrcFile(file1, {.read = true});
        REQUIRE(noa::all(file_to_read.shape() == shape));
        REQUIRE(noa::all(file_to_read.spacing().as<f32>() == spacing.as<f32>()));

        const auto to_read = std::make_unique<f32[]>(size);
        const auto s1 = Span(to_read.get(), shape);
        file_to_read.read_all(s1, {.clamp = true});
        REQUIRE(test::allclose_abs(s0, s1, 1e-8));
    }

    AND_THEN("reading files") {
        nio::mkdir(test_dir);
        REQUIRE(nio::copy_file(data_file, file2));

        // Writing permissions should not be necessary.
        fs::permissions(file2, fs::perms::owner_write | fs::perms::group_write |
                               fs::perms::others_write, fs::perm_options::remove);
        MrcFile file;
        REQUIRE_THROWS_AS(file.open(file2, {.read = true, .write = true}), noa::Exception);
        fs::remove(file2.string() + "~"); // Remove backup copy from this attempt.
        REQUIRE_FALSE(file.is_open());

        // There should be no backup since it is read only.
        file.open(file2, {.read = true});
        REQUIRE(file.is_open());
        REQUIRE_FALSE(nio::is_file(file2.string() + "~"));

        // Any writing operation should fail.
        const auto n_elements_per_slice = file.shape()[2] * file.shape()[3];
        const auto ptr = std::make_unique<f32[]>(static_cast<size_t>(n_elements_per_slice));
        const auto s0 = Span(ptr.get(), n_elements_per_slice);
        REQUIRE_THROWS_AS(file.write_slice(s0.as_const().as_4d(), {}), noa::Exception);
    }

    AND_THEN("writing to an existing file") {
        nio::mkdir(test_dir);
        REQUIRE(copy_file(data_file, file2));

        // Get the shape.
        const auto shape = MrcFile(file2, {.read = true}).shape().set<0>(3).set<1>(1);

        auto image_file = MrcFile(file2, {.write = true}, {
            .shape = shape,
            .dtype = nio::Encoding::I16,
        });
        REQUIRE(image_file.is_open());
        REQUIRE(fs::is_regular_file(file2.string() + "~")); // backup since we open in writing mode

        const auto n_elements = static_cast<size_t>(shape.n_elements());
        const auto to_write = std::make_unique<f32[]>(n_elements);
        const auto s0 = Span(to_write.get(), shape);
        auto randomizer = test::Randomizer<f32>(-1000, 1000);
        for (auto& e: s0.as_1d())
            e = randomizer.get();
        image_file.write_slice(s0.as_const(), {.clamp = false});
        image_file.close();

        image_file.open(file2, {.read = true});
        const auto to_read = std::make_unique<f32[]>(n_elements);
        const auto s1 = Span(to_read.get(), shape);
        image_file.read_slice(s1.subregion(0), {.bd_offset = {0, 0}, .clamp = false});
        image_file.read_slice(s1.subregion(1), {.bd_offset = {1, 0}, .clamp = false});
        image_file.read_slice(s1.subregion(2), {.bd_offset = {2, 0}, .clamp = false});

        f32 diff{};
        // cast to i16 is necessary: it happens during write_slice() since the encoding is i16.
        for (size_t i{}; i < s1.size(); ++i)
            diff += static_cast<f32>(static_cast<i16>(to_write[i])) - to_read[i];
        REQUIRE_THAT(diff, Catch::Matchers::WithinULP(0.f, 4));
    }
    std::error_code er;
    fs::remove_all(test_dir, er); // silence error
}
