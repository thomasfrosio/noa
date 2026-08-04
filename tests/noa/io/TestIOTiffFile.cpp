#include <noa/io/IO.hpp>
#include <noa/io/ImageFile.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;
namespace nio = ::noa::io;
namespace fs = std::filesystem;

TEST_CASE("io::ImageFile - TIFF: simple write and read") {
    const Path test_dir = fs::current_path() / "test_TIFFFile";
    fs::remove_all(test_dir);

    const Path file0 = test_dir / "file1.tif";

    using TiffFile = nio::BasicImageFile<nio::ImageFileEncoderTiff>;
    constexpr auto shape = Shape<isize, 4>{3, 1, 64, 65};
    constexpr auto dtype = nio::DataType::I32;

    auto data0 = std::make_unique<i32[]>(static_cast<size_t>(shape.n_elements()));
    for (i32 i{}; auto& e: Span(data0.get(), shape.n_elements()))
        e = i++;

    auto file = TiffFile(file0, {.write = true}, {
        .shape = shape,
        .spacing = {1., 2., 3.},
        .dtype = "i32",
        .compression = nio::Compression::LZW,
        .stats = {.min = -1., .max = 1.},
    });
    file.write_all(Span(data0.get(), file.shape()).as_const());
    file.close();

    file.open(file0, {.read = true});
    REQUIRE(shape == file.shape());
    REQUIRE(dtype == file.dtype());
    REQUIRE((file.is_compressed() and file.compression() == nio::Compression::LZW));
    REQUIRE(file.encoder_name() == "tiff");
    REQUIRE(file.spacing() == Vec{1., 2., 3.});
    REQUIRE((file.stats().min == -1. and file.stats().max == 1.));

    auto data1 = std::make_unique<i32[]>(static_cast<size_t>(shape.n_elements()));
    file.read_all(Span(data1.get(), file.shape()), {.n_threads = 4});

    REQUIRE(test::allclose_abs(data0.get(), data1.get(), shape.n_elements()));

    std::error_code er;
    fs::remove_all(test_dir, er); // silence error
}
