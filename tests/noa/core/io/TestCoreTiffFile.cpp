#include <noa/core/io/IO.hpp>
#include <noa/core/io/ImageFile.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;
namespace nio = ::noa::io;
namespace fs = std::filesystem;

TEST_CASE("core::io::ImageFile - TIFF: simple write and read") {
    const Path test_dir = fs::current_path() / "test_TIFFFile";
    fs::remove_all(test_dir);

    const Path file0 = test_dir / "file1.tif";

    using TiffFile = nio::BasicImageFile<nio::EncoderTiff>;
    constexpr auto shape = Shape<i64, 4>{4, 1, 256, 256};
    constexpr auto dtype = nio::Encoding::F32;

    auto data0 = std::make_unique<f32[]>(static_cast<size_t>(shape.n_elements()));
    for (f32 i{}; auto& e: Span(data0.get(), shape.n_elements()))
        e = i++;

    auto file = TiffFile(file0, {.write = true}, {.shape = shape, .dtype = nio::Encoding::F32});
    file.write_all(Span(data0.get(), file.shape()).as_const());
    file.close();

    file.open(file0, {.read = true});
    REQUIRE(noa::all(shape == file.shape()));
    REQUIRE(noa::all(dtype == file.dtype()));

    auto data1 = std::make_unique<f32[]>(static_cast<size_t>(shape.n_elements()));
    file.read_all(Span(data1.get(), file.shape()), {.n_threads = 4});

    REQUIRE(test::allclose_abs(data0.get(), data1.get(), shape.n_elements()));

    std::error_code er;
    fs::remove_all(test_dir, er); // silence error
}
