#include <noa/unified/io/ImageFile.hpp>
#include "noa/unified/Factory.hpp"
#include <noa/unified/Cast.hpp>
#include <noa/core/io/OS.hpp>
#include <catch2/catch.hpp>

#include "Utils.hpp"

namespace fs = std::filesystem;
namespace nio = noa::io;
using namespace noa::types;


TEST_CASE("unified::ImageFile", "[noa]") {
    const auto cwd = fs::current_path() / "test_image_file" / "test0.mrc";

    auto a0 = Array<f32>({10, 1, 64, 64});
    auto randomizer = test::Randomizer<f32>(-128, 128);
    for (auto& e: a0.span_1d_contiguous())
        e = randomizer.get();

    {
        auto file = nio::ImageFile(cwd, {.write=true});
        file.write(a0.view());
        REQUIRE(noa::all(file.shape() == a0.shape()));
        REQUIRE(noa::all(file.pixel_size() == Vec{0., 0., 0.}));
        REQUIRE(file.encoding_format() == nio::Encoding::F32);
    }
    {
        auto file = nio::ImageFile(cwd, {.read=true});
        auto a1 = file.read<f32>({});
        REQUIRE(test::allclose_abs(a0, a1));

        auto a2 = a0.subregion(3);
        auto a3 = noa::like(a2);
        REQUIRE_THROWS_AS(file.read_slice(a3, 10, false), noa::Exception);
        REQUIRE_THROWS_AS(file.read_slice(a0, 4, false), noa::Exception);
        file.read_slice(a3, 5, false);
        REQUIRE_FALSE(test::allclose_abs(a2, a3));

        file.read_slice(a3, 3, false);
        REQUIRE(test::allclose_abs(a2, a3));
    }
    {
        auto a1 = nio::read_data<f32>(cwd);
        REQUIRE(test::allclose_abs(a0, a1));
    }

    {
        auto a1 = noa::like<f16>(a0);
        noa::cast(a0, a1);
        noa::cast(a1, a0);

        nio::write(a1, Vec{1.7, 1.8}, cwd, {.encoding_format=nio::Encoding::F16});
        auto a2 = nio::read_data<f32>(cwd);
        REQUIRE(test::allclose_abs(a0, a2));
    }

    fs::remove_all(cwd.parent_path());
}
