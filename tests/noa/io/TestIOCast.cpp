#include <noa/runtime/Factory.hpp>

#include <noa/io/IO.hpp>
#include <noa/io/Cast.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

namespace fs = std::filesystem;
namespace nio = noa::io;
using namespace noa::types;


TEST_CASE("io::cast") {
    auto dtype = GENERATE(nio::DataType::F32, nio::DataType::I64);

    const auto path = fs::current_path() / "test_image_file";
    for (auto extension: std::array{".mrc", ".tiff"}) {
        const auto file_0 = path / fmt::format("test0{}", extension);
        const auto file_1 = path / fmt::format("test1{}", extension);

        {
            auto a0 = Array<f32>({10, 1, 160, 64});
            auto s0 = Vec{1., 2., 3.};
            auto randomizer = test::Randomizer<i32>(-128, 128);
            for (auto& e: a0.span_1d_contiguous())
                e = randomizer.get();

            noa::write_image(a0, file_0, {.spacing = s0, .dtype = dtype});
            noa::write_image(a0.reinterpret_as<std::byte>(), "f32", file_1, {.spacing = s0, .dtype = dtype});

            auto a1 = nio::read_image<f32>(file_0).data;
            auto a2 = nio::read_image<std::byte>(file_1, {.dtype = "f32"}).data;

            REQUIRE(test::allclose_abs(a0, a1));
            REQUIRE(test::allclose_abs(a0, a2.reinterpret_as<f32>()));
        }
        {
            auto a0 = Array<f32>({10, 1, 160, 65});
            auto randomizer = test::Randomizer<f32>(-128, 128);
            for (auto& e: a0.span_1d_contiguous())
                e = randomizer.get();

            auto a1 = noa::like<i32>(a0);
            auto a2 = noa::like<i32>(a0);
            noa::cast(a0, a1);
            nio::cast(a0, a2.reinterpret_as<std::byte>(), "i32");
            REQUIRE(test::allclose_abs(a1, a2));
        }
    }
    fs::remove_all(path);
}
