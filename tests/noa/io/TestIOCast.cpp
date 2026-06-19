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
            auto a0 = Array<f32, 3>({10, 160, 64});
            auto s0 = Vec{1., 2., 3.};
            auto randomizer = test::Randomizer<i32>(-128, 128);
            for (auto& e: a0.span_1d())
                e = static_cast<f32>(randomizer.get());

            noa::write_image(a0, file_0, {.enforce_2d_stack = true, .spacing = s0, .dtype = dtype});
            noa::write_image(a0.as<const std::byte>(), "f32", file_1, {.enforce_2d_stack = true, .spacing = s0, .dtype = dtype});

            auto a1 = nio::read_image<f32, 3>(file_0, {.enforce_2d_stack = true}).data;
            auto a2 = nio::read_image<std::byte, 1>(file_1, {.enforce_2d_stack = true, .dtype = "f32"}).data;

            REQUIRE(test::allclose_abs(a0, a1));
            REQUIRE(test::allclose_abs(a0.as_1d(), a2.as<f32>()));
        }
        {
            auto a0 = Array<f32, 3>({10, 160, 65});
            auto randomizer = test::Randomizer<f32>(-128, 128);
            for (auto& e: a0.span_1d())
                e = randomizer.get();

            auto a1 = noa::empty_like<i32>(a0);
            auto a2 = noa::empty_like<i32>(a0);
            noa::cast(a0, a1);
            nio::cast(a0, a2.as<std::byte>(), "i32");
            REQUIRE(test::allclose_abs(a1, a2));
        }
    }
    fs::remove_all(path);
}
