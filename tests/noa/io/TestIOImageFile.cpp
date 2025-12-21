#include <noa/runtime/Factory.hpp>

#include <noa/io/IO.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

namespace fs = std::filesystem;
namespace nio = noa::io;
using namespace noa::types;


TEST_CASE("io::ImageFile") {
    const auto cwd = fs::current_path() / "test_image_file" / "test0.mrc";

    auto a0 = Array<f32>({10, 1, 64, 64});
    auto s0 = Vec{1., 2., 3.};
    auto randomizer = test::Randomizer<f32>(-128, 128);
    for (auto& e: a0.span_1d_contiguous())
        e = randomizer.get();

    {
        noa::write_image(a0, cwd, {.spacing = s0});

        auto file = nio::ImageFile(cwd, {.read = true});
        REQUIRE(file.shape() == a0.shape());
        REQUIRE(file.spacing().as<f32>() == s0.as<f32>());
        REQUIRE(file.dtype() == nio::DataType::F32);
    }
    {
        auto&& [a1, h1] = noa::read_image<f32>(cwd);
        REQUIRE(test::allclose_abs(a0, a1));
        REQUIRE(h1.spacing.as<f32>() == s0.as<f32>());
    }
    {
        auto a1 = nio::read_image<f32>(cwd).data;
        REQUIRE(test::allclose_abs(a0, a1));
    }

    {
        auto a1 = noa::like<f16>(a0);
        noa::cast(a0, a1);
        noa::cast(a1, a0);

        nio::write_image(a1, cwd, {.dtype = "f16"});
        auto a2 = nio::read_image<f32>(cwd).data;
        REQUIRE(test::allclose_abs(a0, a2));
    }

    fs::remove_all(cwd.parent_path());
}
