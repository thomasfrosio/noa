#include <noa/core/io/BinaryFile.hpp>
#include <catch2/catch.hpp>

#include "Utils.hpp"

TEST_CASE("io::BinaryFile", "[noa][io]") {
    using namespace noa::types;
    using noa::io::Open;
    using noa::io::BinaryFile;

    const i64 n_elements = 1024;
    const std::unique_ptr lhs = test::random<f32>(n_elements, test::Randomizer<f32>(-10, 10));
    auto lhs_span = SpanContiguous<const f32, 1>(lhs.get(), n_elements);

    // Write something to a tmp file.
    auto file = BinaryFile({.write=true}, {.default_to_cwd=true});
    auto filename = file.path();
    file.write(lhs_span);
    file.close();

    // Read from the file.
    const std::unique_ptr rhs = test::zero<f32>(n_elements);
    auto rhs_span = SpanContiguous<f32, 1>(rhs.get(), n_elements);
    file.open(filename, {.read=true}, {.close_delete=true});
    file.read(rhs_span);
    file.close();
    REQUIRE(not noa::io::is_file(filename));

    REQUIRE(test::allclose_abs(lhs_span, rhs_span));
}
