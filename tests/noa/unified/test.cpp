#include <iostream>
#include <noa/unified/Array.hpp>
#include <noa/core/Enums.hpp>
#include <noa/Geometry.hpp>
#include <noa/IO.hpp>
#include <noa/Array.hpp>

#include <catch2/catch.hpp>

#include "Utils.hpp"

using namespace noa::types;

TEST_CASE("test") {
    // noa::Stream::current("cpu").set_thread_limit(1);
    //
    // const auto input = noa::io::read_data<f32>(test::NOA_DATA_PATH / "geometry" / "transform2D_cubic_input.mrc");
    //
    // const auto center = Vec{239., 240.};
    // const auto scale = Vec{0.9, 1.2};
    // const auto cvalue = 1.3f;
    // const auto rotate = noa::deg2rad(-45.);
    // const auto inv_matrix = noa::geometry::affine2truncated(noa::inverse(
    //     noa::geometry::translate(center) *
    //     noa::geometry::linear2affine(noa::geometry::rotate(rotate)) *
    //     noa::geometry::linear2affine(noa::geometry::scale(scale)) *
    //     noa::geometry::translate(-center)
    // ).as<f32>());
    //
    // const auto output = noa::like(input);
    // noa::geometry::transform_2d(input, output, inv_matrix, {
    //     .interp = noa::Interp::LANCZOS8,
    //     .border = noa::Border::VALUE,
    //     .cvalue = cvalue,
    // });
    // noa::io::write(output, test::NOA_DATA_PATH / "geometry" / "test1.mrc");
}
