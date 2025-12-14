#include <iostream>
#include <noa/Array.hpp>
#include <noa/Signal.hpp>
#include <noa/Geometry.hpp>
#include <noa/IO.hpp>

#include "Catch.hpp"
#include "Utils.hpp"
#include "noa/core/utils/Zip.hpp"
#include "noa/unified/fft/Factory.hpp"

namespace {
    using namespace noa::types;
    namespace ns = noa::signal;
    namespace ng = noa::geometry;
    namespace nf = noa::fft;
}

TEST_CASE("test::Array BDHW", "[.]") {
    auto xmap = noa::read_image<f32>("~/Tmp/noa/test_peak/xmap.mrc").data;


    constexpr nf::Layout LAYOUT = "h";
    constexpr nf::Layout LAYOUT2 = "h";
    constexpr bool A = LAYOUT == nf::Layout::H2H;
    constexpr auto r = LAYOUT.is_any("h");
}
