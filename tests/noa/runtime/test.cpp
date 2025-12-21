#include <iostream>

#include <noa/Runtime.hpp>
#include <noa/Signal.hpp>
#include <noa/Geometry.hpp>
#include <noa/IO.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

namespace {
    using namespace noa::types;
    namespace ns = noa::signal;
    namespace ng = noa::geometry;
    namespace nf = noa::fft;
}

TEST_CASE("test::Array BDHW", "[.]") {
    auto xmap = noa::read_image<f32>("~/Tmp/noa/test_peak/xmap.mrc").data;
    auto shape = Shape4{1, 2, 3, 4};

    noa::
}
