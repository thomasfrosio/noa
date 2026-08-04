#include <noa/Runtime.hpp>
#include <noa/xform/Texture.hpp>
#include <noa/xform/Transform.hpp>

#include "Catch.hpp"

TEST_CASE("xform::Texture", "[.]") {
    using namespace noa::types;
    namespace nx = noa::xform;

    auto devices = std::vector<Device>{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto device: devices) {
        auto array = noa::random<f32>(noa::Uniform{-1.f, 1.f}, {3, 1, 128, 128}, {.device = device});
        auto texture = nx::Texture<f32>(array, device, nx::Interp::CUBIC); // FIXME CTAD

        auto output = noa::like(array);
        nx::transform_2d(texture, output, nx::rotate_z(noa::deg2rad(45.)));
    }
}
