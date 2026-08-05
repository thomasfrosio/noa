#include <noa/Runtime.hpp>
#include <noa/xform/Texture.hpp>
#include <noa/xform/Transform.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

TEST_CASE("xform::Texture - update") {
    using namespace noa::types;
    namespace nx = noa::xform;

    auto devices = std::vector<Device>{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    for (auto device: devices) {
        auto array = noa::random<f32, 6>(noa::Uniform{-1.f, 1.f}, {5, 4, 3, 2, 64, 64}, {.device = device, .allocator = Allocator::MANAGED});
        auto t1 = nx::Texture2D<f32, 6>(array, device, nx::Interp::CUBIC);

        auto t2 = nx::Texture2D<f32, 6>(array.shape(), device, nx::Interp::CUBIC);
        t2.update(array);

        static_assert(noa::traits::texture<decltype(t1)>);
        static_assert(noa::traits::array_size_v<decltype(t1)> == 6);
        [[maybe_unused]] nx::Texture2D<f32, 6, ArrayOwnership::VIEW> t3 = t1.view();
        [[maybe_unused]] nx::Texture2D<const f32, 6> t4 = t1;

        auto o1 = noa::empty_like(array);
        nx::transform_2d(t1, o1, nx::rotate_z(noa::rad2deg(45.)));

        auto o2 = noa::empty_like(array);
        nx::transform_2d(t2, o2, nx::rotate_z(noa::rad2deg(45.)));

        REQUIRE(test::allclose_abs(o1.as_1d(), o2.as_1d(), 1e-5));
    }
}
