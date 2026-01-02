#include <noa/Runtime.hpp>
#include <noa/Xform.hpp>
#include <noa/FFT.hpp>
#include "Catch.hpp"

namespace nx = noa::xform;

TEST_CASE("xxtest", "[.]") {
    // Create an array of uninitialized values of two 1024x1024 images on the GPU.
    auto images = noa::zeros<float>({1, 1, 512, 512}, {.device = "cpu", .allocator = "managed"});

    // auto matmul = noa::like(images);
    // noa::matmul(images, images, matmul);
    auto rfft = noa::fft::r2c(images);
    auto device = images.device();
    images = {};
    rfft = {};

    device.reset();
    images = noa::zeros<float>({1, 1, 512, 512}, {.device = device, .allocator = "managed"});
    // noa::matmul(images, images, matmul);
    rfft = noa::fft::r2c(images);
    fmt::println("dot product\n");
}
