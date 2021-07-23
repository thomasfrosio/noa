#include <noa/common/Math.h>
#include <noa/common/files/MRCFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Set.h>

#include <noa/gpu/cuda/math/Generics.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/transform/Rotate.h>
#include <noa/gpu/cuda/fourier/Transforms.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cuda::transform::rotate() - python", "[noa][cuda][transform]") {
    size2_t shape(513, 513);
    size_t elements = getElements(shape);
    memory::PtrHost<float> image(elements);
    memory::set(image.begin(), image.end(), 0.f);
    image[256 * shape.x + 256] = 2;
    image[256 * shape.x + 258] = 1;

    float rotation = math::toRad(90.f);
    float2_t rotation_center = float2_t(shape / 2UL);

    cuda::Stream stream;
    cuda::memory::PtrDevice<float> d_image(elements);
    cuda::memory::PtrDevice<float> d_transformed_image(elements);
    cuda::memory::copy(image.get(), d_image.get(), elements, stream);

    cuda::transform::rotate2D(d_image.get(), shape.x, d_transformed_image.get(), shape.x, shape,
                              rotation, rotation_center, INTERP_LINEAR, BORDER_ZERO, stream);
    cuda::memory::copy(d_transformed_image.get(), image.get(), elements, stream);
    stream.synchronize();

    MRCFile file(test::PATH_TEST_DATA / "transform" / "image2D_test_again_python.mrc", io::WRITE);
    file.setShape(size3_t(shape.x, shape.y, 1));
    file.writeAll(image.get());
}

TEST_CASE("cuda::transform::rotate()", "[noa][cuda][transform]") {
    MRCFile file(test::PATH_TEST_DATA / "transform" / "image2D.mrc", io::READ);
    size3_t shape = file.getShape();
    size_t elements = getElements(shape);
    memory::PtrHost<float> image(elements);

    file.readAll(image.get());

    float rotation = math::toRad(90.f);
    float2_t rotation_center = float2_t(128, 128);

    cuda::Stream stream;
    cuda::memory::PtrDevice<float> d_image(elements);
    cuda::memory::PtrDevice<float> d_transformed_image(elements);
    cuda::memory::copy(image.get(), d_image.get(), elements, stream);

    cuda::transform::rotate2D(d_image.get(), shape.x, d_transformed_image.get(), shape.x, size2_t(shape.x, shape.y),
                              rotation, rotation_center, INTERP_LINEAR, BORDER_MIRROR, stream);
    cuda::memory::copy(d_transformed_image.get(), image.get(), elements, stream);
    stream.synchronize();

    file.open(test::PATH_TEST_DATA / "transform" / "image2D_transformed_linear.mrc", io::WRITE);
    file.writeAll(image.get());
}

TEST_CASE("cuda::transform::rotate(), fft", "[noa][cuda][transform]") {
    size3_t shape(256, 256, 1);
    size3_t shape_fft(getShapeFFT(shape));
    size_t elements_fft = getElements(shape_fft);
    memory::PtrHost<float> image(elements_fft);
    for (size_t idx = 0; idx < elements_fft; ++idx)
        image[idx] = static_cast<float>(idx);

    MRCFile file(test::PATH_TEST_DATA / "transform" / "image2D_fft_in.mrc", io::WRITE);
    file.setShape(shape_fft);
    file.writeAll(image.get());

    float rotation = math::toRad(0.f);

    cuda::Stream stream;
    cuda::memory::PtrDevice<float> d_image(elements_fft);
    cuda::memory::PtrDevice<float> d_transformed_image(elements_fft);
    cuda::memory::copy(image.get(), d_image.get(), elements_fft, stream);

    cuda::transform::rotate2DFT(d_image.get(), shape_fft.x,
                                d_transformed_image.get(), shape_fft.x,
                                shape.x, rotation, 0.5f, stream);
    cuda::memory::copy(d_transformed_image.get(), image.get(), elements_fft, stream);
    stream.synchronize();

    file.open(test::PATH_TEST_DATA / "transform" / "image2D_fft_out.mrc", io::WRITE);
    file.writeAll(image.get());
}
