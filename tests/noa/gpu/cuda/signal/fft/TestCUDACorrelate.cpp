#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/fft/Transforms.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/signal/fft/Correlate.h>
#include <noa/gpu/cuda/signal/Shape.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

TEMPLATE_TEST_CASE("cuda::signal::fft::{xmap|xpeak2D}()", "[noa][cuda]", float, double) {
    using namespace ::noa;
    using real_t = TestType;
    using complex_t = Complex<real_t>;

    size4_t shape = test::getRandomShape(2);
    shape[2] += 200;
    shape[3] += 200;
    const size4_t strides = shape.strides();
    const size_t elements = shape.elements();
    const size4_t shape_fft = shape.fft();
    const size4_t strides_fft = shape_fft.strides();

    const bool normalize = GENERATE(true, false);
    const fft::Norm norm_mode = GENERATE(fft::NORM_FORWARD, fft::NORM_BACKWARD, fft::NORM_ORTHO);

    test::Randomizer<float> randomizer{-30, 30};
    const float2_t lhs_center{shape[2] / 2, shape[3] / 2};
    const float2_t radius{25, 25};
    const float taper = 7;
    float2_t rhs_center = lhs_center;
    float2_t shift0{randomizer.get(), randomizer.get()};
    rhs_center += shift0;

    cuda::Stream stream(cuda::Stream::DEFAULT);
    cuda::memory::PtrDevice<real_t> lhs(elements, stream);
    cuda::memory::PtrDevice<real_t> rhs(elements, stream);

    cuda::signal::rectangle<real_t>(nullptr, {}, lhs.share(), strides, shape, lhs_center, radius, taper,
                                    float22_t{}, math::multiply_t{}, real_t{1}, false, stream);
    cuda::signal::rectangle<real_t>(nullptr, {}, rhs.share(), strides, shape, rhs_center, radius, taper,
                                    float22_t{}, math::multiply_t{}, real_t{1}, false, stream);

    cuda::memory::PtrDevice<complex_t> lhs_fft(shape.fft().elements(), stream);
    cuda::memory::PtrDevice<complex_t> rhs_fft(shape.fft().elements(), stream);

    cuda::fft::r2c(lhs.share(), strides, lhs_fft.share(), strides_fft, shape, norm_mode, stream);
    cuda::fft::r2c(rhs.share(), strides, rhs_fft.share(), strides_fft, shape, norm_mode, stream);

    cuda::signal::fft::xmap<fft::H2F>(lhs_fft.share(), strides_fft,
                                      rhs_fft.share(), strides_fft,
                                      rhs.share(), strides,
                                      shape, normalize, norm_mode, stream);

    float2_t shift1 = cuda::signal::fft::xpeak2D<fft::F2F>(rhs.share(), strides, shape, {}, {1, 1}, stream);
    shift1 -= lhs_center;

    INFO(shift0);
    INFO(-shift1);
    REQUIRE_THAT(shift1[0], Catch::WithinAbs(static_cast<double>(-shift0[0]), 5e-2));
    REQUIRE_THAT(shift1[1], Catch::WithinAbs(static_cast<double>(-shift0[1]), 5e-2));
}

TEMPLATE_TEST_CASE("cuda::signal::fft::{xmap|xpeak2D}(), batched", "[noa][cuda]", float, double) {
    using namespace ::noa;
    using real_t = TestType;
    using complex_t = Complex<real_t>;

    size4_t shape = test::getRandomShapeBatched(2);
    shape[2] += 200;
    shape[3] += 200;
    const size4_t strides = shape.strides();
    const size_t elements = shape.elements();
    const size4_t shape_fft = shape.fft();
    const size4_t strides_fft = shape_fft.strides();

    const bool normalize = false;
    const fft::Norm norm_mode = fft::NORM_FORWARD;

    test::Randomizer<float> randomizer{-30, 30};
    const float2_t lhs_center{shape[2] / 2, shape[3] / 2};
    const float2_t radius{25, 25};
    const float taper = 7;
    float2_t rhs_center = lhs_center;
    float2_t shift0{randomizer.get(), randomizer.get()};
    rhs_center += shift0;

    cuda::Stream stream(cuda::Stream::DEFAULT);
    cuda::memory::PtrDevice<real_t> lhs(elements, stream);
    cuda::memory::PtrDevice<real_t> rhs(elements, stream);

    cuda::signal::rectangle<real_t>(nullptr, {}, lhs.share(), strides, shape, lhs_center, radius, taper,
                                    float22_t{}, math::multiply_t{}, real_t{1}, false, stream);
    cuda::signal::rectangle<real_t>(nullptr, {}, rhs.share(), strides, shape, rhs_center, radius, taper,
                                    float22_t{}, math::multiply_t{}, real_t{1}, false, stream);

    cuda::memory::PtrDevice<complex_t> lhs_fft(shape.fft().elements(), stream);
    cuda::memory::PtrDevice<complex_t> rhs_fft(shape.fft().elements(), stream);

    cuda::fft::r2c(lhs.share(), strides, lhs_fft.share(), strides_fft, shape, norm_mode, stream);
    cuda::fft::r2c(rhs.share(), strides, rhs_fft.share(), strides_fft, shape, norm_mode, stream);

    cuda::signal::fft::xmap<fft::H2FC>(lhs_fft.share(), strides_fft,
                                       rhs_fft.share(), strides_fft,
                                       rhs.share(), strides,
                                       shape, normalize, norm_mode, stream);

    cpu::memory::PtrHost<float2_t> shift1{shape[0]};
    cuda::signal::fft::xpeak2D<fft::FC2FC>(rhs.share(), strides, shape, shift1.share(), {}, {1, 1}, stream);
    stream.synchronize();

    for (size_t i = 0 ; i < shape[0]; ++i) {
        float2_t shift = shift1[i];
        shift -= lhs_center;
        INFO(i);
        INFO(shift0);
        INFO(-shift);
        REQUIRE_THAT(shift[0], Catch::WithinAbs(static_cast<double>(-shift0[0]), 5e-2));
        REQUIRE_THAT(shift[1], Catch::WithinAbs(static_cast<double>(-shift0[1]), 5e-2));
    }
}

TEMPLATE_TEST_CASE("cuda::signal::fft::{xmap|xpeak3D}()", "[noa][cuda]", float, double) {
    using namespace ::noa;
    using real_t = TestType;
    using complex_t = Complex<real_t>;

    size4_t shape = test::getRandomShape(3);
    shape[1] += 100;
    shape[2] += 100;
    shape[3] += 100;
    const size4_t strides = shape.strides();
    const size_t elements = shape.elements();
    const size4_t shape_fft = shape.fft();
    const size4_t strides_fft = shape_fft.strides();

    const bool normalize = GENERATE(true, false);
    const fft::Norm norm_mode = GENERATE(fft::NORM_FORWARD, fft::NORM_BACKWARD, fft::NORM_ORTHO);

    test::Randomizer<float> randomizer{-20, 20};
    const float3_t lhs_center{shape[1] / 2, shape[2] / 2, shape[3] / 2};
    const float3_t radius{20, 20, 20};
    const float taper = 7;
    float3_t rhs_center = lhs_center;
    float3_t shift0{randomizer.get(), randomizer.get(), randomizer.get()};
    rhs_center += shift0;

    cuda::Stream stream(cuda::Stream::DEFAULT);
    cuda::memory::PtrDevice<real_t> lhs(elements, stream);
    cuda::memory::PtrDevice<real_t> rhs(elements, stream);

    cuda::signal::rectangle<real_t>(nullptr, {}, lhs.share(), strides, shape, lhs_center, radius, taper,
                                    float33_t{}, math::multiply_t{}, real_t{1}, false, stream);
    cuda::signal::rectangle<real_t>(nullptr, {}, rhs.share(), strides, shape, rhs_center, radius, taper,
                                    float33_t{}, math::multiply_t{}, real_t{1}, false, stream);

    cuda::memory::PtrDevice<complex_t> lhs_fft(shape.fft().elements(), stream);
    cuda::memory::PtrDevice<complex_t> rhs_fft(shape.fft().elements(), stream);

    cuda::fft::r2c(lhs.share(), strides, lhs_fft.share(), strides_fft, shape, norm_mode, stream);
    cuda::fft::r2c(rhs.share(), strides, rhs_fft.share(), strides_fft, shape, norm_mode, stream);

    cuda::signal::fft::xmap<fft::H2F>(lhs_fft.share(), strides_fft,
                                      rhs_fft.share(), strides_fft,
                                      rhs.share(), strides,
                                      shape, normalize, norm_mode, stream);

    float3_t shift1 = cuda::signal::fft::xpeak3D<fft::F2F>(rhs.share(), strides, shape, {}, {1, 1, 1}, stream);
    shift1 -= lhs_center;

    INFO(shift0);
    INFO(-shift1);
    REQUIRE_THAT(shift1[0], Catch::WithinAbs(static_cast<double>(-shift0[0]), 5e-2));
    REQUIRE_THAT(shift1[1], Catch::WithinAbs(static_cast<double>(-shift0[1]), 5e-2));
    REQUIRE_THAT(shift1[2], Catch::WithinAbs(static_cast<double>(-shift0[2]), 5e-2));
}

TEMPLATE_TEST_CASE("cuda::signal::fft::{xmap|xpeak3D}(), batched", "[noa][cuda]", float, double) {
    using namespace ::noa;
    using real_t = TestType;
    using complex_t = Complex<real_t>;

    size4_t shape = test::getRandomShapeBatched(3);
    shape[1] += 100;
    shape[2] += 100;
    shape[3] += 100;
    const size4_t strides = shape.strides();
    const size_t elements = shape.elements();
    const size4_t shape_fft = shape.fft();
    const size4_t strides_fft = shape_fft.strides();

    const bool normalize = false;
    const fft::Norm norm_mode = fft::NORM_FORWARD;

    test::Randomizer<float> randomizer{-20, 20};
    const float3_t lhs_center{shape[1] / 2, shape[2] / 2, shape[3] / 2};
    const float3_t radius{20, 20, 20};
    const float taper = 7;
    float3_t rhs_center = lhs_center;
    float3_t shift0{randomizer.get(), randomizer.get(), randomizer.get()};
    rhs_center += shift0;

    cuda::Stream stream(cuda::Stream::DEFAULT);
    cuda::memory::PtrDevice<real_t> lhs(elements, stream);
    cuda::memory::PtrDevice<real_t> rhs(elements, stream);

    cuda::signal::rectangle<real_t>(nullptr, {}, lhs.share(), strides, shape, lhs_center, radius, taper,
                                    float33_t{}, math::multiply_t{}, real_t{1}, false, stream);
    cuda::signal::rectangle<real_t>(nullptr, {}, rhs.share(), strides, shape, rhs_center, radius, taper,
                                    float33_t{}, math::multiply_t{}, real_t{1}, false, stream);

    cuda::memory::PtrDevice<complex_t> lhs_fft(shape.fft().elements(), stream);
    cuda::memory::PtrDevice<complex_t> rhs_fft(shape.fft().elements(), stream);

    cuda::fft::r2c(lhs.share(), strides, lhs_fft.share(), strides_fft, shape, norm_mode, stream);
    cuda::fft::r2c(rhs.share(), strides, rhs_fft.share(), strides_fft, shape, norm_mode, stream);

    cuda::signal::fft::xmap<fft::H2FC>(lhs_fft.share(), strides_fft,
                                       rhs_fft.share(), strides_fft,
                                       rhs.share(), strides,
                                       shape, normalize, norm_mode, stream);

    cpu::memory::PtrHost<float3_t> shift1{shape[0]};
    cuda::signal::fft::xpeak3D<fft::FC2FC>(rhs.share(), strides, shape, shift1.share(), {}, {1, 1, 1}, stream);
    stream.synchronize();

    for (auto shift: shift1) {
        shift -= lhs_center;
        INFO(shape[0]);
        INFO(shift0);
        INFO(-shift);
        REQUIRE_THAT(shift[0], Catch::WithinAbs(static_cast<double>(-shift0[0]), 5e-2));
        REQUIRE_THAT(shift[1], Catch::WithinAbs(static_cast<double>(-shift0[1]), 5e-2));
        REQUIRE_THAT(shift[2], Catch::WithinAbs(static_cast<double>(-shift0[2]), 5e-2));
    }
}
