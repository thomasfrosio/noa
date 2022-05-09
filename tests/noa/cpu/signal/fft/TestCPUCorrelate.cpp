#include <noa/cpu/fft/Transforms.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/signal/fft/Correlate.h>
#include <noa/cpu/signal/Shape.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

TEST_CASE("cpu::signal::fft::xmap()", "[noa][cpu]") {
    using namespace ::noa;

    const size4_t shape{1, 1, 256, 256};
    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();

    const size4_t shape_fft = shape.fft();
    const size4_t stride_fft = shape_fft.stride();

    cpu::memory::PtrHost<float> target{elements};
    cpu::memory::PtrHost<float> data{elements};

    cpu::Stream stream;

    test::Randomizer<float> randomizer{-30, 30};
    const float2_t target_center{128, 128};
    const float2_t radius{25, 25};
    float2_t data_center = target_center;
    float2_t shift0{randomizer.get(), randomizer.get()};
    data_center += shift0;

    cpu::signal::rectangle<false, float>(nullptr, {}, target.share(), stride, shape, target_center, radius, 7, stream);
    cpu::signal::rectangle<false, float>(nullptr, {}, data.share(), stride, shape, data_center, radius, 7, stream);

    cpu::memory::PtrHost<cfloat_t> target_fft{shape.fft().elements()};
    cpu::memory::PtrHost<cfloat_t> data_fft{shape.fft().elements()};

    cpu::fft::r2c(target.share(), stride, target_fft.share(), stride_fft,
                  shape, cpu::fft::ESTIMATE, fft::NORM_FORWARD, stream);
    cpu::fft::r2c(data.share(), stride, data_fft.share(), stride_fft,
                  shape, cpu::fft::ESTIMATE, fft::NORM_FORWARD, stream);

    cpu::signal::fft::xmap<fft::H2F>(target_fft.share(), stride_fft,
                                     data_fft.share(), stride_fft,
                                     data.share(), stride,
                                     shape, fft::NORM_FORWARD, stream);

    float2_t shift1 = cpu::signal::fft::xpeak2D<fft::F2F>(data.share(), stride, shape, stream);
    shift1 -= target_center;

    INFO(target_center);
    INFO(shift0);
    INFO(-shift1);
    REQUIRE_THAT(shift1[0], Catch::WithinAbs(static_cast<double>(-shift0[0]), 5e-3));
    REQUIRE_THAT(shift1[1], Catch::WithinAbs(static_cast<double>(-shift0[1]), 5e-3));
}
