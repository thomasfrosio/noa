#include <noa/cpu/fft/Transforms.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/signal/fft/Correlate.h>
#include <noa/cpu/signal/Shape.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

TEMPLATE_TEST_CASE("cpu::signal::fft::{xmap|xpeak2D}()", "[noa][cpu]", float, double) {
    using namespace ::noa;
    using real_t = TestType;
    using complex_t = Complex<real_t>;

    size4_t shape = test::getRandomShape(2);
    shape[2] += 200;
    shape[3] += 200;
    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();
    const size4_t shape_fft = shape.fft();
    const size4_t stride_fft = shape_fft.stride();

    const bool normalize = GENERATE(true, false);
    const fft::Norm norm_mode = GENERATE(fft::NORM_FORWARD, fft::NORM_BACKWARD, fft::NORM_ORTHO);
    const cpu::fft::Flag fft_flag = cpu::fft::ESTIMATE;

    test::Randomizer<float> randomizer{-30, 30};
    const float2_t lhs_center{shape[2] / 2, shape[3] / 2};
    const float2_t radius{25, 25};
    const float taper = 7;
    float2_t rhs_center = lhs_center;
    float2_t shift0{randomizer.get(), randomizer.get()};
    rhs_center += shift0;

    cpu::Stream stream{cpu::Stream::DEFAULT};
    cpu::memory::PtrHost<real_t> lhs{elements};
    cpu::memory::PtrHost<real_t> rhs{elements};

    cpu::signal::rectangle<false, real_t>(nullptr, {}, lhs.share(), stride, shape, lhs_center, radius, taper, stream);
    cpu::signal::rectangle<false, real_t>(nullptr, {}, rhs.share(), stride, shape, rhs_center, radius, taper, stream);

    cpu::memory::PtrHost<complex_t> lhs_fft{shape.fft().elements()};
    cpu::memory::PtrHost<complex_t> rhs_fft{shape.fft().elements()};

    cpu::fft::r2c(lhs.share(), stride, lhs_fft.share(), stride_fft, shape, fft_flag, norm_mode, stream);
    cpu::fft::r2c(rhs.share(), stride, rhs_fft.share(), stride_fft, shape, fft_flag, norm_mode, stream);

    cpu::signal::fft::xmap<fft::H2F>(lhs_fft.share(), stride_fft,
                                     rhs_fft.share(), stride_fft,
                                     rhs.share(), stride,
                                     shape, normalize, norm_mode, stream);

    float2_t shift1 = cpu::signal::fft::xpeak2D<fft::F2F>(rhs.share(), stride, shape, stream);
    shift1 -= lhs_center;

    INFO(shift0);
    INFO(-shift1);
    REQUIRE_THAT(shift1[0], Catch::WithinAbs(static_cast<double>(-shift0[0]), 5e-2));
    REQUIRE_THAT(shift1[1], Catch::WithinAbs(static_cast<double>(-shift0[1]), 5e-2));
}

TEMPLATE_TEST_CASE("cpu::signal::fft::{xmap|xpeak2D}(), batched", "[noa][cpu]", float, double) {
    using namespace ::noa;
    using real_t = TestType;
    using complex_t = Complex<real_t>;

    size4_t shape = test::getRandomShapeBatched(2);
    shape[2] += 200;
    shape[3] += 200;
    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();
    const size4_t shape_fft = shape.fft();
    const size4_t stride_fft = shape_fft.stride();

    const bool normalize = false;
    const fft::Norm norm_mode = fft::NORM_FORWARD;
    const cpu::fft::Flag fft_flag = cpu::fft::ESTIMATE;

    test::Randomizer<float> randomizer{-30, 30};
    const float2_t lhs_center{shape[2] / 2, shape[3] / 2};
    const float2_t radius{25, 25};
    const float taper = 7;
    float2_t rhs_center = lhs_center;
    float2_t shift0{randomizer.get(), randomizer.get()};
    rhs_center += shift0;

    cpu::Stream stream{cpu::Stream::DEFAULT};
    cpu::memory::PtrHost<real_t> lhs{elements};
    cpu::memory::PtrHost<real_t> rhs{elements};

    cpu::signal::rectangle<false, real_t>(nullptr, {}, lhs.share(), stride, shape, lhs_center, radius, taper, stream);
    cpu::signal::rectangle<false, real_t>(nullptr, {}, rhs.share(), stride, shape, rhs_center, radius, taper, stream);

    cpu::memory::PtrHost<complex_t> lhs_fft{shape.fft().elements()};
    cpu::memory::PtrHost<complex_t> rhs_fft{shape.fft().elements()};

    cpu::fft::r2c(lhs.share(), stride, lhs_fft.share(), stride_fft, shape, fft_flag, norm_mode, stream);
    cpu::fft::r2c(rhs.share(), stride, rhs_fft.share(), stride_fft, shape, fft_flag, norm_mode, stream);

    cpu::signal::fft::xmap<fft::H2FC>(lhs_fft.share(), stride_fft,
                                      rhs_fft.share(), stride_fft,
                                      rhs.share(), stride,
                                      shape, normalize, norm_mode, stream);

    cpu::memory::PtrHost<float2_t> shift1{shape[0]};
    cpu::signal::fft::xpeak2D<fft::FC2FC>(rhs.share(), stride, shape, shift1.share(), stream);
    stream.synchronize();

    for (auto shift: shift1) {
        shift -= lhs_center;
        INFO(shape[0]);
        INFO(shift0);
        INFO(-shift);
        REQUIRE_THAT(shift[0], Catch::WithinAbs(static_cast<double>(-shift0[0]), 5e-2));
        REQUIRE_THAT(shift[1], Catch::WithinAbs(static_cast<double>(-shift0[1]), 5e-2));
    }
}

TEMPLATE_TEST_CASE("cpu::signal::fft::{xmap|xpeak3D}()", "[noa][cpu]", float, double) {
    using namespace ::noa;
    using real_t = TestType;
    using complex_t = Complex<real_t>;

    size4_t shape = test::getRandomShape(3);
    shape[1] += 100;
    shape[2] += 100;
    shape[3] += 100;
    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();
    const size4_t shape_fft = shape.fft();
    const size4_t stride_fft = shape_fft.stride();

    const bool normalize = GENERATE(true, false);
    const fft::Norm norm_mode = GENERATE(fft::NORM_FORWARD, fft::NORM_BACKWARD, fft::NORM_ORTHO);
    const cpu::fft::Flag fft_flag = cpu::fft::ESTIMATE;

    test::Randomizer<float> randomizer{-20, 20};
    const float3_t lhs_center{shape[1] / 2, shape[2] / 2, shape[3] / 2};
    const float3_t radius{20, 20, 20};
    const float taper = 7;
    float3_t rhs_center = lhs_center;
    float3_t shift0{randomizer.get(), randomizer.get(), randomizer.get()};
    rhs_center += shift0;

    cpu::Stream stream{cpu::Stream::DEFAULT};
    cpu::memory::PtrHost<real_t> lhs{elements};
    cpu::memory::PtrHost<real_t> rhs{elements};

    cpu::signal::rectangle<false, real_t>(nullptr, {}, lhs.share(), stride, shape, lhs_center, radius, taper, stream);
    cpu::signal::rectangle<false, real_t>(nullptr, {}, rhs.share(), stride, shape, rhs_center, radius, taper, stream);

    cpu::memory::PtrHost<complex_t> lhs_fft{shape.fft().elements()};
    cpu::memory::PtrHost<complex_t> rhs_fft{shape.fft().elements()};

    cpu::fft::r2c(lhs.share(), stride, lhs_fft.share(), stride_fft, shape, fft_flag, norm_mode, stream);
    cpu::fft::r2c(rhs.share(), stride, rhs_fft.share(), stride_fft, shape, fft_flag, norm_mode, stream);

    cpu::signal::fft::xmap<fft::H2F>(lhs_fft.share(), stride_fft,
                                     rhs_fft.share(), stride_fft,
                                     rhs.share(), stride,
                                     shape, normalize, norm_mode, stream);

    float3_t shift1 = cpu::signal::fft::xpeak3D<fft::F2F>(rhs.share(), stride, shape, stream);
    shift1 -= lhs_center;

    INFO(shift0);
    INFO(-shift1);
    REQUIRE_THAT(shift1[0], Catch::WithinAbs(static_cast<double>(-shift0[0]), 5e-2));
    REQUIRE_THAT(shift1[1], Catch::WithinAbs(static_cast<double>(-shift0[1]), 5e-2));
    REQUIRE_THAT(shift1[2], Catch::WithinAbs(static_cast<double>(-shift0[2]), 5e-2));
}

TEMPLATE_TEST_CASE("cpu::signal::fft::{xmap|xpeak3D}(), batched", "[noa][cpu]", float, double) {
    using namespace ::noa;
    using real_t = TestType;
    using complex_t = Complex<real_t>;

    size4_t shape = test::getRandomShapeBatched(3);
    shape[1] += 100;
    shape[2] += 100;
    shape[3] += 100;
    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();
    const size4_t shape_fft = shape.fft();
    const size4_t stride_fft = shape_fft.stride();

    const bool normalize = false;
    const fft::Norm norm_mode = fft::NORM_FORWARD;
    const cpu::fft::Flag fft_flag = cpu::fft::ESTIMATE;

    test::Randomizer<float> randomizer{-20, 20};
    const float3_t lhs_center{shape[1] / 2, shape[2] / 2, shape[3] / 2};
    const float3_t radius{20, 20, 20};
    const float taper = 7;
    float3_t rhs_center = lhs_center;
    float3_t shift0{randomizer.get(), randomizer.get(), randomizer.get()};
    rhs_center += shift0;

    cpu::Stream stream{cpu::Stream::DEFAULT};
    cpu::memory::PtrHost<real_t> lhs{elements};
    cpu::memory::PtrHost<real_t> rhs{elements};

    cpu::signal::rectangle<false, real_t>(nullptr, {}, lhs.share(), stride, shape, lhs_center, radius, taper, stream);
    cpu::signal::rectangle<false, real_t>(nullptr, {}, rhs.share(), stride, shape, rhs_center, radius, taper, stream);

    cpu::memory::PtrHost<complex_t> lhs_fft{shape.fft().elements()};
    cpu::memory::PtrHost<complex_t> rhs_fft{shape.fft().elements()};

    cpu::fft::r2c(lhs.share(), stride, lhs_fft.share(), stride_fft, shape, fft_flag, norm_mode, stream);
    cpu::fft::r2c(rhs.share(), stride, rhs_fft.share(), stride_fft, shape, fft_flag, norm_mode, stream);

    cpu::signal::fft::xmap<fft::H2FC>(lhs_fft.share(), stride_fft,
                                      rhs_fft.share(), stride_fft,
                                      rhs.share(), stride,
                                      shape, normalize, norm_mode, stream);

    cpu::memory::PtrHost<float3_t> shift1{shape[0]};
    cpu::signal::fft::xpeak3D<fft::FC2FC>(rhs.share(), stride, shape, shift1.share(), stream);
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

//TEST_CASE("cpu::signal::fft, register translation", "[noa][cpu]") {
//    using namespace ::noa;
//    io::ImageFile file{test::NOA_DATA_PATH / "signal" / "fft" / "tilt1.mrc", io::READ};
//    const size4_t shape_stack = file.shape();
//    const size4_t shape{1, 1, shape_stack[2], shape_stack[3]};
//    const size4_t stride = shape.stride();
//    const size_t elements = stride[0];
//
//    const size4_t shape_fft = shape.fft();
//    const size4_t stride_fft = shape_fft.stride();
//
//    const float2_t center = {shape[2] / 2, shape[3] / 2};
//
//    cpu::memory::PtrHost<float> target{elements};
//    cpu::memory::PtrHost<float> data{elements};
//    file.readSlice(target.get(), 21, 22);
//    file.readSlice(data.get(), 22, 23);
//
//    cpu::Stream stream{cpu::Stream::DEFAULT};
//
//    cpu::memory::PtrHost<cfloat_t> target_fft{shape.fft().elements()};
//    cpu::memory::PtrHost<cfloat_t> data_fft{shape.fft().elements()};
//
//    cpu::fft::r2c(target.share(), stride, target_fft.share(), stride_fft,
//                  shape, cpu::fft::ESTIMATE, fft::NORM_FORWARD, stream);
//    cpu::fft::r2c(data.share(), stride, data_fft.share(), stride_fft,
//                  shape, cpu::fft::ESTIMATE, fft::NORM_FORWARD, stream);
//
//    cpu::signal::fft::standardize<fft::H2H>(target_fft.share(), stride_fft,
//                                            target_fft.share(), stride_fft,
//                                            shape, fft::NORM_FORWARD, stream);
//    cpu::signal::fft::standardize<fft::H2H>(data_fft.share(), stride_fft,
//                                            data_fft.share(), stride_fft,
//                                            shape, fft::NORM_FORWARD, stream);
//    cpu::signal::fft::xmap<fft::H2F>(target_fft.share(), stride_fft,
//                                     data_fft.share(), stride_fft,
//                                     data.share(), stride,
//                                     shape, true, fft::NORM_FORWARD, stream);
//
//    float2_t shift1 = cpu::signal::fft::xpeak2D<fft::F2F>(data.share(), stride, shape, stream);
//    shift1 -= center;
//
//    cpu::fft::c2r(target_fft.share(), stride_fft, target.share(), stride,
//                  shape, cpu::fft::ESTIMATE, fft::NORM_FORWARD, stream);
//
//    cpu::geometry::shift2D(target.share(), stride, shape, data.share(), stride, shape, -shift1,
//                           INTERP_LINEAR, BORDER_ZERO, 0.f, stream);
//    cpu::signal::rectangle(data.share(), stride, data.share(), stride, shape,
//                           center - shift1, center - 50, 30, stream);
//
//    stream.synchronize();
//    file.open(test::NOA_DATA_PATH / "signal" / "fft" / "tilt1_shifted.mrc", io::WRITE);
//    file.shape(shape);
//    file.writeAll(data.get());
//    file.close();
//}
