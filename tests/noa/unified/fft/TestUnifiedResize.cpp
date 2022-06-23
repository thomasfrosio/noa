#include <noa/common/io/ImageFile.h>
#include <noa/unified/FFT.h>

#include <catch2/catch.hpp>
#include "Helpers.h"
using namespace ::noa;

TEST_CASE("unified::fft::resize, crop", "[noa][unified]") {
    using real_t = float;
    using complex_t = Complex<real_t>;

    const Device cpu{};
    StreamGuard stream{cpu};
    ArrayOption options{cpu, Allocator::DEFAULT};

    io::ImageFile file{test::NOA_DATA_PATH / "signal" / "fft" / "tilt1_slice21.mrc", io::READ};
    const size4_t shape = file.shape();

    Array<real_t> src{shape, options};
    Array<complex_t> src_fft{shape.fft(), options};
    file.readAll(src.get());

    fft::r2c(src, src_fft);
    src = nullptr;

    const size4_t shape_bin2{shape[0], shape[1], shape[2] / 7, shape[3] / 7};
    Array<complex_t> dst_fft{shape_bin2.fft(), options};
    fft::resize<fft::H2H>(src_fft, shape, dst_fft, shape_bin2);
    src_fft = nullptr;

    Array<real_t> dst{shape_bin2, options};
    fft::c2r(dst_fft, dst);
    dst_fft = nullptr;

    file.open(test::NOA_DATA_PATH / "signal" / "fft" / "tilt1_slice21_cropped2.mrc", io::WRITE);
    file.shape(dst.shape());
    file.writeAll(dst.eval().get());
}

TEST_CASE("unified::fft::resize, crop entire stack", "[noa][unified]") {
    using real_t = float;
    using complex_t = Complex<real_t>;

    const Device cpu{};
    StreamGuard stream{cpu};
    ArrayOption options{cpu, Allocator::DEFAULT};

    io::ImageFile file{test::NOA_DATA_PATH / "signal" / "fft" / "tilt1.mrc", io::READ};
    const size4_t shape = file.shape();

    Array<real_t> src{shape, options};
    Array<complex_t> src_fft{shape.fft(), options};
    file.readAll(src.get());

    fft::r2c(src, src_fft);
    src = nullptr;

    const size4_t shape_bin2{shape[0], shape[1], shape[2] / 7, shape[3] / 7};
    Array<complex_t> dst_fft{shape_bin2.fft(), options};
    fft::resize<fft::H2H>(src_fft, shape, dst_fft, shape_bin2);
    src_fft = nullptr;

    Array<real_t> dst{shape_bin2, options};
    fft::c2r(dst_fft, dst);
    dst_fft = nullptr;

    file.open(test::NOA_DATA_PATH / "signal" / "fft" / "tilt1_cropped2.mrc", io::WRITE);
    file.shape(dst.shape());
    file.writeAll(dst.eval().get());
}