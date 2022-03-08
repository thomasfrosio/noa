#include <noa/common/io//ImageFile.h>

#include <noa/cpu/fft/Resize.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/memory/PtrManaged.h>
#include <noa/gpu/cuda/fft/Resize.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::fft::resize(), non-redundant", "[noa][cuda][fft]",
                   half_t, float, cfloat_t, chalf_t, double, cdouble_t) {
    test::Randomizer<TestType> randomizer(1., 5.);
    test::Randomizer<size_t> randomizer_int(0, 32);
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShape(ndim);
    size4_t shape_padded(shape);
    if (ndim > 2) shape_padded[1] += randomizer_int.get();
    if (ndim > 1) shape_padded[2] += randomizer_int.get();
    shape_padded[3] += randomizer_int.get();

    INFO(shape);
    INFO(shape_padded);
    cuda::Stream gpu_stream(cuda::Stream::SERIAL);
    cpu::Stream cpu_stream(cpu::Stream::SERIAL);

    AND_THEN("pad then crop") {
        const size4_t stride = shape.fft().strides();
        const size4_t stride_padded = shape_padded.fft().strides();
        const size_t elements = shape.fft().elements();
        const size_t elements_padded = shape_padded.fft().elements();
        cuda::memory::PtrManaged<TestType> d_original(elements, gpu_stream);
        cuda::memory::PtrManaged<TestType> d_pad(elements_padded, gpu_stream);
        cuda::memory::PtrManaged<TestType> d_crop(elements, gpu_stream);
        cpu::memory::PtrHost<TestType> h_original(elements);
        cpu::memory::PtrHost<TestType> h_pad(elements_padded);

        test::randomize(h_original.get(), h_original.elements(), randomizer);
        cuda::memory::copy(h_original.get(), d_original.get(), h_original.elements(), gpu_stream);

        cuda::fft::resize<fft::H2H>(d_original.get(), stride, shape,
                                    d_pad.get(), stride_padded, shape_padded, gpu_stream);
        cpu::fft::resize<fft::H2H>(h_original.get(), stride, shape,
                                   h_pad.get(), stride_padded, shape_padded, cpu_stream);
        gpu_stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, d_pad.get(), h_pad.get(), elements_padded, 1e-10));

        cuda::fft::resize<fft::H2H>(d_pad.get(), stride_padded, shape_padded,
                                    d_crop.get(), stride, shape, gpu_stream);
        gpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, h_original.get(), d_crop.get(), h_original.elements(), 1e-10));
    }

    AND_THEN("padFull then cropFull") {
        const size4_t stride = shape.strides();
        const size4_t stride_padded = shape_padded.strides();
        const size_t elements = shape.elements();
        const size_t elements_padded = shape_padded.elements();
        cuda::memory::PtrManaged<TestType> d_original(elements, gpu_stream);
        cuda::memory::PtrManaged<TestType> d_pad(elements_padded, gpu_stream);
        cuda::memory::PtrManaged<TestType> d_crop(elements, gpu_stream);
        cpu::memory::PtrHost<TestType> h_original(elements);
        cpu::memory::PtrHost<TestType> h_pad(elements_padded);

        test::randomize(h_original.get(), h_original.elements(), randomizer);
        cuda::memory::copy(h_original.get(), d_original.get(), h_original.elements(), gpu_stream);

        cuda::fft::resize<fft::F2F>(d_original.get(), stride, shape,
                                    d_pad.get(), stride_padded, shape_padded, gpu_stream);
        cpu::fft::resize<fft::F2F>(h_original.get(), stride, shape,
                                   h_pad.get(), stride_padded, shape_padded, cpu_stream);
        gpu_stream.synchronize();
        cpu_stream.synchronize();


        io::ImageFile file(test::PATH_NOA_DATA / "test_resize_cuda.mrc", io::WRITE);
        file.shape(shape_padded);
        file.writeAll(d_pad.get());
        file.close();

        file.open(test::PATH_NOA_DATA / "test_resize_cpu.mrc", io::WRITE);
        file.shape(shape_padded);
        file.writeAll(h_pad.get());
        file.close();

        REQUIRE(test::Matcher(test::MATCH_ABS, d_pad.get(), h_pad.get(), elements_padded, 1e-10));

        cuda::fft::resize<fft::F2F>(d_pad.get(), stride_padded, shape_padded,
                                    d_crop.get(), stride, shape, gpu_stream);
        gpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, h_original.get(), d_crop.get(), h_original.elements(), 1e-10));
    }
}
