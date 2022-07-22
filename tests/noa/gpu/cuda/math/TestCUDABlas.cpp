#include <noa/cpu/math/Blas.h>
#include <noa/cpu/memory/PtrHost.h>

#include <noa/gpu/cuda/math/Blas.h>
#include <noa/gpu/cuda/math/Ewise.h>
#include <noa/gpu/cuda/math/Random.h>
#include <noa/gpu/cuda/math/Reduce.h>
#include <noa/gpu/cuda/memory/PtrManaged.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::math::dot()", "[noa][cuda]", float, double, cfloat_t, cdouble_t) {
    using real_t = traits::value_type_t<TestType>;
    test::Randomizer<size_t> randomizer(4096, 1048576);
    const size4_t shape{1, 1, 1, randomizer.get()};

    cuda::Stream stream;
    cuda::memory::PtrManaged<TestType> lhs(shape.elements(), stream);
    cuda::memory::PtrManaged<TestType> rhs(shape.elements(), stream);
    cuda::math::randomize(math::uniform_t{}, lhs.share(), lhs.elements(), real_t{-5}, real_t{5}, stream);
    cuda::math::randomize(math::uniform_t{}, rhs.share(), rhs.elements(), real_t{-5}, real_t{5}, stream);
    stream.synchronize();

    // Compute output:
    const TestType output0 = cuda::math::dot(lhs.share(), shape.strides(), shape,
                                             rhs.share(), shape.strides(), shape, stream);

    // Compute expected:
    cuda::math::ewise(lhs.share(), shape.strides(),
                      rhs.share(), shape.strides(),
                      lhs.share(), shape.strides(), shape, math::multiply_t{}, stream);
    const TestType output1 = cuda::math::sum(lhs.share(), shape.strides(), shape, stream);

    stream.synchronize();
    if constexpr (traits::is_complex_v<TestType>) {
        REQUIRE_THAT(output0.real, Catch::WithinRel(output1.real));
        REQUIRE_THAT(output0.imag, Catch::WithinRel(output1.imag));
    } else {
        REQUIRE_THAT(output0, Catch::WithinRel(output1));
    }
}

TEMPLATE_TEST_CASE("cuda::math::dot(), batches", "[noa][cuda]", float, double, cfloat_t, cdouble_t) {
    using real_t = traits::value_type_t<TestType>;
    const size_t batches = test::Randomizer<size_t>(1, 5).get();

    test::Randomizer<size_t> randomizer(4096, 1048576);
    const size4_t shape{batches, 1, 1, randomizer.get()};

    cuda::Stream stream;
    cuda::memory::PtrManaged<TestType> lhs(shape.elements(), stream);
    cuda::memory::PtrManaged<TestType> rhs(shape.elements(), stream);
    cuda::memory::PtrManaged<TestType> out(batches, stream);
    cuda::math::randomize(math::uniform_t{}, lhs.share(), lhs.elements(), real_t{-5}, real_t{5}, stream);
    cuda::math::randomize(math::uniform_t{}, rhs.share(), rhs.elements(), real_t{-5}, real_t{5}, stream);
    stream.synchronize();

    // Compute output:
    cuda::math::dot(lhs.share(), shape.strides(), shape,
                    rhs.share(), shape.strides(), shape,
                    out.share(), stream);

    // Compute expected:
    cuda::memory::PtrManaged<TestType> exp(batches, stream);
    const size4_t reduced_shape{batches, 1, 1, 1};
    cuda::math::ewise(lhs.share(), shape.strides(), rhs.share(), shape.strides(), lhs.share(), shape.strides(), shape, math::multiply_t{}, stream);
    cuda::math::sum(lhs.share(), shape.strides(), shape, exp.share(), reduced_shape.strides(), reduced_shape, stream);

    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, out.get(), exp.get(), exp.elements(),
                          std::is_same_v<real_t, double> ? 1e-7 : 5e-3));
}

TEMPLATE_TEST_CASE("cuda::math::matmul()", "[noa][cuda]", float, double, cfloat_t, cdouble_t) { //
    using real_t = traits::value_type_t<TestType>;
    const size_t batches = test::Randomizer<size_t>(1, 4).get();

    test::Randomizer<size_t> randomizer(8, 256);
    const size_t m = randomizer.get(), n = randomizer.get(), k = randomizer.get();
    const size4_t lhs_shape{batches, 1, m, k};
    const size4_t rhs_shape{batches, 1, k, n};
    const size4_t out_shape{batches, 1, m, n};

    cpu::Stream cpu_stream;
    cuda::Stream gpu_stream;
    cuda::memory::PtrManaged<TestType> lhs(lhs_shape.elements(), gpu_stream);
    cuda::memory::PtrManaged<TestType> rhs(rhs_shape.elements(), gpu_stream);
    cuda::memory::PtrManaged<TestType> out_gpu(out_shape.elements(), gpu_stream);
    cpu::memory::PtrHost<TestType> out_cpu(out_shape.elements());
    cuda::math::randomize(math::uniform_t{}, lhs.share(), lhs.elements(), real_t{-5}, real_t{5}, gpu_stream);
    cuda::math::randomize(math::uniform_t{}, rhs.share(), rhs.elements(), real_t{-5}, real_t{5}, gpu_stream);
    gpu_stream.synchronize();

    cuda::math::matmul(lhs.share(), lhs_shape.strides(), lhs_shape,
                       rhs.share(), rhs_shape.strides(), rhs_shape,
                       out_gpu.share(), out_shape.strides(), out_shape, gpu_stream);
    cpu::math::matmul(lhs.share(), lhs_shape.strides(), lhs_shape,
                      rhs.share(), rhs_shape.strides(), rhs_shape,
                      out_cpu.share(), out_shape.strides(), out_shape, cpu_stream);

    cpu_stream.synchronize();
    gpu_stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, out_cpu.get(), out_gpu.get(), out_gpu.elements(),
                          std::is_same_v<real_t, double> ? 1e-7 : 5e-3));
}
