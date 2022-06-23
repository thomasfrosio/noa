#include <noa/cpu/math/Blas.h>
#include <noa/cpu/math/Ewise.h>
#include <noa/cpu/math/Random.h>
#include <noa/cpu/math/Reduce.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

namespace {
    template<typename T>
    void naiveMatmul_(const T* lhs, size4_t lhs_stride,
                      const T* rhs, size4_t rhs_stride,
                      T* out, size4_t out_stride,
                      size4_t bmnk) {
        std::fill(out, out + out_stride[0] * bmnk[0], T{0});

        for (size_t batch = 0; batch < bmnk[0]; ++batch) {
            for (size_t m = 0; m < bmnk[1]; ++m) {
                for (size_t n = 0; n < bmnk[2]; ++n) {
                    for (size_t k = 0; k < bmnk[3]; ++k) {
                        out[indexing::at(batch, 0, m, n, out_stride)] +=
                                lhs[indexing::at(batch, 0, m, k, lhs_stride)] *
                                rhs[indexing::at(batch, 0, k, n, rhs_stride)];
                    }
                }
            }
        }
    }

    template<typename T>
    void naiveDot_(const T* lhs, size_t lhs_stride,
                   const T* rhs, size_t rhs_stride,
                   size_t n, size_t batches, T* out) {
        for (size_t batch = 0; batch < batches; ++batch) {
            T sum{0};
            for (size_t i = 0; i < n; ++i)
                sum += lhs[batch * n + i * lhs_stride] * rhs[batch * n +i * rhs_stride];
            out[batch] = sum;
        }
    }
}

TEMPLATE_TEST_CASE("cpu::math::dot()", "[noa][cpu]", float, double, cfloat_t, cdouble_t) {
    using real_t = traits::value_type_t<TestType>;
    const size_t batches = test::Randomizer<size_t>(1, 5).get();

    test::Randomizer<size_t> randomizer(4096, 1048576);
    const size4_t shape{batches, 1, 1, randomizer.get()};

    cpu::Stream stream(cpu::Stream::ASYNC);
    cpu::memory::PtrHost<TestType> lhs(shape.elements());
    cpu::memory::PtrHost<TestType> rhs(shape.elements());
    cpu::memory::PtrHost<TestType> out(batches);
    cpu::math::randomize(math::uniform_t{}, lhs.share(), lhs.elements(), real_t{-5}, real_t{5}, stream);
    cpu::math::randomize(math::uniform_t{}, rhs.share(), rhs.elements(), real_t{-5}, real_t{5}, stream);
    stream.synchronize();

    // Compute output:
    cpu::math::dot(lhs.share(), shape.stride(), shape,
                   rhs.share(), shape.stride(), shape,
                   out.share(), stream);

    // Compute expected:
    cpu::memory::PtrHost<TestType> exp(batches);
    const size4_t reduced_shape{batches, 1, 1, 1};
    stream.synchronize();
    cpu::math::ewise(lhs.share(), shape.stride(), rhs.share(), shape.stride(), lhs.share(), shape.stride(), shape, math::multiply_t{}, stream);
    cpu::math::sum(lhs.share(), shape.stride(), shape, exp.share(), reduced_shape.stride(), reduced_shape, stream);

    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, out.get(), exp.get(), exp.elements(),
                          std::is_same_v<real_t, double> ? 1e-7 : 5e-3));
}

TEMPLATE_TEST_CASE("cpu::math::dot() vs matmul", "[noa][cpu]", float, double) {
    using real_t = traits::value_type_t<TestType>;

    test::Randomizer<size_t> randomizer(4096, 1048576);
    const size_t n = randomizer.get();
    const size4_t lhs_shape{1, 1, 1, n};
    const size4_t rhs_shape{1, 1, n, 1};
    const size4_t out_shape{1, 1, 1, 1};

    cpu::Stream stream(cpu::Stream::ASYNC);
    cpu::memory::PtrHost<TestType> lhs(n);
    cpu::memory::PtrHost<TestType> rhs(n);

    cpu::math::randomize(math::uniform_t{}, lhs.share(), lhs.elements(), real_t{-5}, real_t{5}, stream);
    cpu::math::randomize(math::uniform_t{}, rhs.share(), rhs.elements(), real_t{-5}, real_t{5}, stream);
    stream.synchronize();

    TestType out_dot = cpu::math::dot(lhs.share(), lhs_shape.stride(), lhs_shape,
                                      rhs.share(), rhs_shape.stride(), rhs_shape, stream);

    cpu::memory::PtrHost<TestType> out(1);
    cpu::math::matmul(lhs.share(), lhs_shape.stride(), lhs_shape,
                      rhs.share(), rhs_shape.stride(), rhs_shape,
                      out.share(), out_shape.stride(), out_shape,
                      stream);

    stream.synchronize();
    REQUIRE_THAT(out_dot, Catch::WithinAbs(static_cast<double>(out[0]), std::is_same_v<real_t, double> ? 1e-7 : 5e-3));
}

TEMPLATE_TEST_CASE("cpu::math::matmul()", "[noa][cpu]", float, double, cfloat_t, cdouble_t) {
    using real_t = traits::value_type_t<TestType>;
    const size_t batches = test::Randomizer<size_t>(1, 4).get();

    test::Randomizer<size_t> randomizer(8, 256);
    const size_t m = randomizer.get(), n = randomizer.get(), k = randomizer.get();
    const size4_t lhs_shape{batches, 1, m, k};
    const size4_t rhs_shape{batches, 1, k, n};
    const size4_t out_shape{batches, 1, m, n};

    cpu::Stream stream(cpu::Stream::ASYNC);
    cpu::memory::PtrHost<TestType> lhs(lhs_shape.elements());
    cpu::memory::PtrHost<TestType> rhs(rhs_shape.elements());
    cpu::memory::PtrHost<TestType> out(out_shape.elements());
    cpu::math::randomize(math::uniform_t{}, lhs.share(), lhs.elements(), real_t{-5}, real_t{5}, stream);
    cpu::math::randomize(math::uniform_t{}, rhs.share(), rhs.elements(), real_t{-5}, real_t{5}, stream);
    stream.synchronize();

    // Compute output:
    cpu::math::matmul(lhs.share(), lhs_shape.stride(), lhs_shape,
                      rhs.share(), rhs_shape.stride(), rhs_shape,
                      out.share(), out_shape.stride(), out_shape, stream);

    // Compute expected:
    cpu::memory::PtrHost<TestType> exp(out_shape.elements());
    naiveMatmul_(lhs.get(), lhs_shape.stride(), rhs.get(), rhs_shape.stride(), exp.get(), out_shape.stride(),
                size4_t{batches, m, n, k});

    stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, out.get(), exp.get(), exp.elements(),
                          std::is_same_v<real_t, double> ? 1e-7 : 5e-3));
}