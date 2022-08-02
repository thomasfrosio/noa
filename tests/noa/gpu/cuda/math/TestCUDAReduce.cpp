#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/math/Reduce.h>
#include <noa/gpu/cuda/math/Reduce.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/PtrManaged.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::math:: reduce all, contiguous", "[noa][cuda][math]",
                   half_t, float, double, int32_t, int64_t, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();
    cpu::Stream cpu_stream;
    cuda::Stream gpu_stream;
    INFO(shape);

    cpu::memory::PtrHost<TestType> h_data(elements);
    cuda::memory::PtrDevice<TestType> d_data(elements, gpu_stream);
    TestType cpu_results, gpu_results;

    test::Randomizer<TestType> randomizer(-100., 100.);
    test::randomize(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.get(), d_data.get(), d_data.size(), gpu_stream);

    if constexpr (!noa::traits::is_complex_v<TestType>) {
        AND_THEN("min") {
            gpu_results = cuda::math::min<TestType>(d_data.share(), stride, shape, gpu_stream);
            cpu_results = cpu::math::min<TestType>(h_data.share(), stride, shape, cpu_stream);
            auto diff = static_cast<double>(math::abs(gpu_results - cpu_results));
            REQUIRE_THAT(diff, Catch::WithinAbs(0., 1e-6));
        }

        AND_THEN("max") {
            gpu_results = cuda::math::max<TestType>(d_data.share(), stride, shape, gpu_stream);
            cpu_results = cpu::math::max<TestType>(h_data.share(), stride, shape, cpu_stream);
            auto diff = static_cast<double>(math::abs(gpu_results - cpu_results));
            REQUIRE_THAT(diff, Catch::WithinAbs(0., 1e-6));
        }
    }

    if constexpr (!std::is_same_v<half_t, TestType>) {
        AND_THEN("sum") {
            gpu_results = cuda::math::sum<TestType>(d_data.share(), stride, shape, gpu_stream);
            cpu_results = cpu::math::sum<TestType>(h_data.share(), stride, shape, cpu_stream);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_results, &cpu_results, 1, 3e-5));
        }

        AND_THEN("mean") {
            gpu_results = cuda::math::mean<TestType>(d_data.share(), stride, shape, gpu_stream);
            cpu_results = cpu::math::mean<TestType>(h_data.share(), stride, shape, cpu_stream);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_results, &cpu_results, 1, 1e-6));
        }

        using real_t = noa::traits::value_type_t<TestType>;
        if constexpr (noa::traits::is_float_v<real_t>) {
            real_t cpu_var, gpu_var;
            AND_THEN("var") {
                gpu_var = cuda::math::var<0, TestType, real_t>(d_data.share(), stride, shape, gpu_stream);
                cpu_var = cpu::math::var<0, TestType, real_t>(h_data.share(), stride, shape, cpu_stream);
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_var, &cpu_var, 1, 1e-6));

                gpu_var = cuda::math::var<1, TestType, real_t>(d_data.share(), stride, shape, gpu_stream);
                cpu_var = cpu::math::var<1, TestType, real_t>(h_data.share(), stride, shape, cpu_stream);
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_var, &cpu_var, 1, 1e-6));
            }

            AND_THEN("std") {
                gpu_var = cuda::math::std<0, TestType, real_t>(d_data.share(), stride, shape, gpu_stream);
                cpu_var = cpu::math::std<0, TestType, real_t>(h_data.share(), stride, shape, cpu_stream);
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_var, &cpu_var, 1, 1e-6));

                gpu_var = cuda::math::std<1, TestType, real_t>(d_data.share(), stride, shape, gpu_stream);
                cpu_var = cpu::math::std<1, TestType, real_t>(h_data.share(), stride, shape, cpu_stream);
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_var, &cpu_var, 1, 1e-6));
            }
        }
    }
}

TEMPLATE_TEST_CASE("cuda::math:: reduce all, padded", "[noa][cuda][math]",
                   half_t, float, double, int32_t, int64_t, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();
    cpu::Stream cpu_stream;
    cuda::Stream gpu_stream;

    cpu::memory::PtrHost<TestType> h_data(elements);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape);
    TestType cpu_results, gpu_results;

    test::Randomizer<TestType> randomizer(-100., 100.);
    test::randomize(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.share(), stride, d_data.share(), d_data.strides(), shape, gpu_stream);

    if constexpr (!noa::traits::is_complex_v<TestType>) {
        AND_THEN("min") {
            gpu_results = cuda::math::min<TestType>(d_data.share(), d_data.strides(), shape, gpu_stream);
            cpu_results = cpu::math::min<TestType>(h_data.share(), stride, shape, cpu_stream);
            auto diff = static_cast<double>(math::abs(gpu_results - cpu_results));
            REQUIRE_THAT(diff, Catch::WithinAbs(0., 1e-6));
        }

        AND_THEN("max") {
            gpu_results = cuda::math::max<TestType>(d_data.share(), d_data.strides(), shape, gpu_stream);
            cpu_results = cpu::math::max<TestType>(h_data.share(), stride, shape, cpu_stream);
            auto diff = static_cast<double>(math::abs(gpu_results - cpu_results));
            REQUIRE_THAT(diff, Catch::WithinAbs(0., 1e-6));
        }
    }

    if constexpr (!std::is_same_v<half_t, TestType>) {
        AND_THEN("sum") {
            gpu_results = cuda::math::sum<TestType>(d_data.share(), d_data.strides(), shape, gpu_stream);
            cpu_results = cpu::math::sum<TestType>(h_data.share(), stride, shape, cpu_stream);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_results, &cpu_results, 1, 3e-5));
        }

        AND_THEN("mean") {
            gpu_results = cuda::math::mean<TestType>(d_data.share(), d_data.strides(), shape, gpu_stream);
            cpu_results = cpu::math::mean<TestType>(h_data.share(), stride, shape, cpu_stream);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_results, &cpu_results, 1, 1e-6));
        }

        using real_t = noa::traits::value_type_t<TestType>;
        if constexpr (noa::traits::is_float_v<real_t>) {
            real_t cpu_var, gpu_var;
            AND_THEN("var") {
                gpu_var = cuda::math::var<0, TestType, real_t>(d_data.share(), d_data.strides(), shape, gpu_stream);
                cpu_var = cpu::math::var<0, TestType, real_t>(h_data.share(), stride, shape, cpu_stream);
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_var, &cpu_var, 1, 1e-6));

                gpu_var = cuda::math::var<1, TestType, real_t>(d_data.share(), d_data.strides(), shape, gpu_stream);
                cpu_var = cpu::math::var<1, TestType, real_t>(h_data.share(), stride, shape, cpu_stream);
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_var, &cpu_var, 1, 1e-6));
            }

            AND_THEN("std") {
                gpu_var = cuda::math::std<0, TestType, real_t>(d_data.share(), d_data.strides(), shape, gpu_stream);
                cpu_var = cpu::math::std<0, TestType, real_t>(h_data.share(), stride, shape, cpu_stream);
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_var, &cpu_var, 1, 1e-6));

                gpu_var = cuda::math::std<1, TestType, real_t>(d_data.share(), d_data.strides(), shape, gpu_stream);
                cpu_var = cpu::math::std<1, TestType, real_t>(h_data.share(), stride, shape, cpu_stream);
                REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_var, &cpu_var, 1, 1e-6));
            }
        }
    }
}

TEMPLATE_TEST_CASE("cuda::math:: reduce all, large, contiguous", "[noa][cuda][math]",
                   float, double, cfloat_t, cdouble_t) {
    const size4_t shape{1, 320, 320, 320};
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();
    cpu::Stream cpu_stream;
    cuda::Stream gpu_stream;

    cpu::memory::PtrHost<TestType> h_data(elements);
    cuda::memory::PtrDevice<TestType> d_data(elements, gpu_stream);
    TestType cpu_results, gpu_results;

    test::Randomizer<TestType> randomizer(-100., 100.);
    test::randomize(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.get(), d_data.get(), d_data.size(), gpu_stream);

    AND_THEN("sum") {
        gpu_results = cuda::math::sum<TestType>(d_data.share(), stride, shape, gpu_stream);
        cpu_results = cpu::math::sum<TestType>(h_data.share(), stride, shape, cpu_stream);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_results, &cpu_results, 1, 3e-5));
    }

    AND_THEN("mean") {
        gpu_results = cuda::math::mean<TestType>(d_data.share(), stride, shape, gpu_stream);
        cpu_results = cpu::math::mean<TestType>(h_data.share(), stride, shape, cpu_stream);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_results, &cpu_results, 1, 1e-6));
    }

    using real_t = noa::traits::value_type_t<TestType>;
    real_t cpu_var, gpu_var;
    AND_THEN("var") {
        gpu_var = cuda::math::var<0, TestType, real_t>(d_data.share(), stride, shape, gpu_stream);
        cpu_var = cpu::math::var<0, TestType, real_t>(h_data.share(), stride, shape, cpu_stream);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_var, &cpu_var, 1, 1e-6));

        gpu_var = cuda::math::var<1, TestType, real_t>(d_data.share(), stride, shape, gpu_stream);
        cpu_var = cpu::math::var<1, TestType, real_t>(h_data.share(), stride, shape, cpu_stream);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_var, &cpu_var, 1, 1e-6));
    }

    AND_THEN("std") {
        gpu_var = cuda::math::std<0, TestType, real_t>(d_data.share(), stride, shape, gpu_stream);
        cpu_var = cpu::math::std<0, TestType, real_t>(h_data.share(), stride, shape, cpu_stream);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_var, &cpu_var, 1, 1e-6));

        gpu_var = cuda::math::std<1, TestType, real_t>(d_data.share(), stride, shape, gpu_stream);
        cpu_var = cpu::math::std<1, TestType, real_t>(h_data.share(), stride, shape, cpu_stream);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_var, &cpu_var, 1, 1e-6));
    }
}

TEMPLATE_TEST_CASE("cuda::math:: reduce all, large, padded", "[noa][cuda][math]",
                   float, double, cfloat_t, cdouble_t) {
    const size4_t shape{1, 320, 320, 320};
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();
    cpu::Stream cpu_stream;
    cuda::Stream gpu_stream;

    cpu::memory::PtrHost<TestType> h_data(elements);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape);
    TestType cpu_results, gpu_results;

    test::Randomizer<TestType> randomizer(-100., 100.);
    test::randomize(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.share(), stride, d_data.share(), d_data.strides(), shape, gpu_stream);

    AND_THEN("sum") {
        gpu_results = cuda::math::sum<TestType>(d_data.share(), d_data.strides(), shape, gpu_stream);
        cpu_results = cpu::math::sum<TestType>(h_data.share(), stride, shape, cpu_stream);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_results, &cpu_results, 1, 3e-5));
    }

    AND_THEN("mean") {
        gpu_results = cuda::math::mean<TestType>(d_data.share(), d_data.strides(), shape, gpu_stream);
        cpu_results = cpu::math::mean<TestType>(h_data.share(), stride, shape, cpu_stream);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_results, &cpu_results, 1, 1e-6));
    }

    using real_t = noa::traits::value_type_t<TestType>;
    real_t cpu_var, gpu_var;
    AND_THEN("var") {
        gpu_var = cuda::math::var<0, TestType, real_t>(d_data.share(), d_data.strides(), shape, gpu_stream);
        cpu_var = cpu::math::var<0, TestType, real_t>(h_data.share(), stride, shape, cpu_stream);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_var, &cpu_var, 1, 1e-6));

        gpu_var = cuda::math::var<1, TestType, real_t>(d_data.share(), d_data.strides(), shape, gpu_stream);
        cpu_var = cpu::math::var<1, TestType, real_t>(h_data.share(), stride, shape, cpu_stream);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_var, &cpu_var, 1, 1e-6));
    }

    AND_THEN("std") {
        gpu_var = cuda::math::std<0, TestType, real_t>(d_data.share(), d_data.strides(), shape, gpu_stream);
        cpu_var = cpu::math::std<0, TestType, real_t>(h_data.share(), stride, shape, cpu_stream);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_var, &cpu_var, 1, 1e-6));

        gpu_var = cuda::math::std<1, TestType, real_t>(d_data.share(), d_data.strides(), shape, gpu_stream);
        cpu_var = cpu::math::std<1, TestType, real_t>(h_data.share(), stride, shape, cpu_stream);
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_var, &cpu_var, 1, 1e-6));
    }
}

TEMPLATE_TEST_CASE("cuda::math:: reduce statistics, contiguous", "[noa][cuda][math]",
                   float, double, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();
    cpu::Stream cpu_stream;
    cuda::Stream gpu_stream;

    cpu::memory::PtrHost<TestType> h_data(elements);
    cuda::memory::PtrDevice<TestType> d_data(elements, gpu_stream);

    test::Randomizer<TestType> randomizer(-100., 100.);
    test::randomize(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.share(), stride, d_data.share(), stride, shape, gpu_stream);

    using real_t = noa::traits::value_type_t<TestType>;
    auto[gpu_sum, gpu_mean, gpu_var, gpu_std] =
        cuda::math::statistics<0, TestType, real_t>(d_data.share(), stride, shape, gpu_stream);
    auto[cpu_sum, cpu_mean, cpu_var, cpu_std] =
        cpu::math::statistics<0, TestType, real_t>(h_data.share(), stride, shape, cpu_stream);
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_sum, &cpu_sum, 1, 3e-5));
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_mean, &cpu_mean, 1, 1e-6));
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_var, &cpu_var, 1, 1e-6));
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_std, &cpu_std, 1, 1e-6));
}

TEMPLATE_TEST_CASE("cuda::math:: reduce statistics, padded", "[noa][cuda][math]",
                   float, double, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.strides();
    const size_t elements = shape.elements();
    cpu::Stream cpu_stream;
    cuda::Stream gpu_stream;

    cpu::memory::PtrHost<TestType> h_data(elements);
    cuda::memory::PtrDevicePadded<TestType> d_data(shape);

    test::Randomizer<TestType> randomizer(-100., 100.);
    test::randomize(h_data.get(), h_data.elements(), randomizer);
    cuda::memory::copy(h_data.share(), stride, d_data.share(), d_data.strides(), shape, gpu_stream);

    using real_t = noa::traits::value_type_t<TestType>;
    auto[gpu_sum, gpu_mean, gpu_var, gpu_std] =
            cuda::math::statistics<0, TestType, real_t>(d_data.share(), d_data.strides(), shape, gpu_stream);
    auto[cpu_sum, cpu_mean, cpu_var, cpu_std] =
            cpu::math::statistics<0, TestType, real_t>(h_data.share(), stride, shape, cpu_stream);
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_sum, &cpu_sum, 1, 3e-5));
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_mean, &cpu_mean, 1, 1e-6));
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_var, &cpu_var, 1, 1e-6));
    REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, &gpu_std, &cpu_std, 1, 1e-6));
}

TEMPLATE_TEST_CASE("cuda::math:: reduce 1 axis", "[assets][noa][cuda][math]",
                   float, double, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t input_shape = test::getRandomShapeBatched(ndim);
    const size4_t input_stride = input_shape.strides();
    const size_t input_elements = input_shape.elements();

    const uint axis = GENERATE(0u, 1u, 2u, 3u);
    size4_t output_shape = input_shape;
    output_shape[axis] = 1;
    const size4_t output_stride = output_shape.strides();
    const size_t output_elements = output_shape.elements();
    INFO(input_shape);
    INFO(output_shape);

    cpu::Stream cpu_stream;
    cuda::Stream gpu_stream;

    cpu::memory::PtrHost<TestType> h_input(input_elements);
    cpu::memory::PtrHost<TestType> h_output(output_elements);
    cuda::memory::PtrManaged<TestType> d_input(input_elements, gpu_stream);
    cuda::memory::PtrManaged<TestType> d_output(output_elements, gpu_stream);

    test::Randomizer<TestType> randomizer(-30., 100.);
    test::randomize(h_input.get(), h_input.elements(), randomizer);
    cuda::memory::copy(h_input.share(), input_stride, d_input.share(), input_stride, input_shape, gpu_stream);

    if constexpr (!noa::traits::is_complex_v<TestType>) {
        AND_THEN("min") {
            cuda::math::min<TestType>(d_input.share(), input_stride, input_shape,
                                      d_output.share(), output_stride, output_shape, gpu_stream);
            cpu::math::min<TestType>(h_input.share(), input_stride, input_shape,
                                     h_output.share(), output_stride, output_shape, cpu_stream);
            gpu_stream.synchronize();
            cpu_stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), d_output.get(), 1, 1e-5));
        }

        AND_THEN("max") {
            cuda::math::max<TestType>(d_input.share(), input_stride, input_shape,
                                      d_output.share(), output_stride, output_shape, gpu_stream);
            cpu::math::max<TestType>(h_input.share(), input_stride, input_shape,
                                     h_output.share(), output_stride, output_shape, cpu_stream);
            gpu_stream.synchronize();
            cpu_stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), d_output.get(), 1, 1e-5));
        }
    }

    AND_THEN("sum") {
        cuda::math::sum<TestType>(d_input.share(), input_stride, input_shape,
                                  d_output.share(), output_stride, output_shape, gpu_stream);
        cpu::math::sum<TestType>(h_input.share(), input_stride, input_shape,
                                 h_output.share(), output_stride, output_shape, cpu_stream);
        gpu_stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), d_output.get(), 1, 1e-5));
    }

    AND_THEN("mean") {
        cuda::math::mean<TestType>(d_input.share(), input_stride, input_shape,
                                   d_output.share(), output_stride, output_shape, gpu_stream);
        cpu::math::mean<TestType>(h_input.share(), input_stride, input_shape,
                                  h_output.share(), output_stride, output_shape, cpu_stream);
        gpu_stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), d_output.get(), 1, 1e-5));
    }

    if constexpr (noa::traits::is_complex_v<TestType>) {
        using real_t = noa::traits::value_type_t<TestType>;
        cpu::memory::PtrHost<real_t> h_output_real(output_elements);
        cuda::memory::PtrManaged<real_t> d_output_real(output_elements, gpu_stream);

        AND_THEN("var") {
            cuda::math::var<0, TestType, real_t>(d_input.share(), input_stride, input_shape,
                                                 d_output_real.share(), output_stride, output_shape, gpu_stream);
            cpu::math::var<0, TestType, real_t>(h_input.share(), input_stride, input_shape,
                                                h_output_real.share(), output_stride, output_shape, cpu_stream);
            gpu_stream.synchronize();
            cpu_stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output_real.get(), d_output_real.get(), 1, 1e-5));

            cuda::math::var<1, TestType, real_t>(d_input.share(), input_stride, input_shape,
                                                 d_output_real.share(), output_stride, output_shape, gpu_stream);
            cpu::math::var<1, TestType, real_t>(h_input.share(), input_stride, input_shape,
                                                h_output_real.share(), output_stride, output_shape, cpu_stream);
            gpu_stream.synchronize();
            cpu_stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output_real.get(), d_output_real.get(), 1, 1e-5));
        }

        AND_THEN("std") {
            cuda::math::std<0, TestType, real_t>(d_input.share(), input_stride, input_shape,
                                                 d_output_real.share(), output_stride, output_shape, gpu_stream);
            cpu::math::std<0, TestType, real_t>(h_input.share(), input_stride, input_shape,
                                                h_output_real.share(), output_stride, output_shape, cpu_stream);
            gpu_stream.synchronize();
            cpu_stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output_real.get(), d_output_real.get(), 1, 1e-5));

            cuda::math::std<1, TestType, real_t>(d_input.share(), input_stride, input_shape,
                                                 d_output_real.share(), output_stride, output_shape, gpu_stream);
            cpu::math::std<1, TestType, real_t>(h_input.share(), input_stride, input_shape,
                                                h_output_real.share(), output_stride, output_shape, cpu_stream);
            gpu_stream.synchronize();
            cpu_stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output_real.get(), d_output_real.get(), 1, 1e-5));
        }
    } else {
        AND_THEN("var") {
            cuda::math::var<0, TestType, TestType>(d_input.share(), input_stride, input_shape,
                                                   d_output.share(), output_stride, output_shape, gpu_stream);
            cpu::math::var<0, TestType, TestType>(h_input.share(), input_stride, input_shape,
                                                  h_output.share(), output_stride, output_shape, cpu_stream);
            gpu_stream.synchronize();
            cpu_stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), d_output.get(), 1, 1e-5));

            cuda::math::var<1, TestType, TestType>(d_input.share(), input_stride, input_shape,
                                                   d_output.share(), output_stride, output_shape, gpu_stream);
            cpu::math::var<1, TestType, TestType>(h_input.share(), input_stride, input_shape,
                                                  h_output.share(), output_stride, output_shape, cpu_stream);
            gpu_stream.synchronize();
            cpu_stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), d_output.get(), 1, 1e-5));
        }

        AND_THEN("std") {
            cuda::math::std<0, TestType, TestType>(d_input.share(), input_stride, input_shape,
                                                   d_output.share(), output_stride, output_shape, gpu_stream);
            cpu::math::std<0, TestType, TestType>(h_input.share(), input_stride, input_shape,
                                                  h_output.share(), output_stride, output_shape, cpu_stream);
            gpu_stream.synchronize();
            cpu_stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), d_output.get(), 1, 1e-5));

            cuda::math::std<1, TestType, TestType>(d_input.share(), input_stride, input_shape,
                                                   d_output.share(), output_stride, output_shape, gpu_stream);
            cpu::math::std<1, TestType, TestType>(h_input.share(), input_stride, input_shape,
                                                  h_output.share(), output_stride, output_shape, cpu_stream);
            gpu_stream.synchronize();
            cpu_stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), d_output.get(), 1, 1e-5));
        }
    }
}

TEMPLATE_TEST_CASE("cuda::math:: reduce innermost axes", "[assets][noa][cuda][math]",
                   float, double, cfloat_t, cdouble_t) {
    const uint ndim = GENERATE(1u, 2u, 3u);
    const size4_t input_shape = test::getRandomShapeBatched(ndim);
    const size4_t input_stride = input_shape.strides();
    const size_t input_elements = input_shape.elements();

    const size4_t output_shape{input_shape[0], 1, 1, 1};
    const size4_t output_stride = output_shape.strides();
    const size_t output_elements = output_shape.elements();
    INFO(input_shape);
    INFO(output_shape);

    cpu::Stream cpu_stream;
    cuda::Stream gpu_stream;

    cpu::memory::PtrHost<TestType> h_input(input_elements);
    cpu::memory::PtrHost<TestType> h_output(output_elements);
    cuda::memory::PtrManaged<TestType> d_input(input_elements, gpu_stream);
    cuda::memory::PtrManaged<TestType> d_output(output_elements, gpu_stream);

    test::Randomizer<TestType> randomizer(-30., 100.);
    test::randomize(h_input.get(), h_input.elements(), randomizer);
    cuda::memory::copy(h_input.share(), input_stride, d_input.share(), input_stride, input_shape, gpu_stream);

    if constexpr (!noa::traits::is_complex_v<TestType>) {
        AND_THEN("min") {
            cuda::math::min(d_input.share(), input_stride, input_shape,
                            d_output.share(), output_stride, output_shape, gpu_stream);
            cpu::math::min(h_input.share(), input_stride, input_shape,
                           h_output.share(), output_stride, output_shape, cpu_stream);
            gpu_stream.synchronize();
            cpu_stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), d_output.get(), output_elements, 1e-5));
        }

        AND_THEN("max") {
            cuda::math::max(d_input.share(), input_stride, input_shape,
                            d_output.share(), output_stride, output_shape, gpu_stream);
            cpu::math::max(h_input.share(), input_stride, input_shape,
                           h_output.share(), output_stride, output_shape, cpu_stream);
            gpu_stream.synchronize();
            cpu_stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), d_output.get(), output_elements, 1e-5));
        }
    }

    AND_THEN("sum") {
        cuda::math::sum(d_input.share(), input_stride, input_shape,
                        d_output.share(), output_stride, output_shape, gpu_stream);
        cpu::math::sum(h_input.share(), input_stride, input_shape,
                       h_output.share(), output_stride, output_shape, cpu_stream);
        const size4_t unbatched{1, input_shape[1], input_shape[2], input_shape[3]};
        gpu_stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), d_output.get(), output_elements, 1e-5));
    }

    AND_THEN("mean") {
        cuda::math::mean(d_input.share(), input_stride, input_shape,
                         d_output.share(), output_stride, output_shape, gpu_stream);
        cpu::math::mean(h_input.share(), input_stride, input_shape,
                        h_output.share(), output_stride, output_shape, cpu_stream);
        const size4_t unbatched{1, input_shape[1], input_shape[2], input_shape[3]};
        gpu_stream.synchronize();
        cpu_stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), d_output.get(), output_elements, 1e-5));
    }

    if constexpr (noa::traits::is_complex_v<TestType>) {
        using real_t = noa::traits::value_type_t<TestType>;
        cpu::memory::PtrHost<real_t> h_output_real(output_elements);
        cuda::memory::PtrManaged<real_t> d_output_real(output_elements, gpu_stream);

        AND_THEN("var") {
            cuda::math::var<0>(d_input.share(), input_stride, input_shape,
                               d_output_real.share(), output_stride, output_shape, gpu_stream);
            cpu::math::var<0>(h_input.share(), input_stride, input_shape,
                              h_output_real.share(), output_stride, output_shape, cpu_stream);
            gpu_stream.synchronize();
            cpu_stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output_real.get(), d_output_real.get(), output_elements, 1e-5));

            cuda::math::var<1>(d_input.share(), input_stride, input_shape,
                               d_output_real.share(), output_stride, output_shape, gpu_stream);
            cpu::math::var<1>(h_input.share(), input_stride, input_shape,
                              h_output_real.share(), output_stride, output_shape, cpu_stream);
            gpu_stream.synchronize();
            cpu_stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output_real.get(), d_output_real.get(), output_elements, 1e-5));
        }

        AND_THEN("std") {
            cuda::math::std<0>(d_input.share(), input_stride, input_shape,
                               d_output_real.share(), output_stride, output_shape, gpu_stream);
            cpu::math::std<0>(h_input.share(), input_stride, input_shape,
                              h_output_real.share(), output_stride, output_shape, cpu_stream);
            gpu_stream.synchronize();
            cpu_stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output_real.get(), d_output_real.get(), output_elements, 1e-5));

            cuda::math::std<1>(d_input.share(), input_stride, input_shape,
                               d_output_real.share(), output_stride, output_shape, gpu_stream);
            cpu::math::std<1>(h_input.share(), input_stride, input_shape,
                              h_output_real.share(), output_stride, output_shape, cpu_stream);
            gpu_stream.synchronize();
            cpu_stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output_real.get(), d_output_real.get(), output_elements, 1e-5));
        }
    } else {
        AND_THEN("var") {
            cuda::math::var<0>(d_input.share(), input_stride, input_shape,
                               d_output.share(), output_stride, output_shape, gpu_stream);
            cpu::math::var<0>(h_input.share(), input_stride, input_shape,
                              h_output.share(), output_stride, output_shape, cpu_stream);
            gpu_stream.synchronize();
            cpu_stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), d_output.get(), output_elements, 1e-5));

            cuda::math::var<1>(d_input.share(), input_stride, input_shape,
                               d_output.share(), output_stride, output_shape, gpu_stream);
            cpu::math::var<1>(h_input.share(), input_stride, input_shape,
                              h_output.share(), output_stride, output_shape, cpu_stream);
            gpu_stream.synchronize();
            cpu_stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), d_output.get(), output_elements, 1e-5));
        }

        AND_THEN("std") {
            cuda::math::std<0>(d_input.share(), input_stride, input_shape,
                               d_output.share(), output_stride, output_shape, gpu_stream);
            cpu::math::std<0>(h_input.share(), input_stride, input_shape,
                              h_output.share(), output_stride, output_shape, cpu_stream);
            gpu_stream.synchronize();
            cpu_stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), d_output.get(), output_elements, 1e-5));

            cuda::math::std<1>(d_input.share(), input_stride, input_shape,
                               d_output.share(), output_stride, output_shape, gpu_stream);
            cpu::math::std<1>(h_input.share(), input_stride, input_shape,
                              h_output.share(), output_stride, output_shape, cpu_stream);
            gpu_stream.synchronize();
            cpu_stream.synchronize();
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, h_output.get(), d_output.get(), output_elements, 1e-5));
        }
    }
}
