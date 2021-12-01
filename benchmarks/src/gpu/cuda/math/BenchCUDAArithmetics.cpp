#include <noa/gpu/cuda/math/Arithmetics.h>

#include <noa/Session.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEST_CASE("cuda::Math: Arithmetics", "[noa][cuda][math]") {
    NOA_BENCHMARK_HEADER("cuda::Math: Arithmetics");
    test::Randomizer<float> randomizer(1., 10.);

    size3_t shape = {128, 128, 128};
    size_t elements = noa::elements(shape);
    size_t batches = 1;
    std::string string_test_size = string::format("shape:{}, elements:{}", shape, elements);

    cuda::Stream stream;
    {
        noa::Session::logger.info("contiguous memory, multiplyBy* functions: shape:{}, elements:{}, batches:{}",
                                  shape, elements, batches);

        cpu::memory::PtrHost<float> data(elements * batches);
        cpu::memory::PtrHost<float> expected(elements * batches);
        cpu::memory::PtrHost<float> values(batches);
        cpu::memory::PtrHost<float> array(elements);
        float value = randomizer.get();

        cuda::memory::PtrDevice<float> d_data(elements * batches);
        cuda::memory::PtrDevice<float> d_values(batches);
        cuda::memory::PtrDevice<float> d_array(elements);
        cuda::memory::PtrDevice<float> d_results(elements * batches);
        cpu::memory::PtrHost<float> cuda_results(elements * batches);

        test::randomize(data.get(), data.elements(), randomizer);
        test::memset(expected.get(), expected.elements(), 0);
        test::randomize(values.get(), values.elements(), randomizer);
        test::randomize(array.get(), array.elements(), randomizer);

        cuda::memory::copy(data.get(), d_data.get(), elements * batches);
        cuda::memory::copy(expected.get(), d_results.get(), elements * batches);
        cuda::memory::copy(values.get(), d_values.get(), batches);
        cuda::memory::copy(array.get(), d_array.get(), elements);

        {
            NOA_BENCHMARK_CUDA_SCOPE(stream, "multiplyByValue - one batch - multiply array by single value");
            cuda::math::multiplyByValue(d_data.get(), value, d_results.get(), elements, stream);
        }

        {
            NOA_BENCHMARK_CUDA_SCOPE(stream, "multiplyByValue - multiply one value per batch");
            cuda::math::multiplyByValue(d_data.get(), d_values.get(), d_results.get(), elements, batches, stream);
        }

        {
            NOA_BENCHMARK_CUDA_SCOPE(stream, "multiplyByArray - element-wise multiply");
            cuda::math::multiplyByArray(d_data.get(), d_array.get(), d_results.get(), elements, batches, stream);
        }
    }

    {
        noa::Session::logger.info("padded memory, multiplyBy* functions: shape:{}, elements:{}, batches:{}",
                                  shape, elements, batches);
        cpu::memory::PtrHost<float> data(elements * batches);
        cpu::memory::PtrHost<float> expected(elements * batches);
        cpu::memory::PtrHost<float> values(batches);
        cpu::memory::PtrHost<float> array(elements);
        float value = randomizer.get();

        size3_t shape_batch = {shape.x, shape.y, shape.z * batches};
        cuda::memory::PtrDevicePadded<float> d_data(shape_batch);
        cuda::memory::PtrDevice<float> d_values(batches);
        cuda::memory::PtrDevicePadded<float> d_array(shape);
        cuda::memory::PtrDevicePadded<float> d_results(shape_batch);
        cpu::memory::PtrHost<float> cuda_results(elements * batches);

        test::randomize(data.get(), data.elements(), randomizer);
        test::memset(expected.get(), expected.elements(), 0);
        test::randomize(values.get(), values.elements(), randomizer);
        test::randomize(array.get(), array.elements(), randomizer);

        cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batch);
        cuda::memory::copy(expected.get(), shape.x, d_results.get(), d_results.pitch(), shape_batch);
        cuda::memory::copy(values.get(), d_values.get(), batches);
        cuda::memory::copy(array.get(), shape.x, d_array.get(), d_array.pitch(), shape);

        {
            NOA_BENCHMARK_CUDA_SCOPE(stream, "multiplyByValue - one batch - multiply array by single value");
            cuda::math::multiplyByValue(d_data.get(), d_data.pitch(), value,
                                        d_results.get(), d_results.pitch(), shape, stream);
        }

        {
            NOA_BENCHMARK_CUDA_SCOPE(stream, "multiplyByValue - multiply one value per batch");
            cuda::math::multiplyByValue(d_data.get(), d_data.pitch(), d_values.get(),
                                        d_results.get(), d_results.pitch(), shape, batches, stream);
        }

        {
            NOA_BENCHMARK_CUDA_SCOPE(stream, "multiplyByArray - element-wise multiply");
            cuda::math::multiplyByArray(d_data.get(), d_data.pitch(), d_array.get(), d_array.pitch(),
                                        d_results.get(), d_array.pitch(), shape, batches, stream);
        }
    }
}
