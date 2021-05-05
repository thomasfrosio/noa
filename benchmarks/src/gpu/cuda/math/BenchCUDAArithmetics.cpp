#include <noa/gpu/cuda/math/Arithmetics.h>

#include <noa/Session.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace Noa;

TEST_CASE("CUDA::Math: Arithmetics", "[noa][cuda][math]") {
    NOA_BENCHMARK_HEADER("CUDA::Math: Arithmetics");
    Test::Randomizer<float> randomizer(1., 10.);

    size3_t shape = {128, 128, 128};
    size_t elements = getElements(shape);
    uint batches = 1;
    std::string string_test_size = String::format("shape:{}, elements:{}", shape, elements);

    CUDA::Stream stream;
    {
        Noa::Session::logger.info("contiguous memory, multiplyBy* functions: shape:{}, elements:{}, batches:{}",
                                  shape, elements, batches);

        Memory::PtrHost<float> data(elements * batches);
        Memory::PtrHost<float> expected(elements * batches);
        Memory::PtrHost<float> values(batches);
        Memory::PtrHost<float> array(elements);
        float value = randomizer.get();

        CUDA::Memory::PtrDevice<float> d_data(elements * batches);
        CUDA::Memory::PtrDevice<float> d_values(batches);
        CUDA::Memory::PtrDevice<float> d_array(elements);
        CUDA::Memory::PtrDevice<float> d_results(elements * batches);
        Memory::PtrHost<float> cuda_results(elements * batches);

        Test::initDataRandom(data.get(), data.elements(), randomizer);
        Test::initDataZero(expected.get(), expected.elements());
        Test::initDataRandom(values.get(), values.elements(), randomizer);
        Test::initDataRandom(array.get(), array.elements(), randomizer);

        CUDA::Memory::copy(data.get(), d_data.get(), elements * batches * sizeof(float));
        CUDA::Memory::copy(expected.get(), d_results.get(), elements * batches * sizeof(float));
        CUDA::Memory::copy(values.get(), d_values.get(), batches * sizeof(float));
        CUDA::Memory::copy(array.get(), d_array.get(), elements * sizeof(float));

        {
            NOA_BENCHMARK_CUDA_SCOPE(stream, "multiplyByValue - one batch - multiply array by single value");
            CUDA::Math::multiplyByValue(d_data.get(), value, d_results.get(), elements, stream);
        }

        {
            NOA_BENCHMARK_CUDA_SCOPE(stream, "multiplyByValue - multiply one value per batch");
            CUDA::Math::multiplyByValue(d_data.get(), d_values.get(), d_results.get(), elements, batches, stream);
        }

        {
            NOA_BENCHMARK_CUDA_SCOPE(stream, "multiplyByArray - element-wise multiply");
            CUDA::Math::multiplyByArray(d_data.get(), d_array.get(), d_results.get(), elements, batches, stream);
        }
    }

    {
        Noa::Session::logger.info("padded memory, multiplyBy* functions: shape:{}, elements:{}, batches:{}",
                                  shape, elements, batches);
        Memory::PtrHost<float> data(elements * batches);
        Memory::PtrHost<float> expected(elements * batches);
        Memory::PtrHost<float> values(batches);
        Memory::PtrHost<float> array(elements);
        float value = randomizer.get();

        size3_t shape_batch = {shape.x, shape.y, shape.z * batches};
        CUDA::Memory::PtrDevicePadded<float> d_data(shape_batch);
        CUDA::Memory::PtrDevice<float> d_values(batches);
        CUDA::Memory::PtrDevicePadded<float> d_array(shape);
        CUDA::Memory::PtrDevicePadded<float> d_results(shape_batch);
        Memory::PtrHost<float> cuda_results(elements * batches);

        Test::initDataRandom(data.get(), data.elements(), randomizer);
        Test::initDataZero(expected.get(), expected.elements());
        Test::initDataRandom(values.get(), values.elements(), randomizer);
        Test::initDataRandom(array.get(), array.elements(), randomizer);

        CUDA::Memory::copy(data.get(), shape.x * sizeof(float), d_data.get(), d_data.pitch(), shape_batch);
        CUDA::Memory::copy(expected.get(), shape.x * sizeof(float), d_results.get(), d_results.pitch(), shape_batch);
        CUDA::Memory::copy(values.get(), d_values.get(), batches * sizeof(float));
        CUDA::Memory::copy(array.get(), shape.x * sizeof(float), d_array.get(), d_array.pitch(), shape);

        {
            NOA_BENCHMARK_CUDA_SCOPE(stream, "multiplyByValue - one batch - multiply array by single value");
            CUDA::Math::multiplyByValue(d_data.get(), d_data.pitchElements(), value,
                                        d_results.get(), d_results.pitchElements(), shape, stream);
        }

        {
            NOA_BENCHMARK_CUDA_SCOPE(stream, "multiplyByValue - multiply one value per batch");
            CUDA::Math::multiplyByValue(d_data.get(), d_data.pitchElements(), d_values.get(),
                                        d_results.get(), d_results.pitchElements(), shape, batches, stream);
        }

        {
            NOA_BENCHMARK_CUDA_SCOPE(stream, "multiplyByArray - element-wise multiply");
            CUDA::Math::multiplyByArray(d_data.get(), d_data.pitchElements(), d_array.get(), d_array.pitchElements(),
                                        d_results.get(), d_array.pitchElements(), shape, batches, stream);
        }
    }
}
