#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Transpose.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Transpose.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cuda::memory::transpose()", "[assets][noa][cuda][memory]") {
    const path_t path_base = test::NOA_DATA_PATH / "memory";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["transpose"]["tests"];
    io::ImageFile file;
    cuda::Stream stream;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto filename_input = path_base / test["input"].as<path_t>();
        const auto filename_expected = path_base / test["expected"].as<path_t>();
        const auto permutation = test["permutation"].as<uint4_t>();
        const auto inplace = test["inplace"].as<bool>();

        file.open(filename_input, io::READ);
        const size4_t shape = file.shape();
        const size4_t stride = shape.stride();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<float> data(elements);
        cpu::memory::PtrHost<float> expected(elements);

        file.open(filename_input, io::READ);
        file.readAll(data.get());
        file.open(filename_expected, io::READ);
        file.readAll(expected.get());

        const size4_t output_shape = cuda::memory::transpose(shape, permutation);
        const size4_t output_stride = output_shape.stride();

        cuda::memory::PtrDevicePadded<float> d_data(shape);
        cuda::memory::PtrDevicePadded<float> d_result(output_shape);
        cpu::memory::PtrHost<float> result(elements);

        if (inplace) {
            cuda::memory::copy(data.get(), stride, d_data.get(), d_data.stride(), shape, stream);
            cuda::memory::transpose(d_data.get(), d_data.stride(), shape,
                                    d_data.get(), d_data.stride(), permutation, stream);
            cuda::memory::copy(d_data.get(), d_data.stride(), data.get(), output_stride, output_shape, stream);
            stream.synchronize();

            REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), data.get(), elements, 1e-8));
        } else {
            cuda::memory::copy(data.get(), stride, d_data.get(), d_data.stride(), shape, stream);
            cuda::memory::transpose(d_data.get(), d_data.stride(), shape,
                                    d_result.get(), d_result.stride(), permutation, stream);
            cuda::memory::copy(d_result.get(), d_result.stride(), result.get(), output_stride, output_shape, stream);
            stream.synchronize();

            REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), result.get(), elements, 1e-8));
        }
    }
}

TEMPLATE_TEST_CASE("cuda::memory::transpose() - random shapes - contiguous layouts", "[noa][cuda][memory]",
                   int, int64_t, double, cfloat_t) {
    const std::array<uint4_t, 6> permutations{uint4_t{0, 1, 2, 3},
                                              uint4_t{0, 1, 3, 2},
                                              uint4_t{0, 3, 1, 2},
                                              uint4_t{0, 3, 2, 1},
                                              uint4_t{0, 2, 1, 3},
                                              uint4_t{0, 2, 3, 1}};
    const uint ndim = GENERATE(2U, 3U);
    const uint number = GENERATE(0U, 1U, 2U, 3U, 4U, 5U);
    const uint4_t permutation = permutations[number];

    test::Randomizer<TestType> randomizer(-5., 5.);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();
    cpu::memory::PtrHost<TestType> h_data(elements);
    test::randomize(h_data.get(), elements, randomizer);

    cpu::Stream cpu_stream;
    cuda::Stream gpu_stream;
    cuda::memory::PtrDevice<TestType> d_data(elements);
    cuda::memory::PtrDevice<TestType> d_result(elements);
    cpu::memory::PtrHost<TestType> h_cuda_result(elements);
    cpu::memory::PtrHost<TestType> h_result(elements);

    const size4_t output_shape = cpu::memory::transpose(shape, permutation);
    const size4_t output_stride = output_shape.stride();

    if (ndim == 2 && !(all(permutation == uint4_t{0, 1, 2, 3}) || all(permutation == uint4_t{0, 1, 3, 2}))) {
        // While this is technically OK, it doesn't make much sense to test these...
        return;
    }

    cpu::memory::transpose(h_data.get(), stride, shape, h_result.get(), output_stride, permutation, cpu_stream);
    cuda::memory::copy(h_data.get(), d_data.get(), elements, gpu_stream);
    cuda::memory::transpose(d_data.get(), stride, shape, d_result.get(), output_stride, permutation, gpu_stream);
    cuda::memory::copy(d_result.get(), h_cuda_result.get(), elements, gpu_stream);
    gpu_stream.synchronize();
    cpu_stream.synchronize();
    REQUIRE(test::Matcher(test::MATCH_ABS, h_result.get(), h_cuda_result.get(), elements, 1e-8));
}
