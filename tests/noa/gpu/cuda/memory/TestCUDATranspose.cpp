#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Transpose.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Transpose.h>

#include <noa/common/files/MRCFile.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cuda::memory::transpose()", "[noa][cuda][memory]") {
    int test_number = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    INFO(test_number);

    path_t filename_data;
    path_t filename_expected;
    size3_t shape;
    uint3_t permutation;
    bool in_place;
    test::assets::memory::getTransposeParams(test_number, &filename_data, &filename_expected,
                                             &shape, &permutation, &in_place);

    size_t elements = getElements(shape);
    memory::PtrHost<float> data(elements);
    memory::PtrHost<float> expected(elements);

    MRCFile file;
    file.open(filename_data, io::READ);
    file.readAll(data.get());
    file.open(filename_expected, io::READ);
    file.readAll(expected.get());

    size3_t new_shape = memory::transpose(shape, permutation);
    cuda::Stream stream(cuda::Stream::SERIAL);
    cuda::memory::PtrDevicePadded<float> d_data(shape);
    cuda::memory::PtrDevicePadded<float> d_result(new_shape);
    memory::PtrHost<float> result(elements);

    if (in_place) {
        cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape, stream);
        cuda::memory::transpose(d_data.get(), d_data.pitch(), shape,
                                d_data.get(), d_data.pitch(), permutation, 1, stream);
        cuda::memory::copy(d_data.get(), d_data.pitch(), data.get(), new_shape.x, new_shape, stream);
        cuda::Stream::synchronize(stream);

        float diff = test::getDifference(expected.get(), data.get(), elements);
        REQUIRE(diff == 0);
    } else {
        cuda::memory::copy(data.get(), shape.x, d_data.get(), d_data.pitch(), shape, stream);
        cuda::memory::transpose(d_data.get(), d_data.pitch(), shape,
                                d_result.get(), d_result.pitch(), permutation, 1, stream);
        cuda::memory::copy(d_result.get(), d_result.pitch(), result.get(), new_shape.x, new_shape, stream);
        cuda::Stream::synchronize(stream);

        float diff = test::getDifference(expected.get(), result.get(), elements);
        REQUIRE(diff == 0);
    }
}

TEMPLATE_TEST_CASE("cuda::memory::transpose() - random shapes - contiguous layouts", "[noa][cuda][memory]",
                   int, long long, float, double) {
    std::array<uint3_t, 6> permutations{uint3_t(0, 1, 2),
                                        uint3_t(0, 2, 1),
                                        uint3_t(1, 0, 2),
                                        uint3_t(1, 2, 0),
                                        uint3_t(2, 0, 1),
                                        uint3_t(2, 1, 0)};
    uint ndim = GENERATE(2U, 3U);
    uint number = GENERATE(0U, 1U, 2U, 3U, 4U, 5U);
    uint3_t permutation = permutations[number];

    test::Randomizer<TestType> randomizer(-5., 5.);
    size3_t shape = test::getRandomShape(ndim);
    uint batches = test::IntRandomizer<uint>(1, 4).get();
    size_t elements = getElements(shape) * batches;
    memory::PtrHost<TestType> h_data(elements);
    test::initDataRandom(h_data.get(), elements, randomizer);

    cuda::Stream stream(cuda::Stream::SERIAL);
    cuda::memory::PtrDevice<TestType> d_data(elements);
    cuda::memory::PtrDevice<TestType> d_result(elements);
    memory::PtrHost<TestType> h_cuda_result(elements);
    memory::PtrHost<TestType> h_result(elements);

    if (ndim == 2 && !(all(permutation == uint3_t(0, 1, 2)) || all(permutation == uint3_t(1, 0, 2)))) {
        REQUIRE_THROWS_AS((cuda::memory::transpose(d_data.get(), shape, d_data.get(), permutation, 1, stream)),
                          noa::Exception);
        REQUIRE_THROWS_AS((memory::transpose(d_data.get(), shape, d_data.get(), permutation, 1)),
                          noa::Exception);
        return;
    }

    cuda::memory::copy(h_data.get(), d_data.get(), elements, stream);
    cuda::memory::transpose(d_data.get(), shape, d_result.get(), permutation, batches, stream);
    cuda::memory::copy(d_result.get(), h_cuda_result.get(), elements, stream);
    memory::transpose(h_data.get(), shape, h_result.get(), permutation, batches);
    cuda::Stream::synchronize(stream);

    TestType diff = test::getDifference(h_result.get(), h_cuda_result.get(), elements);
    REQUIRE(diff == 0);
}
