//#include <noa/gpu/cuda/math/Reductions.h>
//
//#include <noa/cpu/memory/PtrHost.h>
//#include <noa/gpu/cuda/memory/PtrDevice.h>
//#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
//#include <noa/gpu/cuda/memory/Copy.h>
//
//#include "Helpers.h"
//#include <catch2/catch.hpp>
//
//using namespace noa;
//
//TEMPLATE_TEST_CASE("cuda::Math: sumMean", "[noa][cuda][math]", float, double, cfloat_t, cdouble_t) {
//    size_t batches = 5;
//    size3_t shape(512, 512, 128);
//    size3_t shape_batched(shape.x, shape.y, shape.z * batches);
//    size_t elements = noa::elements(shape); // 33554432
//
//    test::Randomizer<TestType> randomizer(1., 10.);
//    cpu::memory::PtrHost<TestType> h_data(elements * batches);
//    test::randomize(h_data.get(), h_data.elements(), randomizer);
//
//    cuda::memory::PtrDevice<TestType> d_results(2 * batches);
//
//    {
//        NOA_BENCHMARK_HEADER("cuda::Math: sumMean - contiguous");
//
//        cuda::Stream stream;
//        cuda::memory::PtrDevice<TestType> d_data(elements * batches);
//        cuda::memory::copy(h_data.get(), d_data.get(), d_data.size(), stream);
//        cuda::math::sumMean(d_data.get(), d_results.get(), d_results.get(), 65536, 1, stream); // warm up
//
//        noa::Session::logger.info("Type: {}, Batches 5", noa::string::typeName<TestType>());
//        size_t i_elements;
//        {
//            i_elements = 1;
//            NOA_BENCHMARK_CUDA_SCOPE(stream, "elements:{}", i_elements);
//            cuda::math::sumMean(d_data.get(), d_results.get(), d_results.get() + batches, i_elements, batches, stream);
//        }
//        {
//            i_elements = 512;
//            NOA_BENCHMARK_CUDA_SCOPE(stream, "elements:{}", i_elements);
//            cuda::math::sumMean(d_data.get(), d_results.get(), d_results.get() + batches, i_elements, batches, stream);
//        }
//        {
//            i_elements = 1024;
//            NOA_BENCHMARK_CUDA_SCOPE(stream, "elements:{}", i_elements);
//            cuda::math::sumMean(d_data.get(), d_results.get(), d_results.get() + batches, i_elements, batches, stream);
//        }
//        {
//            i_elements = 8192;
//            NOA_BENCHMARK_CUDA_SCOPE(stream, "elements:{}", i_elements);
//            cuda::math::sumMean(d_data.get(), d_results.get(), d_results.get() + batches, i_elements, batches, stream);
//        }
//        {
//            i_elements = 65536;
//            NOA_BENCHMARK_CUDA_SCOPE(stream, "elements:{}", i_elements);
//            cuda::math::sumMean(d_data.get(), d_results.get(), d_results.get() + batches, i_elements, batches, stream);
//        }
//        {
//            i_elements = 524288;
//            NOA_BENCHMARK_CUDA_SCOPE(stream, "elements:{}", i_elements);
//            cuda::math::sumMean(d_data.get(), d_results.get(), d_results.get() + batches, i_elements, batches, stream);
//        }
//        {
//            i_elements = 2097152;
//            NOA_BENCHMARK_CUDA_SCOPE(stream, "elements:{}", i_elements);
//            cuda::math::sumMean(d_data.get(), d_results.get(), d_results.get() + batches, i_elements, batches, stream);
//        }
//        {
//            i_elements = 16777216;
//            NOA_BENCHMARK_CUDA_SCOPE(stream, "elements:{}", i_elements);
//            cuda::math::sumMean(d_data.get(), d_results.get(), d_results.get() + batches, i_elements, batches, stream);
//        }
//        {
//            i_elements = 33554432;
//            NOA_BENCHMARK_CUDA_SCOPE(stream, "elements:{}", i_elements);
//            cuda::math::sumMean(d_data.get(), d_results.get(), d_results.get() + batches, i_elements, batches, stream);
//        }
//    }
//
//    {
//        NOA_BENCHMARK_HEADER("cuda::Math: sumMean - padded");
//
//        cuda::memory::PtrDevicePadded<TestType> d_data(shape_batched);
//
//        cuda::Stream stream;
//        cuda::memory::copy(h_data.get(), shape.x, d_data.get(), d_data.pitch(), shape_batched, stream);
//        cuda::math::sumMean(d_data.get(), d_data.pitch(), d_results.get(), d_results.get(),
//                            size3_t(256, 256, 1), 1, stream); // warm up
//
//        noa::Session::logger.info("Type: {}, Batches 5", noa::string::typeName<TestType>());
//        size3_t i_shape;
//        {
//            i_shape = 1;
//            NOA_BENCHMARK_CUDA_SCOPE(stream, "shape:{}, elements:{}", i_shape, noa::elements(i_shape));
//            cuda::math::sumMean(d_data.get(), d_data.pitch(), d_results.get(), d_results.get() + batches,
//                                i_shape, batches, stream);
//        }
//        {
//            i_shape = {32, 16, 1};
//            NOA_BENCHMARK_CUDA_SCOPE(stream, "shape:{}, elements:{}", i_shape, noa::elements(i_shape));
//            cuda::math::sumMean(d_data.get(), d_data.pitch(), d_results.get(), d_results.get() + batches,
//                                i_shape, batches, stream);
//        }
//        {
//            i_shape = {32, 32, 1};
//            NOA_BENCHMARK_CUDA_SCOPE(stream, "shape:{}, elements:{}", i_shape, noa::elements(i_shape));
//            cuda::math::sumMean(d_data.get(), d_data.pitch(), d_results.get(), d_results.get() + batches,
//                                i_shape, batches, stream);
//        }
//        {
//            i_shape = {128, 64, 1};
//            NOA_BENCHMARK_CUDA_SCOPE(stream, "shape:{}, elements:{}", i_shape, noa::elements(i_shape));
//            cuda::math::sumMean(d_data.get(), d_data.pitch(), d_results.get(), d_results.get() + batches,
//                                i_shape, batches, stream);
//        }
//        {
//            i_shape = {256, 256, 1};
//            NOA_BENCHMARK_CUDA_SCOPE(stream, "shape:{}, elements:{}", i_shape, noa::elements(i_shape));
//            cuda::math::sumMean(d_data.get(), d_data.pitch(), d_results.get(), d_results.get() + batches,
//                                i_shape, batches, stream);
//        }
//        {
//            i_shape = {64, 64, 128};
//            NOA_BENCHMARK_CUDA_SCOPE(stream, "shape:{}, elements:{}", i_shape, noa::elements(i_shape));
//            cuda::math::sumMean(d_data.get(), d_data.pitch(), d_results.get(), d_results.get() + batches,
//                                i_shape, batches, stream);
//        }
//        {
//            i_shape = {1024, 64, 32};
//            NOA_BENCHMARK_CUDA_SCOPE(stream, "shape:{}, elements:{}", i_shape, noa::elements(i_shape));
//            cuda::math::sumMean(d_data.get(), d_data.pitch(), d_results.get(), d_results.get() + batches,
//                                i_shape, batches, stream);
//        }
//        {
//            i_shape = {256, 256, 256};
//            NOA_BENCHMARK_CUDA_SCOPE(stream, "shape:{}, elements:{}", i_shape, noa::elements(i_shape));
//            cuda::math::sumMean(d_data.get(), d_data.pitch(), d_results.get(), d_results.get() + batches,
//                                i_shape, batches, stream);
//        }
//        {
//            i_shape = {1024, 1024, 32};
//            NOA_BENCHMARK_CUDA_SCOPE(stream, "shape:{}, elements:{}", i_shape, noa::elements(i_shape));
//            cuda::math::sumMean(d_data.get(), d_data.pitch(), d_results.get(), d_results.get() + batches,
//                                i_shape, batches, stream);
//        }
//    }
//}
