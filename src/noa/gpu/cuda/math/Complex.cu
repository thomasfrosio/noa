#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/math/Complex.h"
#include "noa/gpu/cuda/util/Block.cuh"
#include "noa/gpu/cuda/util/EwiseBinary.cuh"
#include "noa/gpu/cuda/util/Pointers.h"

namespace {
    using namespace ::noa;

    constexpr uint32_t ELEMENTS_PER_THREAD = 4;
    constexpr uint32_t BLOCK_SIZE = 128;
    constexpr uint32_t BLOCK_WORK_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;

    constexpr dim3 BLOCK_SIZE_2D(32, 8);
    constexpr dim3 ELEMENTS_PER_THREAD_2D(1, 4);
    constexpr dim3 BLOCK_WORK_SIZE_2D(BLOCK_SIZE_2D.x * ELEMENTS_PER_THREAD_2D.x,
                                      BLOCK_SIZE_2D.y * ELEMENTS_PER_THREAD_2D.y);

    template<typename T, int32_t VEC_SIZE>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void decompose1D_(AccessorRestrict<const Complex<T>, 1, uint32_t> complex,
                      AccessorRestrict<T, 1, uint32_t> real,
                      AccessorRestrict<T, 1, uint32_t> imag,
                      uint32_t elements) {
        const uint32_t base = BLOCK_WORK_SIZE * blockIdx.x;

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int32_t i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                const uint32_t gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements) {
                    const Complex<T> tmp = complex[gid];
                    real[gid] = tmp.real;
                    imag[gid] = tmp.imag;
                }
            }
        } else {
            NOA_ASSERT(complex.stride(0) == 1 && real.stride(0) == 1 && imag.stride(0) == 1);
            const uint32_t remaining = elements - base;
            const auto complex_ = complex.offset(base);
            const auto real_ = real.offset(base);
            const auto imag_ = imag.offset(base);

            if (remaining < BLOCK_WORK_SIZE) {
                #pragma unroll
                for (int32_t i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const uint32_t gid = BLOCK_SIZE * i + threadIdx.x;
                    if (gid < remaining) {
                        const Complex<T> tmp = complex_[gid];
                        real_[gid] = tmp.real;
                        imag_[gid] = tmp.imag;
                    }
                }
            } else {
                Complex<T> args[ELEMENTS_PER_THREAD];
                T ireal[ELEMENTS_PER_THREAD];
                T iimag[ELEMENTS_PER_THREAD];

                using namespace ::noa::cuda::util;
                block::vectorizedLoad<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(complex_.get(), args, threadIdx.x);
                #pragma unroll
                for (int32_t i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    ireal[i] = args[i].real;
                    iimag[i] = args[i].imag;
                }
                block::vectorizedStore<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(ireal, real_.get(), threadIdx.x);
                block::vectorizedStore<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(iimag, imag_.get(), threadIdx.x);
            }
        }
    }

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void decompose4D_(AccessorRestrict<const Complex<T>, 4, uint32_t> complex,
                      AccessorRestrict<T, 4, uint32_t> real,
                      AccessorRestrict<T, 4, uint32_t> imag,
                      uint2_t shape, uint32_t blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        const auto complex_ = complex[gid[0]][gid[1]];
        const auto real_ = real[gid[0]][gid[1]];
        const auto imag_ = imag[gid[0]][gid[1]];

        #pragma unroll
        for (int32_t k = 0; k < ELEMENTS_PER_THREAD_2D.y; ++k) {
            #pragma unroll
            for (int32_t l = 0; l < ELEMENTS_PER_THREAD_2D.x; ++l) {
                const uint32_t ik = gid[2] + BLOCK_SIZE_2D.y * k;
                const uint32_t il = gid[3] + BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1]) {
                    const Complex<T> tmp = complex_(ik, il);
                    real_(ik, il) = tmp.real;
                    imag_(ik, il) = tmp.imag;
                }
            }
        }
    }
}

namespace noa::cuda::math {
    template<typename T, typename>
    void decompose(const shared_t<Complex<T>[]>& input, dim4_t input_strides,
                   const shared_t<T[]>& real, dim4_t real_strides,
                   const shared_t<T[]>& imag, dim4_t imag_strides,
                   dim4_t shape, Stream& stream) {
        NOA_ASSERT(real.get() != imag.get() && all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(real.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(imag.get(), stream.device());

        if (all(input_strides > 0)) {
            const auto order = indexing::order(input_strides, shape);
            input_strides = indexing::reorder(input_strides, order);
            real_strides = indexing::reorder(real_strides, order);
            imag_strides = indexing::reorder(imag_strides, order);
            shape = indexing::reorder(shape, order);
        }

        const bool4_t is_contiguous = indexing::isContiguous(input_strides, shape) &&
                                      indexing::isContiguous(real_strides, shape) &&
                                      indexing::isContiguous(imag_strides, shape);
        if (is_contiguous[0] && is_contiguous[1] && is_contiguous[2]) {
            const auto elements = safe_cast<uint32_t>(shape.elements());
            const uint32_t blocks = noa::math::divideUp(elements, BLOCK_WORK_SIZE);
            const int32_t vec_size = is_contiguous[3] ? std::min({util::maxVectorCount(real.get()),
                                                                  util::maxVectorCount(imag.get()),
                                                                  util::maxVectorCount(input.get())}) : 1;

            const AccessorRestrict<const Complex<T>, 1, uint32_t> input_accessor(input.get(), input_strides[3]);
            const AccessorRestrict<T, 1, uint32_t> real_accessor(real.get(), real_strides[3]);
            const AccessorRestrict<T, 1, uint32_t> imag_accessor(imag.get(), imag_strides[3]);
            if (vec_size == 4) {
                stream.enqueue("memory::decompose", decompose1D_<T, 4>, {blocks, BLOCK_SIZE},
                               input_accessor, real_accessor, imag_accessor, elements);
            } else if (vec_size == 2) {
                stream.enqueue("memory::decompose", decompose1D_<T, 2>, {blocks, BLOCK_SIZE},
                               input_accessor, real_accessor, imag_accessor, elements);
            } else {
                stream.enqueue("memory::decompose", decompose1D_<T, 1>, {blocks, BLOCK_SIZE},
                               input_accessor, real_accessor, imag_accessor, elements);
            }
        } else {
            const uint2_t i_shape(shape.get(2));
            const uint32_t blocks_x = noa::math::divideUp(i_shape[1], BLOCK_WORK_SIZE_2D.x);
            const uint32_t blocks_y = noa::math::divideUp(i_shape[0], BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);

            const AccessorRestrict<const Complex<T>, 4, uint32_t> input_accessor(input.get(), safe_cast<uint4_t>(input_strides));
            const AccessorRestrict<T, 4, uint32_t> real_accessor(real.get(), safe_cast<uint4_t>(real_strides));
            const AccessorRestrict<T, 4, uint32_t> imag_accessor(imag.get(), safe_cast<uint4_t>(imag_strides));
            stream.enqueue("memory::decompose", decompose4D_<T>, {blocks, BLOCK_SIZE_2D},
                           input_accessor, real_accessor, imag_accessor, i_shape, blocks_x);
        }
        stream.attach(input, real, imag);
    }

    template<typename T, typename U>
    void complex(const shared_t<T[]>& real, dim4_t real_strides,
                 const shared_t<T[]>& imag, dim4_t imag_strides,
                 const shared_t<Complex<T>[]>& output, dim4_t output_strides,
                 dim4_t shape, Stream& stream) {
        util::ewise::binary<true>("memory::complex",
                                  real.get(), real_strides,
                                  imag.get(), imag_strides,
                                  output.get(), output_strides,
                                  shape, true, stream, []__device__(T r, T i) { return Complex<T>{r, i}; });
        stream.attach(real, imag, output);
    }

    #define NOA_INSTANTIATE_COMPLEX_(T)                                                                                                                     \
    template void decompose<T, void>(const shared_t<Complex<T>[]>&, dim4_t, const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&);   \
    template void complex<T, void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, const shared_t<Complex<T>[]>&, dim4_t, dim4_t, Stream&)

    NOA_INSTANTIATE_COMPLEX_(half_t);
    NOA_INSTANTIATE_COMPLEX_(float);
    NOA_INSTANTIATE_COMPLEX_(double);
}
