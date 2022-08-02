#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/gpu/cuda/math/Complex.h"
#include "noa/gpu/cuda/util/Block.cuh"
#include "noa/gpu/cuda/util/EwiseBinary.cuh"
#include "noa/gpu/cuda/util/Pointers.h"

namespace {
    using namespace ::noa;

    constexpr uint ELEMENTS_PER_THREAD = 4;
    constexpr uint BLOCK_SIZE = 128;
    constexpr uint BLOCK_WORK_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;

    constexpr dim3 BLOCK_SIZE_2D(32, 8);
    constexpr dim3 ELEMENTS_PER_THREAD_2D(1, 4);
    constexpr dim3 BLOCK_WORK_SIZE_2D(BLOCK_SIZE_2D.x * ELEMENTS_PER_THREAD_2D.x,
                                      BLOCK_SIZE_2D.y * ELEMENTS_PER_THREAD_2D.y);

    template<typename T, int VEC_SIZE>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void decompose1D_(const Complex<T>* __restrict__ complex, uint complex_stride,
                      T* __restrict__ real, uint real_stride,
                      T* __restrict__ imag, uint imag_stride,
                      uint elements) {
        const uint base = BLOCK_WORK_SIZE * blockIdx.x;

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                const uint gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements) {
                    const Complex<T> tmp = complex[gid * complex_stride];
                    real[gid * real_stride] = tmp.real;
                    imag[gid * imag_stride] = tmp.imag;
                }
            }
        } else {
            NOA_ASSERT(complex_stride == 1 && real_stride == 1 && imag_stride == 1);
            const uint remaining = elements - base;
            complex += base * complex_stride;
            real += base * real_stride;
            imag += base * imag_stride;

            if (remaining < BLOCK_WORK_SIZE) {
                #pragma unroll
                for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    const uint gid = BLOCK_SIZE * i + threadIdx.x;
                    if (gid < remaining) {
                        const Complex<T> tmp = complex[gid * complex_stride];
                        real[gid * real_stride] = tmp.real;
                        imag[gid * imag_stride] = tmp.imag;
                    }
                }
            } else {
                Complex<T> args[ELEMENTS_PER_THREAD];
                T ireal[ELEMENTS_PER_THREAD];
                T iimag[ELEMENTS_PER_THREAD];

                cuda::util::block::vectorizedLoad<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(complex, args, threadIdx.x);
                #pragma unroll
                for (uint i = 0; i < ELEMENTS_PER_THREAD; ++i) {
                    ireal[i] = args[i].real;
                    iimag[i] = args[i].imag;
                }
                cuda::util::block::vectorizedStore<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(ireal, real, threadIdx.x);
                cuda::util::block::vectorizedStore<BLOCK_SIZE, ELEMENTS_PER_THREAD, VEC_SIZE>(iimag, imag, threadIdx.x);
            }
        }
    }

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void decompose4D_(const Complex<T>* __restrict__ complex, uint4_t complex_strides,
                      T* __restrict__ real, uint4_t real_strides,
                      T* __restrict__ imag, uint4_t imag_strides,
                      uint2_t shape, uint blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        complex += indexing::at(gid[0], gid[1], complex_strides);
        real += indexing::at(gid[0], gid[1], real_strides);
        imag += indexing::at(gid[0], gid[1], imag_strides);

        #pragma unroll
        for (int k = 0; k < ELEMENTS_PER_THREAD_2D.y; ++k) {
            #pragma unroll
            for (int l = 0; l < ELEMENTS_PER_THREAD_2D.x; ++l) {
                const uint ik = gid[2] + BLOCK_SIZE_2D.y * k;
                const uint il = gid[3] + BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1]) {
                    const Complex<T> tmp = complex[ik * complex_strides[2] + il * complex_strides[3]];
                    real[ik * real_strides[2] + il * real_strides[3]] = tmp.real;
                    imag[ik * imag_strides[2] + il * imag_strides[3]] = tmp.imag;
                }
            }
        }
    }
}

namespace noa::cuda::math {
    template<typename T, typename>
    void decompose(const shared_t<Complex<T>[]>& input, size4_t input_strides,
                   const shared_t<T[]>& real, size4_t real_strides,
                   const shared_t<T[]>& imag, size4_t imag_strides,
                   size4_t shape, Stream& stream) {
        NOA_ASSERT(reinterpret_cast<const T*>(input.get()) != real.get());
        NOA_ASSERT(reinterpret_cast<const T*>(input.get()) != imag.get());

        if (all(input_strides > 0)) {
            const size4_t order = indexing::order(input_strides, shape);
            input_strides = indexing::reorder(input_strides, order);
            real_strides = indexing::reorder(real_strides, order);
            imag_strides = indexing::reorder(imag_strides, order);
            shape = indexing::reorder(shape, order);
        }

        const bool4_t is_contiguous = indexing::isContiguous(input_strides, shape) &&
                                      indexing::isContiguous(real_strides, shape) &&
                                      indexing::isContiguous(imag_strides, shape);
        if (is_contiguous[0] && is_contiguous[1] && is_contiguous[2]) {
            const uint elements = shape.elements();
            const uint blocks = noa::math::divideUp(static_cast<uint>(elements), BLOCK_WORK_SIZE);
            const int vec_size = is_contiguous[3] ? std::min({util::maxVectorCount(real.get()),
                                                              util::maxVectorCount(imag.get()),
                                                              util::maxVectorCount(input.get())}) : 1;
            if (vec_size == 4) {
                return stream.enqueue("memory::decompose", decompose1D_<T, 4>, {blocks, BLOCK_SIZE},
                                      input.get(), input_strides[3],
                                      real.get(), real_strides[3],
                                      imag.get(), imag_strides[3], elements);
            } else if (vec_size == 2) {
                return stream.enqueue("memory::decompose", decompose1D_<T, 2>, {blocks, BLOCK_SIZE},
                                      input.get(), input_strides[3],
                                      real.get(), real_strides[3],
                                      imag.get(), imag_strides[3], elements);
            } else {
                return stream.enqueue("memory::decompose", decompose1D_<T, 1>, {blocks, BLOCK_SIZE},
                                      input.get(), input_strides[3],
                                      real.get(), real_strides[3],
                                      imag.get(), imag_strides[3], elements);
            }
        } else {
            const uint2_t i_shape(shape.get(2));
            const uint blocks_x = noa::math::divideUp(i_shape[1], BLOCK_WORK_SIZE_2D.x);
            const uint blocks_y = noa::math::divideUp(i_shape[0], BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            stream.enqueue("memory::decompose", decompose4D_<T>, {blocks, BLOCK_SIZE_2D},
                           input.get(), uint4_t(input_strides),
                           real.get(), uint4_t(real_strides),
                           imag.get(), uint4_t(imag_strides), i_shape, blocks_x);
        }
        stream.attach(input, real, imag);
    }

    template<typename T, typename U>
    void complex(const shared_t<T[]>& real, size4_t real_strides,
                 const shared_t<T[]>& imag, size4_t imag_strides,
                 const shared_t<Complex<T>[]>& output, size4_t output_strides,
                 size4_t shape, Stream& stream) {
        NOA_ASSERT(reinterpret_cast<T*>(output.get()) != real.get());
        NOA_ASSERT(reinterpret_cast<T*>(output.get()) != imag.get());
        util::ewise::binary<true>("memory::complex",
                                  real.get(), real_strides,
                                  imag.get(), imag_strides,
                                  output.get(), output_strides,
                                  shape, true, stream, []__device__(T r, T i) { return Complex<T>{r, i}; });
        stream.attach(real, imag, output);
    }

    #define NOA_INSTANTIATE_COMPLEX_(T)                                                                                                                         \
    template void decompose<T, void>(const shared_t<Complex<T>[]>&, size4_t, const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&);   \
    template void complex<T, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, const shared_t<Complex<T>[]>&, size4_t, size4_t, Stream&)

    NOA_INSTANTIATE_COMPLEX_(half_t);
    NOA_INSTANTIATE_COMPLEX_(float);
    NOA_INSTANTIATE_COMPLEX_(double);
}
