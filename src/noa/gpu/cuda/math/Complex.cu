#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
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
    void decompose4D_(const Complex<T>* __restrict__ complex, uint4_t complex_stride,
                      T* __restrict__ real, uint4_t real_stride,
                      T* __restrict__ imag, uint4_t imag_stride,
                      uint2_t shape, uint blocks_x) {
        const uint2_t index = indexes(blockIdx.x, blocks_x);
        const int4_t gid(blockIdx.z,
                         blockIdx.y,
                         BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x);
        complex += at(gid[0], gid[1], complex_stride);
        real += at(gid[0], gid[1], real_stride);
        imag += at(gid[0], gid[1], imag_stride);

        #pragma unroll
        for (int k = 0; k < ELEMENTS_PER_THREAD_2D.y; ++k) {
            #pragma unroll
            for (int l = 0; l < ELEMENTS_PER_THREAD_2D.x; ++l) {
                const uint ik = gid[2] + BLOCK_SIZE_2D.y * k;
                const uint il = gid[3] + BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1]) {
                    const Complex<T> tmp = complex[ik * complex_stride[2] + il * complex_stride[3]];
                    real[ik * real_stride[2] + il * real_stride[3]] = tmp.real;
                    imag[ik * imag_stride[2] + il * imag_stride[3]] = tmp.imag;
                }
            }
        }
    }
}

namespace noa::cuda::math {
    template<typename T>
    void decompose(const Complex<T>* complex, size4_t complex_stride,
                   T* real, size4_t real_stride,
                   T* imag, size4_t imag_stride,
                   size4_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(reinterpret_cast<const T*>(complex) != real);
        NOA_ASSERT(reinterpret_cast<const T*>(complex) != imag);

        const bool4_t is_contiguous = isContiguous(complex_stride, shape) &&
                                      isContiguous(real_stride, shape) &&
                                      isContiguous(imag_stride, shape);
        if (is_contiguous[0] && is_contiguous[1] && is_contiguous[2]) {
            const uint elements = shape.elements();
            const uint blocks = noa::math::divideUp(static_cast<uint>(elements), BLOCK_WORK_SIZE);
            const int vec_size = is_contiguous[3] ? std::min(std::min(util::maxVectorCount(real),
                                                                      util::maxVectorCount(imag)),
                                                             util::maxVectorCount(complex)) : 1;
            if (vec_size == 4) {
                return stream.enqueue("memory::decompose", decompose1D_<T, 4>, {blocks, BLOCK_SIZE},
                                      complex, complex_stride[3], real, real_stride[3], imag, imag_stride[3], elements);
            } else if (vec_size == 2) {
                return stream.enqueue("memory::decompose", decompose1D_<T, 2>, {blocks, BLOCK_SIZE},
                                      complex, complex_stride[3], real, real_stride[3], imag, imag_stride[3], elements);
            } else {
                return stream.enqueue("memory::decompose", decompose1D_<T, 1>, {blocks, BLOCK_SIZE},
                                      complex, complex_stride[3], real, real_stride[3], imag, imag_stride[3], elements);
            }
        } else {
            NOA_PROFILE_FUNCTION();
            const uint2_t i_shape{shape.get() + 2};
            const uint blocks_x = noa::math::divideUp(i_shape[1], BLOCK_WORK_SIZE_2D.x);
            const uint blocks_y = noa::math::divideUp(i_shape[0], BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            stream.enqueue("memory::decompose", decompose4D_<T>, {blocks, BLOCK_SIZE_2D},
                           complex, uint4_t{complex_stride}, real, uint4_t{real_stride},
                           imag, uint4_t{imag_stride}, i_shape, blocks_x);
        }
    }

    template<typename T>
    void complex(const T* real, size4_t real_stride,
                 const T* imag, size4_t imag_stride,
                 Complex<T>* output, size4_t output_stride,
                 size4_t shape, Stream& stream) {
        NOA_ASSERT(reinterpret_cast<T*>(output) != real);
        NOA_ASSERT(reinterpret_cast<T*>(output) != imag);
        util::ewise::binary<true>("memory::complex", real, real_stride, imag, imag_stride, output, output_stride,
                                  shape, stream, []__device__(T r, T i) { return Complex<T>{r, i}; });
    }

    #define NOA_INSTANTIATE_COMPLEX_(T)                                                                     \
    template void decompose<T>(const Complex<T>*, size4_t, T*, size4_t, T*, size4_t, size4_t, Stream&);     \
    template void complex<T>(const T*, size4_t, const T*, size4_t, Complex<T>*, size4_t, size4_t, Stream&)

    NOA_INSTANTIATE_COMPLEX_(half_t);
    NOA_INSTANTIATE_COMPLEX_(float);
    NOA_INSTANTIATE_COMPLEX_(double);
}
