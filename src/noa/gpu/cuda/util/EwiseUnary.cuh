#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Traits.h"
#include "noa/gpu/cuda/util/Block.cuh"
#include "noa/gpu/cuda/util/Pointers.h"

// -- Unary -- //
namespace noa::cuda::util::ewise::details {
    struct UnaryConfig {
        static constexpr uint ELEMENTS_PER_THREAD = 4;
        static constexpr uint BLOCK_SIZE = 128;
        static constexpr uint BLOCK_WORK_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;

        // Still the same threads per block and elements per thread, but using a 2D block.
        // The goal is waste as fewer threads as possible, assuming 2D/3D/4D arrays have a
        // similar number of elements in their two innermost dimensions.
        static constexpr uint ELEMENTS_PER_THREAD_2D = ELEMENTS_PER_THREAD / 2;
        static constexpr dim3 BLOCK_SIZE_2D{32, BLOCK_SIZE / 32, 1};
        static constexpr dim3 BLOCK_WORK_SIZE_2D{BLOCK_SIZE_2D.x * ELEMENTS_PER_THREAD_2D,
                                                 BLOCK_SIZE_2D.y * ELEMENTS_PER_THREAD_2D, 1};
    };

    template<typename T, typename U, typename UnaryOp, int VEC_SIZE, bool RESTRICT>
    __global__ __launch_bounds__(UnaryConfig::BLOCK_SIZE)
    void unary1D_(accessor_t<RESTRICT, const T*> input, uint2_t input_stride,
                  accessor_t<RESTRICT, U*> output, uint2_t output_stride,
                  uint elements, UnaryOp unary_op) {
        using iptr_t = typename accessor_t<RESTRICT, const T*>::ptr_type;
        using optr_t = typename accessor_t<RESTRICT, U*>::ptr_type;
        const uint base = UnaryConfig::BLOCK_WORK_SIZE * blockIdx.x;
        iptr_t input_ptr = input.get();
        optr_t output_ptr = output.get();

        input_ptr += blockIdx.y * input_stride[0];
        output_ptr += blockIdx.y * output_stride[0];

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int i = 0; i < UnaryConfig::ELEMENTS_PER_THREAD; ++i) {
                const uint gid = base + UnaryConfig::BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements)
                    output_ptr[gid * output_stride[1]] = static_cast<U>(unary_op(input_ptr[gid * input_stride[1]]));
            }
        } else { // assume contiguous
            input_ptr += base;
            output_ptr += base;
            const uint remaining = elements - base;
            if (remaining < UnaryConfig::BLOCK_WORK_SIZE) {
                #pragma unroll
                for (int i = 0; i < UnaryConfig::ELEMENTS_PER_THREAD; ++i) {
                    const uint gid = UnaryConfig::BLOCK_SIZE * i + threadIdx.x;
                    if (gid < remaining)
                        output_ptr[gid] = static_cast<U>(unary_op(input_ptr[gid]));
                }
            } else { // this block has BLOCK_WORK_SIZE elements to handle, so we can use vectorized memory accesses
                T args[UnaryConfig::ELEMENTS_PER_THREAD];
                U results[UnaryConfig::ELEMENTS_PER_THREAD];

                block::vectorizedLoad<UnaryConfig::BLOCK_SIZE, UnaryConfig::ELEMENTS_PER_THREAD, VEC_SIZE>(
                        input_ptr, args, threadIdx.x);
                #pragma unroll
                for (uint i = 0; i < UnaryConfig::ELEMENTS_PER_THREAD; ++i)
                    results[i] = static_cast<U>(unary_op(args[i]));
                block::vectorizedStore<UnaryConfig::BLOCK_SIZE, UnaryConfig::ELEMENTS_PER_THREAD, VEC_SIZE>(
                        results, output_ptr, threadIdx.x);
            }
        }
    }

    template<typename T, typename U, typename UnaryOp, bool RESTRICT>
    __global__ __launch_bounds__(UnaryConfig::BLOCK_SIZE)
    void unary4D_(accessor_t<RESTRICT, const T*> input, uint4_t input_stride,
                  accessor_t<RESTRICT, U*> output, uint4_t output_stride,
                  uint2_t shape, UnaryOp unary_op, uint blocks_x) {
        using iptr_t = typename accessor_t<RESTRICT, const T*>::ptr_type;
        using optr_t = typename accessor_t<RESTRICT, U*>::ptr_type;
        iptr_t input_ptr = input.get();
        optr_t output_ptr = output.get();

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid(blockIdx.z,
                         blockIdx.y,
                         UnaryConfig::BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         UnaryConfig::BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x);
        input_ptr += indexing::at(gid[0], gid[1], input_stride);
        output_ptr += indexing::at(gid[0], gid[1], output_stride);

        #pragma unroll
        for (int k = 0; k < UnaryConfig::ELEMENTS_PER_THREAD_2D; ++k) {
            #pragma unroll
            for (int l = 0; l < UnaryConfig::ELEMENTS_PER_THREAD_2D; ++l) {
                const uint ik = gid[2] + UnaryConfig::BLOCK_SIZE_2D.y * k;
                const uint il = gid[3] + UnaryConfig::BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1])
                    output_ptr[ik * output_stride[2] + il * output_stride[3]] =
                            static_cast<U>(unary_op(input_ptr[ik * input_stride[2] + il * input_stride[3]]));
            }
        }
    }
}

namespace noa::cuda::util::ewise {
    /// Apply an unary operator, element-wise.
    /// \tparam RESTRICT        Whether \p input and \p output can be accessed using the __restrict__ attribute,
    ///                         implying no aliasing between these two pointers.
    /// \param[in] name         Name of the function. Used for logging if kernel launch fails.
    /// \param[in] input        On the \b device. Input array to transform.
    /// \param[out] output      On the \b device. Transformed array.
    /// \param elements         Number of elements to transform.
    /// \param[in,out] stream   Stream on which to enqueue this function. No synchronization is performed on the stream.
    /// \param unary_op         Unary operator, such as, op(\p T) -> \p U.
    ///                         The output is explicitly casted to \p U.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool RESTRICT = false, typename T, typename U, typename UnaryOp>
    void unary(const char* name,
               const shared_t<const T[]>& input,
               const shared_t<U[]>& output,
               size_t elements, Stream& stream, UnaryOp unary_op) {
        NOA_PROFILE_FUNCTION();
        using namespace details;
        accessor_t<RESTRICT, const T*> input_accessor(input.get());
        accessor_t<RESTRICT, U*> output_accessor(output.get());

        const uint2_t stride{0, 1};
        const uint blocks = noa::math::divideUp(static_cast<uint>(elements), UnaryConfig::BLOCK_WORK_SIZE);
        const int vec_size = std::min(maxVectorCount(input.get()), maxVectorCount(output.get()));
        if (vec_size == 4) {
            return stream.enqueue(name, unary1D_<T, U, UnaryOp, 4, RESTRICT>, {blocks, UnaryConfig::BLOCK_SIZE},
                                  input_accessor, stride, output_accessor, stride, elements, unary_op);
        } else if (vec_size == 2) {
            return stream.enqueue(name, unary1D_<T, U, UnaryOp, 2, RESTRICT>, {blocks, UnaryConfig::BLOCK_SIZE},
                                  input_accessor, stride, output_accessor, stride, elements, unary_op);
        } else {
            return stream.enqueue(name, unary1D_<T, U, UnaryOp, 1, RESTRICT>, {blocks, UnaryConfig::BLOCK_SIZE},
                                  input_accessor, stride, output_accessor, stride, elements, unary_op);
        }
        stream.attach(input, output);
    }

    /// Apply an unary operator, element-wise.
    /// \tparam RESTRICT        Whether \p input and \p output can be accessed using the __restrict__ attribute,
    ///                         implying no aliasing between these two pointers.
    /// \param[in] name         Name of the function. Used for logging if kernel launch fails.
    /// \param[in] input        On the \b device. Input array to transform.
    /// \param input_stride     Rightmost stride, of \p input.
    /// \param[out] output      On the \b device. Transformed array.
    /// \param output_stride    Rightmost stride, of \p output.
    /// \param elements         Rightmost shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function. No synchronization is performed on the stream.
    /// \param unary_op         Unary operator, such as, op(\p T) -> \p U.
    ///                         The output is explicitly casted to \p U.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool RESTRICT = false, typename T, typename U, typename UnaryOp>
    void unary(const char* name,
               const shared_t<const T[]>& input, size4_t input_stride,
               const shared_t<U[]>& output, size4_t output_stride,
               size4_t shape, Stream& stream, UnaryOp unary_op) {
        NOA_PROFILE_FUNCTION();
        using namespace details;
        accessor_t<RESTRICT, const T*> input_accessor(input.get());
        accessor_t<RESTRICT, U*> output_accessor(output.get());

        const bool4_t is_contiguous = indexing::isContiguous(input_stride, shape) &&
                                      indexing::isContiguous(output_stride, shape);
        if (is_contiguous[1] && is_contiguous[2]) { // 1D-like
            // Keep batches separated in a different Grid.Y if they're not contiguous.
            const uint4_t uint_shape{shape};
            const uint elements = is_contiguous[0] ? uint_shape.elements() : uint3_t{uint_shape.get() + 1}.elements();
            const dim3 blocks(noa::math::divideUp(elements, UnaryConfig::BLOCK_WORK_SIZE),
                              is_contiguous[0] ? 1 : shape[0]);

            uint vec_size = is_contiguous[3] ? std::min(maxVectorCount(input.get()), maxVectorCount(output.get())) : 1;
            if (blocks.y > 1) // make sure the beginning of each batch preserves the alignment
                vec_size = input_stride[0] % vec_size || output_stride[0] % vec_size ? 1 : vec_size;

            const uint2_t uint_input_stride{input_stride[0], input_stride[3]};
            const uint2_t uint_output_stride{output_stride[0], output_stride[3]};
            const LaunchConfig config{blocks, UnaryConfig::BLOCK_SIZE};
            if (vec_size == 4) {
                return stream.enqueue(name, unary1D_<T, U, UnaryOp, 4, RESTRICT>, config,
                                      input_accessor, uint_input_stride,
                                      output_accessor, uint_output_stride,
                                      elements, unary_op);
            } else if (vec_size == 2) {
                return stream.enqueue(name, unary1D_<T, U, UnaryOp, 2, RESTRICT>, config,
                                      input_accessor, uint_input_stride,
                                      output_accessor, uint_output_stride,
                                      elements, unary_op);
            } else {
                return stream.enqueue(name, unary1D_<T, U, UnaryOp, 1, RESTRICT>, config,
                                      input_accessor, uint_input_stride,
                                      output_accessor, uint_output_stride,
                                      elements, unary_op);
            }
        } else { // multi-dimensional, non-contiguous array
            const uint2_t i_shape{shape.get() + 2};
            const uint blocks_x = noa::math::divideUp(i_shape[1], UnaryConfig::BLOCK_WORK_SIZE_2D.x);
            const uint blocks_y = noa::math::divideUp(i_shape[0], UnaryConfig::BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, UnaryConfig::BLOCK_SIZE_2D};
            stream.enqueue(name, unary4D_<T, U, UnaryOp, RESTRICT>, config,
                           input_accessor, uint4_t{input_stride},
                           output_accessor, uint4_t{output_stride},
                           i_shape, unary_op, blocks_x);
        }
        stream.attach(input, output);
    }
}
