#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Traits.h"
#include "noa/gpu/cuda/util/Block.cuh"
#include "noa/gpu/cuda/util/Pointers.h"

namespace noa::cuda::util::ewise::details {
    struct BinaryConfig {
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

    // Extracting the value_type from the U doesn't work: while const seems to be removed according to static_assert,
    // the compiler complains the value is const and cannot be reassigned. Passing the value_type explicitly seems
    // to work.
    template<typename T, typename U, typename UV, typename V, typename Binary, int VEC_SIZE, bool RESTRICT>
    __global__ __launch_bounds__(BinaryConfig::BLOCK_SIZE)
    void binaryValue1D_(accessor_t<RESTRICT, const T*> lhs, uint2_t lhs_stride, U rhs,
                        accessor_t<RESTRICT, V*> output, uint2_t output_stride,
                        uint elements, Binary binary_op) {
        using iptr_t = typename accessor_t<RESTRICT, const T*>::ptr_type;
        using optr_t = typename accessor_t<RESTRICT, V*>::ptr_type;
        const uint batch = blockIdx.y;
        const uint base = BinaryConfig::BLOCK_WORK_SIZE * blockIdx.x;

        iptr_t lhs_ptr = lhs.get() + batch * lhs_stride[0];
        optr_t output_ptr = output.get() + batch * output_stride[0];

        UV value;
        if constexpr (traits::is_accessor_v<U>)
            value = rhs[batch];
        else
            value = rhs;

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int i = 0; i < BinaryConfig::ELEMENTS_PER_THREAD; ++i) {
                const uint gid = base + BinaryConfig::BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements)
                    output_ptr[gid * output_stride[1]] =
                            static_cast<V>(binary_op(lhs_ptr[gid * lhs_stride[1]], value));
            }
        } else {
            lhs_ptr += base;
            output_ptr += base;
            const uint remaining = elements - base;
            if (remaining < BinaryConfig::BLOCK_WORK_SIZE) {
                for (int i = 0; i < BinaryConfig::ELEMENTS_PER_THREAD; ++i) {
                    const uint offset = BinaryConfig::BLOCK_SIZE * i + threadIdx.x;
                    if (offset < remaining)
                        output_ptr[offset] = static_cast<V>(binary_op(lhs_ptr[offset], value));
                }
            } else {
                T args[BinaryConfig::ELEMENTS_PER_THREAD];
                V results[BinaryConfig::ELEMENTS_PER_THREAD];
                block::vectorizedLoad<BinaryConfig::BLOCK_SIZE, BinaryConfig::ELEMENTS_PER_THREAD, VEC_SIZE>(
                        lhs_ptr, args, threadIdx.x);
                #pragma unroll
                for (uint i = 0; i < BinaryConfig::ELEMENTS_PER_THREAD; ++i)
                    results[i] = static_cast<V>(binary_op(args[i], value));
                block::vectorizedStore<BinaryConfig::BLOCK_SIZE, BinaryConfig::ELEMENTS_PER_THREAD, VEC_SIZE>(
                        results, output_ptr, threadIdx.x);
            }
        }
    }

    template<typename T, typename U, typename UV, typename V, typename BinaryOp, bool RESTRICT>
    __global__ __launch_bounds__(BinaryConfig::BLOCK_SIZE)
    void binaryValue4D_(accessor_t<RESTRICT, const T*> lhs, uint4_t lhs_stride, U rhs,
                        accessor_t<RESTRICT, V*> output, uint4_t output_stride,
                        uint2_t shape, BinaryOp binary_op, uint blocks_x) {
        using iptr_t = typename accessor_t<RESTRICT, const T*>::ptr_type;
        using optr_t = typename accessor_t<RESTRICT, V*>::ptr_type;
        iptr_t lhs_ptr = lhs.get();
        optr_t output_ptr = output.get();

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid(blockIdx.z,
                         blockIdx.y,
                         BinaryConfig::BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BinaryConfig::BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x);
        lhs_ptr += indexing::at(gid[0], gid[1], lhs_stride);
        output_ptr += indexing::at(gid[0], gid[1], output_stride);

        UV value;
        if constexpr (traits::is_accessor_v<U>)
            value = rhs[gid[0]];
        else
            value = rhs;

        #pragma unroll
        for (int k = 0; k < BinaryConfig::ELEMENTS_PER_THREAD_2D; ++k) {
            #pragma unroll
            for (int l = 0; l < BinaryConfig::ELEMENTS_PER_THREAD_2D; ++l) {
                const uint ik = gid[2] + BinaryConfig::BLOCK_SIZE_2D.y * k;
                const uint il = gid[3] + BinaryConfig::BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1])
                    output_ptr[ik * output_stride[2] + il * output_stride[3]] =
                            static_cast<V>(binary_op(lhs_ptr[ik * lhs_stride[2] + il * lhs_stride[3]], value));
            }
        }
    }

    template<typename T, typename U, typename V, typename Binary, int VEC_SIZE, bool RESTRICT>
    __global__ __launch_bounds__(BinaryConfig::BLOCK_SIZE)
    void binaryArray1D_(accessor_t<RESTRICT, const T*> lhs, uint2_t lhs_stride,
                        accessor_t<RESTRICT, const U*> rhs, uint2_t rhs_stride,
                        accessor_t<RESTRICT, V*> output, uint2_t output_stride,
                        uint elements, Binary binary_op) {
        using iptr_t = typename accessor_t<RESTRICT, const T*>::ptr_type;
        using aptr_t = typename accessor_t<RESTRICT, const U*>::ptr_type;
        using optr_t = typename accessor_t<RESTRICT, V*>::ptr_type;
        const uint batch = blockIdx.y;
        const uint base = BinaryConfig::BLOCK_WORK_SIZE * blockIdx.x;
        iptr_t lhs_ptr = lhs.get() + batch * lhs_stride[0];
        aptr_t rhs_ptr = rhs.get() + batch * rhs_stride[0];
        optr_t output_ptr = output.get() + batch * output_stride[0];

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int i = 0; i < BinaryConfig::ELEMENTS_PER_THREAD; ++i) {
                const uint gid = base + BinaryConfig::BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements) {
                    output_ptr[gid * output_stride[1]] =
                            static_cast<V>(binary_op(lhs_ptr[gid * lhs_stride[1]],
                                                     rhs_ptr[gid * rhs_stride[1]]));
                }
            }
        } else {
            const uint remaining = elements - base;
            lhs_ptr += base;
            rhs_ptr += base;
            output_ptr += base;
            if (remaining < BinaryConfig::BLOCK_WORK_SIZE) {
                #pragma unroll
                for (int i = 0; i < BinaryConfig::ELEMENTS_PER_THREAD; ++i) {
                    const uint offset = BinaryConfig::BLOCK_SIZE * i + threadIdx.x;
                    if (offset < remaining)
                        output_ptr[offset] = static_cast<V>(binary_op(lhs_ptr[offset], rhs_ptr[offset]));
                }
            } else {
                T ilhs[BinaryConfig::ELEMENTS_PER_THREAD];
                U irhs[BinaryConfig::ELEMENTS_PER_THREAD];
                V results[BinaryConfig::ELEMENTS_PER_THREAD];
                block::vectorizedLoad<BinaryConfig::BLOCK_SIZE, BinaryConfig::ELEMENTS_PER_THREAD, VEC_SIZE>(
                        lhs_ptr, ilhs, threadIdx.x);
                block::vectorizedLoad<BinaryConfig::BLOCK_SIZE, BinaryConfig::ELEMENTS_PER_THREAD, VEC_SIZE>(
                        rhs_ptr, irhs, threadIdx.x);
                #pragma unroll
                for (uint i = 0; i < BinaryConfig::ELEMENTS_PER_THREAD; ++i)
                    results[i] = static_cast<V>(binary_op(ilhs[i], irhs[i]));
                block::vectorizedStore<BinaryConfig::BLOCK_SIZE, BinaryConfig::ELEMENTS_PER_THREAD, VEC_SIZE>(
                        results, output_ptr, threadIdx.x);
            }
        }
    }

    template<typename T, typename U, typename V, typename BinaryOp, bool RESTRICT>
    __global__ __launch_bounds__(BinaryConfig::BLOCK_SIZE)
    void binaryArray4D_(accessor_t<RESTRICT, const T*> lhs, uint4_t lhs_stride,
                        accessor_t<RESTRICT, const U*> rhs, uint4_t rhs_stride,
                        accessor_t<RESTRICT, V*> output, uint4_t output_stride,
                        uint2_t shape, BinaryOp binary_op, uint blocks_x) {
        using iptr_t = typename accessor_t<RESTRICT, const T*>::ptr_type;
        using aptr_t = typename accessor_t<RESTRICT, const U*>::ptr_type;
        using optr_t = typename accessor_t<RESTRICT, V*>::ptr_type;
        iptr_t lhs_ptr = lhs.get();
        aptr_t rhs_ptr = rhs.get();
        optr_t output_ptr = output.get();

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid(blockIdx.z,
                         blockIdx.y,
                         BinaryConfig::BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BinaryConfig::BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x);
        lhs_ptr += indexing::at(gid[0], gid[1], lhs_stride);
        rhs_ptr += indexing::at(gid[0], gid[1], rhs_stride);
        output_ptr += indexing::at(gid[0], gid[1], output_stride);

        #pragma unroll
        for (int k = 0; k < BinaryConfig::ELEMENTS_PER_THREAD_2D; ++k) {
            #pragma unroll
            for (int l = 0; l < BinaryConfig::ELEMENTS_PER_THREAD_2D; ++l) {
                const uint ik = gid[2] + BinaryConfig::BLOCK_SIZE_2D.y * k;
                const uint il = gid[3] + BinaryConfig::BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1]) {
                    output_ptr[ik * output_stride[2] + il * output_stride[3]] =
                            static_cast<V>(binary_op(lhs_ptr[ik * lhs_stride[2] + il * lhs_stride[3]],
                                                     rhs_ptr[ik * rhs_stride[2] + il * rhs_stride[3]]));
                }
            }
        }
    }
}

namespace noa::cuda::util::ewise {
    /// Apply a binary operator, element-wise.
    /// \tparam RESTRICT        Whether \p lhs and \p output can be accessed using the __restrict__ attribute,
    ///                         implying no aliasing between these two pointers.
    /// \param[in] name         Name of the function. Used for logging if kernel launch fails.
    /// \param[in] lhs          On the \b device. Left-hand side.
    /// \param rhs              Right-hand side.
    /// \param[out] output      On the \b device. Transformed array.
    /// \param elements         Number of elements to transform.
    /// \param[in,out] stream   Stream on which to enqueue this function. No synchronization is performed on the stream.
    /// \param binary_op        Binary operator, such as, op(\p T, \p U) -> \p V.
    ///                         The output is explicitly casted to \p V.
    /// \note This function is asynchronous relative to the host and may return before completion.
    ///       One must make sure \p lhs and \p output stay valid until completion.
    template<bool RESTRICT = false, typename T, typename U, typename V, typename BinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<U>>>
    void binary(const char* name,
                const T* lhs, U rhs, V* output,
                size_t elements, Stream& stream, BinaryOp binary_op) {
        NOA_PROFILE_FUNCTION();
        using namespace details;
        accessor_t<RESTRICT, const T*> lhs_accessor(lhs);
        accessor_t<RESTRICT, V*> output_accessor(output);

        const uint2_t stride{0, 1};
        const uint blocks = noa::math::divideUp(static_cast<uint>(elements), BinaryConfig::BLOCK_WORK_SIZE);
        const int vec_size = std::min(maxVectorCount(lhs), maxVectorCount(output));
        const LaunchConfig config{blocks, BinaryConfig::BLOCK_SIZE};
        if (vec_size == 4) {
            return stream.enqueue(name, binaryValue1D_<T, U, U, V, BinaryOp, 4, RESTRICT>, config,
                                  lhs_accessor, stride, rhs, output_accessor, stride, elements, binary_op);
        } else if (vec_size == 2) {
            return stream.enqueue(name, binaryValue1D_<T, U, U, V, BinaryOp, 2, RESTRICT>, config,
                                  lhs_accessor, stride, rhs, output_accessor, stride, elements, binary_op);
        } else {
            return stream.enqueue(name, binaryValue1D_<T, U, U, V, BinaryOp, 1, RESTRICT>, config,
                                  lhs_accessor, stride, rhs, output_accessor, stride, elements, binary_op);
        }
    }

    /// Apply a binary operator, element-wise.
    /// \tparam RESTRICT        Whether \p lhs and \p output can be accessed using the __restrict__ attribute,
    ///                         implying no aliasing between these two pointers.
    /// \param[in] name         Name of the function. Used for logging if kernel launch fails.
    /// \param[in] lhs          On the \b device. Input array to transform.
    /// \param lhs_stride       Rightmost stride of \p lhs.
    /// \param[in] rhs          Right-hand side argument for the binary operator.
    /// \param[out] output      On the \b device. Transformed array.
    /// \param output_stride    Rightmost stride of \p output.
    /// \param shape            Rightmost shape of \p lhs and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function. No synchronization is performed on the stream.
    /// \param binary_op        Binary operator. The output is explicitly casted to \p V.
    /// \note This function is asynchronous relative to the host and may return before completion.
    ///       One must make sure \p lhs and \p output stay valid until completion.
    template<bool RESTRICT = false, typename T, typename U, typename V, typename BinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<U>>>
    void binary(const char* name,
                const T* lhs, size4_t lhs_stride, U rhs,
                V* output, size4_t output_stride, size4_t shape,
                Stream& stream, BinaryOp binary_op) {
        NOA_PROFILE_FUNCTION();
        using namespace details;
        accessor_t<RESTRICT, const T*> lhs_accessor(lhs);
        accessor_t<RESTRICT, V*> output_accessor(output);
        using value_t = std::remove_const_t<U>;

        const bool4_t is_contiguous = indexing::isContiguous(lhs_stride, shape) &&
                                      indexing::isContiguous(output_stride, shape);
        if (is_contiguous[1] && is_contiguous[2]) { // 1D-like
            // Keep batches separated in a different Grid.Y if they're not contiguous or
            // if we need to apply a different value to each batch.
            const uint4_t uint_shape{shape};
            uint elements, blocks_y;
            if (!is_contiguous[0]) {
                elements = uint_shape[1] * uint_shape[2] * uint_shape[3];
                blocks_y = shape[0];
            } else {
                elements = uint_shape.elements();
                blocks_y = 1;
            }
            const dim3 blocks(noa::math::divideUp(elements, BinaryConfig::BLOCK_WORK_SIZE), blocks_y);

            uint vec_size = is_contiguous[3] ? std::min(maxVectorCount(lhs), maxVectorCount(output)) : 1;
            if (blocks.y > 1) // make sure the beginning of each batch preserves the alignment
                vec_size = lhs_stride[0] % vec_size || output_stride[0] % vec_size ? 1 : vec_size;

            const uint2_t uint_lhs_stride{lhs_stride[0], lhs_stride[3]};
            const uint2_t uint_output_stride{output_stride[0], output_stride[3]};
            const LaunchConfig config{blocks, BinaryConfig::BLOCK_SIZE};
            if (vec_size == 4) {
                return stream.enqueue(name, binaryValue1D_<T, value_t, value_t, V, BinaryOp, 4, RESTRICT>, config,
                                      lhs_accessor, uint_lhs_stride, rhs,
                                      output_accessor, uint_output_stride, elements, binary_op);
            } else if (vec_size == 2) {
                return stream.enqueue(name, binaryValue1D_<T, value_t, value_t, V, BinaryOp, 2, RESTRICT>, config,
                                      lhs_accessor, uint_lhs_stride, rhs,
                                      output_accessor, uint_output_stride, elements, binary_op);
            } else {
                return stream.enqueue(name, binaryValue1D_<T, value_t, value_t, V, BinaryOp, 1, RESTRICT>, config,
                                      lhs_accessor, uint_lhs_stride, rhs,
                                      output_accessor, uint_output_stride, elements, binary_op);
            }
        } else {
            const uint2_t i_shape{shape.get() + 2};
            const uint blocks_x = noa::math::divideUp(i_shape[1], BinaryConfig::BLOCK_WORK_SIZE_2D.x);
            const uint blocks_y = noa::math::divideUp(i_shape[0], BinaryConfig::BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, BinaryConfig::BLOCK_SIZE_2D};
            stream.enqueue(name, binaryValue4D_<T, value_t, value_t, V, BinaryOp, RESTRICT>, config,
                           lhs_accessor, uint4_t{lhs_stride}, rhs,
                           output_accessor, uint4_t{output_stride}, i_shape, binary_op, blocks_x);
        }
    }

    /// Apply a binary operator, element-wise.
    /// \tparam RESTRICT        Whether \p lhs and \p output can be accessed using the __restrict__ attribute,
    ///                         implying no aliasing between these two pointers.
    /// \param[in] name         Name of the function. Used for logging if kernel launch fails.
    /// \param[in] lhs          On the \b device. Input array to transform.
    /// \param lhs_stride       Rightmost stride of \p lhs.
    /// \param[in] rhs          Right-hand side argument for the binary operator. One value per batch.
    /// \param[out] output      On the \b device. Transformed array.
    /// \param output_stride    Rightmost stride of \p output.
    /// \param shape            Rightmost shape of \p lhs and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function. No synchronization is performed on the stream.
    /// \param binary_op        Binary operator. The output is explicitly casted to \p V.
    /// \note This function is asynchronous relative to the host and may return before completion.
    ///       One must make sure \p lhs, \p rhs and \p output stay valid until completion.
    template<bool RESTRICT = false, typename T, typename U, typename V, typename BinaryOp>
    void binary(const char* name,
                const T* lhs, size4_t lhs_stride, const U* rhs,
                V* output, size4_t output_stride, size4_t shape,
                Stream& stream, BinaryOp binary_op) {
        NOA_PROFILE_FUNCTION();
        using namespace details;
        accessor_t<RESTRICT, const T*> lhs_accessor(lhs);
        accessor_t<RESTRICT, V*> output_accessor(output);

        using accessor_type = accessor_t<RESTRICT, const U*>;
        accessor_type rhs_accessor(rhs);

        const bool4_t is_contiguous = indexing::isContiguous(lhs_stride, shape) &&
                                      indexing::isContiguous(output_stride, shape);
        if (is_contiguous[1] && is_contiguous[2]) { // 1D-like
            // Keep batches separated in a different Grid.Y if they're not contiguous or
            // if we need to apply a different value to each batch.
            const uint4_t uint_shape{shape};
            uint elements, blocks_y;
            elements = uint_shape[1] * uint_shape[2] * uint_shape[3];
            blocks_y = shape[0];
            const dim3 blocks(noa::math::divideUp(elements, BinaryConfig::BLOCK_WORK_SIZE), blocks_y);

            uint vec_size = is_contiguous[3] ? std::min(maxVectorCount(lhs), maxVectorCount(output)) : 1;
            if (blocks.y > 1) // make sure the beginning of each batch preserves the alignment
                vec_size = lhs_stride[0] % vec_size || output_stride[0] % vec_size ? 1 : vec_size;

            const uint2_t uint_lhs_stride{lhs_stride[0], lhs_stride[3]};
            const uint2_t uint_output_stride{output_stride[0], output_stride[3]};
            const LaunchConfig config{blocks, BinaryConfig::BLOCK_SIZE};
            if (vec_size == 4) {
                return stream.enqueue(name, binaryValue1D_<T, accessor_type, U, V, BinaryOp, 4, RESTRICT>, config,
                                      lhs_accessor, uint_lhs_stride, rhs_accessor,
                                      output_accessor, uint_output_stride, elements, binary_op);
            } else if (vec_size == 2) {
                return stream.enqueue(name, binaryValue1D_<T, accessor_type, U, V, BinaryOp, 2, RESTRICT>, config,
                                      lhs_accessor, uint_lhs_stride, rhs_accessor,
                                      output_accessor, uint_output_stride, elements, binary_op);
            } else {
                return stream.enqueue(name, binaryValue1D_<T, accessor_type, U, V, BinaryOp, 1, RESTRICT>, config,
                                      lhs_accessor, uint_lhs_stride, rhs_accessor,
                                      output_accessor, uint_output_stride, elements, binary_op);
            }
        } else {
            const uint2_t i_shape{shape.get() + 2};
            const uint blocks_x = noa::math::divideUp(i_shape[1], BinaryConfig::BLOCK_WORK_SIZE_2D.x);
            const uint blocks_y = noa::math::divideUp(i_shape[0], BinaryConfig::BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, BinaryConfig::BLOCK_SIZE_2D};
            stream.enqueue(name, binaryValue4D_<T, accessor_type, U, V, BinaryOp, RESTRICT>, config,
                           lhs_accessor, uint4_t{lhs_stride}, rhs_accessor,
                           output_accessor, uint4_t{output_stride}, i_shape, binary_op, blocks_x);
        }
    }

    /// Apply a binary operator, element-wise.
    /// \tparam RESTRICT        Whether \p lhs, \p rhs and \p output can be accessed using the __restrict__ attribute,
    ///                         implying no aliasing between these pointers.
    /// \param[in] name         Name of the function. Used for logging if kernel launch fails.
    /// \param[in] lhs          On the \b device. Left-hand side argument.
    /// \param[in] rhs          On the \b device. Right-hand side argument.
    /// \param[out] output      On the \b device. Transformed array.
    /// \param elements         Number of elements to transform.
    /// \param[in,out] stream   Stream on which to enqueue this function. No synchronization is performed on the stream.
    /// \param binary_op        Binary operator, such as, op(\p T, \p U) -> \p V.
    ///                         The output is explicitly casted to \p V.
    /// \note This function is asynchronous relative to the host and may return before completion.
    ///       One must make sure \p lhs, \p rhs and \p output stay valid until completion.
    template<bool RESTRICT = false, typename T, typename U, typename V, typename BinaryOp>
    void binary(const char* name,
                const T* lhs,
                const U* rhs,
                const V* output,
                size_t elements, Stream& stream, BinaryOp binary_op) {
        NOA_PROFILE_FUNCTION();
        using namespace details;
        accessor_t<RESTRICT, const T*> lhs_accessor(lhs);
        accessor_t<RESTRICT, const U*> rhs_accessor(rhs);
        accessor_t<RESTRICT, V*> output_accessor(output);

        const uint2_t stride{0, 1};
        const uint blocks = noa::math::divideUp(static_cast<uint>(elements), BinaryConfig::BLOCK_WORK_SIZE);
        const int vec_size = std::min({maxVectorCount(lhs),
                                       maxVectorCount(rhs),
                                       maxVectorCount(output)});
        const LaunchConfig config{blocks, BinaryConfig::BLOCK_SIZE};
        if (vec_size == 4) {
            return stream.enqueue(name, binaryArray1D_<T, U, V, BinaryOp, 4, RESTRICT>, config,
                                  lhs_accessor, stride, rhs_accessor, stride,
                                  output_accessor, stride, elements, binary_op);
        } else if (vec_size == 2) {
            return stream.enqueue(name, binaryArray1D_<T, U, V, BinaryOp, 2, RESTRICT>, config,
                                  lhs_accessor, stride, rhs_accessor, stride,
                                  output_accessor, stride, elements, binary_op);
        } else {
            return stream.enqueue(name, binaryArray1D_<T, U, V, BinaryOp, 1, RESTRICT>, config,
                                  lhs_accessor, stride, rhs_accessor, stride,
                                  output_accessor, stride, elements, binary_op);
        }
    }

    /// Apply a binary operator, element-wise.
    /// \tparam RESTRICT        Whether \p lhs, \p rhs and \p output can be accessed using the __restrict__ attribute,
    ///                         implying no aliasing between these pointers.
    /// \param[in] name         Name of the function. Used for logging if kernel launch fails.
    /// \param[in] lhs          On the \b device. Left-hand side argument.
    /// \param lhs_stride       Rightmost stride of \p lhs.
    /// \param[in] rhs          On the \b device. Right-hand side argument.
    /// \param rhs_stride       Rightmost stride of \p rhs.
    /// \param[out] output      On the \b device. Transformed array.
    /// \param output_stride    Rightmost stride of \p output.
    /// \param shape            Rightmost shape of \p lhs, \p rhs and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function. No synchronization is performed on the stream.
    /// \param binary_op        Binary operator, such as, op(\p T, \p U) -> \p V.
    ///                         The output is explicitly casted to \p V.
    /// \note This function is asynchronous relative to the host and may return before completion.
    ///       One must make sure \p lhs, \p rhs and \p output stay valid until completion.
    template<bool RESTRICT = false, typename T, typename U, typename V, typename BinaryOp>
    void binary(const char* name,
                const T* lhs, size4_t lhs_stride,
                const U* rhs, size4_t rhs_stride,
                V* output, size4_t output_stride,
                size4_t shape, Stream& stream, BinaryOp binary_op) {
        NOA_PROFILE_FUNCTION();
        using namespace details;
        accessor_t<RESTRICT, const T*> lhs_accessor(lhs);
        accessor_t<RESTRICT, const U*> rhs_accessor(rhs);
        accessor_t<RESTRICT, V*> output_accessor(output);
        const bool4_t is_contiguous = indexing::isContiguous(lhs_stride, shape) &&
                                      indexing::isContiguous(rhs_stride, shape) &&
                                      indexing::isContiguous(output_stride, shape);

        if (is_contiguous[1] && is_contiguous[2]) { // 1D-like
            // Keep batches separated in a different Grid.Y if they're not contiguous.
            const uint4_t uint_shape{shape};
            const uint elements = is_contiguous[0] ? uint_shape.elements() : uint3_t{uint_shape.get() + 1}.elements();
            const dim3 blocks(noa::math::divideUp(elements, BinaryConfig::BLOCK_WORK_SIZE),
                              is_contiguous[0] ? 1 : shape[0]);

            uint vec_size = is_contiguous[3] ? std::min({maxVectorCount(lhs),
                                                         maxVectorCount(rhs),
                                                         maxVectorCount(output)}) : 1;
            if (blocks.y > 1) { // make sure the beginning of each batch preserves the alignment
                const bool is_not_multiple = lhs_stride[0] % vec_size ||
                                             rhs_stride[0] % vec_size ||
                                             output_stride[0] % vec_size;
                vec_size = is_not_multiple ? 1 : vec_size;
            }

            const uint2_t uint_lhs_stride{lhs_stride[0], lhs_stride[3]};
            const uint2_t uint_rhs_stride{rhs_stride[0], rhs_stride[3]};
            const uint2_t uint_output_stride{output_stride[0], output_stride[3]};
            const LaunchConfig config{blocks, BinaryConfig::BLOCK_SIZE};
            if (vec_size == 4) {
                return stream.enqueue(name, binaryArray1D_<T, U, V, BinaryOp, 4, RESTRICT>, config,
                                      lhs_accessor, uint_lhs_stride, rhs_accessor, uint_rhs_stride,
                                      output_accessor, uint_output_stride, elements, binary_op);
            } else if (vec_size == 2) {
                return stream.enqueue(name, binaryArray1D_<T, U, V, BinaryOp, 2, RESTRICT>, config,
                                      lhs_accessor, uint_lhs_stride, rhs_accessor, uint_rhs_stride,
                                      output_accessor, uint_output_stride, elements, binary_op);
            } else {
                return stream.enqueue(name, binaryArray1D_<T, U, V, BinaryOp, 1, RESTRICT>, config,
                                      lhs_accessor, uint_lhs_stride, rhs_accessor, uint_rhs_stride,
                                      output_accessor, uint_output_stride, elements, binary_op);
            }
        } else {
            const uint2_t i_shape{shape.get() + 2};
            const uint blocks_x = noa::math::divideUp(i_shape[1], BinaryConfig::BLOCK_WORK_SIZE_2D.x);
            const uint blocks_y = noa::math::divideUp(i_shape[0], BinaryConfig::BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, BinaryConfig::BLOCK_SIZE_2D};
            stream.enqueue(name, binaryArray4D_<T, U, V, BinaryOp, RESTRICT>, config,
                           lhs_accessor, uint4_t{lhs_stride}, rhs_accessor, uint4_t{rhs_stride},
                           output_accessor, uint4_t{output_stride}, i_shape, binary_op, blocks_x);
        }
    }
}
