#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Traits.h"
#include "noa/gpu/cuda/util/Block.cuh"
#include "noa/gpu/cuda/util/Pointers.h"

namespace noa::cuda::util::ewise::details {
    struct TrinaryConfig {
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
    template<typename T, typename U, typename UV, typename V, typename TrinaryOp, int VEC_SIZE, bool RESTRICT>
    __global__ __launch_bounds__(TrinaryConfig::BLOCK_SIZE)
    void trinaryValue1D_(accessor_t<RESTRICT, const T*> lhs, uint2_t lhs_stride, U mhs, U rhs,
                         accessor_t<RESTRICT, V*> output, uint2_t output_stride,
                         uint elements, TrinaryOp trinary_op) {
        using iptr_t = typename accessor_t<RESTRICT, const T*>::ptr_type;
        using optr_t = typename accessor_t<RESTRICT, V*>::ptr_type;
        const uint batch = blockIdx.y;
        const uint base = TrinaryConfig::BLOCK_WORK_SIZE * blockIdx.x;
        iptr_t lhs_ptr = lhs.get() + batch * lhs_stride[0];;
        optr_t output_ptr = output.get() + batch * output_stride[0];;

        UV value1, value2;
        if constexpr (traits::is_accessor_v<U>) {
            value1 = mhs[batch];
            value2 = rhs[batch];
        } else {
            value1 = mhs;
            value2 = rhs;
        }

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int i = 0; i < TrinaryConfig::ELEMENTS_PER_THREAD; ++i) {
                const uint gid = base + TrinaryConfig::BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements) {
                    output_ptr[gid * output_stride[1]] =
                            static_cast<V>(trinary_op(lhs_ptr[gid * lhs_stride[1]], value1, value2));
                }
            }
        } else {
            lhs_ptr += base;
            output_ptr += base;
            const uint remaining = elements - base;
            if (remaining < TrinaryConfig::BLOCK_WORK_SIZE) {
                #pragma unroll
                for (int i = 0; i < TrinaryConfig::ELEMENTS_PER_THREAD; ++i) {
                    const uint offset = TrinaryConfig::BLOCK_SIZE * i + threadIdx.x;
                    if (offset < remaining)
                        output_ptr[offset] = static_cast<V>(trinary_op(lhs_ptr[offset], value1, value2));
                }
            } else {
                T args[TrinaryConfig::ELEMENTS_PER_THREAD];
                V results[TrinaryConfig::ELEMENTS_PER_THREAD];
                block::vectorizedLoad<TrinaryConfig::BLOCK_SIZE, TrinaryConfig::ELEMENTS_PER_THREAD, VEC_SIZE>(
                        lhs_ptr, args, threadIdx.x);
                #pragma unroll
                for (uint i = 0; i < TrinaryConfig::ELEMENTS_PER_THREAD; ++i)
                    results[i] = static_cast<V>(trinary_op(args[i], value1, value2));
                block::vectorizedStore<TrinaryConfig::BLOCK_SIZE, TrinaryConfig::ELEMENTS_PER_THREAD, VEC_SIZE>(
                        results, output_ptr, threadIdx.x);
            }
        }
    }

    template<typename T, typename U, typename UV, typename V, typename TrinaryOp, bool RESTRICT>
    __global__ __launch_bounds__(TrinaryConfig::BLOCK_SIZE)
    void trinaryValue4D_(accessor_t<RESTRICT, const T*> lhs, uint4_t lhs_stride, U mhs, U rhs,
                         accessor_t<RESTRICT, V*> output, uint4_t output_stride,
                         uint2_t shape, TrinaryOp trinary_op, uint blocks_x) {
        using iptr_t = typename accessor_t<RESTRICT, const T*>::ptr_type;
        using optr_t = typename accessor_t<RESTRICT, V*>::ptr_type;
        iptr_t lhs_ptr = lhs.get();
        optr_t output_ptr = output.get();

        const uint2_t index = indexes(blockIdx.x, blocks_x);
        const int4_t gid(blockIdx.z,
                         blockIdx.y,
                         TrinaryConfig::BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         TrinaryConfig::BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x);
        lhs_ptr += at(gid[0], gid[1], lhs_stride);
        output_ptr += at(gid[0], gid[1], output_stride);

        UV value1, value2;
        if constexpr (traits::is_accessor_v<U>) {
            value1 = mhs[gid[0]];
            value2 = rhs[gid[0]];
        } else {
            value1 = mhs;
            value2 = rhs;
        }

        #pragma unroll
        for (int k = 0; k < TrinaryConfig::ELEMENTS_PER_THREAD_2D; ++k) {
            #pragma unroll
            for (int l = 0; l < TrinaryConfig::ELEMENTS_PER_THREAD_2D; ++l) {
                const uint ik = gid[2] + TrinaryConfig::BLOCK_SIZE_2D.y * k;
                const uint il = gid[3] + TrinaryConfig::BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1]) {
                    output_ptr[ik * output_stride[2] + il * output_stride[3]] =
                            static_cast<V>(trinary_op(lhs_ptr[ik * lhs_stride[2] + il * lhs_stride[3]],
                                                      value1, value2));
                }
            }
        }
    }

    template<typename T, typename U, typename V, typename W, typename Trinary, int VEC_SIZE, bool RESTRICT>
    __global__ __launch_bounds__(TrinaryConfig::BLOCK_SIZE)
    void trinaryArray1D_(accessor_t<RESTRICT, const T*> lhs, uint2_t lhs_stride,
                         accessor_t<RESTRICT, const U*> mhs, uint2_t mhs_stride,
                         accessor_t<RESTRICT, const V*> rhs, uint2_t rhs_stride,
                         accessor_t<RESTRICT, W*> output, uint2_t output_stride,
                         uint elements, Trinary trinary_op) {
        using lptr_t = typename accessor_t<RESTRICT, const T*>::ptr_type;
        using mptr_t = typename accessor_t<RESTRICT, const U*>::ptr_type;
        using rptr_t = typename accessor_t<RESTRICT, const V*>::ptr_type;
        using optr_t = typename accessor_t<RESTRICT, W*>::ptr_type;
        const uint batch = blockIdx.y;
        const uint base = TrinaryConfig::BLOCK_WORK_SIZE * blockIdx.x;
        lptr_t lhs_ptr = lhs.get() + batch * lhs_stride[0];
        mptr_t mhs_ptr = mhs.get() + batch * mhs_stride[0];
        rptr_t rhs_ptr = rhs.get() + batch * rhs_stride[0];
        optr_t output_ptr = output.get() + batch * output_stride[0];

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int i = 0; i < TrinaryConfig::ELEMENTS_PER_THREAD; ++i) {
                const uint gid = base + TrinaryConfig::BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements) {
                    output_ptr[gid * output_stride[1]] =
                            static_cast<W>(trinary_op(lhs_ptr[gid * lhs_stride[1]],
                                                      mhs_ptr[gid * mhs_stride[1]],
                                                      rhs_ptr[gid * rhs_stride[1]]));
                }
            }
        } else {
            const uint remaining = elements - base;
            lhs_ptr += base;
            mhs_ptr += base;
            rhs_ptr += base;
            output_ptr += base;
            if (remaining < TrinaryConfig::BLOCK_WORK_SIZE) {
                #pragma unroll
                for (int i = 0; i < TrinaryConfig::ELEMENTS_PER_THREAD; ++i) {
                    const uint offset = TrinaryConfig::BLOCK_SIZE * i + threadIdx.x;
                    if (offset < remaining) {
                        output_ptr[offset] = static_cast<W>(trinary_op(
                                lhs_ptr[offset], mhs_ptr[offset], rhs_ptr[offset]));
                    }
                }
            } else {
                T ilhs[TrinaryConfig::ELEMENTS_PER_THREAD];
                U imhs[TrinaryConfig::ELEMENTS_PER_THREAD];
                V irhs[TrinaryConfig::ELEMENTS_PER_THREAD];
                W results[TrinaryConfig::ELEMENTS_PER_THREAD];
                block::vectorizedLoad<TrinaryConfig::BLOCK_SIZE, TrinaryConfig::ELEMENTS_PER_THREAD, VEC_SIZE>(
                        lhs_ptr, ilhs, threadIdx.x);
                block::vectorizedLoad<TrinaryConfig::BLOCK_SIZE, TrinaryConfig::ELEMENTS_PER_THREAD, VEC_SIZE>(
                        mhs_ptr, imhs, threadIdx.x);
                block::vectorizedLoad<TrinaryConfig::BLOCK_SIZE, TrinaryConfig::ELEMENTS_PER_THREAD, VEC_SIZE>(
                        rhs_ptr, irhs, threadIdx.x);
                #pragma unroll
                for (uint i = 0; i < TrinaryConfig::ELEMENTS_PER_THREAD; ++i)
                    results[i] = static_cast<W>(trinary_op(ilhs[i], imhs[i], irhs[i]));
                block::vectorizedStore<TrinaryConfig::BLOCK_SIZE, TrinaryConfig::ELEMENTS_PER_THREAD, VEC_SIZE>(
                        results, output_ptr, threadIdx.x);
            }
        }
    }

    template<typename T, typename U, typename V, typename W, typename TrinaryOp, bool RESTRICT>
    __global__ __launch_bounds__(TrinaryConfig::BLOCK_SIZE)
    void trinaryArray4D_(accessor_t<RESTRICT, const T*> lhs, uint4_t lhs_stride,
                         accessor_t<RESTRICT, const U*> mhs, uint4_t mhs_stride,
                         accessor_t<RESTRICT, const V*> rhs, uint4_t rhs_stride,
                         accessor_t<RESTRICT, W*> output, uint4_t output_stride,
                         uint2_t shape, TrinaryOp trinary_op, uint blocks_x) {
        using lptr_t = typename accessor_t<RESTRICT, const T*>::ptr_type;
        using mptr_t = typename accessor_t<RESTRICT, const U*>::ptr_type;
        using rptr_t = typename accessor_t<RESTRICT, const V*>::ptr_type;
        using optr_t = typename accessor_t<RESTRICT, W*>::ptr_type;
        lptr_t lhs_ptr = lhs.get();
        mptr_t mhs_ptr = mhs.get();
        rptr_t rhs_ptr = rhs.get();
        optr_t output_ptr = output.get();

        const uint2_t index = indexes(blockIdx.x, blocks_x);
        const int4_t gid(blockIdx.z,
                         blockIdx.y,
                         TrinaryConfig::BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         TrinaryConfig::BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x);
        lhs_ptr += at(gid[0], gid[1], lhs_stride);
        mhs_ptr += at(gid[0], gid[1], mhs_stride);
        rhs_ptr += at(gid[0], gid[1], rhs_stride);
        output_ptr += at(gid[0], gid[1], output_stride);

        #pragma unroll
        for (int k = 0; k < TrinaryConfig::ELEMENTS_PER_THREAD_2D; ++k) {
            #pragma unroll
            for (int l = 0; l < TrinaryConfig::ELEMENTS_PER_THREAD_2D; ++l) {
                const uint ik = gid[2] + TrinaryConfig::BLOCK_SIZE_2D.y * k;
                const uint il = gid[3] + TrinaryConfig::BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1]) {
                    output_ptr[ik * output_stride[2] + il * output_stride[3]] =
                            static_cast<W>(trinary_op(lhs_ptr[ik * lhs_stride[2] + il * lhs_stride[3]],
                                                      mhs_ptr[ik * mhs_stride[2] + il * mhs_stride[3]],
                                                      rhs_ptr[ik * rhs_stride[2] + il * rhs_stride[3]]));
                }
            }
        }
    }
}

namespace noa::cuda::util::ewise {
    /// Apply a trinary operator, element-wise.
    /// \tparam RESTRICT        Whether \p lhs and \p output can be accessed using the __restrict__ attribute,
    ///                         implying no aliasing between these two pointers.
    /// \param[in] name         Name of the function. Used for logging if kernel launch fails.
    /// \param[in] lhs          On the \b device. Left-hand side argument.
    /// \param mhs              Middle-hand side argument.
    /// \param rhs              Right-hand side argument.
    /// \param[out] output      On the \b device. Transformed array.
    /// \param elements         Number of elements to transform.
    /// \param[in,out] stream   Stream on which to enqueue this function. No synchronization is performed on the stream.
    /// \param trinary_op       Trinary operator, such as, op(\p T, \p U, \p U) -> \p V.
    ///                         The output is explicitly casted to \p V.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool RESTRICT = false, typename T, typename U, typename V, typename TrinaryOp,
             typename = std::enable_if_t<!std::is_pointer_v<U>>>
    void trinary(const char* name, const T* lhs, U mhs, U rhs, V* output,
                 size_t elements, Stream& stream, TrinaryOp trinary_op) {
        NOA_PROFILE_FUNCTION();
        accessor_t<RESTRICT, const T*> lhs_accessor(lhs);
        accessor_t<RESTRICT, V*> output_accessor(output);

        using namespace details;
        const uint2_t stride{0, 1};
        const uint blocks = noa::math::divideUp(static_cast<uint>(elements), TrinaryConfig::BLOCK_WORK_SIZE);
        const int vec_size = std::min(maxVectorCount(lhs), maxVectorCount(output));
        const LaunchConfig config{blocks, TrinaryConfig::BLOCK_SIZE};
        if (vec_size == 4) {
            return stream.enqueue(name, trinaryValue1D_<T, U, U, V, TrinaryOp, 4, RESTRICT>, config,
                                  lhs_accessor, stride, mhs, rhs,
                                  output_accessor, stride, elements, trinary_op);
        } else if (vec_size == 2) {
            return stream.enqueue(name, trinaryValue1D_<T, U, U, V, TrinaryOp, 2, RESTRICT>, config,
                                  lhs_accessor, stride, mhs, rhs,
                                  output_accessor, stride, elements, trinary_op);
        } else {
            return stream.enqueue(name, trinaryValue1D_<T, U, U, V, TrinaryOp, 1, RESTRICT>, config,
                                  lhs_accessor, stride, mhs, rhs,
                                  output_accessor, stride, elements, trinary_op);
        }
    }

    /// Apply a trinary operator, element-wise.
    /// \tparam RESTRICT        Whether \p lhs and \p output can be accessed using the __restrict__ attribute,
    ///                         implying no aliasing between these two pointers.
    /// \param[in] name         Name of the function. Used for logging if kernel launch fails.
    /// \param[in] lhs          On the \b device. Left-hand side argument.
    /// \param lhs_stride       Rightmost stride of \p lhs.
    /// \param value1           Middle-hand side argument.
    ///                         If \p U is not a pointer: the same value is applied to every batch.
    ///                         If \p U is a pointer: one value per batch.
    /// \param value2           Right-hand side argument.
    ///                         If \p U is not a pointer: the same value is applied to every batch.
    ///                         If \p U is a pointer: one value per batch.
    /// \param[out] output      On the \b device. Transformed array.
    /// \param output_stride    Rightmost stride of \p output.
    /// \param shape            Rightmost shape of \p lhs and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function. No synchronization is performed on the stream.
    /// \param trinary_op       Trinary operator. The output is explicitly casted to \p V.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool RESTRICT = false, typename T, typename U, typename V, typename TrinaryOp>
    void trinary(const char* name, const T* lhs, size4_t lhs_stride, U values1, U values2,
                 V* output, size4_t output_stride, size4_t shape, Stream& stream, TrinaryOp trinary_op) {
        NOA_PROFILE_FUNCTION();
        using namespace details;

        accessor_t<RESTRICT, const T*> lhs_accessor(lhs);
        accessor_t<RESTRICT, V*> output_accessor(output);
        constexpr bool NEED_BATCH = std::is_pointer_v<U>;
        using wrapper_t = std::conditional_t<std::is_pointer_v<U>, accessor_t<RESTRICT, U>, U>;
        using value_t = std::remove_const_t<std::remove_pointer_t<U>>;
        wrapper_t values1_accessor(values1);
        wrapper_t values2_accessor(values2);

        const bool4_t is_contiguous = isContiguous(lhs_stride, shape) && isContiguous(output_stride, shape);
        if (is_contiguous[1] && is_contiguous[2]) {
            const uint4_t uint_shape{shape};
            uint elements, blocks_y;
            if (NEED_BATCH || !is_contiguous[0]) {
                elements = uint_shape[1] * uint_shape[2] * uint_shape[3];
                blocks_y = shape[0];
            } else {
                elements = uint_shape.elements();
                blocks_y = 1;
            }
            const dim3 blocks(noa::math::divideUp(elements, TrinaryConfig::BLOCK_WORK_SIZE), blocks_y);

            uint vec_size = is_contiguous[3] ? std::min(maxVectorCount(lhs), maxVectorCount(output)) : 1;
            if (blocks.y > 1)
                vec_size = lhs_stride[0] % vec_size || output_stride[0] % vec_size ? 1 : vec_size;

            const uint2_t uint_lhs_stride{lhs_stride[0], lhs_stride[3]};
            const uint2_t uint_output_stride{output_stride[0], output_stride[3]};
            const LaunchConfig config{blocks, TrinaryConfig::BLOCK_SIZE};
            if (vec_size == 4) {
                return stream.enqueue(name, trinaryValue1D_<T, wrapper_t, value_t, V, TrinaryOp, 4, RESTRICT>, config,
                                      lhs_accessor, uint_lhs_stride, values1_accessor, values2_accessor,
                                      output_accessor, uint_output_stride, elements, trinary_op);
            } else if (vec_size == 2) {
                return stream.enqueue(name, trinaryValue1D_<T, wrapper_t, value_t, V, TrinaryOp, 2, RESTRICT>, config,
                                      lhs_accessor, uint_lhs_stride, values1_accessor, values2_accessor,
                                      output_accessor, uint_output_stride, elements, trinary_op);
            } else {
                return stream.enqueue(name, trinaryValue1D_<T, wrapper_t, value_t, V, TrinaryOp, 1, RESTRICT>, config,
                                      lhs_accessor, uint_lhs_stride, values1_accessor, values2_accessor,
                                      output_accessor, uint_output_stride, elements, trinary_op);
            }
        } else {
            const uint2_t i_shape{shape.get() + 2};
            const uint blocks_x = noa::math::divideUp(i_shape[1], TrinaryConfig::BLOCK_WORK_SIZE_2D.x);
            const uint blocks_y = noa::math::divideUp(i_shape[0], TrinaryConfig::BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, TrinaryConfig::BLOCK_SIZE_2D};
            stream.enqueue(name, trinaryValue4D_<T, wrapper_t, value_t, V, TrinaryOp, RESTRICT>, config,
                           lhs_accessor, uint4_t{lhs_stride}, values1_accessor, values2_accessor,
                           output_accessor, uint4_t{output_stride}, i_shape, trinary_op, blocks_x);
        }
    }

    /// Apply a trinary operator, element-wise.
    /// \tparam RESTRICT        Whether \p lhs, \p mhs, \p rhs and \p output can be accessed using the
    ///                         __restrict__ attribute, implying no aliasing between these pointers.
    /// \param[in] name         Name of the function. Used for logging if kernel launch fails.
    /// \param[in] lhs          On the \b device. Left-hand side argument.
    /// \param[in] mhs          On the \b device. Middle-hand side argument.
    /// \param[in] rhs          On the \b device. Right-hand side argument.
    /// \param[out] output      On the \b device. Transformed array.
    /// \param elements         Number of elements to transform.
    /// \param[in,out] stream   Stream on which to enqueue this function. No synchronization is performed on the stream.
    /// \param trinary_op       Trinary operator, such as, op(\p T, \p U, \p V) -> \p W.
    ///                         The output is explicitly casted to \p W.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool RESTRICT = false, typename T, typename U, typename V, typename W, typename TrinaryOp>
    void trinary(const char* name, const T* lhs, const U* mhs, const V* rhs, W* output,
                size_t elements, Stream& stream, TrinaryOp trinary_op) {
        NOA_PROFILE_FUNCTION();
        using namespace details;
        accessor_t<RESTRICT, const T*> lhs_accessor(lhs);
        accessor_t<RESTRICT, const U*> mhs_accessor(mhs);
        accessor_t<RESTRICT, const V*> rhs_accessor(rhs);
        accessor_t<RESTRICT, W*> output_accessor(output);

        const uint2_t stride{0, 1};
        const uint blocks = noa::math::divideUp(static_cast<uint>(elements), TrinaryConfig::BLOCK_WORK_SIZE);
        const int vec_size = std::min(std::min(std::min(maxVectorCount(lhs), maxVectorCount(mhs)),
                                               maxVectorCount(rhs)), maxVectorCount(output));
        const LaunchConfig config{blocks, TrinaryConfig::BLOCK_SIZE};
        if (vec_size == 4) {
            return stream.enqueue(name, trinaryArray1D_<T, U, V, W, TrinaryOp, 4, RESTRICT>, config,
                                  lhs_accessor, stride, mhs_accessor, stride,
                                  rhs_accessor, stride, output_accessor, stride,
                                  elements, trinary_op);
        } else if (vec_size == 2) {
            return stream.enqueue(name, trinaryArray1D_<T, U, V, W, TrinaryOp, 2, RESTRICT>, config,
                                  lhs_accessor, stride, mhs_accessor, stride,
                                  rhs_accessor, stride, output_accessor, stride,
                                  elements, trinary_op);
        } else {
            return stream.enqueue(name, trinaryArray1D_<T, U, V, W, TrinaryOp, 1, RESTRICT>, config,
                                  lhs_accessor, stride, mhs_accessor, stride,
                                  rhs_accessor, stride, output_accessor, stride,
                                  elements, trinary_op);
        }
    }

    /// Apply a trinary operator, element-wise.
    /// \tparam RESTRICT        Whether \p lhs, \p mhs, \p rhs and \p output can be accessed using the
    ///                         __restrict__ attribute, implying no aliasing between these pointers.
    /// \param[in] name         Name of the function. Used for logging if kernel launch fails.
    /// \param[in] lhs          On the \b device. Left-hand side argument.
    /// \param lhs_stride       Rightmost stride of \p lhs.
    /// \param[in] mhs          On the \b device. Middle-hand side argument.
    /// \param mhs_stride       Rightmost stride of \p mhs.
    /// \param[in] rhs          On the \b device. Right-hand side argument.
    /// \param rhs_stride       Rightmost stride of \p rhs.
    /// \param[out] output      On the \b device. Transformed array.
    /// \param shape            Rightmost shape of \p lhs, \p mhs, \p rhs and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function. No synchronization is performed on the stream.
    /// \param trinary_op       Trinary operator, such as, op(\p T, \p U, \p V) -> \p W.
    ///                         The output is explicitly casted to \p W.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool RESTRICT = false, typename T, typename U, typename V, typename W, typename TrinaryOp>
    void trinary(const char* name, const T* lhs, size4_t lhs_stride,
                 const U* mhs, size4_t mhs_stride,
                 const V* rhs, size4_t rhs_stride,
                 W* output, size4_t output_stride, size4_t shape,
                 Stream& stream, TrinaryOp trinary_op) {
        NOA_PROFILE_FUNCTION();
        using namespace details;
        accessor_t<RESTRICT, const T*> lhs_accessor(lhs);
        accessor_t<RESTRICT, const U*> mhs_accessor(mhs);
        accessor_t<RESTRICT, const V*> rhs_accessor(rhs);
        accessor_t<RESTRICT, W*> output_accessor(output);

        const bool4_t is_contiguous = isContiguous(lhs_stride, shape) && isContiguous(mhs_stride, shape) &&
                                      isContiguous(rhs_stride, shape) && isContiguous(output_stride, shape);
        if (is_contiguous[1] && is_contiguous[2]) {
            const uint4_t uint_shape{shape};
            const uint elements = is_contiguous[0] ? uint_shape.elements() : uint3_t{uint_shape.get() + 1}.elements();
            const dim3 blocks(noa::math::divideUp(elements, TrinaryConfig::BLOCK_WORK_SIZE),
                              is_contiguous[0] ? 1 : shape[0]);

            uint vec_size = 1;
            if (is_contiguous[3]) {
                vec_size = std::min(std::min(maxVectorCount(lhs), maxVectorCount(mhs)), maxVectorCount(rhs));
                vec_size = std::min(vec_size, maxVectorCount(output));
            }

            if (blocks.y > 1)
                vec_size = lhs_stride[0] % vec_size || mhs_stride[0] % vec_size ||
                           rhs_stride[0] % vec_size || output_stride[0] % vec_size ?
                           1 : vec_size;

            const uint2_t uint_lhs_stride{lhs_stride[0], lhs_stride[3]};
            const uint2_t uint_mhs_stride{mhs_stride[0], mhs_stride[3]};
            const uint2_t uint_rhs_stride{rhs_stride[0], rhs_stride[3]};
            const uint2_t uint_output_stride{output_stride[0], output_stride[3]};
            const LaunchConfig config{blocks, TrinaryConfig::BLOCK_SIZE};
            if (vec_size == 4) {
                return stream.enqueue(name, trinaryArray1D_<T, U, V, W, TrinaryOp, 4, RESTRICT>, config,
                                      lhs_accessor, uint_lhs_stride, mhs_accessor, uint_mhs_stride,
                                      rhs_accessor, uint_rhs_stride, output_accessor, uint_output_stride,
                                      elements, trinary_op);
            } else if (vec_size == 2) {
                return stream.enqueue(name, trinaryArray1D_<T, U, V, W, TrinaryOp, 2, RESTRICT>, config,
                                      lhs_accessor, uint_lhs_stride, mhs_accessor, uint_mhs_stride,
                                      rhs_accessor, uint_rhs_stride, output_accessor, uint_output_stride,
                                      elements, trinary_op);
            } else {
                return stream.enqueue(name, trinaryArray1D_<T, U, V, W, TrinaryOp, 1, RESTRICT>, config,
                                      lhs_accessor, uint_lhs_stride, mhs_accessor, uint_mhs_stride,
                                      rhs_accessor, uint_rhs_stride, output_accessor, uint_output_stride,
                                      elements, trinary_op);
            }
        } else {
            const uint2_t i_shape{shape.get() + 2};
            const uint blocks_x = noa::math::divideUp(i_shape[1], TrinaryConfig::BLOCK_WORK_SIZE_2D.x);
            const uint blocks_y = noa::math::divideUp(i_shape[0], TrinaryConfig::BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, TrinaryConfig::BLOCK_SIZE_2D};
            stream.enqueue(name, trinaryArray4D_<T, U, V, W, TrinaryOp, RESTRICT>, config,
                           lhs_accessor, uint4_t{lhs_stride}, mhs_accessor, uint4_t{mhs_stride},
                           rhs_accessor, uint4_t{rhs_stride}, output_accessor, uint4_t{output_stride},
                           i_shape, trinary_op, blocks_x);
        }
    }
}
