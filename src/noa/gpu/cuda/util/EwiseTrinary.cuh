#pragma once

#include "noa/common/Definitions.h"
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

    template<typename lhs_val_t, typename mhs_val_t, typename rhs_val_t,
             typename out_val_t, typename trinary_t,
             int VEC_SIZE, bool RESTRICT>
    __global__ __launch_bounds__(TrinaryConfig::BLOCK_SIZE)
    void trinaryValue1D_(accessor_t<RESTRICT, const lhs_val_t*> lhs, uint2_t lhs_stride,
                         mhs_val_t mhs, rhs_val_t rhs,
                         accessor_t<RESTRICT, out_val_t*> output, uint2_t output_stride,
                         uint elements, trinary_t trinary_op) {
        constexpr uint BLOCK_SIZE = TrinaryConfig::BLOCK_SIZE;
        constexpr uint BLOCK_WORK_SIZE = TrinaryConfig::BLOCK_WORK_SIZE;
        constexpr uint EPT = TrinaryConfig::ELEMENTS_PER_THREAD;

        using iptr_t = typename accessor_t<RESTRICT, const lhs_val_t*>::ptr_type;
        using optr_t = typename accessor_t<RESTRICT, out_val_t*>::ptr_type;
        const uint batch = blockIdx.y;
        const uint base = BLOCK_WORK_SIZE * blockIdx.x;
        iptr_t lhs_ = lhs.get() + batch * lhs_stride[0];;
        optr_t out_ = output.get() + batch * output_stride[0];;

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int i = 0; i < EPT; ++i) {
                const uint gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements) {
                    out_[gid * output_stride[1]] =
                            static_cast<out_val_t>(trinary_op(lhs_[gid * lhs_stride[1]], mhs, rhs));
                }
            }
        } else {
            lhs_ += base;
            out_ += base;
            const uint remaining = elements - base;
            if (remaining < BLOCK_WORK_SIZE) {
                #pragma unroll
                for (int i = 0; i < EPT; ++i) {
                    const uint offset = BLOCK_SIZE * i + threadIdx.x;
                    if (offset < remaining)
                        out_[offset] = static_cast<out_val_t>(trinary_op(lhs_[offset], mhs, rhs));
                }
            } else {
                lhs_val_t args[EPT];
                out_val_t results[EPT];
                block::vectorizedLoad<BLOCK_SIZE, EPT, VEC_SIZE>(lhs_, args, threadIdx.x);
                #pragma unroll
                for (uint i = 0; i < EPT; ++i)
                    results[i] = static_cast<out_val_t>(trinary_op(args[i], mhs, rhs));
                block::vectorizedStore<BLOCK_SIZE, EPT, VEC_SIZE>(results, out_, threadIdx.x);
            }
        }
    }

    template<typename lhs_val_t, typename mhs_val_t, typename rhs_val_t,
             typename out_val_t, typename trinary_t, bool RESTRICT>
    __global__ __launch_bounds__(TrinaryConfig::BLOCK_SIZE)
    void trinaryValue4D_(accessor_t<RESTRICT, const lhs_val_t*> lhs, uint4_t lhs_stride,
                         mhs_val_t mhs, rhs_val_t rhs,
                         accessor_t<RESTRICT, out_val_t*> output, uint4_t output_stride,
                         uint2_t shape, trinary_t trinary_op, uint blocks_x) {
        using iptr_t = typename accessor_t<RESTRICT, const lhs_val_t*>::ptr_type;
        using optr_t = typename accessor_t<RESTRICT, out_val_t*>::ptr_type;
        iptr_t lhs_ = lhs.get();
        optr_t out_ = output.get();

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         TrinaryConfig::BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         TrinaryConfig::BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        lhs_ += indexing::at(gid[0], gid[1], lhs_stride);
        out_ += indexing::at(gid[0], gid[1], output_stride);

        #pragma unroll
        for (int k = 0; k < TrinaryConfig::ELEMENTS_PER_THREAD_2D; ++k) {
            #pragma unroll
            for (int l = 0; l < TrinaryConfig::ELEMENTS_PER_THREAD_2D; ++l) {
                const uint ik = gid[2] + TrinaryConfig::BLOCK_SIZE_2D.y * k;
                const uint il = gid[3] + TrinaryConfig::BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1]) {
                    out_[ik * output_stride[2] + il * output_stride[3]] =
                            static_cast<out_val_t>(trinary_op(lhs_[ik * lhs_stride[2] + il * lhs_stride[3]],
                                                              mhs, rhs));
                }
            }
        }
    }

    template<typename lhs_val_t, typename mhs_val_t, typename rhs_val_t,
             typename out_val_t, typename trinary_t,
             int VEC_SIZE, bool RESTRICT>
    __global__ __launch_bounds__(TrinaryConfig::BLOCK_SIZE)
    void trinaryArray1D_(accessor_t<RESTRICT, const lhs_val_t*> lhs, uint2_t lhs_stride,
                         accessor_t<RESTRICT, const mhs_val_t*> mhs, uint2_t mhs_stride,
                         accessor_t<RESTRICT, const rhs_val_t*> rhs, uint2_t rhs_stride,
                         accessor_t<RESTRICT, out_val_t*> output, uint2_t output_stride,
                         uint elements, trinary_t trinary_op) {
        constexpr uint BLOCK_SIZE = TrinaryConfig::BLOCK_SIZE;
        constexpr uint BLOCK_WORK_SIZE = TrinaryConfig::BLOCK_WORK_SIZE;
        constexpr uint EPT = TrinaryConfig::ELEMENTS_PER_THREAD;

        using lptr_t = typename accessor_t<RESTRICT, const lhs_val_t*>::ptr_type;
        using mptr_t = typename accessor_t<RESTRICT, const mhs_val_t*>::ptr_type;
        using rptr_t = typename accessor_t<RESTRICT, const rhs_val_t*>::ptr_type;
        using optr_t = typename accessor_t<RESTRICT, out_val_t*>::ptr_type;
        const uint batch = blockIdx.y;
        const uint base = BLOCK_WORK_SIZE * blockIdx.x;
        lptr_t lhs_ = lhs.get() + batch * lhs_stride[0];
        mptr_t mhs_ = mhs.get() + batch * mhs_stride[0];
        rptr_t rhs_ = rhs.get() + batch * rhs_stride[0];
        optr_t out_ = output.get() + batch * output_stride[0];

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int i = 0; i < EPT; ++i) {
                const uint gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements) {
                    out_[gid * output_stride[1]] =
                            static_cast<out_val_t>(trinary_op(lhs_[gid * lhs_stride[1]],
                                                              mhs_[gid * mhs_stride[1]],
                                                              rhs_[gid * rhs_stride[1]]));
                }
            }
        } else {
            const uint remaining = elements - base;
            lhs_ += base;
            mhs_ += base;
            rhs_ += base;
            out_ += base;
            if (remaining < BLOCK_WORK_SIZE) {
                #pragma unroll
                for (int i = 0; i < EPT; ++i) {
                    const uint offset = BLOCK_SIZE * i + threadIdx.x;
                    if (offset < remaining) {
                        out_[offset] = static_cast<out_val_t>(trinary_op(
                                lhs_[offset], mhs_[offset], rhs_[offset]));
                    }
                }
            } else {
                lhs_val_t ilhs[EPT];
                mhs_val_t imhs[EPT];
                rhs_val_t irhs[EPT];
                out_val_t results[EPT];
                block::vectorizedLoad<BLOCK_SIZE, EPT, VEC_SIZE>(lhs_, ilhs, threadIdx.x);
                block::vectorizedLoad<BLOCK_SIZE, EPT, VEC_SIZE>(mhs_, imhs, threadIdx.x);
                block::vectorizedLoad<BLOCK_SIZE, EPT, VEC_SIZE>(rhs_, irhs, threadIdx.x);
                #pragma unroll
                for (uint i = 0; i < EPT; ++i)
                    results[i] = static_cast<out_val_t>(trinary_op(ilhs[i], imhs[i], irhs[i]));
                block::vectorizedStore<BLOCK_SIZE, EPT, VEC_SIZE>(results, out_, threadIdx.x);
            }
        }
    }

    template<typename lhs_val_t, typename mhs_val_t, typename rhs_val_t,
             typename out_val_t, typename trinary_t, bool RESTRICT>
    __global__ __launch_bounds__(TrinaryConfig::BLOCK_SIZE)
    void trinaryArray4D_(accessor_t<RESTRICT, const lhs_val_t*> lhs, uint4_t lhs_stride,
                         accessor_t<RESTRICT, const mhs_val_t*> mhs, uint4_t mhs_stride,
                         accessor_t<RESTRICT, const rhs_val_t*> rhs, uint4_t rhs_stride,
                         accessor_t<RESTRICT, out_val_t*> output, uint4_t output_stride,
                         uint2_t shape, trinary_t trinary_op, uint blocks_x) {
        using lptr_t = typename accessor_t<RESTRICT, const lhs_val_t*>::ptr_type;
        using mptr_t = typename accessor_t<RESTRICT, const mhs_val_t*>::ptr_type;
        using rptr_t = typename accessor_t<RESTRICT, const rhs_val_t*>::ptr_type;
        using optr_t = typename accessor_t<RESTRICT, out_val_t*>::ptr_type;
        lptr_t lhs_ = lhs.get();
        mptr_t mhs_ = mhs.get();
        rptr_t rhs_ = rhs.get();
        optr_t out_ = output.get();

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         TrinaryConfig::BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         TrinaryConfig::BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        lhs_ += indexing::at(gid[0], gid[1], lhs_stride);
        mhs_ += indexing::at(gid[0], gid[1], mhs_stride);
        rhs_ += indexing::at(gid[0], gid[1], rhs_stride);
        out_ += indexing::at(gid[0], gid[1], output_stride);

        #pragma unroll
        for (int k = 0; k < TrinaryConfig::ELEMENTS_PER_THREAD_2D; ++k) {
            #pragma unroll
            for (int l = 0; l < TrinaryConfig::ELEMENTS_PER_THREAD_2D; ++l) {
                const uint ik = gid[2] + TrinaryConfig::BLOCK_SIZE_2D.y * k;
                const uint il = gid[3] + TrinaryConfig::BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1]) {
                    out_[ik * output_stride[2] + il * output_stride[3]] =
                            static_cast<out_val_t>(trinary_op(lhs_[ik * lhs_stride[2] + il * lhs_stride[3]],
                                                              mhs_[ik * mhs_stride[2] + il * mhs_stride[3]],
                                                              rhs_[ik * rhs_stride[2] + il * rhs_stride[3]]));
                }
            }
        }
    }
}

namespace noa::cuda::util::ewise {
    /// Apply a trinary operator, element-wise.
    /// \tparam RESTRICT        Whether the pointers can be accessed using the __restrict__ attribute.
    /// \param[in] name         Name of the function. Used for logging if kernel launch fails.
    /// \param[in] lhs          On the \b device. Left-hand side argument.
    /// \param lhs_stride       Rightmost stride of \p lhs.
    /// \param mhs              Middle-hand side argument.
    /// \param rhs              Right-hand side argument.
    /// \param[out] output      On the \b device. Transformed array.
    /// \param output_stride    Rightmost stride of \p output.
    /// \param shape            Rightmost shape of \p lhs and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \param trinary_op       Trinary operator. The output is explicitly casted to \p V.
    /// \note This function is asynchronous relative to the host and may return before completion.
    ///       One must make sure \p lhs and \p output stay valid until completion.
    template<bool RESTRICT = false,
             typename lhs_val_t, typename mhs_t, typename rhs_t,
             typename out_val_t, typename trinary_t,
             typename = std::enable_if_t<noa::traits::is_data_v<mhs_t> && noa::traits::is_data_v<rhs_t>>>
    void trinary(const char* name,
                 const lhs_val_t* lhs, size4_t lhs_stride,
                 mhs_t mhs, rhs_t rhs,
                 out_val_t* output, size4_t output_stride,
                 size4_t shape, Stream& stream, trinary_t trinary_op) {
        using namespace details;
        using mhs_val_t = std::remove_const_t<mhs_t>;
        using rhs_val_t = std::remove_const_t<rhs_t>;
        accessor_t<RESTRICT, const lhs_val_t*> lhs_accessor(lhs);
        accessor_t<RESTRICT, out_val_t*> output_accessor(output);

        const bool4_t is_contiguous = indexing::isContiguous(lhs_stride, shape) &&
                                      indexing::isContiguous(output_stride, shape);
        if (is_contiguous[1] && is_contiguous[2]) {
            const uint4_t uint_shape{shape};
            uint elements, blocks_y;
            if (!is_contiguous[0]) {
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
                return stream.enqueue(
                        name, trinaryValue1D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, 4, RESTRICT>,
                        config, lhs_accessor, uint_lhs_stride, mhs, rhs,
                        output_accessor, uint_output_stride, elements, trinary_op);
            } else if (vec_size == 2) {
                return stream.enqueue(
                        name, trinaryValue1D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, 2, RESTRICT>,
                        config, lhs_accessor, uint_lhs_stride, mhs, rhs,
                        output_accessor, uint_output_stride, elements, trinary_op);
            } else {
                return stream.enqueue(
                        name, trinaryValue1D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, 1, RESTRICT>,
                        config, lhs_accessor, uint_lhs_stride, mhs, rhs,
                        output_accessor, uint_output_stride, elements, trinary_op);
            }
        } else {
            const uint2_t i_shape{shape.get() + 2};
            const uint blocks_x = noa::math::divideUp(i_shape[1], TrinaryConfig::BLOCK_WORK_SIZE_2D.x);
            const uint blocks_y = noa::math::divideUp(i_shape[0], TrinaryConfig::BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, TrinaryConfig::BLOCK_SIZE_2D};
            stream.enqueue(name, trinaryValue4D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, RESTRICT>,
                           config, lhs_accessor, uint4_t{lhs_stride}, mhs, rhs,
                           output_accessor, uint4_t{output_stride}, i_shape, trinary_op, blocks_x);
        }
    }

    /// Apply a trinary operator, element-wise.
    /// \tparam RESTRICT        Whether the pointers can be accessed using the __restrict__ attribute.
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
    /// \param trinary_op       Trinary operator. The output is explicitly casted to the output type.
    /// \note This function is asynchronous relative to the host and may return before completion.
    ///       One must make sure \p lhs, \p mhs, \p rhs and \p output stay valid until completion.
    template<bool RESTRICT = false,
             typename lhs_val_t, typename mhs_val_t, typename rhs_val_t,
             typename out_val_t, typename trinary_t>
    void trinary(const char* name,
                 const lhs_val_t* lhs, size4_t lhs_stride,
                 const mhs_val_t* mhs, size4_t mhs_stride,
                 const rhs_val_t* rhs, size4_t rhs_stride,
                 out_val_t* output, size4_t output_stride, size4_t shape,
                 Stream& stream, trinary_t trinary_op) {
        using namespace details;
        accessor_t<RESTRICT, const lhs_val_t*> lhs_accessor(lhs);
        accessor_t<RESTRICT, const mhs_val_t*> mhs_accessor(mhs);
        accessor_t<RESTRICT, const rhs_val_t*> rhs_accessor(rhs);
        accessor_t<RESTRICT, out_val_t*> output_accessor(output);

        const bool4_t is_contiguous = indexing::isContiguous(lhs_stride, shape) &&
                                      indexing::isContiguous(mhs_stride, shape) &&
                                      indexing::isContiguous(rhs_stride, shape) &&
                                      indexing::isContiguous(output_stride, shape);
        if (is_contiguous[1] && is_contiguous[2]) {
            const uint4_t uint_shape{shape};
            const uint elements = is_contiguous[0] ? uint_shape.elements() : uint3_t{uint_shape.get() + 1}.elements();
            const dim3 blocks(noa::math::divideUp(elements, TrinaryConfig::BLOCK_WORK_SIZE),
                              is_contiguous[0] ? 1 : shape[0]);

            uint vec_size = is_contiguous[3] ? std::min({maxVectorCount(lhs), maxVectorCount(mhs),
                                                         maxVectorCount(rhs), maxVectorCount(output)}) : 1;
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
                return stream.enqueue(
                        name, trinaryArray1D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, 4, RESTRICT>,
                        config,
                        lhs_accessor, uint_lhs_stride, mhs_accessor, uint_mhs_stride,
                        rhs_accessor, uint_rhs_stride, output_accessor, uint_output_stride,
                        elements, trinary_op);
            } else if (vec_size == 2) {
                return stream.enqueue(
                        name, trinaryArray1D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, 2, RESTRICT>,
                        config,
                        lhs_accessor, uint_lhs_stride, mhs_accessor, uint_mhs_stride,
                        rhs_accessor, uint_rhs_stride, output_accessor, uint_output_stride,
                        elements, trinary_op);
            } else {
                return stream.enqueue(
                        name, trinaryArray1D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, 1, RESTRICT>,
                        config,
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
            stream.enqueue(name,
                           trinaryArray4D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, RESTRICT>, config,
                           lhs_accessor, uint4_t{lhs_stride}, mhs_accessor, uint4_t{mhs_stride},
                           rhs_accessor, uint4_t{rhs_stride}, output_accessor, uint4_t{output_stride},
                           i_shape, trinary_op, blocks_x);
        }
    }
}
