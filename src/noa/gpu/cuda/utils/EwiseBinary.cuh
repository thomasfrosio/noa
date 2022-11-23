#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/utils/Traits.h"
#include "noa/gpu/cuda/utils/Block.cuh"
#include "noa/gpu/cuda/utils/Pointers.h"

namespace noa::cuda::utils::ewise::details {
    struct BinaryConfig {
        static constexpr uint32_t ELEMENTS_PER_THREAD = 4;
        static constexpr uint32_t BLOCK_SIZE = 128;
        static constexpr uint32_t BLOCK_WORK_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;

        // Still the same threads per block and elements per thread, but using a 2D block.
        // The goal is waste as fewer threads as possible, assuming 2D/3D/4D arrays have a
        // similar number of elements in their two innermost dimensions.
        static constexpr uint32_t ELEMENTS_PER_THREAD_2D = ELEMENTS_PER_THREAD / 2;
        static constexpr dim3 BLOCK_SIZE_2D{32, BLOCK_SIZE / 32, 1};
        static constexpr dim3 BLOCK_WORK_SIZE_2D{BLOCK_SIZE_2D.x * ELEMENTS_PER_THREAD_2D,
                                                 BLOCK_SIZE_2D.y * ELEMENTS_PER_THREAD_2D, 1};
    };

    template<typename LhsVal, typename RhsVal,
             typename OutVal, typename BinaryOp,
             int VEC_SIZE, AccessorTraits TRAITS>
    __global__ __launch_bounds__(BinaryConfig::BLOCK_SIZE)
    void binaryValueLeft1D_(Accessor<const LhsVal, 2, uint32_t, TRAITS> lhs,
                            RhsVal rhs,
                            Accessor<OutVal, 2, uint32_t, TRAITS> output,
                            uint32_t elements, BinaryOp binary_op) {
        constexpr uint32_t BLOCK_SIZE = BinaryConfig::BLOCK_SIZE;
        constexpr uint32_t BLOCK_WORK_SIZE = BinaryConfig::BLOCK_WORK_SIZE;
        constexpr uint32_t EPT = BinaryConfig::ELEMENTS_PER_THREAD;

        const uint32_t batch = blockIdx.y;
        const uint32_t base = BLOCK_WORK_SIZE * blockIdx.x;
        const auto lhs_ = lhs[batch];
        const auto out_ = output[batch];

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (uint32_t i = 0; i < EPT; ++i) {
                const uint32_t gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements)
                    out_[gid] = static_cast<OutVal>(binary_op(lhs_[gid], rhs));
            }
        } else {
            NOA_ASSERT(lhs_.stride(0) == 1 && out_.stride(0) == 1);
            using lptr_t = typename decltype(lhs)::pointer_type;
            using optr_t = typename decltype(output)::pointer_type;
            lptr_t lhs_ptr = lhs_.get() + base;
            optr_t out_ptr = out_.get() + base;

            const uint32_t remaining = elements - base;
            if (remaining < BLOCK_WORK_SIZE) {
                for (uint32_t i = 0; i < EPT; ++i) {
                    const uint32_t offset = BLOCK_SIZE * i + threadIdx.x;
                    if (offset < remaining)
                        out_ptr[offset] = static_cast<OutVal>(binary_op(lhs_ptr[offset], rhs));
                }
            } else {
                LhsVal args[EPT];
                OutVal results[EPT];
                block::vectorizedLoad<BLOCK_SIZE, EPT, VEC_SIZE>(lhs_ptr, args, threadIdx.x);
                #pragma unroll
                for (uint32_t i = 0; i < EPT; ++i)
                    results[i] = static_cast<OutVal>(binary_op(args[i], rhs));
                block::vectorizedStore<BLOCK_SIZE, EPT, VEC_SIZE>(results, out_ptr, threadIdx.x);
            }
        }
    }

    template<typename LhsVal, typename RhsVal,
             typename OutVal, typename BinaryOp,
             int VEC_SIZE, AccessorTraits TRAITS>
    __global__ __launch_bounds__(BinaryConfig::BLOCK_SIZE)
    void binaryValueRight1D_(LhsVal lhs,
                             Accessor<const RhsVal, 2, uint32_t, TRAITS> rhs,
                             Accessor<OutVal, 2, uint32_t, TRAITS> output,
                             uint32_t elements, BinaryOp binary_op) {
        constexpr uint32_t BLOCK_SIZE = BinaryConfig::BLOCK_SIZE;
        constexpr uint32_t BLOCK_WORK_SIZE = BinaryConfig::BLOCK_WORK_SIZE;
        constexpr uint32_t EPT = BinaryConfig::ELEMENTS_PER_THREAD;

        const uint32_t batch = blockIdx.y;
        const uint32_t base = BLOCK_WORK_SIZE * blockIdx.x;
        const auto rhs_ = rhs[batch];
        const auto out_ = output[batch];

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (uint32_t i = 0; i < EPT; ++i) {
                const uint32_t gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements)
                    out_[gid] = static_cast<OutVal>(binary_op(lhs, rhs_[gid]));
            }
        } else {
            NOA_ASSERT(rhs_.stride(0) == 1 && out_.stride(0) == 1);
            using rptr_t = typename decltype(rhs)::pointer_type;
            using optr_t = typename decltype(output)::pointer_type;
            rptr_t rhs_ptr = rhs_.get() + base;
            optr_t out_ptr = out_.get() + base;

            const uint32_t remaining = elements - base;
            if (remaining < BLOCK_WORK_SIZE) {
                for (uint32_t i = 0; i < EPT; ++i) {
                    const uint32_t offset = BLOCK_SIZE * i + threadIdx.x;
                    if (offset < remaining)
                        out_ptr[offset] = static_cast<OutVal>(binary_op(lhs, rhs_ptr[offset]));
                }
            } else {
                RhsVal args[EPT];
                OutVal results[EPT];
                block::vectorizedLoad<BLOCK_SIZE, EPT, VEC_SIZE>(rhs_ptr, args, threadIdx.x);
                #pragma unroll
                for (uint32_t i = 0; i < EPT; ++i)
                    results[i] = static_cast<OutVal>(binary_op(lhs, args[i]));
                block::vectorizedStore<BLOCK_SIZE, EPT, VEC_SIZE>(results, out_ptr, threadIdx.x);
            }
        }
    }

    template<typename LhsVal, typename RhsVal,
             typename OutVal, typename BinaryOp, AccessorTraits TRAITS>
    __global__ __launch_bounds__(BinaryConfig::BLOCK_SIZE)
    void binaryValueLeft4D_(Accessor<const LhsVal, 4, uint32_t, TRAITS> lhs,
                            RhsVal rhs,
                            Accessor<OutVal, 4, uint32_t, TRAITS> out,
                            uint2_t shape, BinaryOp binary_op, uint32_t blocks_x) {

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint4_t gid{blockIdx.z,
                          blockIdx.y,
                          BinaryConfig::BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                          BinaryConfig::BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        const auto lhs_ = lhs[gid[0]][gid[1]];
        const auto out_ = out[gid[0]][gid[1]];

        #pragma unroll
        for (uint32_t k = 0; k < BinaryConfig::ELEMENTS_PER_THREAD_2D; ++k) {
            #pragma unroll
            for (uint32_t l = 0; l < BinaryConfig::ELEMENTS_PER_THREAD_2D; ++l) {
                const uint32_t ik = gid[2] + BinaryConfig::BLOCK_SIZE_2D.y * k;
                const uint32_t il = gid[3] + BinaryConfig::BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1])
                    out_(ik, il) = static_cast<OutVal>(binary_op(lhs_(ik, il), rhs));
            }
        }
    }

    template<typename LhsVal, typename RhsVal,
             typename OutVal, typename BinaryOp, AccessorTraits TRAITS>
    __global__ __launch_bounds__(BinaryConfig::BLOCK_SIZE)
    void binaryValueRight4D_(LhsVal lhs,
                             Accessor<const RhsVal, 4, uint32_t, TRAITS> rhs,
                             Accessor<OutVal, 4, uint32_t, TRAITS> out,
                             uint2_t shape, BinaryOp binary_op, uint32_t blocks_x) {

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint4_t gid{blockIdx.z,
                          blockIdx.y,
                          BinaryConfig::BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                          BinaryConfig::BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        const auto rhs_ = rhs[gid[0]][gid[1]];
        const auto out_ = out[gid[0]][gid[1]];

        #pragma unroll
        for (uint32_t k = 0; k < BinaryConfig::ELEMENTS_PER_THREAD_2D; ++k) {
            #pragma unroll
            for (uint32_t l = 0; l < BinaryConfig::ELEMENTS_PER_THREAD_2D; ++l) {
                const uint32_t ik = gid[2] + BinaryConfig::BLOCK_SIZE_2D.y * k;
                const uint32_t il = gid[3] + BinaryConfig::BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1])
                    out_(ik, il) = static_cast<OutVal>(binary_op(lhs, rhs_(ik, il)));
            }
        }
    }

    template<typename LhsVal, typename RhsVal,
             typename OutVal, typename BinaryOp, int VEC_SIZE, AccessorTraits TRAITS>
    __global__ __launch_bounds__(BinaryConfig::BLOCK_SIZE)
    void binaryArray1D_(Accessor<const LhsVal, 2, uint32_t, TRAITS> lhs,
                        Accessor<const RhsVal, 2, uint32_t, TRAITS> rhs,
                        Accessor<OutVal, 2, uint32_t, TRAITS> out,
                        uint32_t elements, BinaryOp binary_op) {
        constexpr uint32_t BLOCK_SIZE = BinaryConfig::BLOCK_SIZE;
        constexpr uint32_t BLOCK_WORK_SIZE = BinaryConfig::BLOCK_WORK_SIZE;
        constexpr uint32_t EPT = BinaryConfig::ELEMENTS_PER_THREAD;

        const uint32_t batch = blockIdx.y;
        const uint32_t base = BLOCK_WORK_SIZE * blockIdx.x;
        const auto lhs_ = lhs[batch];
        const auto rhs_ = rhs[batch];
        const auto out_ = out[batch];

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (uint32_t i = 0; i < EPT; ++i) {
                const uint32_t gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements)
                    out_[gid] = static_cast<OutVal>(binary_op(lhs_[gid], rhs_[gid]));
            }
        } else {
            NOA_ASSERT(lhs_.stride(0) == 1 && rhs_.stride(0) == 1 && out_.stride(0) == 1);
            using lptr_t = typename decltype(lhs)::pointer_type;
            using rptr_t = typename decltype(rhs)::pointer_type;
            using optr_t = typename decltype(out)::pointer_type;
            lptr_t lhs_ptr = lhs_.get() + base;
            rptr_t rhs_ptr = rhs_.get() + base;
            optr_t out_ptr = out_.get() + base;

            const uint32_t remaining = elements - base;
            if (remaining < BLOCK_WORK_SIZE) {
                #pragma unroll
                for (uint32_t i = 0; i < EPT; ++i) {
                    const uint32_t offset = BLOCK_SIZE * i + threadIdx.x;
                    if (offset < remaining)
                        out_ptr[offset] = static_cast<OutVal>(binary_op(lhs_ptr[offset], rhs_ptr[offset]));
                }
            } else {
                LhsVal ilhs[EPT];
                RhsVal irhs[EPT];
                OutVal results[EPT];
                block::vectorizedLoad<BLOCK_SIZE, EPT, VEC_SIZE>(lhs_ptr, ilhs, threadIdx.x);
                block::vectorizedLoad<BLOCK_SIZE, EPT, VEC_SIZE>(rhs_ptr, irhs, threadIdx.x);
                #pragma unroll
                for (uint32_t i = 0; i < EPT; ++i)
                    results[i] = static_cast<OutVal>(binary_op(ilhs[i], irhs[i]));
                block::vectorizedStore<BLOCK_SIZE, EPT, VEC_SIZE>(results, out_ptr, threadIdx.x);
            }
        }
    }

    template<typename LhsVal, typename RhsVal,
             typename OutVal, typename BinaryOp, AccessorTraits TRAITS>
    __global__ __launch_bounds__(BinaryConfig::BLOCK_SIZE)
    void binaryArray4D_(Accessor<const LhsVal, 4, uint32_t, TRAITS> lhs,
                        Accessor<const RhsVal, 4, uint32_t, TRAITS> rhs,
                        Accessor<OutVal, 4, uint32_t, TRAITS> out,
                        uint2_t shape, BinaryOp binary_op, uint32_t blocks_x) {

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint4_t gid(blockIdx.z,
                          blockIdx.y,
                          BinaryConfig::BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                          BinaryConfig::BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x);
        const auto lhs_ = lhs[gid[0]][gid[1]];
        const auto rhs_ = rhs[gid[0]][gid[1]];
        const auto out_ = out[gid[0]][gid[1]];

        #pragma unroll
        for (uint32_t k = 0; k < BinaryConfig::ELEMENTS_PER_THREAD_2D; ++k) {
            #pragma unroll
            for (uint32_t l = 0; l < BinaryConfig::ELEMENTS_PER_THREAD_2D; ++l) {
                const uint32_t ik = gid[2] + BinaryConfig::BLOCK_SIZE_2D.y * k;
                const uint32_t il = gid[3] + BinaryConfig::BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1])
                    out_(ik, il) = static_cast<OutVal>(binary_op(lhs_(ik, il), rhs_(ik, il)));
            }
        }
    }
}

namespace noa::cuda::utils::ewise {
    // Applies a binary operator, element-wise.
    // RESTRICT:        Whether the pointers can be accessed using the __restrict__ attribute.
    // name:            Name of the function. Used for logging if kernel launch fails.
    // lhs:             On the device. Left-hand side argument for the binary operator.
    // lhs_strides:     Strides of lhs.
    // rhs:             Right-hand side argument for the binary operator.
    // output:          On the device. Transformed array.
    // output_strides:  Strides of output.
    // shape:           Shape of lhs and output.
    // swap_layout:     Swap the memory layout to optimize output writes.
    //                  If false, assume rightmost order is the fastest order.
    // stream:          Stream on which to enqueue this function.
    // binary_op:       Binary operator. The output is explicitly cast to the output type.
    // This function is asynchronous relative to the host and may return before completion.
    // One must make sure input and output pointers stay valid until completion.
    template<bool RESTRICT = false,
             typename LhsVal, typename Rhs,
             typename OutVal, typename BinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<Rhs>>>
    void binary(const char* name,
                const LhsVal* lhs, dim4_t lhs_strides, Rhs rhs,
                OutVal* output, dim4_t output_strides, dim4_t shape,
                bool swap_layout, Stream& stream,
                BinaryOp binary_op) {
        using namespace details;
        using RhsVal = noa::traits::remove_ref_cv_t<Rhs>;
        constexpr AccessorTraits TRAITS = RESTRICT ? AccessorTraits::RESTRICT : AccessorTraits::DEFAULT;
        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(lhs, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());

        if (swap_layout) {
            const auto order = indexing::order(output_strides, shape);
            shape = indexing::reorder(shape, order);
            output_strides = indexing::reorder(output_strides, order);
            lhs_strides = indexing::reorder(lhs_strides, order);
        }

        const bool4_t is_contiguous = indexing::isContiguous(lhs_strides, shape) &&
                                      indexing::isContiguous(output_strides, shape);
        if (is_contiguous[1] && is_contiguous[2]) { // 1D-like
            // Keep batches separated in a different Grid.Y if they're not contiguous or
            // if we need to apply a different value to each batch.
            uint32_t elements, blocks_y;
            if (!is_contiguous[0]) {
                elements = safe_cast<uint32_t>(shape[1] * shape[2] * shape[3]);
                blocks_y = shape[0];
            } else {
                elements = safe_cast<uint32_t>(shape.elements());
                blocks_y = 1;
            }
            const dim3 blocks(noa::math::divideUp(elements, BinaryConfig::BLOCK_WORK_SIZE), blocks_y);
            const auto uint_lhs_strides = safe_cast<uint2_t>(dim2_t{lhs_strides[0], lhs_strides[3]});
            const auto uint_output_strides = safe_cast<uint2_t>(dim2_t{output_strides[0], output_strides[3]});
            const LaunchConfig config{blocks, BinaryConfig::BLOCK_SIZE};

            uint32_t vec_size = is_contiguous[3] ? std::min(maxVectorCount(lhs), maxVectorCount(output)) : 1;
            if (blocks.y > 1) // make sure the beginning of each batch preserves the alignment
                vec_size = uint_lhs_strides[0] % vec_size || uint_output_strides[0] % vec_size ? 1 : vec_size;

            const Accessor<const LhsVal, 2, uint32_t, TRAITS> lhs_accessor(lhs, uint_lhs_strides);
            const Accessor<OutVal, 2, uint32_t, TRAITS> output_accessor(output, uint_output_strides);

            if (vec_size == 4) {
                return stream.enqueue(
                        name, binaryValueLeft1D_<LhsVal, RhsVal, OutVal, BinaryOp, 4, TRAITS>, config,
                        lhs_accessor, rhs, output_accessor, elements, binary_op);
            } else if (vec_size == 2) {
                return stream.enqueue(
                        name, binaryValueLeft1D_<LhsVal, RhsVal, OutVal, BinaryOp, 2, TRAITS>, config,
                        lhs_accessor, rhs, output_accessor, elements, binary_op);
            } else {
                return stream.enqueue(
                        name, binaryValueLeft1D_<LhsVal, RhsVal, OutVal, BinaryOp, 1, TRAITS>, config,
                        lhs_accessor, rhs, output_accessor, elements, binary_op);
            }
        } else {
            const auto i_shape = safe_cast<uint2_t>(dim2_t(shape.get(2)));
            const uint32_t blocks_x = noa::math::divideUp(i_shape[1], BinaryConfig::BLOCK_WORK_SIZE_2D.x);
            const uint32_t blocks_y = noa::math::divideUp(i_shape[0], BinaryConfig::BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, BinaryConfig::BLOCK_SIZE_2D};

            const Accessor<const LhsVal, 4, uint32_t, TRAITS> lhs_accessor(lhs, safe_cast<uint4_t>(lhs_strides));
            const Accessor<OutVal, 4, uint32_t, TRAITS> output_accessor(output, safe_cast<uint4_t>(output_strides));

            stream.enqueue(name, binaryValueLeft4D_<LhsVal, RhsVal, OutVal, BinaryOp, TRAITS>, config,
                           lhs_accessor, rhs, output_accessor, i_shape, binary_op, blocks_x);
        }
    }

    // Applies a binary operator, element-wise.
    // RESTRICT:       Whether the pointers can be accessed using the __restrict__ attribute.
    // name:           Name of the function. Used for logging if kernel launch fails.
    // lhs:            Left-hand side argument for the binary operator.
    // rhs:            On the device. Right-hand side argument for the binary operator.
    // rhs_strides:    Strides of rhs.
    // output:         On the device. Transformed array.
    // output_strides: Strides of output.
    // shape:          Shape of rhs and output.
    // swap_layout:    Swap the memory layout to optimize output writes.
    //                 If false, assume rightmost order is fastest order.
    // stream:         Stream on which to enqueue this function.
    // binary_op:      Binary operator. The output is explicitly casted to the output type.
    // This function is asynchronous relative to the host and may return before completion.
    // One must make sure input and output pointers stay valid until completion.
    template<bool RESTRICT = false,
             typename lhs_t, typename RhsVal,
             typename OutVal, typename BinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<lhs_t>>>
    void binary(const char* name,
                lhs_t lhs, const RhsVal* rhs, dim4_t rhs_strides,
                OutVal* output, dim4_t output_strides,
                dim4_t shape, bool swap_layout, Stream& stream,
                BinaryOp binary_op) {
        using namespace details;
        using LhsVal = noa::traits::remove_ref_cv_t<lhs_t>;
        constexpr AccessorTraits TRAITS = RESTRICT ? AccessorTraits::RESTRICT : AccessorTraits::DEFAULT;
        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(rhs, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());

        if (swap_layout) {
            const auto order = indexing::order(output_strides, shape);
            shape = indexing::reorder(shape, order);
            output_strides = indexing::reorder(output_strides, order);
            rhs_strides = indexing::reorder(rhs_strides, order);
        }

        const bool4_t is_contiguous = indexing::isContiguous(rhs_strides, shape) &&
                                      indexing::isContiguous(output_strides, shape);
        if (is_contiguous[1] && is_contiguous[2]) { // 1D-like
            // Keep batches separated in a different Grid.Y if they're not contiguous or
            // if we need to apply a different value to each batch.
            uint32_t elements, blocks_y;
            if (!is_contiguous[0]) {
                elements = safe_cast<uint32_t>(shape[1] * shape[2] * shape[3]);
                blocks_y = shape[0];
            } else {
                elements = safe_cast<uint32_t>(shape.elements());
                blocks_y = 1;
            }
            const dim3 blocks(noa::math::divideUp(elements, BinaryConfig::BLOCK_WORK_SIZE), blocks_y);
            const auto uint_rhs_strides = safe_cast<uint2_t>(dim2_t{rhs_strides[0], rhs_strides[3]});
            const auto uint_output_strides = safe_cast<uint2_t>(dim2_t{output_strides[0], output_strides[3]});
            const LaunchConfig config{blocks, BinaryConfig::BLOCK_SIZE};

            uint32_t vec_size = is_contiguous[3] ? std::min(maxVectorCount(rhs), maxVectorCount(output)) : 1;
            if (blocks.y > 1) // make sure the beginning of each batch preserves the alignment
                vec_size = uint_rhs_strides[0] % vec_size || uint_output_strides[0] % vec_size ? 1 : vec_size;

            const Accessor<const RhsVal, 2, uint32_t, TRAITS> rhs_accessor(rhs, uint_rhs_strides);
            const Accessor<OutVal, 2, uint32_t, TRAITS> output_accessor(output, uint_output_strides);

            if (vec_size == 4) {
                return stream.enqueue(
                        name, binaryValueRight1D_<LhsVal, RhsVal, OutVal, BinaryOp, 4, TRAITS>, config,
                        lhs, rhs_accessor, output_accessor, elements, binary_op);
            } else if (vec_size == 2) {
                return stream.enqueue(
                        name, binaryValueRight1D_<LhsVal, RhsVal, OutVal, BinaryOp, 2, TRAITS>, config,
                        lhs, rhs_accessor, output_accessor, elements, binary_op);
            } else {
                return stream.enqueue(
                        name, binaryValueRight1D_<LhsVal, RhsVal, OutVal, BinaryOp, 1, TRAITS>, config,
                        lhs, rhs_accessor, output_accessor, elements, binary_op);
            }
        } else {
            const auto i_shape = safe_cast<uint2_t>(dim2_t(shape.get(2)));
            const uint32_t blocks_x = noa::math::divideUp(i_shape[1], BinaryConfig::BLOCK_WORK_SIZE_2D.x);
            const uint32_t blocks_y = noa::math::divideUp(i_shape[0], BinaryConfig::BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, BinaryConfig::BLOCK_SIZE_2D};

            const Accessor<const RhsVal, 4, uint32_t, TRAITS> rhs_accessor(rhs, safe_cast<uint4_t>(rhs_strides));
            const Accessor<OutVal, 4, uint32_t, TRAITS> output_accessor(output, safe_cast<uint4_t>(output_strides));

            stream.enqueue(name, binaryValueRight4D_<LhsVal, RhsVal, OutVal, BinaryOp, TRAITS>, config,
                           lhs, rhs_accessor, output_accessor, i_shape, binary_op, blocks_x);
        }
    }

    // Apply a binary operator, element-wise.
    // RESTRICT:        Whether the pointers can be accessed using the __restrict__ attribute.
    // name:            Name of the function. Used for logging if kernel launch fails.
    // lhs:             On the device. Left-hand side argument.
    // lhs_strides:     Strides of lhs.
    // rhs:             On the device. Right-hand side argument.
    // rhs_strides:     Strides of rhs.
    // output:          On the device. Transformed array.
    // output_strides:  Strides of output.
    // shape:           Shape of lhs, rhs and output.
    // swap_layout:     Swap the memory layout to optimize output writes.
    //                  If false, assume rightmost order is the fastest order.
    // stream:          Stream on which to enqueue this function.
    // binary_op:       Binary operator. The output is explicitly cast to the output type.
    // This function is asynchronous relative to the host and may return before completion.
    // One must make sure input and output pointers stay valid until completion.
    template<bool RESTRICT = false,
             typename LhsVal, typename RhsVal,
             typename OutVal, typename BinaryOp>
    void binary(const char* name,
                const LhsVal* lhs, dim4_t lhs_strides,
                const RhsVal* rhs, dim4_t rhs_strides,
                OutVal* output, dim4_t output_strides,
                dim4_t shape, bool swap_layout, Stream& stream,
                BinaryOp binary_op) {
        using namespace details;
        constexpr AccessorTraits TRAITS = RESTRICT ? AccessorTraits::RESTRICT : AccessorTraits::DEFAULT;
        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(lhs, stream.device());
        NOA_ASSERT_DEVICE_PTR(rhs, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());

        if (swap_layout) {
            const auto order = indexing::order(output_strides, shape);
            shape = indexing::reorder(shape, order);
            output_strides = indexing::reorder(output_strides, order);
            lhs_strides = indexing::reorder(lhs_strides, order);
            rhs_strides = indexing::reorder(rhs_strides, order);
        }

        const bool4_t is_contiguous = indexing::isContiguous(lhs_strides, shape) &&
                                      indexing::isContiguous(rhs_strides, shape) &&
                                      indexing::isContiguous(output_strides, shape);
        if (is_contiguous[1] && is_contiguous[2]) { // 1D-like
            // Keep batches separated in a different Grid.Y if they're not contiguous.
            const auto elements = safe_cast<uint32_t>(
                    is_contiguous[0] ? shape.elements() : dim3_t{shape.get(1)}.elements());
            const dim3 blocks(noa::math::divideUp(elements, BinaryConfig::BLOCK_WORK_SIZE),
                              is_contiguous[0] ? 1 : shape[0]);
            const auto uint_lhs_strides = safe_cast<uint2_t>(dim2_t{lhs_strides[0], lhs_strides[3]});
            const auto uint_rhs_strides = safe_cast<uint2_t>(dim2_t{rhs_strides[0], rhs_strides[3]});
            const auto uint_output_strides = safe_cast<uint2_t>(dim2_t{output_strides[0], output_strides[3]});
            const LaunchConfig config{blocks, BinaryConfig::BLOCK_SIZE};

            uint32_t vec_size = is_contiguous[3] ? std::min({maxVectorCount(lhs),
                                                             maxVectorCount(rhs),
                                                             maxVectorCount(output)}) : 1;
            if (blocks.y > 1) { // make sure the beginning of each batch preserves the alignment
                const bool is_not_multiple = uint_lhs_strides[0] % vec_size ||
                                             uint_rhs_strides[0] % vec_size ||
                                             uint_output_strides[0] % vec_size;
                vec_size = is_not_multiple ? 1 : vec_size;
            }

            const Accessor<const LhsVal, 2, uint32_t, TRAITS> lhs_accessor(lhs, uint_lhs_strides);
            const Accessor<const RhsVal, 2, uint32_t, TRAITS> rhs_accessor(rhs, uint_rhs_strides);
            const Accessor<OutVal, 2, uint32_t, TRAITS> output_accessor(output, uint_output_strides);

            if (vec_size == 4) {
                return stream.enqueue(
                        name, binaryArray1D_<LhsVal, RhsVal, OutVal, BinaryOp, 4, TRAITS>, config,
                        lhs_accessor, rhs_accessor, output_accessor, elements, binary_op);
            } else if (vec_size == 2) {
                return stream.enqueue(
                        name, binaryArray1D_<LhsVal, RhsVal, OutVal, BinaryOp, 2, TRAITS>, config,
                        lhs_accessor, rhs_accessor, output_accessor, elements, binary_op);
            } else {
                return stream.enqueue(
                        name, binaryArray1D_<LhsVal, RhsVal, OutVal, BinaryOp, 1, TRAITS>, config,
                        lhs_accessor, rhs_accessor, output_accessor, elements, binary_op);
            }
        } else {
            const auto i_shape = safe_cast<uint2_t>(dim2_t(shape.get(2)));
            const uint32_t blocks_x = noa::math::divideUp(i_shape[1], BinaryConfig::BLOCK_WORK_SIZE_2D.x);
            const uint32_t blocks_y = noa::math::divideUp(i_shape[0], BinaryConfig::BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, BinaryConfig::BLOCK_SIZE_2D};

            const Accessor<const LhsVal, 4, uint32_t, TRAITS> lhs_accessor(lhs, safe_cast<uint4_t>(lhs_strides));
            const Accessor<const RhsVal, 4, uint32_t, TRAITS> rhs_accessor(rhs, safe_cast<uint4_t>(rhs_strides));
            const Accessor<OutVal, 4, uint32_t, TRAITS> output_accessor(output, safe_cast<uint4_t>(output_strides));

            stream.enqueue(name, binaryArray4D_<LhsVal, RhsVal, OutVal, BinaryOp, TRAITS>, config,
                           lhs_accessor, rhs_accessor, output_accessor, i_shape, binary_op, blocks_x);
        }
    }
}
