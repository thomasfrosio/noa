#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/utils/Traits.h"
#include "noa/gpu/cuda/utils/Block.cuh"
#include "noa/gpu/cuda/utils/Pointers.h"

namespace noa::cuda::utils::ewise::details {
    struct TrinaryConfig {
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

    template<typename lhs_val_t, typename mhs_val_t, typename rhs_val_t,
             typename out_val_t, typename trinary_t,
             int VEC_SIZE, AccessorTraits TRAITS>
    __global__ __launch_bounds__(TrinaryConfig::BLOCK_SIZE)
    void trinaryValue1D_(Accessor<const lhs_val_t, 2, uint32_t, TRAITS> lhs,
                         mhs_val_t mhs,
                         rhs_val_t rhs,
                         Accessor<out_val_t, 2, uint32_t, TRAITS> output,
                         uint32_t elements, trinary_t trinary_op) {
        constexpr uint32_t BLOCK_SIZE = TrinaryConfig::BLOCK_SIZE;
        constexpr uint32_t BLOCK_WORK_SIZE = TrinaryConfig::BLOCK_WORK_SIZE;
        constexpr uint32_t EPT = TrinaryConfig::ELEMENTS_PER_THREAD;

        const uint32_t batch = blockIdx.y;
        const uint32_t base = BLOCK_WORK_SIZE * blockIdx.x;
        const auto lhs_ = lhs[batch];
        const auto out_ = output[batch];

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int32_t i = 0; i < EPT; ++i) {
                const uint32_t gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements)
                    out_[gid] = static_cast<out_val_t>(trinary_op(lhs_[gid], mhs, rhs));
            }
        } else {
            NOA_ASSERT(lhs_.stride(0) == 1 && out_.stride(0) == 1);
            using lptr_t = typename decltype(lhs)::pointer_type;
            using optr_t = typename decltype(output)::pointer_type;
            lptr_t lhs_ptr = lhs_.get() + base;
            optr_t out_ptr = out_.get() + base;

            const uint32_t remaining = elements - base;
            if (remaining < BLOCK_WORK_SIZE) {
                #pragma unroll
                for (int32_t i = 0; i < EPT; ++i) {
                    const uint32_t offset = BLOCK_SIZE * i + threadIdx.x;
                    if (offset < remaining)
                    out_ptr[offset] = static_cast<out_val_t>(trinary_op(lhs_ptr[offset], mhs, rhs));
                }
            } else {
                lhs_val_t args[EPT];
                out_val_t results[EPT];
                block::vectorizedLoad<BLOCK_SIZE, EPT, VEC_SIZE>(lhs_ptr, args, threadIdx.x);
                #pragma unroll
                for (uint32_t i = 0; i < EPT; ++i)
                    results[i] = static_cast<out_val_t>(trinary_op(args[i], mhs, rhs));
                block::vectorizedStore<BLOCK_SIZE, EPT, VEC_SIZE>(results, out_ptr, threadIdx.x);
            }
        }
    }

    template<typename lhs_val_t, typename mhs_val_t, typename rhs_val_t,
             typename out_val_t, typename trinary_t, AccessorTraits TRAITS>
    __global__ __launch_bounds__(TrinaryConfig::BLOCK_SIZE)
    void trinaryValue4D_(Accessor<const lhs_val_t, 4, uint32_t, TRAITS> lhs,
                         mhs_val_t mhs, rhs_val_t rhs,
                         Accessor<out_val_t, 4, uint32_t, TRAITS> out,
                         uint2_t shape, trinary_t trinary_op, uint32_t blocks_x) {

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         TrinaryConfig::BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         TrinaryConfig::BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        const auto lhs_ = lhs[gid[0]][gid[1]];
        const auto out_ = out[gid[0]][gid[1]];

        #pragma unroll
        for (int32_t k = 0; k < TrinaryConfig::ELEMENTS_PER_THREAD_2D; ++k) {
            #pragma unroll
            for (int32_t l = 0; l < TrinaryConfig::ELEMENTS_PER_THREAD_2D; ++l) {
                const uint32_t ik = gid[2] + TrinaryConfig::BLOCK_SIZE_2D.y * k;
                const uint32_t il = gid[3] + TrinaryConfig::BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1])
                    out_(ik, il) = static_cast<out_val_t>(trinary_op(lhs_(ik, il), mhs, rhs));
            }
        }
    }

    template<typename lhs_val_t, typename mhs_val_t, typename rhs_val_t,
             typename out_val_t, typename trinary_t,
             int VEC_SIZE, AccessorTraits TRAITS>
    __global__ __launch_bounds__(TrinaryConfig::BLOCK_SIZE)
    void trinaryArrayValue1D_(Accessor<const lhs_val_t, 2, uint32_t, TRAITS> lhs,
                              Accessor<const mhs_val_t, 2, uint32_t, TRAITS> mhs,
                              rhs_val_t rhs,
                              Accessor<out_val_t, 2, uint32_t, TRAITS> out,
                              uint32_t elements, trinary_t trinary_op) {
        constexpr uint32_t BLOCK_SIZE = TrinaryConfig::BLOCK_SIZE;
        constexpr uint32_t BLOCK_WORK_SIZE = TrinaryConfig::BLOCK_WORK_SIZE;
        constexpr uint32_t EPT = TrinaryConfig::ELEMENTS_PER_THREAD;

        const uint32_t batch = blockIdx.y;
        const uint32_t base = BLOCK_WORK_SIZE * blockIdx.x;
        const auto lhs_ = lhs[batch];
        const auto mhs_ = mhs[batch];
        const auto out_ = out[batch];

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int32_t i = 0; i < EPT; ++i) {
                const uint32_t gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements)
                    out_[gid] = static_cast<out_val_t>(trinary_op(lhs_[gid], mhs_[gid], rhs));
            }
        } else {
            NOA_ASSERT(lhs_.stride(0) == 1 && mhs_.stride(0) == 1 && out_.stride(0) == 1);
            using lptr_t = typename decltype(lhs)::pointer_type;
            using mptr_t = typename decltype(mhs)::pointer_type;
            using optr_t = typename decltype(out)::pointer_type;
            lptr_t lhs_ptr = lhs_.get() + base;
            mptr_t mhs_ptr = mhs_.get() + base;
            optr_t out_ptr = out_.get() + base;

            const uint32_t remaining = elements - base;
            if (remaining < BLOCK_WORK_SIZE) {
                #pragma unroll
                for (int32_t i = 0; i < EPT; ++i) {
                    const uint32_t offset = BLOCK_SIZE * i + threadIdx.x;
                    if (offset < remaining)
                        out_ptr[offset] = static_cast<out_val_t>(trinary_op(lhs_ptr[offset], mhs_ptr[offset], rhs));
                }
            } else {
                lhs_val_t ilhs[EPT];
                mhs_val_t imhs[EPT];
                out_val_t results[EPT];
                block::vectorizedLoad<BLOCK_SIZE, EPT, VEC_SIZE>(lhs_ptr, ilhs, threadIdx.x);
                block::vectorizedLoad<BLOCK_SIZE, EPT, VEC_SIZE>(mhs_ptr, imhs, threadIdx.x);
                #pragma unroll
                for (uint32_t i = 0; i < EPT; ++i)
                    results[i] = static_cast<out_val_t>(trinary_op(ilhs[i], imhs[i], rhs));
                block::vectorizedStore<BLOCK_SIZE, EPT, VEC_SIZE>(results, out_ptr, threadIdx.x);
            }
        }
    }

    template<typename lhs_val_t, typename mhs_val_t, typename rhs_val_t,
             typename out_val_t, typename trinary_t, AccessorTraits TRAITS>
    __global__ __launch_bounds__(TrinaryConfig::BLOCK_SIZE)
    void trinaryArrayValue4D_(Accessor<const lhs_val_t, 4, uint32_t, TRAITS> lhs,
                              Accessor<const mhs_val_t, 4, uint32_t, TRAITS> mhs,
                              rhs_val_t rhs,
                              Accessor<out_val_t, 4, uint32_t, TRAITS> out,
                              uint2_t shape, trinary_t trinary_op, uint32_t blocks_x) {

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         TrinaryConfig::BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         TrinaryConfig::BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        const auto lhs_ = lhs[gid[0]][gid[1]];
        const auto mhs_ = mhs[gid[0]][gid[1]];
        const auto out_ = out[gid[0]][gid[1]];

        #pragma unroll
        for (int32_t k = 0; k < TrinaryConfig::ELEMENTS_PER_THREAD_2D; ++k) {
            #pragma unroll
            for (int32_t l = 0; l < TrinaryConfig::ELEMENTS_PER_THREAD_2D; ++l) {
                const uint32_t ik = gid[2] + TrinaryConfig::BLOCK_SIZE_2D.y * k;
                const uint32_t il = gid[3] + TrinaryConfig::BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1])
                    out_(ik, il) = static_cast<out_val_t>(trinary_op(lhs_(ik, il), mhs_(ik, il), rhs));
            }
        }
    }

    template<typename lhs_val_t, typename mhs_val_t, typename rhs_val_t,
             typename out_val_t, typename trinary_t,
             int VEC_SIZE, AccessorTraits TRAITS>
    __global__ __launch_bounds__(TrinaryConfig::BLOCK_SIZE)
    void trinaryValueArray1D_(Accessor<const lhs_val_t, 2, uint32_t, TRAITS> lhs,
                              mhs_val_t mhs,
                              Accessor<const rhs_val_t, 2, uint32_t, TRAITS> rhs,
                              Accessor<out_val_t, 2, uint32_t, TRAITS> out,
                              uint32_t elements, trinary_t trinary_op) {
        constexpr uint32_t BLOCK_SIZE = TrinaryConfig::BLOCK_SIZE;
        constexpr uint32_t BLOCK_WORK_SIZE = TrinaryConfig::BLOCK_WORK_SIZE;
        constexpr uint32_t EPT = TrinaryConfig::ELEMENTS_PER_THREAD;

        const uint32_t batch = blockIdx.y;
        const uint32_t base = BLOCK_WORK_SIZE * blockIdx.x;
        const auto lhs_ = lhs[batch];
        const auto rhs_ = rhs[batch];
        const auto out_ = out[batch];

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int32_t i = 0; i < EPT; ++i) {
                const uint32_t gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements)
                    out_[gid] = static_cast<out_val_t>(trinary_op(lhs_[gid], mhs, rhs_[gid]));
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
                for (int32_t i = 0; i < EPT; ++i) {
                    const uint32_t offset = BLOCK_SIZE * i + threadIdx.x;
                    if (offset < remaining)
                        out_ptr[offset] = static_cast<out_val_t>(trinary_op(lhs_ptr[offset], mhs, rhs_ptr[offset]));
                }
            } else {
                lhs_val_t ilhs[EPT];
                rhs_val_t irhs[EPT];
                out_val_t results[EPT];
                block::vectorizedLoad<BLOCK_SIZE, EPT, VEC_SIZE>(lhs_ptr, ilhs, threadIdx.x);
                block::vectorizedLoad<BLOCK_SIZE, EPT, VEC_SIZE>(rhs_ptr, irhs, threadIdx.x);
                #pragma unroll
                for (uint32_t i = 0; i < EPT; ++i)
                    results[i] = static_cast<out_val_t>(trinary_op(ilhs[i], mhs, irhs[i]));
                block::vectorizedStore<BLOCK_SIZE, EPT, VEC_SIZE>(results, out_ptr, threadIdx.x);
            }
        }
    }

    template<typename lhs_val_t, typename mhs_val_t, typename rhs_val_t,
             typename out_val_t, typename trinary_t, AccessorTraits TRAITS>
    __global__ __launch_bounds__(TrinaryConfig::BLOCK_SIZE)
    void trinaryValueArray4D_(Accessor<const lhs_val_t, 4, uint32_t, TRAITS> lhs,
                              mhs_val_t mhs,
                              Accessor<const rhs_val_t, 4, uint32_t, TRAITS> rhs,
                              Accessor<out_val_t, 4, uint32_t, TRAITS> out,
                              uint2_t shape, trinary_t trinary_op, uint32_t blocks_x) {

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         TrinaryConfig::BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         TrinaryConfig::BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        const auto lhs_ = lhs[gid[0]][gid[1]];
        const auto rhs_ = rhs[gid[0]][gid[1]];
        const auto out_ = out[gid[0]][gid[1]];

        #pragma unroll
        for (int32_t k = 0; k < TrinaryConfig::ELEMENTS_PER_THREAD_2D; ++k) {
            #pragma unroll
            for (int32_t l = 0; l < TrinaryConfig::ELEMENTS_PER_THREAD_2D; ++l) {
                const uint32_t ik = gid[2] + TrinaryConfig::BLOCK_SIZE_2D.y * k;
                const uint32_t il = gid[3] + TrinaryConfig::BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1])
                    out_(ik, il) = static_cast<out_val_t>(trinary_op(lhs_(ik, il), mhs, rhs_(ik, il)));
            }
        }
    }

    template<typename lhs_val_t, typename mhs_val_t, typename rhs_val_t,
             typename out_val_t, typename trinary_t,
             int VEC_SIZE, AccessorTraits TRAITS>
    __global__ __launch_bounds__(TrinaryConfig::BLOCK_SIZE)
    void trinaryArray1D_(Accessor<const lhs_val_t, 2, uint32_t, TRAITS> lhs,
                         Accessor<const mhs_val_t, 2, uint32_t, TRAITS> mhs,
                         Accessor<const rhs_val_t, 2, uint32_t, TRAITS> rhs,
                         Accessor<out_val_t, 2, uint32_t, TRAITS> out,
                         uint32_t elements, trinary_t trinary_op) {
        constexpr uint32_t BLOCK_SIZE = TrinaryConfig::BLOCK_SIZE;
        constexpr uint32_t BLOCK_WORK_SIZE = TrinaryConfig::BLOCK_WORK_SIZE;
        constexpr uint32_t EPT = TrinaryConfig::ELEMENTS_PER_THREAD;

        const uint32_t batch = blockIdx.y;
        const uint32_t base = BLOCK_WORK_SIZE * blockIdx.x;
        const auto lhs_ = lhs[batch];
        const auto mhs_ = mhs[batch];
        const auto rhs_ = rhs[batch];
        const auto out_ = out[batch];

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int32_t i = 0; i < EPT; ++i) {
                const uint32_t gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements)
                    out_[gid] = static_cast<out_val_t>(trinary_op(lhs_[gid], mhs_[gid], rhs_[gid]));
            }
        } else {
            NOA_ASSERT(lhs_.stride(0) == 1 && mhs_.stride(0) == 1 &&
                       rhs_.stride(0) == 1 && out_.stride(0) == 1);
            using lptr_t = typename decltype(lhs)::pointer_type;
            using mptr_t = typename decltype(mhs)::pointer_type;
            using rptr_t = typename decltype(rhs)::pointer_type;
            using optr_t = typename decltype(out)::pointer_type;
            lptr_t lhs_ptr = lhs_.get() + base;
            mptr_t mhs_ptr = mhs_.get() + base;
            rptr_t rhs_ptr = rhs_.get() + base;
            optr_t out_ptr = out_.get() + base;

            const uint32_t remaining = elements - base;
            if (remaining < BLOCK_WORK_SIZE) {
                #pragma unroll
                for (int32_t i = 0; i < EPT; ++i) {
                    const uint32_t offset = BLOCK_SIZE * i + threadIdx.x;
                    if (offset < remaining) {
                        out_ptr[offset] = static_cast<out_val_t>(trinary_op(
                                lhs_ptr[offset], mhs_ptr[offset], rhs_ptr[offset]));
                    }
                }
            } else {
                lhs_val_t ilhs[EPT];
                mhs_val_t imhs[EPT];
                rhs_val_t irhs[EPT];
                out_val_t results[EPT];
                block::vectorizedLoad<BLOCK_SIZE, EPT, VEC_SIZE>(lhs_ptr, ilhs, threadIdx.x);
                block::vectorizedLoad<BLOCK_SIZE, EPT, VEC_SIZE>(mhs_ptr, imhs, threadIdx.x);
                block::vectorizedLoad<BLOCK_SIZE, EPT, VEC_SIZE>(rhs_ptr, irhs, threadIdx.x);
                #pragma unroll
                for (uint32_t i = 0; i < EPT; ++i)
                    results[i] = static_cast<out_val_t>(trinary_op(ilhs[i], imhs[i], irhs[i]));
                block::vectorizedStore<BLOCK_SIZE, EPT, VEC_SIZE>(results, out_ptr, threadIdx.x);
            }
        }
    }

    template<typename lhs_val_t, typename mhs_val_t, typename rhs_val_t,
             typename out_val_t, typename trinary_t, AccessorTraits TRAITS>
    __global__ __launch_bounds__(TrinaryConfig::BLOCK_SIZE)
    void trinaryArray4D_(Accessor<const lhs_val_t, 4, uint32_t, TRAITS> lhs,
                         Accessor<const mhs_val_t, 4, uint32_t, TRAITS> mhs,
                         Accessor<const rhs_val_t, 4, uint32_t, TRAITS> rhs,
                         Accessor<out_val_t, 4, uint32_t, TRAITS> out,
                         uint2_t shape, trinary_t trinary_op, uint32_t blocks_x) {

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         TrinaryConfig::BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         TrinaryConfig::BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        const auto lhs_ = lhs[gid[0]][gid[1]];
        const auto mhs_ = mhs[gid[0]][gid[1]];
        const auto rhs_ = rhs[gid[0]][gid[1]];
        const auto out_ = out[gid[0]][gid[1]];

        #pragma unroll
        for (int32_t k = 0; k < TrinaryConfig::ELEMENTS_PER_THREAD_2D; ++k) {
            #pragma unroll
            for (int32_t l = 0; l < TrinaryConfig::ELEMENTS_PER_THREAD_2D; ++l) {
                const uint32_t ik = gid[2] + TrinaryConfig::BLOCK_SIZE_2D.y * k;
                const uint32_t il = gid[3] + TrinaryConfig::BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1])
                    out_(ik, il) = static_cast<out_val_t>(trinary_op(lhs_(ik, il), mhs_(ik, il), rhs_(ik, il)));
            }
        }
    }
}

namespace noa::cuda::utils::ewise {
    // Apply a trinary operator, element-wise.
    // RESTRICT:        Whether the pointers can be accessed using the __restrict__ attribute.
    // name:            Name of the function. Used for logging if kernel launch fails.
    // lhs:             On the device. Left-hand side argument.
    // lhs_strides:     Strides of lhs.
    // mhs:             Middle-hand side argument.
    // rhs:             Right-hand side argument.
    // output:          On the device. Transformed array.
    // output_strides:  Strides of output.
    // shape:           Shape of lhs and output.
    // swap_layout:     Swap the memory layout to optimize output writes.
    //                  If false, assume rightmost order is the fastest order.
    // stream:          Stream on which to enqueue this function.
    // trinary_op:      Trinary operator. The output is explicitly casted to V.
    // This function is asynchronous relative to the host and may return before completion.
    // One must make sure lhs and output stay valid until completion.
    template<bool RESTRICT = false,
             typename lhs_val_t, typename mhs_t, typename rhs_t,
             typename out_val_t, typename trinary_t,
             typename = std::enable_if_t<noa::traits::is_data_v<mhs_t> && noa::traits::is_data_v<rhs_t>>>
    void trinary(const char* name,
                 const lhs_val_t* lhs, dim4_t lhs_strides,
                 mhs_t mhs, rhs_t rhs,
                 out_val_t* output, dim4_t output_strides,
                 dim4_t shape, bool swap_layout, Stream& stream,
                 trinary_t trinary_op) {
        using namespace details;
        using mhs_val_t = noa::traits::remove_ref_cv_t<mhs_t>;
        using rhs_val_t = noa::traits::remove_ref_cv_t<rhs_t>;
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
        if (is_contiguous[1] && is_contiguous[2]) {
            uint32_t elements, blocks_y;
            if (!is_contiguous[0]) {
                elements = safe_cast<uint32_t>(shape[1] * shape[2] * shape[3]);
                blocks_y = shape[0];
            } else {
                elements = safe_cast<uint32_t>(shape.elements());
                blocks_y = 1;
            }
            const dim3 blocks(noa::math::divideUp(elements, TrinaryConfig::BLOCK_WORK_SIZE), blocks_y);
            const auto uint_lhs_strides = safe_cast<uint2_t>(dim2_t{lhs_strides[0], lhs_strides[3]});
            const auto uint_output_strides = safe_cast<uint2_t>(dim2_t{output_strides[0], output_strides[3]});
            const LaunchConfig config{blocks, TrinaryConfig::BLOCK_SIZE};

            uint32_t vec_size = is_contiguous[3] ? std::min(maxVectorCount(lhs), maxVectorCount(output)) : 1;
            if (blocks.y > 1)
                vec_size = uint_lhs_strides[0] % vec_size || uint_output_strides[0] % vec_size ? 1 : vec_size;

            const Accessor<const lhs_val_t, 2, uint32_t, TRAITS> lhs_accessor(lhs, uint_lhs_strides);
            const Accessor<out_val_t, 2, uint32_t, TRAITS> output_accessor(output, uint_output_strides);

            if (vec_size == 4) {
                return stream.enqueue(
                        name, trinaryValue1D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, 4, TRAITS>,
                        config, lhs_accessor, mhs, rhs, output_accessor, elements, trinary_op);
            } else if (vec_size == 2) {
                return stream.enqueue(
                        name, trinaryValue1D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, 2, TRAITS>,
                        config, lhs_accessor, mhs, rhs, output_accessor, elements, trinary_op);
            } else {
                return stream.enqueue(
                        name, trinaryValue1D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, 1, TRAITS>,
                        config, lhs_accessor, mhs, rhs, output_accessor, elements, trinary_op);
            }
        } else {
            const auto i_shape = safe_cast<uint2_t>(dim2_t(shape.get(2)));
            const uint32_t blocks_x = noa::math::divideUp(i_shape[1], TrinaryConfig::BLOCK_WORK_SIZE_2D.x);
            const uint32_t blocks_y = noa::math::divideUp(i_shape[0], TrinaryConfig::BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, TrinaryConfig::BLOCK_SIZE_2D};

            const Accessor<const lhs_val_t, 4, uint32_t, TRAITS> lhs_accessor(lhs, safe_cast<uint4_t>(lhs_strides));
            const Accessor<out_val_t, 4, uint32_t, TRAITS> output_accessor(output, safe_cast<uint4_t>(output_strides));

            stream.enqueue(name, trinaryValue4D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, TRAITS>,
                           config, lhs_accessor, mhs, rhs, output_accessor, i_shape, trinary_op, blocks_x);
        }
    }

    // Apply a trinary operator, element-wise.
    // RESTRICT:        Whether the pointers can be accessed using the __restrict__ attribute.
    // name:            Name of the function. Used for logging if kernel launch fails.
    // lhs:             On the device. Left-hand side argument.
    // lhs_strides:     Strides of lhs.
    // mhs:             On the device. Middle-hand side argument.
    // mhs_strides:     Strides of mhs.
    // rhs:             Right-hand side argument.
    // output:          On the device. Transformed array.
    // output_strides:  Strides of output.
    // shape:           Shape of lhs, mhs and output.
    // swap_layout:     Swap the memory layout to optimize output writes.
    //                  If false, assume rightmost order is the fastest order.
    // stream:          Stream on which to enqueue this function.
    // trinary_op:      Trinary operator. The output is explicitly casted to V.
    // This function is asynchronous relative to the host and may return before completion.
    // One must make sure lhs, mhs and output stay valid until completion.
    template<bool RESTRICT = false,
             typename lhs_val_t, typename mhs_val_t, typename rhs_t,
             typename out_val_t, typename trinary_t,
             typename = std::enable_if_t<noa::traits::is_data_v<rhs_t>>>
    void trinary(const char* name,
                 const lhs_val_t* lhs, dim4_t lhs_strides,
                 const mhs_val_t* mhs, dim4_t mhs_strides, rhs_t rhs,
                 out_val_t* output, dim4_t output_strides,
                 dim4_t shape, bool swap_layout, Stream& stream,
                 trinary_t trinary_op) {
        using namespace details;
        using rhs_val_t = noa::traits::remove_ref_cv_t<rhs_t>;
        constexpr AccessorTraits TRAITS = RESTRICT ? AccessorTraits::RESTRICT : AccessorTraits::DEFAULT;
        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(lhs, stream.device());
        NOA_ASSERT_DEVICE_PTR(mhs, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());

        if (swap_layout) {
            const auto order = indexing::order(output_strides, shape);
            shape = indexing::reorder(shape, order);
            output_strides = indexing::reorder(output_strides, order);
            lhs_strides = indexing::reorder(lhs_strides, order);
            mhs_strides = indexing::reorder(mhs_strides, order);
        }

        const bool4_t is_contiguous = indexing::isContiguous(lhs_strides, shape) &&
                                      indexing::isContiguous(mhs_strides, shape) &&
                                      indexing::isContiguous(output_strides, shape);
        if (is_contiguous[1] && is_contiguous[2]) {
            uint32_t elements, blocks_y;
            if (!is_contiguous[0]) {
                elements = safe_cast<uint32_t>(shape[1] * shape[2] * shape[3]);
                blocks_y = shape[0];
            } else {
                elements = safe_cast<uint32_t>(shape.elements());
                blocks_y = 1;
            }
            const dim3 blocks(noa::math::divideUp(elements, TrinaryConfig::BLOCK_WORK_SIZE), blocks_y);
            const auto uint_lhs_strides = safe_cast<uint2_t>(dim2_t{lhs_strides[0], lhs_strides[3]});
            const auto uint_mhs_strides = safe_cast<uint2_t>(dim2_t{mhs_strides[0], mhs_strides[3]});
            const auto uint_output_strides = safe_cast<uint2_t>(dim2_t{output_strides[0], output_strides[3]});
            const LaunchConfig config{blocks, TrinaryConfig::BLOCK_SIZE};

            uint32_t vec_size = is_contiguous[3] ? std::min({maxVectorCount(lhs),
                                                         maxVectorCount(mhs),
                                                         maxVectorCount(output)}) : 1;
            if (blocks.y > 1) {
                vec_size = uint_lhs_strides[0] % vec_size ||
                           uint_mhs_strides[0] % vec_size ||
                           uint_output_strides[0] % vec_size ? 1 : vec_size;
            }

            const Accessor<const lhs_val_t, 2, uint32_t, TRAITS> lhs_accessor(lhs, uint_lhs_strides);
            const Accessor<const mhs_val_t, 2, uint32_t, TRAITS> mhs_accessor(mhs, uint_mhs_strides);
            const Accessor<out_val_t, 2, uint32_t, TRAITS> output_accessor(output, uint_output_strides);

            if (vec_size == 4) {
                return stream.enqueue(
                        name, trinaryArrayValue1D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, 4, TRAITS>,
                        config, lhs_accessor, mhs_accessor, rhs, output_accessor, elements, trinary_op);
            } else if (vec_size == 2) {
                return stream.enqueue(
                        name, trinaryArrayValue1D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, 2, TRAITS>,
                        config, lhs_accessor, mhs_accessor, rhs, output_accessor, elements, trinary_op);
            } else {
                return stream.enqueue(
                        name, trinaryArrayValue1D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, 1, TRAITS>,
                        config, lhs_accessor, mhs_accessor, rhs, output_accessor, elements, trinary_op);
            }
        } else {
            const auto i_shape = safe_cast<uint2_t>(dim2_t(shape.get(2)));
            const uint32_t blocks_x = noa::math::divideUp(i_shape[1], TrinaryConfig::BLOCK_WORK_SIZE_2D.x);
            const uint32_t blocks_y = noa::math::divideUp(i_shape[0], TrinaryConfig::BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, TrinaryConfig::BLOCK_SIZE_2D};

            const Accessor<const lhs_val_t, 4, uint32_t, TRAITS> lhs_accessor(lhs, safe_cast<uint4_t>(lhs_strides));
            const Accessor<const mhs_val_t, 4, uint32_t, TRAITS> mhs_accessor(mhs, safe_cast<uint4_t>(mhs_strides));
            const Accessor<out_val_t, 4, uint32_t, TRAITS> output_accessor(output, safe_cast<uint4_t>(output_strides));

            stream.enqueue(name, trinaryArrayValue4D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, TRAITS>,
                           config, lhs_accessor, mhs_accessor, rhs, output_accessor, i_shape, trinary_op, blocks_x);
        }
    }

    // Apply a trinary operator, element-wise.
    // RESTRICT:        Whether the pointers can be accessed using the __restrict__ attribute.
    // name:            Name of the function. Used for logging if kernel launch fails.
    // lhs:             On the device. Left-hand side argument.
    // lhs_strides:     Strides of lhs.
    // mhs:             Middle-hand side argument.
    // rhs:             On the device. Right-hand side argument.
    // rhs_strides:     Strides of rhs.
    // output:          On the device. Transformed array.
    // output_strides:  Strides of output.
    // shape:           Shape of lhs, rhs and output.
    // swap_layout:     Swap the memory layout to optimize output writes.
    //                  If false, assume rightmost order is the fastest order.
    // stream:          Stream on which to enqueue this function.
    // trinary_op:      Trinary operator. The output is explicitly cast to V.
    // This function is asynchronous relative to the host and may return before completion.
    // One must make sure lhs, rhs and output stay valid until completion.
    template<bool RESTRICT = false,
             typename lhs_val_t, typename mhs_t, typename rhs_val_t,
             typename out_val_t, typename trinary_t,
             typename = std::enable_if_t<noa::traits::is_data_v<mhs_t>>>
    void trinary(const char* name,
                 const lhs_val_t* lhs, dim4_t lhs_strides, mhs_t mhs,
                 const rhs_val_t* rhs, dim4_t rhs_strides,
                 out_val_t* output, dim4_t output_strides,
                 dim4_t shape, bool swap_layout, Stream& stream,
                 trinary_t trinary_op) {
        using namespace details;
        using mhs_val_t = noa::traits::remove_ref_cv_t<mhs_t>;
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
        if (is_contiguous[1] && is_contiguous[2]) {
            uint32_t elements, blocks_y;
            if (!is_contiguous[0]) {
                elements = safe_cast<uint32_t>(shape[1] * shape[2] * shape[3]);
                blocks_y = shape[0];
            } else {
                elements = safe_cast<uint32_t>(shape.elements());
                blocks_y = 1;
            }
            const dim3 blocks(noa::math::divideUp(elements, TrinaryConfig::BLOCK_WORK_SIZE), blocks_y);
            const auto uint_lhs_strides = safe_cast<uint2_t>(dim2_t{lhs_strides[0], lhs_strides[3]});
            const auto uint_rhs_strides = safe_cast<uint2_t>(dim2_t{rhs_strides[0], rhs_strides[3]});
            const auto uint_output_strides = safe_cast<uint2_t>(dim2_t{output_strides[0], output_strides[3]});
            const LaunchConfig config{blocks, TrinaryConfig::BLOCK_SIZE};

            uint32_t vec_size = is_contiguous[3] ? std::min({maxVectorCount(lhs),
                                                             maxVectorCount(rhs),
                                                             maxVectorCount(output)}) : 1;
            if (blocks.y > 1) {
                vec_size = uint_lhs_strides[0] % vec_size ||
                           uint_rhs_strides[0] % vec_size ||
                           uint_output_strides[0] % vec_size ? 1 : vec_size;
            }

            const Accessor<const lhs_val_t, 2, uint32_t, TRAITS> lhs_accessor(lhs, uint_lhs_strides);
            const Accessor<const rhs_val_t, 2, uint32_t, TRAITS> rhs_accessor(rhs, uint_rhs_strides);
            const Accessor<out_val_t, 2, uint32_t, TRAITS> output_accessor(output, uint_output_strides);

            if (vec_size == 4) {
                return stream.enqueue(
                        name, trinaryValueArray1D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, 4, TRAITS>,
                        config, lhs_accessor, mhs, rhs_accessor, output_accessor, elements, trinary_op);
            } else if (vec_size == 2) {
                return stream.enqueue(
                        name, trinaryValueArray1D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, 2, TRAITS>,
                        config, lhs_accessor, mhs, rhs_accessor, output_accessor, elements, trinary_op);
            } else {
                return stream.enqueue(
                        name, trinaryValueArray1D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, 1, TRAITS>,
                        config, lhs_accessor, mhs, rhs_accessor, output_accessor, elements, trinary_op);
            }
        } else {
            const auto i_shape = safe_cast<uint2_t>(dim2_t(shape.get(2)));
            const uint32_t blocks_x = noa::math::divideUp(i_shape[1], TrinaryConfig::BLOCK_WORK_SIZE_2D.x);
            const uint32_t blocks_y = noa::math::divideUp(i_shape[0], TrinaryConfig::BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, TrinaryConfig::BLOCK_SIZE_2D};

            const Accessor<const lhs_val_t, 4, uint32_t, TRAITS> lhs_accessor(lhs, safe_cast<uint4_t>(lhs_strides));
            const Accessor<const rhs_val_t, 4, uint32_t, TRAITS> rhs_accessor(rhs, safe_cast<uint4_t>(rhs_strides));
            const Accessor<out_val_t, 4, uint32_t, TRAITS> output_accessor(output, safe_cast<uint4_t>(output_strides));

            stream.enqueue(name, trinaryValueArray4D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, TRAITS>,
                           config, lhs_accessor, mhs, rhs_accessor, output_accessor, i_shape, trinary_op, blocks_x);
        }
    }

    // Apply a trinary operator, element-wise.
    // RESTRICT:    Whether the pointers can be accessed using the __restrict__ attribute.
    // name:        Name of the function. Used for logging if kernel launch fails.
    // lhs:         On the device. Left-hand side argument.
    // lhs_strides: Strides of lhs.
    // mhs:         On the device. Middle-hand side argument.
    // mhs_strides: Strides of mhs.
    // rhs:         On the device. Right-hand side argument.
    // rhs_strides: Strides of rhs.
    // output:      On the device. Transformed array.
    // shape:       Shape of lhs, mhs, rhs and output.
    // swap_layout: Swap the memory layout to optimize output writes.
    //              If false, assume rightmost order is the fastest order.
    // stream:      Stream on which to enqueue this function. No synchronization is performed on the stream.
    // trinary_op:  Trinary operator. The output is explicitly cast to the output type.
    // This function is asynchronous relative to the host and may return before completion.
    // One must make sure lhs, mhs, rhs and output stay valid until completion.
    template<bool RESTRICT = false,
             typename lhs_val_t, typename mhs_val_t, typename rhs_val_t,
             typename out_val_t, typename trinary_t>
    void trinary(const char* name,
                 const lhs_val_t* lhs, dim4_t lhs_strides,
                 const mhs_val_t* mhs, dim4_t mhs_strides,
                 const rhs_val_t* rhs, dim4_t rhs_strides,
                 out_val_t* output, dim4_t output_strides, dim4_t shape,
                 bool swap_layout, Stream& stream,
                 trinary_t trinary_op) {
        using namespace details;
        constexpr AccessorTraits TRAITS = RESTRICT ? AccessorTraits::RESTRICT : AccessorTraits::DEFAULT;
        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(lhs, stream.device());
        NOA_ASSERT_DEVICE_PTR(mhs, stream.device());
        NOA_ASSERT_DEVICE_PTR(rhs, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());

        if (swap_layout) {
            const auto order = indexing::order(output_strides, shape);
            shape = indexing::reorder(shape, order);
            output_strides = indexing::reorder(output_strides, order);
            lhs_strides = indexing::reorder(lhs_strides, order);
            mhs_strides = indexing::reorder(mhs_strides, order);
            rhs_strides = indexing::reorder(rhs_strides, order);
        }

        const bool4_t is_contiguous = indexing::isContiguous(lhs_strides, shape) &&
                                      indexing::isContiguous(mhs_strides, shape) &&
                                      indexing::isContiguous(rhs_strides, shape) &&
                                      indexing::isContiguous(output_strides, shape);
        if (is_contiguous[1] && is_contiguous[2]) {
            const auto elements = safe_cast<uint32_t>(
                    is_contiguous[0] ? shape.elements() : dim3_t(shape.get(1)).elements());
            const dim3 blocks(noa::math::divideUp(elements, TrinaryConfig::BLOCK_WORK_SIZE),
                              is_contiguous[0] ? 1 : shape[0]);
            const auto uint_lhs_strides = safe_cast<uint2_t>(dim2_t{lhs_strides[0], lhs_strides[3]});
            const auto uint_mhs_strides = safe_cast<uint2_t>(dim2_t{mhs_strides[0], mhs_strides[3]});
            const auto uint_rhs_strides = safe_cast<uint2_t>(dim2_t{rhs_strides[0], rhs_strides[3]});
            const auto uint_output_strides = safe_cast<uint2_t>(dim2_t{output_strides[0], output_strides[3]});
            const LaunchConfig config{blocks, TrinaryConfig::BLOCK_SIZE};

            uint32_t vec_size = is_contiguous[3] ? std::min({maxVectorCount(lhs), maxVectorCount(mhs),
                                                             maxVectorCount(rhs), maxVectorCount(output)}) : 1;
            if (blocks.y > 1) {
                vec_size = uint_lhs_strides[0] % vec_size || uint_mhs_strides[0] % vec_size ||
                           uint_rhs_strides[0] % vec_size || uint_output_strides[0] % vec_size ?
                           1 : vec_size;
            }

            const Accessor<const lhs_val_t, 2, uint32_t, TRAITS> lhs_accessor(lhs, uint_lhs_strides);
            const Accessor<const mhs_val_t, 2, uint32_t, TRAITS> mhs_accessor(mhs, uint_mhs_strides);
            const Accessor<const rhs_val_t, 2, uint32_t, TRAITS> rhs_accessor(rhs, uint_rhs_strides);
            const Accessor<out_val_t, 2, uint32_t, TRAITS> output_accessor(output, uint_output_strides);

            if (vec_size == 4) {
                return stream.enqueue(
                        name, trinaryArray1D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, 4, TRAITS>,
                        config, lhs_accessor, mhs_accessor, rhs_accessor, output_accessor, elements, trinary_op);
            } else if (vec_size == 2) {
                return stream.enqueue(
                        name, trinaryArray1D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, 2, TRAITS>,
                        config, lhs_accessor, mhs_accessor, rhs_accessor, output_accessor, elements, trinary_op);
            } else {
                return stream.enqueue(
                        name, trinaryArray1D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, 1, TRAITS>,
                        config, lhs_accessor, mhs_accessor, rhs_accessor, output_accessor, elements, trinary_op);
            }
        } else {
            const auto i_shape = safe_cast<uint2_t>(dim2_t(shape.get(2)));
            const uint32_t blocks_x = noa::math::divideUp(i_shape[1], TrinaryConfig::BLOCK_WORK_SIZE_2D.x);
            const uint32_t blocks_y = noa::math::divideUp(i_shape[0], TrinaryConfig::BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, TrinaryConfig::BLOCK_SIZE_2D};

            const Accessor<const lhs_val_t, 4, uint32_t, TRAITS> lhs_accessor(lhs, safe_cast<uint4_t>(lhs_strides));
            const Accessor<const mhs_val_t, 4, uint32_t, TRAITS> mhs_accessor(mhs, safe_cast<uint4_t>(mhs_strides));
            const Accessor<const rhs_val_t, 4, uint32_t, TRAITS> rhs_accessor(rhs, safe_cast<uint4_t>(rhs_strides));
            const Accessor<out_val_t, 4, uint32_t, TRAITS> output_accessor(output, safe_cast<uint4_t>(output_strides));

            stream.enqueue(name,
                           trinaryArray4D_<lhs_val_t, mhs_val_t, rhs_val_t, out_val_t, trinary_t, TRAITS>, config,
                           lhs_accessor, mhs_accessor, rhs_accessor, output_accessor, i_shape, trinary_op, blocks_x);
        }
    }
}
