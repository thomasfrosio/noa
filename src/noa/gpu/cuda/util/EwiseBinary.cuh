#pragma once

#include "noa/common/Definitions.h"
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

    template<typename lhs_val_t, typename rhs_val_t,
             typename out_val_t, typename binary_t,
             int VEC_SIZE, bool RESTRICT>
    __global__ __launch_bounds__(BinaryConfig::BLOCK_SIZE)
    void binaryValueLeft1D_(accessor_t<RESTRICT, const lhs_val_t*> lhs, uint2_t lhs_strides,
                            rhs_val_t rhs,
                            accessor_t<RESTRICT, out_val_t*> output, uint2_t output_strides,
                            uint elements, binary_t binary_op) {
        constexpr uint BLOCK_SIZE = BinaryConfig::BLOCK_SIZE;
        constexpr uint BLOCK_WORK_SIZE = BinaryConfig::BLOCK_WORK_SIZE;
        constexpr uint EPT = BinaryConfig::ELEMENTS_PER_THREAD;

        using iptr_t = typename accessor_t<RESTRICT, const lhs_val_t*>::ptr_type;
        using optr_t = typename accessor_t<RESTRICT, out_val_t*>::ptr_type;
        const uint batch = blockIdx.y;
        const uint base = BLOCK_WORK_SIZE * blockIdx.x;

        iptr_t lhs_ = lhs.get() + batch * lhs_strides[0];
        optr_t out_ = output.get() + batch * output_strides[0];

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int i = 0; i < EPT; ++i) {
                const uint gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements)
                    out_[gid * output_strides[1]] = static_cast<out_val_t>(binary_op(lhs_[gid * lhs_strides[1]], rhs));
            }
        } else {
            lhs_ += base;
            out_ += base;
            const uint remaining = elements - base;
            if (remaining < BLOCK_WORK_SIZE) {
                for (int i = 0; i < EPT; ++i) {
                    const uint offset = BLOCK_SIZE * i + threadIdx.x;
                    if (offset < remaining)
                        out_[offset] = static_cast<out_val_t>(binary_op(lhs_[offset], rhs));
                }
            } else {
                lhs_val_t args[EPT];
                out_val_t results[EPT];
                block::vectorizedLoad<BLOCK_SIZE, EPT, VEC_SIZE>(lhs_, args, threadIdx.x);
                #pragma unroll
                for (uint i = 0; i < EPT; ++i)
                    results[i] = static_cast<out_val_t>(binary_op(args[i], rhs));
                block::vectorizedStore<BLOCK_SIZE, EPT, VEC_SIZE>(results, out_, threadIdx.x);
            }
        }
    }

    template<typename lhs_val_t, typename rhs_val_t,
             typename out_val_t, typename binary_t,
             int VEC_SIZE, bool RESTRICT>
    __global__ __launch_bounds__(BinaryConfig::BLOCK_SIZE)
    void binaryValueRight1D_(lhs_val_t lhs,
                             accessor_t<RESTRICT, const rhs_val_t*> rhs, uint2_t rhs_strides,
                             accessor_t<RESTRICT, out_val_t*> output, uint2_t output_strides,
                             uint elements, binary_t binary_op) {
        constexpr uint BLOCK_SIZE = BinaryConfig::BLOCK_SIZE;
        constexpr uint BLOCK_WORK_SIZE = BinaryConfig::BLOCK_WORK_SIZE;
        constexpr uint EPT = BinaryConfig::ELEMENTS_PER_THREAD;

        using iptr_t = typename accessor_t<RESTRICT, const rhs_val_t*>::ptr_type;
        using optr_t = typename accessor_t<RESTRICT, out_val_t*>::ptr_type;
        const uint batch = blockIdx.y;
        const uint base = BLOCK_WORK_SIZE * blockIdx.x;

        iptr_t rhs_ = rhs.get() + batch * rhs_strides[0];
        optr_t out_ = output.get() + batch * output_strides[0];

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int i = 0; i < EPT; ++i) {
                const uint gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements)
                    out_[gid * output_strides[1]] = static_cast<out_val_t>(binary_op(lhs, rhs_[gid * rhs_strides[1]]));
            }
        } else {
            rhs_ += base;
            out_ += base;
            const uint remaining = elements - base;
            if (remaining < BLOCK_WORK_SIZE) {
                for (int i = 0; i < EPT; ++i) {
                    const uint offset = BLOCK_SIZE * i + threadIdx.x;
                    if (offset < remaining)
                        out_[offset] = static_cast<out_val_t>(binary_op(lhs, rhs_[offset]));
                }
            } else {
                rhs_val_t args[EPT];
                out_val_t results[EPT];
                block::vectorizedLoad<BLOCK_SIZE, EPT, VEC_SIZE>(rhs_, args, threadIdx.x);
                #pragma unroll
                for (uint i = 0; i < EPT; ++i)
                    results[i] = static_cast<out_val_t>(binary_op(lhs, args[i]));
                block::vectorizedStore<BLOCK_SIZE, EPT, VEC_SIZE>(results, out_, threadIdx.x);
            }
        }
    }

    template<typename lhs_val_t, typename rhs_val_t,
             typename out_val_t, typename binary_t, bool RESTRICT>
    __global__ __launch_bounds__(BinaryConfig::BLOCK_SIZE)
    void binaryValueLeft4D_(accessor_t<RESTRICT, const lhs_val_t*> lhs, uint4_t lhs_strides,
                            rhs_val_t rhs,
                            accessor_t<RESTRICT, out_val_t*> output, uint4_t output_strides,
                            uint2_t shape, binary_t binary_op, uint blocks_x) {
        using iptr_t = typename accessor_t<RESTRICT, const lhs_val_t*>::ptr_type;
        using optr_t = typename accessor_t<RESTRICT, out_val_t*>::ptr_type;
        iptr_t lhs_ = lhs.get();
        optr_t out_ = output.get();

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         BinaryConfig::BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BinaryConfig::BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        lhs_ += indexing::at(gid[0], gid[1], lhs_strides);
        out_ += indexing::at(gid[0], gid[1], output_strides);

        #pragma unroll
        for (int k = 0; k < BinaryConfig::ELEMENTS_PER_THREAD_2D; ++k) {
            #pragma unroll
            for (int l = 0; l < BinaryConfig::ELEMENTS_PER_THREAD_2D; ++l) {
                const uint ik = gid[2] + BinaryConfig::BLOCK_SIZE_2D.y * k;
                const uint il = gid[3] + BinaryConfig::BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1])
                    out_[ik * output_strides[2] + il * output_strides[3]] =
                            static_cast<out_val_t>(binary_op(lhs_[ik * lhs_strides[2] + il * lhs_strides[3]], rhs));
            }
        }
    }

    template<typename lhs_val_t, typename rhs_val_t,
             typename out_val_t, typename binary_t, bool RESTRICT>
    __global__ __launch_bounds__(BinaryConfig::BLOCK_SIZE)
    void binaryValueRight4D_(lhs_val_t lhs,
                             accessor_t<RESTRICT, const rhs_val_t*> rhs, uint4_t rhs_strides,
                             accessor_t<RESTRICT, out_val_t*> output, uint4_t output_strides,
                             uint2_t shape, binary_t binary_op, uint blocks_x) {
        using iptr_t = typename accessor_t<RESTRICT, const rhs_val_t*>::ptr_type;
        using optr_t = typename accessor_t<RESTRICT, out_val_t*>::ptr_type;
        iptr_t rhs_ = rhs.get();
        optr_t out_ = output.get();

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         BinaryConfig::BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BinaryConfig::BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        rhs_ += indexing::at(gid[0], gid[1], rhs_strides);
        out_ += indexing::at(gid[0], gid[1], output_strides);

        #pragma unroll
        for (int k = 0; k < BinaryConfig::ELEMENTS_PER_THREAD_2D; ++k) {
            #pragma unroll
            for (int l = 0; l < BinaryConfig::ELEMENTS_PER_THREAD_2D; ++l) {
                const uint ik = gid[2] + BinaryConfig::BLOCK_SIZE_2D.y * k;
                const uint il = gid[3] + BinaryConfig::BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1])
                    out_[ik * output_strides[2] + il * output_strides[3]] =
                            static_cast<out_val_t>(binary_op(lhs, rhs_[ik * rhs_strides[2] + il * rhs_strides[3]]));
            }
        }
    }

    template<typename lhs_val_t, typename rhs_val_t,
             typename out_val_t, typename binary_t, int VEC_SIZE, bool RESTRICT>
    __global__ __launch_bounds__(BinaryConfig::BLOCK_SIZE)
    void binaryArray1D_(accessor_t<RESTRICT, const lhs_val_t*> lhs, uint2_t lhs_strides,
                        accessor_t<RESTRICT, const rhs_val_t*> rhs, uint2_t rhs_strides,
                        accessor_t<RESTRICT, out_val_t*> output, uint2_t output_strides,
                        uint elements, binary_t binary_op) {
        constexpr uint BLOCK_SIZE = BinaryConfig::BLOCK_SIZE;
        constexpr uint BLOCK_WORK_SIZE = BinaryConfig::BLOCK_WORK_SIZE;
        constexpr uint EPT = BinaryConfig::ELEMENTS_PER_THREAD;

        using iptr_t = typename accessor_t<RESTRICT, const lhs_val_t*>::ptr_type;
        using aptr_t = typename accessor_t<RESTRICT, const rhs_val_t*>::ptr_type;
        using optr_t = typename accessor_t<RESTRICT, out_val_t*>::ptr_type;
        const uint batch = blockIdx.y;
        const uint base = BLOCK_WORK_SIZE * blockIdx.x;
        iptr_t lhs_ = lhs.get() + batch * lhs_strides[0];
        aptr_t rhs_ = rhs.get() + batch * rhs_strides[0];
        optr_t out_ = output.get() + batch * output_strides[0];

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int i = 0; i < EPT; ++i) {
                const uint gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements) {
                    out_[gid * output_strides[1]] =
                            static_cast<out_val_t>(binary_op(lhs_[gid * lhs_strides[1]], rhs_[gid * rhs_strides[1]]));
                }
            }
        } else {
            const uint remaining = elements - base;
            lhs_ += base;
            rhs_ += base;
            out_ += base;
            if (remaining < BLOCK_WORK_SIZE) {
                #pragma unroll
                for (int i = 0; i < EPT; ++i) {
                    const uint offset = BLOCK_SIZE * i + threadIdx.x;
                    if (offset < remaining)
                        out_[offset] = static_cast<out_val_t>(binary_op(lhs_[offset], rhs_[offset]));
                }
            } else {
                lhs_val_t ilhs[EPT];
                rhs_val_t irhs[EPT];
                out_val_t results[EPT];
                block::vectorizedLoad<BLOCK_SIZE, EPT, VEC_SIZE>(lhs_, ilhs, threadIdx.x);
                block::vectorizedLoad<BLOCK_SIZE, EPT, VEC_SIZE>(rhs_, irhs, threadIdx.x);
                #pragma unroll
                for (uint i = 0; i < EPT; ++i)
                    results[i] = static_cast<out_val_t>(binary_op(ilhs[i], irhs[i]));
                block::vectorizedStore<BLOCK_SIZE, EPT, VEC_SIZE>(results, out_, threadIdx.x);
            }
        }
    }

    template<typename lhs_val_t, typename rhs_val_t,
             typename out_val_t, typename binary_t, bool RESTRICT>
    __global__ __launch_bounds__(BinaryConfig::BLOCK_SIZE)
    void binaryArray4D_(accessor_t<RESTRICT, const lhs_val_t*> lhs, uint4_t lhs_strides,
                        accessor_t<RESTRICT, const rhs_val_t*> rhs, uint4_t rhs_strides,
                        accessor_t<RESTRICT, out_val_t*> output, uint4_t output_strides,
                        uint2_t shape, binary_t binary_op, uint blocks_x) {
        using iptr_t = typename accessor_t<RESTRICT, const lhs_val_t*>::ptr_type;
        using aptr_t = typename accessor_t<RESTRICT, const rhs_val_t*>::ptr_type;
        using optr_t = typename accessor_t<RESTRICT, out_val_t*>::ptr_type;
        iptr_t lhs_ = lhs.get();
        aptr_t rhs_ = rhs.get();
        optr_t out_ = output.get();

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid(blockIdx.z,
                         blockIdx.y,
                         BinaryConfig::BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BinaryConfig::BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x);
        lhs_ += indexing::at(gid[0], gid[1], lhs_strides);
        rhs_ += indexing::at(gid[0], gid[1], rhs_strides);
        out_ += indexing::at(gid[0], gid[1], output_strides);

        #pragma unroll
        for (int k = 0; k < BinaryConfig::ELEMENTS_PER_THREAD_2D; ++k) {
            #pragma unroll
            for (int l = 0; l < BinaryConfig::ELEMENTS_PER_THREAD_2D; ++l) {
                const uint ik = gid[2] + BinaryConfig::BLOCK_SIZE_2D.y * k;
                const uint il = gid[3] + BinaryConfig::BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1]) {
                    out_[ik * output_strides[2] + il * output_strides[3]] =
                            static_cast<out_val_t>(binary_op(lhs_[ik * lhs_strides[2] + il * lhs_strides[3]],
                                                             rhs_[ik * rhs_strides[2] + il * rhs_strides[3]]));
                }
            }
        }
    }
}

namespace noa::cuda::util::ewise {
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
             typename lhs_val_t, typename rhs_t,
             typename out_val_t, typename binary_t,
             typename = std::enable_if_t<noa::traits::is_data_v<rhs_t>>>
    void binary(const char* name,
                const lhs_val_t* lhs, size4_t lhs_strides, rhs_t rhs,
                out_val_t* output, size4_t output_strides, size4_t shape,
                bool swap_layout, Stream& stream,
                binary_t binary_op) {
        using namespace details;
        using rhs_val_t = std::remove_const_t<rhs_t>;
        accessor_t<RESTRICT, const lhs_val_t*> lhs_accessor(lhs);
        accessor_t<RESTRICT, out_val_t*> output_accessor(output);

        if (swap_layout) {
            const size4_t order = indexing::order(output_strides, shape);
            shape = indexing::reorder(shape, order);
            output_strides = indexing::reorder(output_strides, order);
            lhs_strides = indexing::reorder(lhs_strides, order);
        }

        const bool4_t is_contiguous = indexing::isContiguous(lhs_strides, shape) &&
                                      indexing::isContiguous(output_strides, shape);
        if (is_contiguous[1] && is_contiguous[2]) { // 1D-like
            // Keep batches separated in a different Grid.Y if they're not contiguous or
            // if we need to apply a different value to each batch.
            const uint4_t uint_shape(shape);
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
                vec_size = lhs_strides[0] % vec_size || output_strides[0] % vec_size ? 1 : vec_size;

            const uint2_t uint_lhs_strides{lhs_strides[0], lhs_strides[3]};
            const uint2_t uint_output_strides{output_strides[0], output_strides[3]};
            const LaunchConfig config{blocks, BinaryConfig::BLOCK_SIZE};
            if (vec_size == 4) {
                return stream.enqueue(
                        name, binaryValueLeft1D_<lhs_val_t, rhs_val_t, out_val_t, binary_t, 4, RESTRICT>, config,
                        lhs_accessor, uint_lhs_strides, rhs, output_accessor, uint_output_strides, elements, binary_op);
            } else if (vec_size == 2) {
                return stream.enqueue(
                        name, binaryValueLeft1D_<lhs_val_t, rhs_val_t, out_val_t, binary_t, 2, RESTRICT>, config,
                        lhs_accessor, uint_lhs_strides, rhs, output_accessor, uint_output_strides, elements, binary_op);
            } else {
                return stream.enqueue(
                        name, binaryValueLeft1D_<lhs_val_t, rhs_val_t, out_val_t, binary_t, 1, RESTRICT>, config,
                        lhs_accessor, uint_lhs_strides, rhs, output_accessor, uint_output_strides, elements, binary_op);
            }
        } else {
            const uint2_t i_shape(shape.get(2));
            const uint blocks_x = noa::math::divideUp(i_shape[1], BinaryConfig::BLOCK_WORK_SIZE_2D.x);
            const uint blocks_y = noa::math::divideUp(i_shape[0], BinaryConfig::BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, BinaryConfig::BLOCK_SIZE_2D};
            stream.enqueue(name, binaryValueLeft4D_<lhs_val_t, rhs_val_t, out_val_t, binary_t, RESTRICT>, config,
                           lhs_accessor, uint4_t(lhs_strides), rhs,
                           output_accessor, uint4_t(output_strides),
                           i_shape, binary_op, blocks_x);
        }
    }

    /// Applies a binary operator, element-wise.
    /// RESTRICT:       Whether the pointers can be accessed using the __restrict__ attribute.
    /// name:           Name of the function. Used for logging if kernel launch fails.
    /// lhs:            Left-hand side argument for the binary operator.
    /// rhs:            On the device. Right-hand side argument for the binary operator.
    /// rhs_strides:    Strides of rhs.
    /// output:         On the device. Transformed array.
    /// output_strides: Strides of output.
    /// shape:          Shape of rhs and output.
    /// swap_layout:    Swap the memory layout to optimize output writes.
    ///                 If false, assume rightmost order is fastest order.
    /// stream:         Stream on which to enqueue this function.
    /// binary_op:      Binary operator. The output is explicitly casted to the output type.
    /// This function is asynchronous relative to the host and may return before completion.
    /// One must make sure input and output pointers stay valid until completion.
    template<bool RESTRICT = false,
             typename lhs_t, typename rhs_val_t,
             typename out_val_t, typename binary_t,
             typename = std::enable_if_t<noa::traits::is_data_v<lhs_t>>>
    void binary(const char* name,
                lhs_t lhs, const rhs_val_t* rhs, size4_t rhs_strides,
                out_val_t* output, size4_t output_strides,
                size4_t shape, bool swap_layout, Stream& stream,
                binary_t binary_op) {
        using namespace details;
        using lhs_val_t = std::remove_const_t<lhs_t>;
        accessor_t<RESTRICT, const rhs_val_t*> rhs_accessor(rhs);
        accessor_t<RESTRICT, out_val_t*> output_accessor(output);

        if (swap_layout) {
            const size4_t order = indexing::order(output_strides, shape);
            shape = indexing::reorder(shape, order);
            output_strides = indexing::reorder(output_strides, order);
            rhs_strides = indexing::reorder(rhs_strides, order);
        }

        const bool4_t is_contiguous = indexing::isContiguous(rhs_strides, shape) &&
                                      indexing::isContiguous(output_strides, shape);
        if (is_contiguous[1] && is_contiguous[2]) { // 1D-like
            // Keep batches separated in a different Grid.Y if they're not contiguous or
            // if we need to apply a different value to each batch.
            const uint4_t uint_shape(shape);
            uint elements, blocks_y;
            if (!is_contiguous[0]) {
                elements = uint_shape[1] * uint_shape[2] * uint_shape[3];
                blocks_y = shape[0];
            } else {
                elements = uint_shape.elements();
                blocks_y = 1;
            }
            const dim3 blocks(noa::math::divideUp(elements, BinaryConfig::BLOCK_WORK_SIZE), blocks_y);

            uint vec_size = is_contiguous[3] ? std::min(maxVectorCount(rhs), maxVectorCount(output)) : 1;
            if (blocks.y > 1) // make sure the beginning of each batch preserves the alignment
                vec_size = rhs_strides[0] % vec_size || output_strides[0] % vec_size ? 1 : vec_size;

            const uint2_t uint_rhs_strides{rhs_strides[0], rhs_strides[3]};
            const uint2_t uint_output_strides{output_strides[0], output_strides[3]};
            const LaunchConfig config{blocks, BinaryConfig::BLOCK_SIZE};
            if (vec_size == 4) {
                return stream.enqueue(
                        name, binaryValueRight1D_<lhs_val_t, rhs_val_t, out_val_t, binary_t, 4, RESTRICT>, config,
                        lhs, rhs_accessor, uint_rhs_strides, output_accessor, uint_output_strides, elements, binary_op);
            } else if (vec_size == 2) {
                return stream.enqueue(
                        name, binaryValueRight1D_<lhs_val_t, rhs_val_t, out_val_t, binary_t, 2, RESTRICT>, config,
                        lhs, rhs_accessor, uint_rhs_strides, output_accessor, uint_output_strides, elements, binary_op);
            } else {
                return stream.enqueue(
                        name, binaryValueRight1D_<lhs_val_t, rhs_val_t, out_val_t, binary_t, 1, RESTRICT>, config,
                        lhs, rhs_accessor, uint_rhs_strides, output_accessor, uint_output_strides, elements, binary_op);
            }
        } else {
            const uint2_t i_shape(shape.get(2));
            const uint blocks_x = noa::math::divideUp(i_shape[1], BinaryConfig::BLOCK_WORK_SIZE_2D.x);
            const uint blocks_y = noa::math::divideUp(i_shape[0], BinaryConfig::BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, BinaryConfig::BLOCK_SIZE_2D};
            stream.enqueue(name, binaryValueRight4D_<lhs_val_t, rhs_val_t, out_val_t, binary_t, RESTRICT>, config,
                           lhs, rhs_accessor, uint4_t(rhs_strides),
                           output_accessor, uint4_t(output_strides), i_shape, binary_op, blocks_x);
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
             typename lhs_val_t, typename rhs_val_t,
             typename out_val_t, typename binary_t>
    void binary(const char* name,
                const lhs_val_t* lhs, size4_t lhs_strides,
                const rhs_val_t* rhs, size4_t rhs_strides,
                out_val_t* output, size4_t output_strides,
                size4_t shape, bool swap_layout, Stream& stream,
                binary_t binary_op) {
        using namespace details;
        accessor_t<RESTRICT, const lhs_val_t*> lhs_accessor(lhs);
        accessor_t<RESTRICT, const rhs_val_t*> rhs_accessor(rhs);
        accessor_t<RESTRICT, out_val_t*> output_accessor(output);

        if (swap_layout) {
            const size4_t order = indexing::order(output_strides, shape);
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
            const uint4_t uint_shape(shape);
            const uint elements = is_contiguous[0] ? uint_shape.elements() : uint3_t{uint_shape.get() + 1}.elements();
            const dim3 blocks(noa::math::divideUp(elements, BinaryConfig::BLOCK_WORK_SIZE),
                              is_contiguous[0] ? 1 : shape[0]);

            uint vec_size = is_contiguous[3] ? std::min({maxVectorCount(lhs),
                                                         maxVectorCount(rhs),
                                                         maxVectorCount(output)}) : 1;
            if (blocks.y > 1) { // make sure the beginning of each batch preserves the alignment
                const bool is_not_multiple = lhs_strides[0] % vec_size ||
                                             rhs_strides[0] % vec_size ||
                                             output_strides[0] % vec_size;
                vec_size = is_not_multiple ? 1 : vec_size;
            }

            const uint2_t uint_lhs_strides{lhs_strides[0], lhs_strides[3]};
            const uint2_t uint_rhs_strides{rhs_strides[0], rhs_strides[3]};
            const uint2_t uint_output_strides{output_strides[0], output_strides[3]};
            const LaunchConfig config{blocks, BinaryConfig::BLOCK_SIZE};
            if (vec_size == 4) {
                return stream.enqueue(
                        name, binaryArray1D_<lhs_val_t, rhs_val_t, out_val_t, binary_t, 4, RESTRICT>, config,
                        lhs_accessor, uint_lhs_strides, rhs_accessor, uint_rhs_strides,
                        output_accessor, uint_output_strides, elements, binary_op);
            } else if (vec_size == 2) {
                return stream.enqueue(
                        name, binaryArray1D_<lhs_val_t, rhs_val_t, out_val_t, binary_t, 2, RESTRICT>, config,
                        lhs_accessor, uint_lhs_strides, rhs_accessor, uint_rhs_strides,
                        output_accessor, uint_output_strides, elements, binary_op);
            } else {
                return stream.enqueue(
                        name, binaryArray1D_<lhs_val_t, rhs_val_t, out_val_t, binary_t, 1, RESTRICT>, config,
                        lhs_accessor, uint_lhs_strides, rhs_accessor, uint_rhs_strides,
                        output_accessor, uint_output_strides, elements, binary_op);
            }
        } else {
            const uint2_t i_shape(shape.get(2));
            const uint blocks_x = noa::math::divideUp(i_shape[1], BinaryConfig::BLOCK_WORK_SIZE_2D.x);
            const uint blocks_y = noa::math::divideUp(i_shape[0], BinaryConfig::BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, BinaryConfig::BLOCK_SIZE_2D};
            stream.enqueue(name, binaryArray4D_<lhs_val_t, rhs_val_t, out_val_t, binary_t, RESTRICT>, config,
                           lhs_accessor, uint4_t(lhs_strides), rhs_accessor, uint4_t(rhs_strides),
                           output_accessor, uint4_t(output_strides), i_shape, binary_op, blocks_x);
        }
    }
}
