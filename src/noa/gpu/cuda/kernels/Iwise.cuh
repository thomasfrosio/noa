#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Interfaces.hpp"
#include "noa/gpu/cuda/kernels/Block.cuh"
#include "noa/gpu/cuda/Types.hpp"

namespace noa::cuda::guts {
    template<typename Config, typename Op, typename Index>
    __global__ __launch_bounds__(Config::block_size)
    void iwise_4d_static(Op op, Vec2<Index> end_hw, u32 n_blocks_x) {
        const Vec2<u32> index = ni::offset2index(blockIdx.x, n_blocks_x);
        auto bdhw = Vec4<Index>::from_values(
                blockIdx.z,
                blockIdx.y,
                Config::block_work_size_y * index[0] + threadIdx.y,
                Config::block_work_size_x * index[1] + threadIdx.x
        );

        using interface = ng::IwiseInterface;
        interface::init(op, thread_uid<3>());

        for (Index h = 0; h < Config::n_elements_per_thread_y; ++h) {
            for (Index w = 0; w < Config::n_elements_per_thread_x; ++w) {
                const Index ih = bdhw[2] + Config::block_size_y * h;
                const Index iw = bdhw[3] + Config::block_size_x * w;
                if (ih < end_hw[0] and iw < end_hw[1])
                    interface::call(op, bdhw[0], bdhw[1], ih, iw);
            }
        }
        interface::final(op, thread_uid<3>());
    }

    template<typename Config, typename Op, typename Index>
    __global__ __launch_bounds__(Config::block_size)
    void iwise_3d_static(Op op, Vec2<Index> end_hw) {
        auto dhw = Vec3<Index>::from_values(
                blockIdx.z,
                Config::block_work_size_y * blockIdx.y + threadIdx.y,
                Config::block_work_size_x * blockIdx.x + threadIdx.x
        );

        using interface = ng::IwiseInterface;
        interface::init(op, thread_uid<3>());

        for (Index h = 0; h < Config::n_elements_per_thread_y; ++h) {
            for (Index w = 0; w < Config::n_elements_per_thread_x; ++w) {
                const Index ih = dhw[1] + Config::block_size_y * h;
                const Index iw = dhw[2] + Config::block_size_x * w;
                if (ih < end_hw[0] and iw < end_hw[1])
                    interface::call(op, dhw[0], ih, iw);
            }
        }
        interface::final(op, thread_uid<3>());
    }

    template<typename Config, typename Op, typename Index>
    __global__ __launch_bounds__(Config::block_size)
    void iwise_2d_static(Op op, Vec2<Index> end_hw) {
        auto hw = Vec2<Index>::from_values(
                Config::block_work_size_y * blockIdx.y + threadIdx.y,
                Config::block_work_size_x * blockIdx.x + threadIdx.x
        );

        using interface = ng::IwiseInterface;
        interface::init(op, thread_uid<2>());

        for (Index h = 0; h < Config::n_elements_per_thread_y; ++h) {
            for (Index w = 0; w < Config::n_elements_per_thread_x; ++w) {
                const Index ih = hw[0] + Config::block_size_y * h;
                const Index iw = hw[1] + Config::block_size_x * w;
                if (ih < end_hw[0] and iw < end_hw[1])
                    interface::call(op, ih, iw);
            }
        }
        interface::final(op, thread_uid<2>());
    }

    template<typename Config, typename Op, typename Index>
    __global__  __launch_bounds__(Config::block_size)
    void iwise_1d_static(Op op, Vec1<Index> end) {
        auto index = Vec1<Index>::from_values(Config::block_work_size_x * blockIdx.x + threadIdx.x);

        using interface = ng::IwiseInterface;
        interface::init(op, thread_uid<1>());

        for (Index w = 0; w < Config::n_elements_per_thread_x; ++w) {
            const Index iw = index[0] + Config::block_size_x * w;
            if (iw < end[0])
                interface::call(op, iw);
        }
        interface::final(op, thread_uid<1>());
    }
}
