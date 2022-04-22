#include <curand_kernel.h>
#include <mutex>
#include <random>

#include "noa/gpu/cuda/math/Random.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/util/Block.cuh"
#include "noa/gpu/cuda/util/Pointers.h"

namespace {
    using namespace ::noa;

    constexpr uint ELEMENTS_PER_THREAD = 4;
    constexpr uint BLOCK_SIZE = 128;
    constexpr uint BLOCK_WORK_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    constexpr uint MAX_GRID_SIZE = 2048;

    using state_t = curandStateMRG32k3a; // FIXME

    __global__ __launch_bounds__(BLOCK_SIZE)
    void init_(state_t* state, int seed_base) {
        const uint tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
        // Each thread gets its own seed, with no subsequence and no offsets.
        curand_init(seed_base + tid, 0, 0, &state[tid]);
    }

    template<typename T, typename F, int VEC_SIZE>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void randomize1D_(state_t* state, F distribution, T* output, uint stride, uint elements) {
        constexpr uint EPT = ELEMENTS_PER_THREAD;
        state_t local_state = state[blockIdx.x * BLOCK_SIZE + threadIdx.x];

        // loop until reach the end
        const uint tid = blockIdx.x * BLOCK_WORK_SIZE;
        const uint grid_size = BLOCK_WORK_SIZE * gridDim.x;
        for (uint cid = tid; cid < elements; cid += grid_size) {
            if constexpr (VEC_SIZE == 1) {
                for (int i = 0; i < EPT; ++i) {
                    const uint iid = cid + BLOCK_SIZE * i + threadIdx.x;
                    if (iid < elements)
                        output[iid * stride] = distribution(local_state);
                }
            } else {
                NOA_ASSERT(stride == 1);
                const uint remaining = elements - cid;
                T* ptr = output + cid;
                if (remaining < BLOCK_WORK_SIZE) {
                    for (int i = 0; i < EPT; ++i) {
                        const uint iid = BLOCK_SIZE * i + threadIdx.x;
                        if (iid < remaining)
                            ptr[iid] = distribution(local_state);
                    }
                } else {
                    T values[EPT];
                    for (auto& value: values)
                        value = distribution(local_state);
                    cuda::util::block::vectorizedStore<BLOCK_SIZE, EPT, VEC_SIZE>(values, ptr, threadIdx.x);
                }
            }
        }
    }

    // TODO Add vectorization store?
    template<typename T, typename F>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void randomize4D_(state_t* state, F distribution, T* output, uint4_t stride, uint4_t shape, uint rows) {
        const uint rows_per_grid = blockDim.y * gridDim.x;
        const uint initial_row = blockDim.y * blockIdx.x;

        state_t local_state = state[blockIdx.x * BLOCK_SIZE + threadIdx.x];

        // Initial reduction. Loop until all rows are consumed.
        for (uint row = initial_row; row < rows; row += rows_per_grid) {
            // Retrieve the 3D block index from the linear Grid.X:
            const uint3_t index = indexing::indexes(row, shape[1], shape[2]); // row -> W,Z,Y
            const uint offset = indexing::at(index, stride);

            // Consume the row:
            for (uint cid = threadIdx.x; cid < shape[3]; cid += blockDim.x)
                output[offset + cid * stride[3]] = distribution(local_state);
        }
    }

    template<typename T>
    struct Uniform_ {
        using gen_t = std::conditional_t<std::is_same_v<double, T>, double, float>;
        gen_t min_, max_;

        template<typename U>
        Uniform_(U min, U max) : min_(static_cast<gen_t>(min)), max_(static_cast<gen_t>(max)) {}

        NOA_ID T operator()(state_t& state) {
            gen_t tmp;
            if (std::is_same_v<gen_t, double>)
                tmp = curand_uniform_double(&state);
            else
                tmp = curand_uniform(&state);
            tmp = tmp * (max_ - min_) + min_;
            return static_cast<T>(tmp);
        }
    };

    template<typename T>
    struct Normal_ {
        using gen_t = std::conditional_t<std::is_same_v<double, T>, double, float>;
        gen_t mean_, stddev_;

        template<typename U>
        Normal_(U mean, U stddev) : mean_(static_cast<gen_t>(mean)), stddev_(static_cast<gen_t>(stddev)) {}

        NOA_ID T operator()(state_t& state) {
            gen_t tmp;
            if (std::is_same_v<gen_t, double>)
                tmp = curand_normal_double(&state);
            else
                tmp = curand_normal(&state);
            tmp = mean_ + tmp * stddev_;
            return static_cast<T>(tmp);
        }
    };

    template<typename T>
    struct LogNormal_ {
        using gen_t = std::conditional_t<std::is_same_v<double, T>, double, float>;
        gen_t mean_, stddev_;

        template<typename U>
        LogNormal_(U mean, U stddev) : mean_(static_cast<gen_t>(mean)), stddev_(static_cast<gen_t>(stddev)) {}

        NOA_ID T operator()(state_t& state) {
            gen_t tmp;
            if (std::is_same_v<gen_t, double>)
                tmp = curand_log_normal_double(&state, mean_, stddev_);
            else
                tmp = curand_log_normal(&state, mean_, stddev_);
            return static_cast<T>(tmp);
        }
    };

    template<typename T>
    struct Poisson_ {
        double lambda_;

        template<typename U>
        explicit Poisson_(U lambda) : lambda_(static_cast<double>(lambda)) {}

        NOA_ID T operator()(state_t& state) {
            return static_cast<T>(curand_poisson(&state, lambda_));
        }
    };

    template<typename T, typename F>
    void launch1D_(cuda::LaunchConfig config, state_t* state, T* output, uint stride, uint elements,
                   F distribution, cuda::Stream& stream) {
        const int vec_size = stride == 1 ? noa::cuda::util::maxVectorCount(output) : 1;
        if (vec_size == 4) {
            stream.enqueue("math::randomize", randomize1D_<T, F, 4>, config,
                           state, distribution, output, stride, elements);
        } else if (vec_size == 2) {
            stream.enqueue("math::randomize", randomize1D_<T, F, 2>, config,
                           state, distribution, output, stride, elements);
        } else {
            stream.enqueue("math::randomize", randomize1D_<T, F, 1>, config,
                           state, distribution, output, stride, elements);
        }
    }

    template<typename D, typename T, typename U>
    void randomize1D_(D, const shared_t<T[]>& output,
                      uint stride, uint elements, U x, U y, cuda::Stream& stream) {
        const uint blocks_x = noa::math::min(noa::math::divideUp(elements, BLOCK_WORK_SIZE), MAX_GRID_SIZE);
        const dim3 blocks(blocks_x);
        const cuda::LaunchConfig config{blocks, BLOCK_SIZE};

        cuda::memory::PtrDevice<state_t> states{blocks_x * BLOCK_SIZE, stream};
        const uint seed = std::random_device{}();
        stream.enqueue("math::randomize::init", init_, config, states.get(), seed);

        if constexpr (std::is_same_v<D, noa::math::Uniform>) {
            Uniform_<T> distribution{x, y};
            launch1D_<T, Uniform_<T>>(config, states.get(), output.get(), stride, elements, distribution, stream);
        } else if constexpr (std::is_same_v<D, noa::math::Normal>) {
            Normal_<T> distribution{x, y};
            launch1D_<T, Normal_<T>>(config, states.get(), output.get(), stride, elements, distribution, stream);
        } else if constexpr (std::is_same_v<D, noa::math::LogNormal>) {
            LogNormal_<T> distribution{x, y};
            launch1D_<T, LogNormal_<T>>(config, states.get(), output.get(), stride, elements, distribution, stream);
        } else if constexpr (std::is_same_v<D, noa::math::Poisson>) {
            (void) y;
            Poisson_<T> distribution{x};
            launch1D_<T, Poisson_<T>>(config, states.get(), output.get(), stride, elements, distribution, stream);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        stream.attach(output);
    }

    template<typename D, typename T, typename U>
    void randomize4D_(D, const shared_t<T[]>& output, size4_t stride, size4_t shape, U x, U y, cuda::Stream& stream) {
        const bool4_t contiguous = indexing::isContiguous(stride, shape);
        if (contiguous[0] && contiguous[1] && contiguous[2])
            return randomize1D_(D{}, output, stride[3], shape.elements(), x, y, stream);

        const uint block_dim_x = shape[3] > 512 ? 128 : 64;
        const dim3 threads(block_dim_x, BLOCK_SIZE / block_dim_x);
        const uint rows = shape[2] * shape[1] * shape[0];
        const dim3 blocks(noa::math::min(noa::math::divideUp(rows, threads.y), MAX_GRID_SIZE));
        const cuda::LaunchConfig config{blocks, BLOCK_SIZE};

        cuda::memory::PtrDevice<state_t> states{blocks.x * BLOCK_SIZE, stream};
        const uint seed = std::random_device{}();
        stream.enqueue("math::randomize::init", init_, config, states.get(), seed);

        const uint4_t stride_{stride};
        const uint4_t shape_{shape};
        if constexpr (std::is_same_v<D, noa::math::Uniform>) {
            Uniform_<T> distribution{x, y};
            stream.enqueue("math::randomize", randomize4D_<T, Uniform_<T>>, config,
                           states.get(), distribution, output.get(), stride_, shape_, rows);
        } else if constexpr (std::is_same_v<D, noa::math::Normal>) {
            Normal_<T> distribution{x, y};
            stream.enqueue("math::randomize", randomize4D_<T, Normal_<T>>, config,
                           states.get(), distribution, output.get(), stride_, shape_, rows);
        } else if constexpr (std::is_same_v<D, noa::math::LogNormal>) {
            LogNormal_<T> distribution{x, y};
            stream.enqueue("math::randomize", randomize4D_<T, LogNormal_<T>>, config,
                           states.get(), distribution, output.get(), stride_, shape_, rows);
        } else if constexpr (std::is_same_v<D, noa::math::Poisson>) {
            (void) y;
            Poisson_<T> distribution{x};
            stream.enqueue("math::randomize", randomize4D_<T, Poisson_<T>>, config,
                           states.get(), distribution, output.get(), stride_, shape_, rows);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
        stream.attach(output);
    }
}

namespace noa::cuda::math {
    template<typename T, typename U>
    void randomize(noa::math::Uniform, const shared_t<T[]>& output, size_t elements,
                   U min, U max, Stream& stream) {
        if constexpr (noa::traits::is_complex_v<T>) {
            using real_t = noa::traits::value_type_t<T>;
            using supported_float = std::conditional_t<std::is_same_v<real_t, half_t>, float, real_t>;
            randomize(noa::math::Uniform{}, std::reinterpret_pointer_cast<real_t[]>(output), elements * 2,
                      static_cast<supported_float>(min), static_cast<supported_float>(max), stream);
        } else {
            randomize1D_(noa::math::Uniform{}, output, 1, elements, min, max, stream);
        }
    }

    template<typename T, typename U>
    void randomize(noa::math::Normal, const shared_t<T[]>& output, size_t elements,
                   U mean, U stddev, Stream& stream) {
        if constexpr (noa::traits::is_complex_v<T>) {
            using real_t = noa::traits::value_type_t<T>;
            using supported_float = std::conditional_t<std::is_same_v<real_t, half_t>, float, real_t>;
            randomize(noa::math::Normal{}, std::reinterpret_pointer_cast<real_t[]>(output), elements * 2,
                      static_cast<supported_float>(mean), static_cast<supported_float>(stddev), stream);
        } else {
            randomize1D_(noa::math::Normal{}, output, 1, elements, mean, stddev, stream);
        }
    }

    template<typename T, typename U>
    void randomize(noa::math::LogNormal, const shared_t<T[]>& output, size_t elements,
                   U mean, U stddev, Stream& stream) {
        if constexpr (noa::traits::is_complex_v<T>) {
            using real_t = noa::traits::value_type_t<T>;
            using supported_float = std::conditional_t<std::is_same_v<real_t, half_t>, float, real_t>;
            randomize(noa::math::LogNormal{}, std::reinterpret_pointer_cast<real_t[]>(output), elements * 2,
                      static_cast<supported_float>(mean), static_cast<supported_float>(stddev), stream);
        } else {
            randomize1D_(noa::math::LogNormal{}, output, 1, elements, mean, stddev, stream);
        }
    }

    template<typename T>
    void randomize(noa::math::Poisson, const shared_t<T[]>& output, size_t elements,
                   float lambda, Stream& stream) {
        if constexpr (noa::traits::is_complex_v<T>) {
            using real_t = noa::traits::value_type_t<T>;
            randomize(noa::math::Poisson{}, std::reinterpret_pointer_cast<real_t[]>(output), elements * 2,
                      lambda, stream);
        } else {
            randomize1D_(noa::math::Poisson{}, output, 1, elements, lambda, 0.f, stream);
        }
    }

    template<typename T, typename U>
    void randomize(noa::math::Uniform, const shared_t<T[]>& output, size4_t stride, size4_t shape,
                   U min, U max, Stream& stream) {
        if constexpr (noa::traits::is_complex_v<T>) {
            using real_t = noa::traits::value_type_t<T>;
            using supported_float = std::conditional_t<std::is_same_v<real_t, half_t>, float, real_t>;
            const auto reinterpreted = indexing::Reinterpret<T>{shape, stride, output.get()}.template as<real_t>();
            randomize(noa::math::Uniform{}, std::reinterpret_pointer_cast<real_t[]>(output),
                      reinterpreted.stride, reinterpreted.shape,
                      static_cast<supported_float>(min), static_cast<supported_float>(max), stream);
        } else {
            randomize4D_(noa::math::Uniform{}, output, stride, shape, min, max, stream);
        }
    }

    template<typename T, typename U>
    void randomize(noa::math::Normal, const shared_t<T[]>& output, size4_t stride, size4_t shape,
                   U mean, U stddev, Stream& stream) {
        if constexpr (noa::traits::is_complex_v<T>) {
            using real_t = noa::traits::value_type_t<T>;
            using supported_float = std::conditional_t<std::is_same_v<real_t, half_t>, float, real_t>;
            const auto reinterpreted = indexing::Reinterpret<T>{shape, stride, output.get()}.template as<real_t>();
            randomize(noa::math::Normal{}, std::reinterpret_pointer_cast<real_t[]>(output),
                      reinterpreted.stride, reinterpreted.shape,
                      static_cast<supported_float>(mean), static_cast<supported_float>(stddev), stream);
        } else {
            randomize4D_(noa::math::Normal{}, output, stride, shape, mean, stddev, stream);
        }
    }

    template<typename T, typename U>
    void randomize(noa::math::LogNormal, const shared_t<T[]>& output, size4_t stride, size4_t shape,
                   U mean, U stddev, Stream& stream) {
        if constexpr (noa::traits::is_complex_v<T>) {
            using real_t = noa::traits::value_type_t<T>;
            using supported_float = std::conditional_t<std::is_same_v<real_t, half_t>, float, real_t>;
            const auto reinterpreted = indexing::Reinterpret<T>{shape, stride, output.get()}.template as<real_t>();
            randomize(noa::math::LogNormal{}, std::reinterpret_pointer_cast<real_t[]>(output),
                      reinterpreted.stride, reinterpreted.shape,
                      static_cast<supported_float>(mean), static_cast<supported_float>(stddev), stream);
        } else {
            randomize4D_(noa::math::LogNormal{}, output, stride, shape, mean, stddev, stream);
        }
    }

    template<typename T>
    void randomize(noa::math::Poisson, const shared_t<T[]>& output, size4_t stride, size4_t shape,
                   float lambda, Stream& stream) {
        if constexpr (noa::traits::is_complex_v<T>) {
            using real_t = noa::traits::value_type_t<T>;
            const auto reinterpreted = indexing::Reinterpret<T>{shape, stride, output.get()}.template as<real_t>();
            randomize(noa::math::Poisson{}, std::reinterpret_pointer_cast<real_t[]>(output),
                      reinterpreted.stride, reinterpreted.shape, lambda, stream);
        } else {
            randomize4D_(noa::math::Poisson{}, output, stride, shape, lambda, 0.f, stream);
        }
    }

    #define INSTANTIATE_RANDOM_(T, U)                                                                           \
    template void randomize<T, U>(noa::math::Uniform, const shared_t<T[]>&, size4_t, size4_t, U, U, Stream&);   \
    template void randomize<T, U>(noa::math::Normal, const shared_t<T[]>&, size4_t, size4_t, U, U, Stream&);    \
    template void randomize<T, U>(noa::math::LogNormal, const shared_t<T[]>&, size4_t, size4_t, U, U, Stream&); \
    template void randomize<T>(noa::math::Poisson, const shared_t<T[]>&, size4_t, size4_t, float, Stream&)

    INSTANTIATE_RANDOM_(int16_t, int16_t);
    INSTANTIATE_RANDOM_(uint16_t, uint16_t);
    INSTANTIATE_RANDOM_(int32_t, int32_t);
    INSTANTIATE_RANDOM_(uint32_t, uint32_t);
    INSTANTIATE_RANDOM_(int64_t, int64_t);
    INSTANTIATE_RANDOM_(uint64_t, uint64_t);
    INSTANTIATE_RANDOM_(half_t, half_t);
    INSTANTIATE_RANDOM_(float, float);
    INSTANTIATE_RANDOM_(double, double);
    INSTANTIATE_RANDOM_(chalf_t, half_t);
    INSTANTIATE_RANDOM_(cfloat_t, float);
    INSTANTIATE_RANDOM_(cdouble_t, double);

    #define INSTANTIATE_RANDOM_HALF_(T)                                                                                         \
    template void randomize<T, float>(noa::math::Uniform, const shared_t<T[]>&, size4_t, size4_t, float, float, Stream&);       \
    template void randomize<T, float>(noa::math::Normal, const shared_t<T[]>&, size4_t, size4_t, float, float, Stream&);        \
    template void randomize<T, float>(noa::math::LogNormal, const shared_t<T[]>&, size4_t, size4_t, float, float, Stream&);     \
    template void randomize<T, double>(noa::math::Uniform, const shared_t<T[]>&, size4_t, size4_t, double, double, Stream&);    \
    template void randomize<T, double>(noa::math::Normal, const shared_t<T[]>&, size4_t, size4_t, double, double, Stream&);     \
    template void randomize<T, double>(noa::math::LogNormal, const shared_t<T[]>&, size4_t, size4_t, double, double, Stream&)

    INSTANTIATE_RANDOM_HALF_(half_t);
    INSTANTIATE_RANDOM_HALF_(chalf_t);
}
