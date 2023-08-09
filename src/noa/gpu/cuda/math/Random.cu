#include <curand_kernel.h>
#include <mutex>
#include <random>

#include "noa/gpu/cuda/math/Random.hpp"
#include "noa/gpu/cuda/memory/AllocatorDevice.hpp"
#include "noa/gpu/cuda/utils/Block.cuh"
#include "noa/gpu/cuda/utils/Pointers.hpp"

namespace {
    using namespace ::noa;

    constexpr u32 ELEMENTS_PER_THREAD = 4;
    constexpr u32 BLOCK_SIZE = 128;
    constexpr u32 BLOCK_WORK_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    constexpr u32 MAX_GRID_SIZE = 4096;

    using state_t = curandStateMRG32k3a; // FIXME

    __global__ __launch_bounds__(BLOCK_SIZE)
    void initialize_(state_t* state, i32 seed_base) {
        const u32 tid = blockIdx.x * BLOCK_SIZE + threadIdx.y * blockDim.x + threadIdx.x;
        // Each thread gets its own seed, with no subsequence and no offsets.
        // FIXME Shouldn't we use the same state for multiple threads, and each thread uses
        //       its tid as subsequence/offset? Because having one state per thread uses (at most)
        //       BLOCK_SIZE * MAX_GRID_SIZE * sizeof(state_t) which is non-negligible.
        curand_init(seed_base + tid, 0, 0, &state[tid]);
    }

    template<typename T, typename F, i32 VEC_SIZE>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void randomize_1d_(state_t* state, F distribution, T* output, u32 strides, u32 elements) {
        constexpr u32 EPT = ELEMENTS_PER_THREAD;
        const u32 state_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
        state_t local_state = state[state_idx];

        // loop until reach the end
        const u32 tid = blockIdx.x * BLOCK_WORK_SIZE;
        const u32 grid_size = BLOCK_WORK_SIZE * gridDim.x;
        for (u32 cid = tid; cid < elements; cid += grid_size) {
            if constexpr (VEC_SIZE == 1) {
                for (i32 i = 0; i < EPT; ++i) {
                    const u32 iid = cid + BLOCK_SIZE * i + threadIdx.x;
                    if (iid < elements)
                        output[iid * strides] = distribution(local_state);
                }
            } else {
                NOA_ASSERT(strides == 1);
                (void) strides;
                const u32 remaining = elements - cid;
                T* ptr = output + cid;
                if (remaining < BLOCK_WORK_SIZE) {
                    for (i32 i = 0; i < EPT; ++i) {
                        const u32 iid = BLOCK_SIZE * i + threadIdx.x;
                        if (iid < remaining)
                            ptr[iid] = distribution(local_state);
                    }
                } else {
                    T values[EPT];
                    for (auto& value: values)
                        value = distribution(local_state);
                    noa::cuda::utils::block_store<BLOCK_SIZE, EPT, VEC_SIZE>(values, ptr, threadIdx.x);
                }
            }
        }
    }

    // TODO Add vectorization store?
    template<typename T, typename F>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void randomize_4d_(state_t* state, F distribution,
                       Accessor<T, 4, u32> output,
                       Shape3<u32> shape_dhw, u32 rows) {
        const u32 rows_per_grid = blockDim.y * gridDim.x;
        const u32 initial_row = blockDim.y * blockIdx.x + threadIdx.y;
        const u32 state_idx = blockIdx.x * BLOCK_SIZE + threadIdx.y * blockDim.x + threadIdx.x;
        state_t local_state = state[state_idx];

        // Initial reduction. Loop until all rows are consumed.
        for (u32 row = initial_row; row < rows; row += rows_per_grid) {
            // Retrieve the 3D block index from the linear Grid.X:
            const Vec3<u32> index = noa::indexing::offset2index(row, shape_dhw[0], shape_dhw[1]); // row -> B,D,W
            const auto output_row = output[index[0]][index[1]][index[2]];

            // Consume the row:
            for (u32 cid = threadIdx.x; cid < shape_dhw[2]; cid += blockDim.x)
                output_row[cid] = distribution(local_state);
        }
    }

    template<typename T>
    struct Uniform_ {
        using gen_t = std::conditional_t<std::is_same_v<f64, T>, f64, f32>;
        gen_t min_, max_;

        template<typename U>
        Uniform_(U min, U max) : min_(static_cast<gen_t>(min)), max_(static_cast<gen_t>(max)) {}

        NOA_ID T operator()(state_t& state) {
            gen_t tmp;
            if (std::is_same_v<gen_t, f64>)
                tmp = curand_uniform_double(&state);
            else
                tmp = curand_uniform(&state);
            tmp = tmp * (max_ - min_) + min_;
            return static_cast<T>(tmp);
        }
    };

    template<typename T>
    struct UniformComplex_ {
        using real_t = nt::value_type_t<T>;
        Uniform_<real_t> distributor_real;
        Uniform_<real_t> distributor_imag;

        template<typename U>
        UniformComplex_(U min, U max)
                : distributor_real(min.real, max.real),
                  distributor_imag(min.imag, max.imag) {}

        NOA_FD T operator()(state_t& state) {
            return T{distributor_real(state), distributor_imag(state)};
        }
    };

    template<typename T>
    struct Normal_ {
        using gen_t = std::conditional_t<std::is_same_v<f64, T>, f64, f32>;
        gen_t mean_, stddev_;

        template<typename U>
        Normal_(U mean, U stddev) : mean_(static_cast<gen_t>(mean)), stddev_(static_cast<gen_t>(stddev)) {}

        NOA_ID T operator()(state_t& state) {
            gen_t tmp;
            if (std::is_same_v<gen_t, f64>)
                tmp = curand_normal_double(&state);
            else
                tmp = curand_normal(&state);
            tmp = mean_ + tmp * stddev_;
            return static_cast<T>(tmp);
        }
    };

    template<typename T>
    struct NormalComplex_ {
        using real_t = nt::value_type_t<T>;
        Normal_<real_t> distributor_real;
        Normal_<real_t> distributor_imag;

        template<typename U>
        NormalComplex_(U mean, U stddev)
                : distributor_real(mean.real, stddev.real),
                  distributor_imag(mean.imag, stddev.imag) {}

        NOA_FD T operator()(state_t& state) {
            return T{distributor_real(state), distributor_imag(state)};
        }
    };

    template<typename T>
    struct LogNormal_ {
        using gen_t = std::conditional_t<std::is_same_v<f64, T>, f64, f32>;
        gen_t mean_, stddev_;

        template<typename U>
        LogNormal_(U mean, U stddev) : mean_(static_cast<gen_t>(mean)), stddev_(static_cast<gen_t>(stddev)) {}

        NOA_ID T operator()(state_t& state) {
            gen_t tmp;
            if (std::is_same_v<gen_t, f64>)
                tmp = curand_log_normal_double(&state, mean_, stddev_);
            else
                tmp = curand_log_normal(&state, mean_, stddev_);
            return static_cast<T>(tmp);
        }
    };

    template<typename T>
    struct LogNormalComplex_ {
        using real_t = nt::value_type_t<T>;
        LogNormal_<real_t> distributor_real;
        LogNormal_<real_t> distributor_imag;

        template<typename U>
        LogNormalComplex_(U mean, U stddev)
                : distributor_real(mean.real, stddev.real),
                  distributor_imag(mean.imag, stddev.imag) {}

        NOA_FD T operator()(state_t& state) {
            return T{distributor_real(state), distributor_imag(state)};
        }
    };

    template<typename T>
    struct Poisson_ {
        double lambda_;

        template<typename U>
        explicit Poisson_(U lambda) : lambda_(static_cast<f64>(lambda)) {}

        NOA_ID T operator()(state_t& state) {
            return static_cast<T>(curand_poisson(&state, lambda_));
        }
    };

    template<typename T, typename F>
    void launch_1d_(cuda::LaunchConfig config, state_t* state, T* output, u32 strides, u32 elements,
                   F distribution, cuda::Stream& stream) {
        const i32 vec_size = strides == 1 ? std::min(noa::cuda::utils::max_vector_count(output), i64{4}) : 1;
        if (vec_size == 4) {
            stream.enqueue(randomize_1d_<T, F, 4>, config, state, distribution, output, strides, elements);
        } else if (vec_size == 2) {
            stream.enqueue(randomize_1d_<T, F, 2>, config, state, distribution, output, strides, elements);
        } else {
            stream.enqueue(randomize_1d_<T, F, 1>, config, state, distribution, output, strides, elements);
        }
    }

    template<typename D, typename T, typename U>
    void randomize_1d_(D, T* output, i64 stride, i64 elements, U x, U y, cuda::Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        const auto s_elements = safe_cast<u32>(elements);
        const auto s_stride = safe_cast<u32>(stride);

        const u32 blocks_x = std::min(noa::math::divide_up(s_elements, BLOCK_WORK_SIZE), MAX_GRID_SIZE);
        const dim3 blocks(blocks_x);
        const noa::cuda::LaunchConfig config{blocks, BLOCK_SIZE};

        const auto states = noa::cuda::memory::AllocatorDevice<state_t>::allocate_async(blocks_x * BLOCK_SIZE, stream);
        const u32 seed = std::random_device{}();
        stream.enqueue(initialize_, config, states.get(), seed);

        if constexpr (std::is_same_v<D, noa::math::uniform_t>) {
            using distributor_t = std::conditional_t<nt::is_complex_v<U>, UniformComplex_<T>, Uniform_<T>>;
            const distributor_t distribution(x, y);
            launch_1d_<T, distributor_t>(config, states.get(), output, s_stride, s_elements, distribution, stream);
        } else if constexpr (std::is_same_v<D, noa::math::normal_t>) {
            using distributor_t = std::conditional_t<nt::is_complex_v<U>, NormalComplex_<T>, Normal_<T>>;
            const distributor_t distribution(x, y);
            launch_1d_<T, distributor_t>(config, states.get(), output, s_stride, s_elements, distribution, stream);
        } else if constexpr (std::is_same_v<D, noa::math::log_normal_t>) {
            using distributor_t = std::conditional_t<nt::is_complex_v<U>, LogNormalComplex_<T>, LogNormal_<T>>;
            const distributor_t distribution(x, y);
            launch_1d_<T, distributor_t>(config, states.get(), output, s_stride, s_elements, distribution, stream);
        } else if constexpr (std::is_same_v<D, noa::math::poisson_t>) {
            (void) y;
            const Poisson_<T> distribution(x);
            launch_1d_<T, Poisson_<T>>(config, states.get(), output, s_stride, s_elements, distribution, stream);
        } else {
            static_assert(nt::always_false_v<T>);
        }
    }

    template<typename D, typename T, typename U>
    void randomize_4d_(
            D, T* output,
            Strides4<i64> strides,
            Shape4<i64> shape,
            U x, U y, noa::cuda::Stream& stream
    ) {
        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(output, stream.device());

        const auto order = noa::indexing::order(strides, shape);
        strides = noa::indexing::reorder(strides, order);
        shape = noa::indexing::reorder(shape, order);

        const auto is_contiguous = noa::indexing::is_contiguous(strides, shape);
        if (is_contiguous[0] && is_contiguous[1] && is_contiguous[2])
            return randomize_1d_(D{}, output, strides[3], shape.elements(), x, y, stream);

        const u32 block_dim_x = shape[3] > 512 ? 128 : 64;
        const dim3 threads(block_dim_x, BLOCK_SIZE / block_dim_x);
        const auto rows = safe_cast<u32>(shape[2] * shape[1] * shape[0]);
        const dim3 blocks(std::min(noa::math::divide_up(rows, threads.y), MAX_GRID_SIZE));
        const noa::cuda::LaunchConfig config{blocks, threads};

        const auto states = noa::cuda::memory::AllocatorDevice<state_t>::allocate_async(blocks.x * BLOCK_SIZE, stream);
        const u32 seed = std::random_device{}();
        stream.enqueue(initialize_, config, states.get(), seed);

        auto u_strides = strides.as_safe<u32>();
        auto u_shape = shape.pop_front().as_safe<u32>();
        const Accessor<T, 4, u32> output_accessor(output, u_strides);
        if constexpr (std::is_same_v<D, noa::math::uniform_t>) {
            using distributor_t = std::conditional_t<nt::is_complex_v<U>, UniformComplex_<T>, Uniform_<T>>;
            const distributor_t distribution(x, y);
            stream.enqueue(randomize_4d_<T, distributor_t>, config,
                           states.get(), distribution, output_accessor, u_shape, rows);
        } else if constexpr (std::is_same_v<D, noa::math::normal_t>) {
            using distributor_t = std::conditional_t<nt::is_complex_v<U>, NormalComplex_<T>, Normal_<T>>;
            const distributor_t distribution(x, y);
            stream.enqueue(randomize_4d_<T, distributor_t>, config,
                           states.get(), distribution, output_accessor, u_shape, rows);
        } else if constexpr (std::is_same_v<D, noa::math::log_normal_t>) {
            using distributor_t = std::conditional_t<nt::is_complex_v<U>, LogNormalComplex_<T>, LogNormal_<T>>;
            const distributor_t distribution(x, y);
            stream.enqueue(randomize_4d_<T, distributor_t>, config,
                           states.get(), distribution, output_accessor, u_shape, rows);
        } else if constexpr (std::is_same_v<D, noa::math::poisson_t>) {
            (void) y;
            const Poisson_<T> distribution(x);
            stream.enqueue(randomize_4d_<T, Poisson_<T>>, config,
                           states.get(), distribution, output_accessor, u_shape, rows);
        } else {
            static_assert(nt::always_false_v<T>);
        }
    }
}

namespace noa::cuda::math {
    template<typename T, typename U, typename>
    void randomize(
            noa::math::uniform_t, T* output,
            const Strides4<i64>& strides, const Shape4<i64>& shape,
            U min, U max, Stream& stream
    ) {
        if constexpr (nt::is_complex_v<T> && nt::is_real_v<U>) {
            using real_t = nt::value_type_t<T>;
            using supported_float = std::conditional_t<std::is_same_v<real_t, f16>, float, real_t>;
            const auto reinterpreted = noa::indexing::Reinterpret(shape, strides, output).template as<real_t>();
            randomize(noa::math::uniform_t{}, reinterpret_cast<real_t*>(output),
                      reinterpreted.strides, reinterpreted.shape,
                      static_cast<supported_float>(min), static_cast<supported_float>(max), stream);
        } else {
            randomize_4d_(noa::math::uniform_t{}, output, strides, shape, min, max, stream);
        }
    }

    template<typename T, typename U, typename>
    void randomize(
            noa::math::normal_t, T* output,
            const Strides4<i64>& strides, const Shape4<i64>& shape,
            U mean, U stddev, Stream& stream
    ) {
        if constexpr (nt::is_complex_v<T> && nt::is_real_v<U>) {
            using real_t = nt::value_type_t<T>;
            using supported_float = std::conditional_t<std::is_same_v<real_t, f16>, float, real_t>;
            const auto reinterpreted = noa::indexing::Reinterpret(shape, strides, output).template as<real_t>();
            randomize(noa::math::normal_t{}, reinterpret_cast<real_t*>(output),
                      reinterpreted.strides, reinterpreted.shape,
                      static_cast<supported_float>(mean), static_cast<supported_float>(stddev), stream);
        } else {
            randomize_4d_(noa::math::normal_t{}, output, strides, shape, mean, stddev, stream);
        }
    }

    template<typename T, typename U, typename>
    void randomize(
            noa::math::log_normal_t, T* output,
            const Strides4<i64>& strides, const Shape4<i64>& shape,
            U mean, U stddev, Stream& stream
    ) {
        if constexpr (nt::is_complex_v<T> && nt::is_real_v<U>) {
            using real_t = nt::value_type_t<T>;
            using supported_float = std::conditional_t<std::is_same_v<real_t, f16>, float, real_t>;
            const auto reinterpreted = noa::indexing::Reinterpret(shape, strides, output).template as<real_t>();
            randomize(noa::math::log_normal_t{}, reinterpret_cast<real_t*>(output),
                      reinterpreted.strides, reinterpreted.shape,
                      static_cast<supported_float>(mean), static_cast<supported_float>(stddev), stream);
        } else {
            randomize_4d_(noa::math::log_normal_t{}, output, strides, shape, mean, stddev, stream);
        }
    }

    template<typename T, typename>
    void randomize(
            noa::math::poisson_t, T* output,
            const Strides4<i64>& strides, const Shape4<i64>& shape,
            float lambda, Stream& stream
    ) {
        if constexpr (nt::is_complex_v<T>) {
            using real_t = nt::value_type_t<T>;
            const auto reinterpreted = noa::indexing::Reinterpret(shape, strides, output).template as<real_t>();
            randomize(noa::math::poisson_t{}, reinterpret_cast<real_t*>(output),
                      reinterpreted.strides, reinterpreted.shape, lambda, stream);
        } else {
            randomize_4d_(noa::math::poisson_t{}, output, strides, shape, lambda, 0.f, stream);
        }
    }

    #define INSTANTIATE_RANDOM_(T, U)               \
    template void randomize<T, U, void>(            \
        noa::math::uniform_t, T*,                   \
        const Strides4<i64>&, const Shape4<i64>&,   \
        U, U, Stream&);                             \
    template void randomize<T, U, void>(            \
        noa::math::normal_t, T*,                    \
        const Strides4<i64>&, const Shape4<i64>&,   \
        U, U, Stream&);                             \
    template void randomize<T, U, void>(            \
        noa::math::log_normal_t, T*,                \
        const Strides4<i64>&, const Shape4<i64>&,   \
        U, U, Stream&)

    INSTANTIATE_RANDOM_(i16, i16);
    INSTANTIATE_RANDOM_(u16, u16);
    INSTANTIATE_RANDOM_(i32, i32);
    INSTANTIATE_RANDOM_(u32, u32);
    INSTANTIATE_RANDOM_(i64, i64);
    INSTANTIATE_RANDOM_(u64, u64);
    INSTANTIATE_RANDOM_(f16, f16);
    INSTANTIATE_RANDOM_(f32, f32);
    INSTANTIATE_RANDOM_(f64, f64);
    INSTANTIATE_RANDOM_(c16, f16);
    INSTANTIATE_RANDOM_(c32, f32);
    INSTANTIATE_RANDOM_(c64, f64);
    INSTANTIATE_RANDOM_(c16, c16);
    INSTANTIATE_RANDOM_(c32, c32);
    INSTANTIATE_RANDOM_(c64, c64);

    #define INSTANTIATE_RANDOM_POISSON_(T)          \
    template void randomize<T, void>(               \
        noa::math::poisson_t, T*,   \
        const Strides4<i64>&, const Shape4<i64>&,   \
        f32, Stream&)

    INSTANTIATE_RANDOM_POISSON_(f16);
    INSTANTIATE_RANDOM_POISSON_(f32);
    INSTANTIATE_RANDOM_POISSON_(f64);

    #define INSTANTIATE_RANDOM_HALF_(T, U)              \
    template void randomize<T, U, void>(                \
        noa::math::uniform_t, T*,       \
        const Strides4<i64>&, const Shape4<i64>&,       \
        U, U, Stream&);                                 \
    template void randomize<T, U, void>(                \
        noa::math::normal_t, T*,        \
        const Strides4<i64>&, const Shape4<i64>&,       \
        U, U, Stream&);                                 \
    template void randomize<T, U, void>(                \
        noa::math::log_normal_t, T*,    \
        const Strides4<i64>&, const Shape4<i64>&,       \
        U, U, Stream&)

    INSTANTIATE_RANDOM_HALF_(f16, f32);
    INSTANTIATE_RANDOM_HALF_(c16, f32);
    INSTANTIATE_RANDOM_HALF_(c16, c32);
}
