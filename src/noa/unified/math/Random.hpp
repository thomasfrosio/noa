#pragma once

#include "noa/cpu/math/Random.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/math/Random.hpp"
#endif

#include "noa/unified/Array.hpp"

namespace noa::math::details {
    template<typename T, typename U>
    constexpr bool is_valid_random_v =
            nt::is_any_v<T, i16, i32, i64, u16, u32, u64, f16, f32, f64, c16, c32, c64> &&
            (nt::is_scalar_v<U> || nt::are_complex_v<T, U>);

    template<typename T, typename U>
    auto cast_to_supported_type(U value) {
        if constexpr (nt::are_complex_v<T, U>) {
            using supported_t = std::conditional_t<nt::is_almost_same_v<T, c16>, c32, T>;
            return static_cast<supported_t>(value);
        } else {
            using real_t = nt::value_type_t<T>;
            using supported_t = std::conditional_t<nt::is_almost_same_v<real_t, f16>, f32, real_t>;
            return static_cast<supported_t>(value);
        }
    }
}

namespace noa::math {
    /// Randomizes an array with uniform random values.
    /// \param output   Array to randomize.
    /// \param min, max Minimum and maximum value of the uniform range.
    /// \note If the output value type is complex and \p Value is real,
    ///       \p output is reinterpreted to the corresponding real type array,
    ///       requiring its width dimension to be contiguous.
    template<typename Output, typename Value, typename = std::enable_if_t<
             nt::is_varray_v<Output> &&
             details::is_valid_random_v<nt::value_type_t<Output>, Value>>>
    void randomize(noa::math::uniform_t, const Output& output, Value min, Value max) {
        NOA_CHECK(!output.is_empty(), "Empty array detected");
        using value_t = nt::value_type_t<Output>;
        const auto min_value = details::cast_to_supported_type<value_t>(min);
        const auto max_value = details::cast_to_supported_type<value_t>(max);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::math::randomize(
                        noa::math::uniform_t{},
                        output.get(), output.strides(), output.shape(),
                        min_value, max_value, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::math::randomize(
                    noa::math::uniform_t{},
                    output.get(), output.strides(), output.shape(),
                    min_value, max_value, cuda_stream);
            cuda_stream.enqueue_attach(output);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Randomizes an array with normal random values.
    /// \param output       Array to randomize.
    /// \param mean, stddev Mean and standard-deviation of the normal range.
    /// \note If the output value type is complex and \p Value is real,
    ///       \p output is reinterpreted to the corresponding real type array,
    ///       requiring its width dimension to be contiguous.
    template<typename Output, typename Value, typename = std::enable_if_t<
             nt::is_varray_v<Output> &&
             details::is_valid_random_v<nt::value_type_t<Output>, Value>>>
    void randomize(noa::math::normal_t, const Output& output, Value mean = Value{0}, Value stddev = Value{1}) {
        NOA_CHECK(!output.is_empty(), "Empty array detected");
        using value_t = nt::value_type_t<Output>;
        const auto mean_value = details::cast_to_supported_type<value_t>(mean);
        const auto stddev_value = details::cast_to_supported_type<value_t>(stddev);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::math::randomize(
                        noa::math::normal_t{},
                        output.get(), output.strides(), output.shape(),
                        mean_value, stddev_value, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::math::randomize(
                    noa::math::normal_t{},
                    output.get(), output.strides(), output.shape(),
                    mean_value, stddev_value, cuda_stream);
            cuda_stream.enqueue_attach(output);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Randomizes an array with log-normal random values.
    /// \param output       Array to randomize.
    /// \param mean, stddev Mean and standard-deviation of the log-normal range.
    /// \note If the output value type is complex and \p Value is real,
    ///       \p output is reinterpreted to the corresponding real type array,
    ///       requiring its width dimension to be contiguous.
    template<typename Output, typename Value, typename = std::enable_if_t<
             nt::is_varray_v<Output> &&
             details::is_valid_random_v<nt::value_type_t<Output>, Value>>>
    void randomize(noa::math::log_normal_t, const Output& output, Value mean = Value{0}, Value stddev = Value{1}) {
        NOA_CHECK(!output.is_empty(), "Empty array detected");
        using value_t = nt::value_type_t<Output>;
        const auto mean_value = details::cast_to_supported_type<value_t>(mean);
        const auto stddev_value = details::cast_to_supported_type<value_t>(stddev);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::math::randomize(
                        noa::math::log_normal_t{},
                        output.get(), output.strides(), output.shape(),
                        mean_value, stddev_value, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::math::randomize(
                    noa::math::log_normal_t{},
                    output.get(), output.strides(), output.shape(),
                    mean_value, stddev_value, cuda_stream);
            cuda_stream.enqueue_attach(output);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Randomizes an array with poisson random values.
    /// \param output   Array to randomize.
    /// \param lambda   Mean value of the poisson range.
    /// \note If the output value type is complex, \p output is reinterpreted to
    ///       the corresponding real type array, requiring its width dimension to be contiguous.
    template<typename Output, typename Real, typename = std::enable_if_t<
             nt::is_varray_v<Output> &&
             details::is_valid_random_v<nt::value_type_t<Output>, Real> &&
             nt::is_real_v<Real>>>
    void randomize(noa::math::poisson_t, const Output& output, Real lambda) {
        NOA_CHECK(!output.is_empty(), "Empty array detected");
        const auto lambda_value = static_cast<f32>(lambda);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::math::randomize(
                        noa::math::poisson_t{},
                        output.get(), output.strides(), output.shape(),
                        lambda_value, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::math::randomize(
                    noa::math::poisson_t{},
                    output.get(), output.strides(), output.shape(),
                    lambda_value, cuda_stream);
            cuda_stream.enqueue_attach(output);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}

namespace noa::math {
    /// Randomizes an array with uniform random values.
    /// \param shape    BDHW shape of the array.
    /// \param min, max Minimum and maximum value of the uniform range.
    template<typename T, typename Value, typename = std::enable_if_t<details::is_valid_random_v<T, Value>>>
    [[nodiscard]] Array<T> random(
            noa::math::uniform_t,
            const Shape4<i64>& shape,
            Value min, Value max,
            ArrayOption option = {}) {
        Array<T> out(shape, option);
        randomize(noa::math::uniform_t{}, out, min, max);
        return out;
    }

    /// Randomizes an array with normal random values.
    /// \param shape        BDHW shape of the array.
    /// \param mean, stddev Mean and standard-deviation of the normal range.
    template<typename T, typename Value, typename = std::enable_if_t<details::is_valid_random_v<T, Value>>>
    [[nodiscard]] Array<T> random(
            noa::math::normal_t,
            const Shape4<i64>& shape,
            Value mean = Value{0},
            Value stddev = Value{1},
            ArrayOption option = {}) {
        Array<T> out(shape, option);
        randomize(noa::math::normal_t{}, out, mean, stddev);
        return out;
    }

    /// Randomizes an array with log-normal random values.
    /// \param shape        BDHW shape of the array.
    /// \param mean, stddev Mean and standard-deviation of the log-normal range.
    template<typename T, typename Value, typename = std::enable_if_t<details::is_valid_random_v<T, Value>>>
    [[nodiscard]] Array<T> random(
            noa::math::log_normal_t,
            const Shape4<i64>& shape,
            Value mean = Value{0},
            Value stddev = Value{1},
            ArrayOption option = {}) {
        Array<T> out(shape, option);
        randomize(noa::math::log_normal_t{}, out, mean, stddev);
        return out;
    }

    /// Randomizes an array with poisson random values.
    /// \param shape    BDHW shape of the array.
    /// \param lambda   Mean value of the poisson range.
    template<typename T, typename Real,
             typename = std::enable_if_t<details::is_valid_random_v<T, Real> && nt::is_real_v<Real>>>
    [[nodiscard]] Array<T> random(
            noa::math::poisson_t,
            const Shape4<i64>& shape,
            Real lambda,
            ArrayOption option = {}) {
        Array<T> out(shape, option);
        randomize(noa::math::poisson_t{}, out, lambda);
        return out;
    }

    /// Randomizes an array with uniform random values.
    /// \param elements Number of elements to generate.
    /// \param min, max Minimum and maximum value of the uniform range.
    template<typename T, typename Value, typename = std::enable_if_t<details::is_valid_random_v<T, Value>>>
    [[nodiscard]] Array<T> random(
            noa::math::uniform_t,
            i64 elements,
            Value min, Value max,
            ArrayOption option = {}) {
        Array<T> out(elements, option);
        randomize(noa::math::uniform_t{}, out, min, max);
        return out;
    }

    /// Randomizes an array with normal random values.
    /// \param elements     Number of elements to generate.
    /// \param mean, stddev Mean and standard-deviation of the normal range.
    template<typename T, typename Value, typename = std::enable_if_t<details::is_valid_random_v<T, Value>>>
    [[nodiscard]] Array<T> random(
            noa::math::normal_t,
            i64 elements,
            Value mean = Value{0},
            Value stddev = Value{1},
            ArrayOption option = {}) {
        Array<T> out(elements, option);
        randomize(noa::math::normal_t{}, out, mean, stddev);
        return out;
    }

    /// Randomizes an array with log-normal random values.
    /// \param elements     Number of elements to generate.
    /// \param mean, stddev Mean and standard-deviation of the log-normal range.
    template<typename T, typename Value, typename = std::enable_if_t<details::is_valid_random_v<T, Value>>>
    [[nodiscard]] Array<T> random(
            noa::math::log_normal_t,
            i64 elements,
            Value mean = Value{0},
            Value stddev = Value{1},
            ArrayOption option = {}) {
        Array<T> out(elements, option);
        randomize(noa::math::log_normal_t{}, out, mean, stddev);
        return out;
    }

    /// Randomizes an array with poisson random values.
    /// \param elements Number of elements to generate.
    /// \param lambda   Mean value of the poisson range.
    template<typename T, typename Real,
             typename = std::enable_if_t<details::is_valid_random_v<T, Real> && nt::is_real_v<Real>>>
    [[nodiscard]] Array<T> random(
            noa::math::poisson_t,
            i64 elements,
            Real lambda,
            ArrayOption option = {}) {
        Array<T> out(elements, option);
        randomize(noa::math::poisson_t{}, out, lambda);
        return out;
    }
}
