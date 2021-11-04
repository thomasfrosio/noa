#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/fft/Exception.h"
#include "noa/gpu/cuda/fft/Filters.h"
#include "noa/gpu/cuda/memory/Copy.h"

// Commons:
namespace {
    using namespace noa;
    constexpr dim3 THREADS(32, 8);

    enum class Type {
        LOWPASS,
        HIGHPASS
    };

    __forceinline__ __device__ float getDistance_(uint idx, uint half, uint dimension) {
        return idx >= half ? static_cast<float>(idx) - static_cast<float>(dimension) : static_cast<float>(idx);
    }

    __forceinline__ __device__ float getNormalizedFrequencySqd(uint3_t gid, uint3_t shape, uint3_t half) {
        float3_t distance_sqd(gid.x,
                              getDistance_(gid.y, half.y, shape.y),
                              getDistance_(gid.z, half.z, shape.z));
        distance_sqd /= float3_t(shape);
        return math::dot(distance_sqd, distance_sqd);
    }
}

// Soft edges (Hann window):
namespace {
    template<Type PASS>
    inline __device__ float getSoftWindow_(float freq_cutoff, float freq_width, float freq) {
        constexpr float PI = math::Constants<float>::PI;
        float filter;
        if constexpr (PASS == Type::LOWPASS) {
            if (freq <= freq_cutoff)
                filter = 1;
            else if (freq_cutoff + freq_width <= freq)
                filter = 0;
            else
                filter = (1.f + math::cos(PI * (freq_cutoff - freq) / freq_width)) * 0.5f;
        } else if constexpr (PASS == Type::HIGHPASS) {
            if (freq_cutoff <= freq)
                filter = 1;
            else if (freq <= freq_cutoff - freq_width)
                filter = 0;
            else
                filter = (1.f + math::cos(PI * (freq - freq_cutoff) / freq_width)) * 0.5f;
        }
        return filter;
    }

    template<Type PASS, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void singlePassSoft_(const T* inputs, uint inputs_pitch, T* outputs, uint outputs_pitch,
                         uint3_t shape, uint3_t half, float freq_cutoff, float freq_width, uint blocks_x) {
        using real_t = noa::traits::value_type_t<T>;

        // Get the current indexes.
        const uint2_t idx = coordinates(blockIdx.x, blocks_x);
        const uint3_t gid(THREADS.x * idx.x + threadIdx.x,
                          THREADS.y * idx.y + threadIdx.y,
                          blockIdx.y);
        if (gid.x >= half.x || gid.y >= shape.y)
            return;

        // Offset to current batch.
        inputs += blockIdx.z * rows(shape) * inputs_pitch;
        outputs += blockIdx.z * rows(shape) * outputs_pitch;

        // Apply filter.
        float frequency = math::sqrt(getNormalizedFrequencySqd(gid, shape, half));
        auto filter = static_cast<real_t>(getSoftWindow_<PASS>(freq_cutoff, freq_width, frequency));
        outputs[(gid.z * shape.y + gid.y) * outputs_pitch + gid.x] =
                inputs[(gid.z * shape.y + gid.y) * inputs_pitch + gid.x] * filter;
    }

    template<Type PASS, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void singlePassSoft_(T* output_filter, uint output_filter_pitch,
                         uint3_t shape, uint3_t half, float freq_cutoff, float freq_width) {
        using real_t = noa::traits::value_type_t<T>;

        const uint3_t gid(THREADS.x * blockIdx.x + threadIdx.x,
                          THREADS.y * blockIdx.y + threadIdx.y,
                          blockIdx.z);
        if (gid.x >= half.x || gid.y >= shape.y)
            return;

        float frequency = math::sqrt(getNormalizedFrequencySqd(gid, shape, half));
        auto filter = static_cast<real_t>(getSoftWindow_<PASS>(freq_cutoff, freq_width, frequency));
        output_filter[(gid.z * shape.y + gid.y) * output_filter_pitch + gid.x] = filter;
    }

    template<typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void bandPassSoft_(const T* inputs, uint inputs_pitch, T* outputs, uint outputs_pitch,
                       uint3_t shape, uint3_t half, float freq_cutoff_1, float freq_cutoff_2,
                       float freq_width_1, float freq_width_2, uint blocks_x) {
        using real_t = noa::traits::value_type_t<T>;

        const uint2_t idx = coordinates(blockIdx.x, blocks_x);
        const uint3_t gid(THREADS.x * idx.x + threadIdx.x,
                          THREADS.y * idx.y + threadIdx.y,
                          blockIdx.y);
        if (gid.x >= half.x || gid.y >= shape.y)
            return;

        inputs += blockIdx.z * rows(shape) * inputs_pitch;
        outputs += blockIdx.z * rows(shape) * outputs_pitch;

        float frequency = math::sqrt(getNormalizedFrequencySqd(gid, shape, half));
        float filter = getSoftWindow_<Type::HIGHPASS>(freq_cutoff_1, freq_width_1, frequency);
        filter *= getSoftWindow_<Type::LOWPASS>(freq_cutoff_2, freq_width_2, frequency);
        outputs[(gid.z * shape.y + gid.y) * outputs_pitch + gid.x] =
                inputs[(gid.z * shape.y + gid.y) * inputs_pitch + gid.x] * static_cast<real_t>(filter);
    }

    template<typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void bandPassSoft_(T* output_filter, uint output_filter_pitch, uint3_t shape, uint3_t half,
                       float freq_cutoff_1, float freq_cutoff_2, float freq_width_1, float freq_width_2) {
        using real_t = noa::traits::value_type_t<T>;

        const uint3_t gid(THREADS.x * blockIdx.x + threadIdx.x,
                          THREADS.y * blockIdx.y + threadIdx.y,
                          blockIdx.z);
        if (gid.x >= half.x || gid.y >= shape.y)
            return;

        float frequency = math::sqrt(getNormalizedFrequencySqd(gid, shape, half));
        float filter = getSoftWindow_<Type::HIGHPASS>(freq_cutoff_1, freq_width_1, frequency);
        filter *= getSoftWindow_<Type::LOWPASS>(freq_cutoff_2, freq_width_2, frequency);
        output_filter[(gid.z * shape.y + gid.y) * output_filter_pitch + gid.x] = static_cast<real_t>(filter);
    }
}

// Hard edges:
namespace {
    template<Type PASS>
    inline __device__ float getHardWindow_(float freq_cutoff_sqd, float freq_sqd) {
        float filter;
        if constexpr (PASS == Type::LOWPASS) {
            if (freq_cutoff_sqd < freq_sqd)
                filter = 0;
            else
                filter = 1;
        } else if constexpr (PASS == Type::HIGHPASS) {
            if (freq_sqd < freq_cutoff_sqd)
                filter = 0;
            else
                filter = 1;
        }
        return filter;
    }

    template<Type PASS, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void singlePassHard_(const T* inputs, uint inputs_pitch, T* outputs, uint outputs_pitch,
                         uint3_t shape, uint3_t half, float freq_cutoff, uint blocks_x) {
        using real_t = noa::traits::value_type_t<T>;

        // Get the current indexes.
        const uint2_t idx = coordinates(blockIdx.x, blocks_x);
        const uint3_t gid(THREADS.x * idx.x + threadIdx.x,
                          THREADS.y * idx.y + threadIdx.y,
                          blockIdx.y);
        if (gid.x >= half.x || gid.y >= shape.y)
            return;

        // Offset to current batch.
        inputs += blockIdx.z * rows(shape) * inputs_pitch;
        outputs += blockIdx.z * rows(shape) * outputs_pitch;

        // Apply filter.
        float freq_cutoff_sqd = freq_cutoff * freq_cutoff;
        float frequency_sqd = getNormalizedFrequencySqd(gid, shape, half);
        auto filter = static_cast<real_t>(getHardWindow_<PASS>(freq_cutoff_sqd, frequency_sqd));
        outputs[(gid.z * shape.y + gid.y) * outputs_pitch + gid.x] =
                inputs[(gid.z * shape.y + gid.y) * inputs_pitch + gid.x] * filter;
    }

    template<Type PASS, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void singlePassHard_(T* output_filter, uint output_filter_pitch,
                         uint3_t shape, uint3_t half, float freq_cutoff) {
        using real_t = noa::traits::value_type_t<T>;

        const uint3_t gid(THREADS.x * blockIdx.x + threadIdx.x,
                          THREADS.y * blockIdx.y + threadIdx.y,
                          blockIdx.z);
        if (gid.x >= half.x || gid.y >= shape.y)
            return;

        float freq_cutoff_sqd = freq_cutoff * freq_cutoff;
        float frequency_sqd = getNormalizedFrequencySqd(gid, shape, half);
        auto filter = static_cast<real_t>(getHardWindow_<PASS>(freq_cutoff_sqd, frequency_sqd));
        output_filter[(gid.z * shape.y + gid.y) * output_filter_pitch + gid.x] = filter;
    }

    template<typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void bandPassHard_(const T* inputs, uint inputs_pitch, T* outputs, uint outputs_pitch,
                       uint3_t shape, uint3_t half, float freq_cutoff_1, float freq_cutoff_2,
                       uint blocks_x) {
        using real_t = noa::traits::value_type_t<T>;

        const uint2_t idx = coordinates(blockIdx.x, blocks_x);
        const uint3_t gid(THREADS.x * idx.x + threadIdx.x,
                          THREADS.y * idx.y + threadIdx.y,
                          blockIdx.y);
        if (gid.x >= half.x || gid.y >= shape.y)
            return;

        inputs += blockIdx.z * rows(shape) * inputs_pitch;
        outputs += blockIdx.z * rows(shape) * outputs_pitch;

        float freq_cutoff_sqd_1 = freq_cutoff_1 * freq_cutoff_1;
        float freq_cutoff_sqd_2 = freq_cutoff_2 * freq_cutoff_2;
        float frequency_sqd = getNormalizedFrequencySqd(gid, shape, half);
        float filter = getHardWindow_<Type::HIGHPASS>(freq_cutoff_sqd_1, frequency_sqd);
        filter *= getHardWindow_<Type::LOWPASS>(freq_cutoff_sqd_2, frequency_sqd);
        outputs[(gid.z * shape.y + gid.y) * outputs_pitch + gid.x] =
                inputs[(gid.z * shape.y + gid.y) * inputs_pitch + gid.x] * static_cast<real_t>(filter);
    }

    template<typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void bandPassHard_(T* output_filter, uint output_filter_pitch, uint3_t shape, uint3_t half,
                       float freq_cutoff_1, float freq_cutoff_2) {
        using real_t = noa::traits::value_type_t<T>;

        const uint3_t gid(THREADS.x * blockIdx.x + threadIdx.x,
                          THREADS.y * blockIdx.y + threadIdx.y,
                          blockIdx.z);
        if (gid.x >= half.x || gid.y >= shape.y)
            return;

        float freq_cutoff_sqd_1 = freq_cutoff_1 * freq_cutoff_1;
        float freq_cutoff_sqd_2 = freq_cutoff_2 * freq_cutoff_2;
        float frequency_sqd = getNormalizedFrequencySqd(gid, shape, half);
        float filter = getHardWindow_<Type::HIGHPASS>(freq_cutoff_sqd_1, frequency_sqd);
        filter *= getHardWindow_<Type::LOWPASS>(freq_cutoff_sqd_2, frequency_sqd);
        output_filter[(gid.z * shape.y + gid.y) * output_filter_pitch + gid.x] = static_cast<real_t>(filter);
    }

    template<Type PASS, typename T>
    void singlePass(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                    size3_t shape, size_t batches,
                    float freq_cutoff, float freq_width, cuda::Stream& stream) {
        NOA_PROFILE_FUNCTION();
        uint3_t u_shape(shape);
        uint3_t half(u_shape / 2U + 1U);

        uint blocks_x = math::divideUp(half.x, THREADS.x);
        uint blocks_y = math::divideUp(u_shape.y, THREADS.y);

        if (inputs) {
            dim3 blocks(blocks_x * blocks_y, u_shape.z, batches);
            if (freq_width > 1e-6f) {
                singlePassSoft_<PASS><<<blocks, THREADS, 0, stream.id()>>>(
                        inputs, inputs_pitch, outputs, outputs_pitch, u_shape, half, freq_cutoff, freq_width, blocks_x);
            } else {
                singlePassHard_<PASS><<<blocks, THREADS, 0, stream.id()>>>(
                        inputs, inputs_pitch, outputs, outputs_pitch, u_shape, half, freq_cutoff, blocks_x);
            }
        } else {
            if constexpr(!traits::is_complex_v<T>) {
                dim3 blocks(blocks_x, blocks_y, u_shape.z);
                if (freq_width > 1e-6f) {
                    singlePassSoft_<PASS><<<blocks, THREADS, 0, stream.id()>>>(
                            outputs, outputs_pitch, u_shape, half, freq_cutoff, freq_width);
                } else {
                    singlePassHard_<PASS><<<blocks, THREADS, 0, stream.id()>>>(
                            outputs, outputs_pitch, u_shape, half, freq_cutoff);
                }
                const size_t elements = outputs_pitch * rows(shape);
                for (size_t batch = 1; batch < batches; ++batch)
                    cuda::memory::copy(outputs, outputs + elements * batch, elements, stream);
            } else {
                NOA_THROW_FUNC("(low|high)pass", "Cannot compute a filter of complex type");
            }
        }
        NOA_THROW_IF(cudaGetLastError());
    }
}

namespace noa::cuda::fft {
    template<typename T>
    void lowpass(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                 size3_t shape, size_t batches,
                 float freq_cutoff, float freq_width, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        singlePass<Type::LOWPASS>(inputs, inputs_pitch, outputs, outputs_pitch,
                                  shape, batches, freq_cutoff, freq_width, stream);
    }

    template<typename T>
    void highpass(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                  size3_t shape, size_t batches,
                  float freq_cutoff, float freq_width, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        singlePass<Type::HIGHPASS>(inputs, inputs_pitch, outputs, outputs_pitch,
                                   shape, batches, freq_cutoff, freq_width, stream);
    }

    template<typename T>
    void bandpass(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                  size3_t shape, size_t batches,
                  float freq_cutoff_1, float freq_cutoff_2, float freq_width_1, float freq_width_2, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        uint3_t u_shape(shape);
        uint3_t half(u_shape / 2U + 1U);

        uint blocks_x = math::divideUp(half.x, THREADS.x);
        uint blocks_y = math::divideUp(u_shape.y, THREADS.y);

        if (inputs) {
            dim3 blocks(blocks_x * blocks_y, u_shape.z, batches);
            if (freq_width_1 > 1e-6f || freq_width_2 > 1e-6f) {
                bandPassSoft_<<<blocks, THREADS, 0, stream.id()>>>(
                        inputs, inputs_pitch, outputs, outputs_pitch, u_shape, half,
                        freq_cutoff_1, freq_cutoff_2, freq_width_1, freq_width_2, blocks_x);
            } else {
                bandPassHard_<<<blocks, THREADS, 0, stream.id()>>>(
                        inputs, inputs_pitch, outputs, outputs_pitch, u_shape, half,
                        freq_cutoff_1, freq_cutoff_2, blocks_x);
            }
        } else {
            if constexpr(!traits::is_complex_v<T>) {
                dim3 blocks(blocks_x, blocks_y, u_shape.z);
                if (freq_width_1 > 1e-6f || freq_width_2 > 1e-6f) {
                    bandPassSoft_<<<blocks, THREADS, 0, stream.id()>>>(
                            outputs, outputs_pitch, u_shape, half,
                            freq_cutoff_1, freq_cutoff_2, freq_width_1, freq_width_2);
                } else {
                    bandPassHard_<<<blocks, THREADS, 0, stream.id()>>>(
                            outputs, outputs_pitch, u_shape, half,
                            freq_cutoff_1, freq_cutoff_2);
                }
                const size_t elements = outputs_pitch * rows(shape);
                for (size_t batch = 1; batch < batches; ++batch)
                    cuda::memory::copy(outputs, outputs + elements * batch, elements, stream);
            } else {
                NOA_THROW_FUNC("(low|high)pass", "Cannot compute a filter of complex type");
            }
        }

        NOA_THROW_IF(cudaGetLastError());
    }

    #define NOA_INSTANTIATE_FILTERS_(T)                                                                 \
    template void lowpass<T>(const T*, size_t, T*, size_t, size3_t, size_t, float, float, Stream&);     \
    template void highpass<T>(const T*, size_t, T*, size_t, size3_t, size_t, float, float, Stream&);    \
    template void bandpass<T>(const T*, size_t, T*, size_t, size3_t, size_t, float, float, float, float, Stream&)

    NOA_INSTANTIATE_FILTERS_(float);
    NOA_INSTANTIATE_FILTERS_(double);
    NOA_INSTANTIATE_FILTERS_(cfloat_t);
    NOA_INSTANTIATE_FILTERS_(cdouble_t);
}
