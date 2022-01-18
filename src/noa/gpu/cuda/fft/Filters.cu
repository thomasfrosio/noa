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

    template<bool IS_CENTERED>
    __forceinline__ __device__ int getFrequency_(int idx, int dim) {
        if constexpr(IS_CENTERED)
            return idx - dim / 2;
        else
            return idx < (dim + 1) / 2 ? idx : idx - dim;
        return 0;
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED>
    __forceinline__ __device__ int getOutputIndex_(int i_idx, [[maybe_unused]] int dim) {
        (void) dim;
        if constexpr (IS_SRC_CENTERED == IS_DST_CENTERED)
            return i_idx;
        else if constexpr (IS_SRC_CENTERED)
            return noa::math::iFFTShift(i_idx, dim);
        else
            return noa::math::FFTShift(i_idx, dim);
        return 0;
    }

    template<bool IS_CENTERED>
    __forceinline__ __device__ float getNormalizedFrequencySqd(int3_t gid, int3_t shape, float3_t norm) {
        float3_t distance_sqd(gid.x,
                              getFrequency_<IS_CENTERED>(gid.y, shape.y),
                              getFrequency_<IS_CENTERED>(gid.z, shape.z));
        distance_sqd *= norm;
        return math::dot(distance_sqd, distance_sqd);
    }
}

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

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, Type PASS, bool HAS_WIDTH, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void singlePass_(const T* inputs, uint3_t input_pitch, T* outputs, uint3_t output_pitch,
                     int3_t shape, float3_t norm, float cutoff, [[maybe_unused]] float width, uint blocks_x) {
        using real_t = noa::traits::value_type_t<T>;

        // Get the current indexes within the input.
        const uint2_t idx = coordinates(blockIdx.x, blocks_x);
        const int3_t gid(THREADS.x * idx.x + threadIdx.x,
                         THREADS.y * idx.y + threadIdx.y,
                         blockIdx.y);
        if (gid.x >= shape.x / 2 + 1 || gid.y >= shape.y)
            return;

        // Offset to current batch.
        inputs += blockIdx.z * elements(input_pitch);
        outputs += blockIdx.z * elements(output_pitch);

        // Offset to current output frequency.
        const int64_t oy = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid.y, shape.y);
        const int64_t oz = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid.z, shape.z);
        outputs += (oz * output_pitch.y + oy) * output_pitch.x + gid.x;

        // Get filter for current input frequency.
        real_t filter;
        if constexpr (HAS_WIDTH) {
            const float frequency = math::sqrt(getNormalizedFrequencySqd<IS_SRC_CENTERED>(gid, shape, norm));
            filter = static_cast<real_t>(getSoftWindow_<PASS>(cutoff, width, frequency));
        } else {
            const float frequency_sqd = getNormalizedFrequencySqd<IS_SRC_CENTERED>(gid, shape, norm);
            filter = static_cast<real_t>(getHardWindow_<PASS>(cutoff * cutoff, frequency_sqd));
            (void) width;
        }

        // Save to output.
        *outputs = inputs ? inputs[(gid.z * input_pitch.y + gid.y) * input_pitch.x + gid.x] * filter : filter;
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, bool HAS_WIDTH, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void bandPass_(const T* inputs, uint3_t input_pitch, T* outputs, uint3_t output_pitch,
                   int3_t shape, float3_t norm,
                   float cutoff_1, float cutoff_2,
                   float width_1, float width_2, uint blocks_x) {
        using real_t = noa::traits::value_type_t<T>;

        // Get the current indexes within the input.
        const uint2_t idx = coordinates(blockIdx.x, blocks_x);
        const int3_t gid(THREADS.x * idx.x + threadIdx.x,
                         THREADS.y * idx.y + threadIdx.y,
                         blockIdx.y);
        if (gid.x >= shape.x / 2 + 1 || gid.y >= shape.y)
            return;

        // Offset to current batch.
        inputs += blockIdx.z * elements(input_pitch);
        outputs += blockIdx.z * elements(output_pitch);

        // Offset to current output frequency.
        const int64_t oy = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid.y, shape.y);
        const int64_t oz = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid.z, shape.z);
        outputs += (oz * output_pitch.y + oy) * output_pitch.x + gid.x;

        // Get filter for current input frequency.
        real_t filter;
        if constexpr (HAS_WIDTH) {
            const float frequency = math::sqrt(getNormalizedFrequencySqd<IS_SRC_CENTERED>(gid, shape, norm));
            filter = static_cast<real_t>(getSoftWindow_<Type::HIGHPASS>(cutoff_1, width_1, frequency) *
                                         getSoftWindow_<Type::LOWPASS>(cutoff_2, width_2, frequency));
        } else {
            const float frequency_sqd = getNormalizedFrequencySqd<IS_SRC_CENTERED>(gid, shape, norm);
            filter = static_cast<real_t>(getSoftWindow_<Type::HIGHPASS>(cutoff_1 * cutoff_1, width_1, frequency_sqd) *
                                         getSoftWindow_<Type::LOWPASS>(cutoff_2 * cutoff_2, width_2, frequency_sqd));
            (void) width_1;
            (void) width_2;
        }

        // Save to output.
        *outputs = inputs ? inputs[(gid.z * input_pitch.y + gid.y) * input_pitch.x + gid.x] * filter : filter;
    }
}

namespace {
    template<Type PASS, ::noa::fft::Remap REMAP, typename T>
    void launchSinglePass_(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                           size3_t shape, size_t batches, float cutoff, float width, cuda::Stream& stream) {
        using Layout = ::noa::fft::Layout;
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL) {
            static_assert(traits::always_false_v<T>);
        }
        NOA_ASSERT(inputs != outputs || IS_SRC_CENTERED == IS_DST_CENTERED);

        const int3_t s_shape(shape);
        float3_t norm(shape / 2 * 2 + size3_t{shape == 1});
        norm = 1.f / norm;

        const uint blocks_x = math::divideUp(s_shape.x / 2 + 1, static_cast<int>(THREADS.x));
        const uint blocks_y = math::divideUp(s_shape.y, static_cast<int>(THREADS.y));
        const dim3 blocks(blocks_x * blocks_y, s_shape.z, batches);
        if (width > 1e-6f) {
            singlePass_<IS_SRC_CENTERED, IS_DST_CENTERED, PASS, true><<<blocks, THREADS, 0, stream.id()>>>(
                    inputs, uint3_t{input_pitch}, outputs, uint3_t{output_pitch},
                    s_shape, norm, cutoff, width, blocks_x);
        } else {
            singlePass_<IS_SRC_CENTERED, IS_DST_CENTERED, PASS, false><<<blocks, THREADS, 0, stream.id()>>>(
                    inputs, uint3_t{input_pitch}, outputs, uint3_t{output_pitch},
                    s_shape, norm, cutoff, width, blocks_x);
        }
        NOA_THROW_IF(cudaGetLastError());
    }
}

namespace noa::cuda::fft {
    template<Remap REMAP, typename T>
    void lowpass(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                 size3_t shape, size_t batches, float cutoff, float width, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        launchSinglePass_<Type::LOWPASS, REMAP>(
                inputs, input_pitch, outputs, output_pitch, shape, batches, cutoff, width, stream);
    }

    template<Remap REMAP, typename T>
    void highpass(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                  size3_t shape, size_t batches, float cutoff, float width, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        launchSinglePass_<Type::HIGHPASS, REMAP>(
                inputs, input_pitch, outputs, output_pitch, shape, batches, cutoff, width, stream);
    }

    template<Remap REMAP, typename T>
    NOA_HOST void bandpass(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                           size_t batches, float cutoff1, float cutoff2, float width1, float width2, Stream& stream) {
        using Layout = ::noa::fft::Layout;
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL) {
            static_assert(traits::always_false_v<T>);
        }
        NOA_ASSERT(inputs != outputs || IS_SRC_CENTERED == IS_DST_CENTERED);
        NOA_PROFILE_FUNCTION();
        const int3_t s_shape(shape);
        float3_t norm(shape / 2 * 2 + size3_t{shape == 1});
        norm = 1.f / norm;

        const uint blocks_x = math::divideUp(s_shape.x / 2 + 1, static_cast<int>(THREADS.x));
        const uint blocks_y = math::divideUp(s_shape.y, static_cast<int>(THREADS.y));
        const dim3 blocks(blocks_x * blocks_y, s_shape.z, batches);
        if (width1 > 1e-6f || width2 > 1e-6f) {
            bandPass_<IS_SRC_CENTERED, IS_DST_CENTERED, true><<<blocks, THREADS, 0, stream.id()>>>(
                    inputs, uint3_t{input_pitch}, outputs, uint3_t{output_pitch}, s_shape, norm,
                    cutoff1, cutoff2, width1, width2, blocks_x);
        } else {
            bandPass_<IS_SRC_CENTERED, IS_DST_CENTERED, false><<<blocks, THREADS, 0, stream.id()>>>(
                    inputs, uint3_t{input_pitch}, outputs, uint3_t{output_pitch}, s_shape, norm,
                    cutoff1, cutoff2, width1, width2, blocks_x);
        }
        NOA_THROW_IF(cudaGetLastError());
    }

    #define NOA_INSTANTIATE_FILTERS_(T)                                                                                         \
    template void lowpass<Remap::H2H, T>(const T*, size3_t, T*, size3_t, size3_t, size_t, float, float, Stream&);               \
    template void highpass<Remap::H2H,T>(const T*, size3_t, T*, size3_t, size3_t, size_t, float, float, Stream&);               \
    template void bandpass<Remap::H2H,T>(const T*, size3_t, T*, size3_t, size3_t, size_t, float, float, float, float, Stream&); \
    template void lowpass<Remap::H2HC, T>(const T*, size3_t, T*, size3_t, size3_t, size_t, float, float, Stream&);              \
    template void highpass<Remap::H2HC,T>(const T*, size3_t, T*, size3_t, size3_t, size_t, float, float, Stream&);              \
    template void bandpass<Remap::H2HC,T>(const T*, size3_t, T*, size3_t, size3_t, size_t, float, float, float, float, Stream&);\
    template void lowpass<Remap::HC2H, T>(const T*, size3_t, T*, size3_t, size3_t, size_t, float, float, Stream&);              \
    template void highpass<Remap::HC2H,T>(const T*, size3_t, T*, size3_t, size3_t, size_t, float, float, Stream&);              \
    template void bandpass<Remap::HC2H,T>(const T*, size3_t, T*, size3_t, size3_t, size_t, float, float, float, float, Stream&);\
    template void lowpass<Remap::HC2HC, T>(const T*, size3_t, T*, size3_t, size3_t, size_t, float, float, Stream&);             \
    template void highpass<Remap::HC2HC,T>(const T*, size3_t, T*, size3_t, size3_t, size_t, float, float, Stream&);             \
    template void bandpass<Remap::HC2HC,T>(const T*, size3_t, T*, size3_t, size3_t, size_t, float, float, float, float, Stream&)

    NOA_INSTANTIATE_FILTERS_(half_t);
    NOA_INSTANTIATE_FILTERS_(float);
    NOA_INSTANTIATE_FILTERS_(double);
    NOA_INSTANTIATE_FILTERS_(chalf_t);
    NOA_INSTANTIATE_FILTERS_(cfloat_t);
    NOA_INSTANTIATE_FILTERS_(cdouble_t);
}
