#include "noa/Math.h"
#include "noa/gpu/cuda/fourier/Exception.h"
#include "noa/gpu/cuda/fourier/Filters.h"

// TODO Test 2D block to reduce idle threads / divergence, e.g. BLOCK_SIZE(32, 8).

// Commons:
namespace {
    enum class Type { LOWPASS, HIGHPASS };

    inline __device__ float getDistanceSquared_(size_t dimension, uint half, size_t idx) {
        float dist = idx >= half ? static_cast<float>(idx) - static_cast<float>(dimension) : static_cast<float>(idx);
        dist /= static_cast<float>(dimension);
        dist *= dist;
        return dist;
    }
}

// Soft edges (Hann window):
namespace {
    using namespace Noa;

    template<Type PASS>
    inline __device__ float getSoftWindow_(float freq_cutoff, float freq_width, float freq) {
        constexpr float PI = Math::Constants<float>::PI;
        float filter;
        if constexpr (PASS == Type::LOWPASS) {
            if (freq <= freq_cutoff)
                filter = 1;
            else if (freq_cutoff + freq_width <= freq)
                filter = 0;
            else
                filter = (1.f + Math::cos(PI * (freq_cutoff - freq) / freq_width)) * 0.5f;
        } else if constexpr (PASS == Type::HIGHPASS) {
            if (freq_cutoff <= freq)
                filter = 1;
            else if (freq <= freq_cutoff - freq_width)
                filter = 0;
            else
                filter = (1.f + Math::cos(PI * (freq - freq_cutoff) / freq_width)) * 0.5f;
        }
        return filter;
    }

    template<Type PASS, typename T>
    __global__ void singlePassSoft_(T* inputs, uint pitch_inputs, T* outputs, uint pitch_outputs,
                                    uint3_t shape, uint3_t half, float freq_cutoff, float freq_width, uint batches) {
        using real_t = Noa::Traits::value_type_t<T>;
        uint y = blockIdx.x, z = blockIdx.y;
        inputs += (z * shape.y + y) * pitch_inputs;
        outputs += (z * shape.y + y) * pitch_outputs;

        uint elements_inputs = 0, elements_outputs = 0;
        if (batches) {
            uint rows = getRows(shape);
            elements_inputs = pitch_inputs * rows;
            elements_outputs = pitch_outputs * rows;
        }

        float3_t distance_sqd;
        distance_sqd.z = getDistanceSquared_(shape.z, half.z, z);
        distance_sqd.y = getDistanceSquared_(shape.y, half.y, y);

        float frequency;
        real_t filter;
        for (uint x = threadIdx.x; x < half.x; x += blockDim.x) {
            distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
            distance_sqd.x *= distance_sqd.x;
            frequency = Math::sqrt(Math::sum(distance_sqd)); // from 0 to 0.5
            filter = static_cast<real_t>(getSoftWindow_<PASS>(freq_cutoff, freq_width, frequency));
            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + x] = inputs[batch * elements_inputs + x] * filter;
        }
    }

    template<Type PASS, typename T>
    __global__ void singlePassSoft_(T* output_filter, uint pitch_output_filter, uint3_t shape, uint3_t half,
                                    float freq_cutoff, float freq_width) {
        using real_t = Noa::Traits::value_type_t<T>;
        uint y = blockIdx.x, z = blockIdx.y;
        output_filter += (z * shape.y + y) * pitch_output_filter;

        float3_t distance_sqd;
        distance_sqd.z = getDistanceSquared_(shape.z, half.z, z);
        distance_sqd.y = getDistanceSquared_(shape.y, half.y, y);

        float frequency, filter;
        for (uint x = threadIdx.x; x < half.x; x += blockDim.x) {
            distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
            distance_sqd.x *= distance_sqd.x;
            frequency = Math::sqrt(Math::sum(distance_sqd)); // from 0 to 0.5
            filter = getSoftWindow_<PASS>(freq_cutoff, freq_width, frequency);
            output_filter[x] = static_cast<real_t>(filter);
        }
    }

    template<typename T>
    __global__ void bandPassSoft_(T* inputs, uint pitch_inputs, T* outputs, uint pitch_outputs,
                                  uint3_t shape, uint3_t half, float freq_cutoff_1, float freq_cutoff_2,
                                  float freq_width_1, float freq_width_2, uint batches) {
        using real_t = Noa::Traits::value_type_t<T>;
        uint y = blockIdx.x, z = blockIdx.y;
        inputs += (z * shape.y + y) * pitch_inputs;
        outputs += (z * shape.y + y) * pitch_outputs;

        uint elements_inputs = 0, elements_outputs = 0;
        if (batches) {
            uint rows = getRows(shape);
            elements_inputs = pitch_inputs * rows;
            elements_outputs = pitch_outputs * rows;
        }

        float3_t distance_sqd;
        distance_sqd.z = getDistanceSquared_(shape.z, half.z, z);
        distance_sqd.y = getDistanceSquared_(shape.y, half.y, y);

        float frequency, filter;
        for (uint x = threadIdx.x; x < half.x; x += blockDim.x) {
            distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
            distance_sqd.x *= distance_sqd.x;
            frequency = Math::sqrt(Math::sum(distance_sqd)); // from 0 to 0.5
            filter = getSoftWindow_<Type::HIGHPASS>(freq_cutoff_1, freq_width_1, frequency);
            filter *= getSoftWindow_<Type::LOWPASS>(freq_cutoff_2, freq_width_2, frequency);
            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + x] =
                        inputs[batch * elements_inputs + x] * static_cast<real_t>(filter);
        }
    }

    template<typename T>
    __global__ void bandPassSoft_(T* output_filter, uint pitch_output_filter, uint3_t shape, uint3_t half,
                                  float freq_cutoff_1, float freq_cutoff_2, float freq_width_1, float freq_width_2) {
        using real_t = Noa::Traits::value_type_t<T>;
        uint y = blockIdx.x, z = blockIdx.y;
        output_filter += (z * shape.y + y) * pitch_output_filter;

        float3_t distance_sqd;
        distance_sqd.z = getDistanceSquared_(shape.z, half.z, z);
        distance_sqd.y = getDistanceSquared_(shape.y, half.y, y);

        float frequency, filter;
        for (uint x = threadIdx.x; x < half.x; x += blockDim.x) {
            distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
            distance_sqd.x *= distance_sqd.x;
            frequency = Math::sqrt(Math::sum(distance_sqd)); // from 0 to 0.5
            filter = getSoftWindow_<Type::HIGHPASS>(freq_cutoff_1, freq_width_1, frequency);
            filter *= getSoftWindow_<Type::LOWPASS>(freq_cutoff_2, freq_width_2, frequency);
            output_filter[x] = static_cast<real_t>(filter);
        }
    }
}

// Hard edges:
namespace {
    using namespace Noa;

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
    __global__ void singlePassHard_(T* inputs, uint pitch_inputs, T* outputs, uint pitch_outputs,
                                    uint3_t shape, uint3_t half, float freq_cutoff, uint batches) {
        using real_t = Noa::Traits::value_type_t<T>;
        uint y = blockIdx.x, z = blockIdx.y;
        inputs += (z * shape.y + y) * pitch_inputs;
        outputs += (z * shape.y + y) * pitch_outputs;

        uint elements_inputs = 0, elements_outputs = 0;
        if (batches) {
            uint rows = getRows(shape);
            elements_inputs = pitch_inputs * rows;
            elements_outputs = pitch_outputs * rows;
        }

        float3_t distance_sqd;
        distance_sqd.z = getDistanceSquared_(shape.z, half.z, z);
        distance_sqd.y = getDistanceSquared_(shape.y, half.y, y);

        float freq_cutoff_sqd = freq_cutoff * freq_cutoff;
        float frequency_sqd;
        real_t filter;
        for (uint x = threadIdx.x; x < half.x; x += blockDim.x) {
            distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
            distance_sqd.x *= distance_sqd.x;
            frequency_sqd = Math::sum(distance_sqd); // from 0 to 0.25
            filter = static_cast<real_t>(getHardWindow_<PASS>(freq_cutoff_sqd, frequency_sqd));
            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + x] = inputs[batch * elements_inputs + x] * filter;
        }
    }

    template<Type PASS, typename T>
    __global__ void singlePassHard_(T* output_filter, uint pitch_output_filter,
                                    uint3_t shape, uint3_t half, float freq_cutoff) {
        using real_t = Noa::Traits::value_type_t<T>;
        uint y = blockIdx.x, z = blockIdx.y;
        output_filter += (z * shape.y + y) * pitch_output_filter;

        float3_t distance_sqd;
        distance_sqd.z = getDistanceSquared_(shape.z, half.z, z);
        distance_sqd.y = getDistanceSquared_(shape.y, half.y, y);

        float freq_cutoff_sqd = freq_cutoff * freq_cutoff;
        float frequency_sqd, filter;
        for (uint x = threadIdx.x; x < half.x; x += blockDim.x) {
            distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
            distance_sqd.x *= distance_sqd.x;
            frequency_sqd = Math::sum(distance_sqd); // from 0 to 0.25
            filter = getHardWindow_<PASS>(freq_cutoff_sqd, frequency_sqd);
            output_filter[x] = static_cast<real_t>(filter);
        }
    }

    template<typename T>
    __global__ void bandPassHard_(T* inputs, uint pitch_inputs, T* outputs, uint pitch_outputs,
                                  uint3_t shape, uint3_t half, float freq_cutoff_1, float freq_cutoff_2, uint batches) {
        using real_t = Noa::Traits::value_type_t<T>;
        uint y = blockIdx.x, z = blockIdx.y;
        inputs += (z * shape.y + y) * pitch_inputs;
        outputs += (z * shape.y + y) * pitch_outputs;

        uint elements_inputs = 0, elements_outputs = 0;
        if (batches) {
            uint rows = getRows(shape);
            elements_inputs = pitch_inputs * rows;
            elements_outputs = pitch_outputs * rows;
        }

        float3_t distance_sqd;
        distance_sqd.z = getDistanceSquared_(shape.z, half.z, z);
        distance_sqd.y = getDistanceSquared_(shape.y, half.y, y);

        float freq_cutoff_sqd_1 = freq_cutoff_1 * freq_cutoff_1;
        float freq_cutoff_sqd_2 = freq_cutoff_2 * freq_cutoff_2;
        float frequency_sqd, filter;
        for (uint x = threadIdx.x; x < half.x; x += blockDim.x) {
            distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
            distance_sqd.x *= distance_sqd.x;
            frequency_sqd = Math::sum(distance_sqd); // from 0 to 0.25
            filter = getHardWindow_<Type::HIGHPASS>(freq_cutoff_sqd_1, frequency_sqd);
            filter *= getHardWindow_<Type::LOWPASS>(freq_cutoff_sqd_2, frequency_sqd);
            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * elements_outputs + x] =
                        inputs[batch * elements_inputs + x] * static_cast<real_t>(filter);
        }
    }

    template<typename T>
    __global__ void bandPassHard_(T* output_filter, uint pitch_output_filter, uint3_t shape, uint3_t half,
                                  float freq_cutoff_1, float freq_cutoff_2) {
        using real_t = Noa::Traits::value_type_t<T>;
        uint y = blockIdx.x, z = blockIdx.y;
        output_filter += (z * shape.y + y) * pitch_output_filter;

        float3_t distance_sqd;
        distance_sqd.z = getDistanceSquared_(shape.z, half.z, z);
        distance_sqd.y = getDistanceSquared_(shape.y, half.y, y);

        float freq_cutoff_sqd_1 = freq_cutoff_1 * freq_cutoff_1;
        float freq_cutoff_sqd_2 = freq_cutoff_2 * freq_cutoff_2;
        float frequency_sqd, filter;
        for (uint x = threadIdx.x; x < half.x; x += blockDim.x) {
            distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
            distance_sqd.x *= distance_sqd.x;
            frequency_sqd = Math::sum(distance_sqd); // from 0 to 0.25
            filter = getHardWindow_<Type::HIGHPASS>(freq_cutoff_sqd_1, frequency_sqd);
            filter *= getHardWindow_<Type::LOWPASS>(freq_cutoff_sqd_2, frequency_sqd);
            output_filter[x] = static_cast<real_t>(filter);
        }
    }
}

namespace Noa::CUDA::Fourier {
    template<typename T>
    void lowpass(T* inputs, size_t pitch_inputs, T* outputs, size_t pitch_outputs, size3_t shape,
                 float freq_cutoff, float freq_width, uint batches, Stream& stream) {
        uint3_t tmp_shape(shape);
        uint3_t half(tmp_shape / 2U + 1U);

        uint threads = Math::min(128U, Math::nextMultipleOf(half.x, 32U));
        dim3 blocks(tmp_shape.y, tmp_shape.z);
        if (freq_width > 1e-8f) {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            singlePassSoft_<Type::LOWPASS>,
                            inputs, pitch_inputs, outputs, pitch_outputs, tmp_shape, half,
                            freq_cutoff, freq_width, batches);
        } else {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            singlePassHard_<Type::LOWPASS>,
                            inputs, pitch_inputs, outputs, pitch_outputs, tmp_shape, half,
                            freq_cutoff, batches);
        }
    }

    template<typename T>
    void lowpass(T* output_lowpass, size_t pitch_output_lowpass, size3_t shape,
                 float freq_cutoff, float freq_width, Stream& stream) {
        uint3_t tmp_shape(shape);
        uint3_t half(tmp_shape / 2U + 1U);

        uint threads = Math::min(128U, Math::nextMultipleOf(half.x, 32U));
        dim3 blocks(tmp_shape.y, tmp_shape.z);
        if (freq_width > 1e-8f) {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            singlePassSoft_<Type::LOWPASS>,
                            output_lowpass, pitch_output_lowpass, tmp_shape, half, freq_cutoff, freq_width);
        } else {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            singlePassHard_<Type::LOWPASS>,
                            output_lowpass, pitch_output_lowpass, tmp_shape, half, freq_cutoff);
        }
    }

    template<typename T>
    void highpass(T* inputs, size_t pitch_inputs, T* outputs, size_t pitch_outputs, size3_t shape,
                  float freq_cutoff, float freq_width, uint batches, Stream& stream) {
        uint3_t tmp_shape(shape);
        uint3_t half(tmp_shape / 2U + 1U);

        uint threads = Math::min(128U, Math::nextMultipleOf(half.x, 32U));
        dim3 blocks(tmp_shape.y, tmp_shape.z);
        if (freq_width > 1e-8f) {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            singlePassSoft_<Type::HIGHPASS>,
                            inputs, pitch_inputs, outputs, pitch_outputs, tmp_shape, half,
                            freq_cutoff, freq_width, batches);
        } else {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            singlePassHard_<Type::HIGHPASS>,
                            inputs, pitch_inputs, outputs, pitch_outputs, tmp_shape, half,
                            freq_cutoff, batches);
        }
    }

    template<typename T>
    void highpass(T* output_highpass, size_t pitch_output_highpass, size3_t shape,
                  float freq_cutoff, float freq_width, Stream& stream) {
        uint3_t tmp_shape(shape);
        uint3_t half(tmp_shape / 2U + 1U);

        uint threads = Math::min(128U, Math::nextMultipleOf(half.x, 32U));
        dim3 blocks(tmp_shape.y, tmp_shape.z);
        if (freq_width > 1e-8f) {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            singlePassSoft_<Type::HIGHPASS>,
                            output_highpass, pitch_output_highpass, tmp_shape, half,
                            freq_cutoff, freq_width);
        } else {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            singlePassHard_<Type::HIGHPASS>,
                            output_highpass, pitch_output_highpass, tmp_shape, half, freq_cutoff);
        }
    }

    template<typename T>
    void bandpass(T* inputs, size_t pitch_inputs, T* outputs, size_t pitch_outputs, size3_t shape,
                  float freq_cutoff_1, float freq_cutoff_2, float freq_width_1, float freq_width_2,
                  uint batches, Stream& stream) {
        uint3_t tmp_shape(shape);
        uint3_t half(tmp_shape / 2U + 1U);

        uint threads = Math::min(128U, Math::nextMultipleOf(half.x, 32U));
        dim3 blocks(tmp_shape.y, tmp_shape.z);
        if (freq_width_1 > 1e-8f || freq_width_2 > 1e-8f) {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            bandPassSoft_,
                            inputs, pitch_inputs, outputs, pitch_outputs, tmp_shape, half,
                            freq_cutoff_1, freq_cutoff_2, freq_width_1, freq_width_2, batches);
        } else {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            bandPassHard_,
                            inputs, pitch_inputs, outputs, pitch_outputs, tmp_shape, half,
                            freq_cutoff_1, freq_cutoff_2, batches);
        }
    }

    template<typename T>
    void bandpass(T* output_bandpass, size_t pitch_output_bandpass, size3_t shape,
                  float freq_cutoff_1, float freq_cutoff_2, float freq_width_1, float freq_width_2, Stream& stream) {
        uint3_t tmp_shape(shape);
        uint3_t half(tmp_shape / 2U + 1U);

        uint threads = Math::min(128U, Math::nextMultipleOf(half.x, 32U));
        dim3 blocks(tmp_shape.y, tmp_shape.z);
        if (freq_width_1 > 1e-8f || freq_width_2 > 1e-8f) {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            bandPassSoft_,
                            output_bandpass, pitch_output_bandpass, tmp_shape, half,
                            freq_cutoff_1, freq_cutoff_2, freq_width_1, freq_width_2);
        } else {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            bandPassHard_,
                            output_bandpass, pitch_output_bandpass, tmp_shape, half,
                            freq_cutoff_1, freq_cutoff_2);
        }
    }

    #define INSTANTIATE_FILTERS(REAL, CPLX)                                                                         \
    template void lowpass<CPLX>(CPLX*, size_t, CPLX*, size_t, size3_t, float, float, uint, Stream&);                \
    template void lowpass<REAL>(REAL*, size_t, REAL*, size_t, size3_t, float, float, uint, Stream&);                \
    template void lowpass<REAL>(REAL*, size_t, size3_t, float, float, Stream&);                                     \
    template void highpass<CPLX>(CPLX*, size_t, CPLX*, size_t, size3_t, float, float, uint, Stream&);               \
    template void highpass<REAL>(REAL*, size_t, REAL*, size_t, size3_t, float, float, uint, Stream&);               \
    template void highpass<REAL>(REAL*, size_t, size3_t, float, float, Stream&);                                    \
    template void bandpass<CPLX>(CPLX*, size_t, CPLX*, size_t, size3_t, float, float, float, float, uint, Stream&); \
    template void bandpass<REAL>(REAL*, size_t, REAL*, size_t, size3_t, float, float, float, float, uint, Stream&); \
    template void bandpass<REAL>(REAL*, size_t, size3_t, float, float, float, float, Stream&)

    INSTANTIATE_FILTERS(float, cfloat_t);
    INSTANTIATE_FILTERS(double, cdouble_t);
}
