#include "noa/cpu/fourier/Filters.h"
#include "noa/Profiler.h"

// Commons:
namespace {
    enum class Type { LOWPASS, HIGHPASS };

    float getDistanceSquared(size_t dimension, uint half, size_t idx) {
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
    float getSoftWindow(float freq_cutoff, float freq_width, float freq) {
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
    void singlePassSoft(T* inputs, T* outputs, size3_t shape, float freq_cutoff, float freq_width, uint batches) {
        using real_t = Noa::Traits::value_type_t<T>;
        size_t elements = getElementsFFT(shape);
        uint3_t half(shape / size_t(2) + size_t(1));

        float3_t distance_sqd;
        float frequency, filter;
        for (size_t z = 0; z < shape.z; ++z) {
            distance_sqd.z = getDistanceSquared(shape.z, half.z, z);
            for (size_t y = 0; y < shape.y; ++y) {
                distance_sqd.y = getDistanceSquared(shape.y, half.y, y);
                size_t offset = (z * shape.y + y) * half.x;
                for (size_t x = 0; x < half.x; ++x) {
                    distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
                    distance_sqd.x *= distance_sqd.x;
                    frequency = Math::sqrt(Math::sum(distance_sqd)); // from 0 to 0.5
                    filter = getSoftWindow<PASS>(freq_cutoff, freq_width, frequency);
                    for (uint batch = 0; batch < batches; ++batch)
                        outputs[batch * elements + offset + x] =
                                inputs[batch * elements + offset + x] * static_cast<real_t>(filter);
                }
            }
        }
    }

    template<typename T>
    void bandPassSoft(T* inputs, T* outputs, size3_t shape, float freq_cutoff_1, float freq_cutoff_2,
                      float freq_width_1, float freq_width_2, uint batches) {
        using real_t = Noa::Traits::value_type_t<T>;
        size_t elements = getElementsFFT(shape);
        uint3_t half(shape / size_t(2) + size_t(1));

        float3_t distance_sqd;
        float frequency, filter;
        for (size_t z = 0; z < shape.z; ++z) {
            distance_sqd.z = getDistanceSquared(shape.z, half.z, z);
            for (size_t y = 0; y < shape.y; ++y) {
                distance_sqd.y = getDistanceSquared(shape.y, half.y, y);
                size_t offset = (z * shape.y + y) * half.x;
                for (size_t x = 0; x < half.x; ++x) {
                    distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
                    distance_sqd.x *= distance_sqd.x;
                    frequency = Math::sqrt(Math::sum(distance_sqd)); // from 0 to 0.5
                    filter = getSoftWindow<Type::HIGHPASS>(freq_cutoff_1, freq_width_1, frequency);
                    filter *= getSoftWindow<Type::LOWPASS>(freq_cutoff_2, freq_width_2, frequency);
                    for (uint batch = 0; batch < batches; ++batch)
                        outputs[batch * elements + offset + x] =
                                inputs[batch * elements + offset + x] * static_cast<real_t>(filter);
                }
            }
        }
    }

    template<Type PASS, typename T>
    void singlePassSoft(T* output_filter, size3_t shape, float freq_cutoff, float freq_width) {
        using real_t = Noa::Traits::value_type_t<T>;
        uint3_t half(shape / size_t(2) + size_t(1));

        float3_t distance_sqd;
        float frequency, filter;
        for (size_t z = 0; z < shape.z; ++z) {
            distance_sqd.z = getDistanceSquared(shape.z, half.z, z);
            for (size_t y = 0; y < shape.y; ++y) {
                distance_sqd.y = getDistanceSquared(shape.y, half.y, y);
                size_t offset = (z * shape.y + y) * half.x;
                for (size_t x = 0; x < half.x; ++x) {
                    distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
                    distance_sqd.x *= distance_sqd.x;
                    frequency = Math::sqrt(Math::sum(distance_sqd));
                    filter = getSoftWindow<PASS>(freq_cutoff, freq_width, frequency);
                    output_filter[offset + x] = static_cast<real_t>(filter);
                }
            }
        }
    }

    template<typename T>
    void bandPassSoft(T* output_filter, size3_t shape, float freq_cutoff_1, float freq_cutoff_2,
                      float freq_width_1, float freq_width_2) {
        using real_t = Noa::Traits::value_type_t<T>;
        uint3_t half(shape / size_t(2) + size_t(1));

        float3_t distance_sqd;
        float frequency, filter;
        for (size_t z = 0; z < shape.z; ++z) {
            distance_sqd.z = getDistanceSquared(shape.z, half.z, z);
            for (size_t y = 0; y < shape.y; ++y) {
                distance_sqd.y = getDistanceSquared(shape.y, half.y, y);
                size_t offset = (z * shape.y + y) * half.x;
                for (size_t x = 0; x < half.x; ++x) {
                    distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
                    distance_sqd.x *= distance_sqd.x;
                    frequency = Math::sqrt(Math::sum(distance_sqd));
                    filter = getSoftWindow<Type::HIGHPASS>(freq_cutoff_1, freq_width_1, frequency);
                    filter *= getSoftWindow<Type::LOWPASS>(freq_cutoff_2, freq_width_2, frequency);
                    output_filter[offset + x] = static_cast<real_t>(filter);
                }
            }
        }
    }
}

// Hard edges:
namespace {
    using namespace Noa;

    template<Type PASS>
    float getHardWindow(float freq_cutoff_sqd, float freq_sqd) {
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
    void singlePassHard(T* inputs, T* outputs, size3_t shape, float freq_cutoff, uint batches) {
        using real_t = Noa::Traits::value_type_t<T>;
        size_t elements = getElementsFFT(shape);
        uint3_t half(shape / size_t(2) + size_t(1));

        float3_t distance_sqd;
        float freq_cutoff_sqd = freq_cutoff * freq_cutoff;
        float filter;
        for (size_t z = 0; z < shape.z; ++z) {
            distance_sqd.z = getDistanceSquared(shape.z, half.z, z);
            for (size_t y = 0; y < shape.y; ++y) {
                distance_sqd.y = getDistanceSquared(shape.y, half.y, y);
                size_t offset = (z * shape.y + y) * half.x;
                for (size_t x = 0; x < half.x; ++x) {
                    distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
                    distance_sqd.x *= distance_sqd.x;
                    filter = getHardWindow<PASS>(freq_cutoff_sqd, Math::sum(distance_sqd));
                    for (uint batch = 0; batch < batches; ++batch)
                        outputs[batch * elements + offset + x] =
                                inputs[batch * elements + offset + x] * static_cast<real_t>(filter);
                }
            }
        }
    }

    template<typename T>
    void bandPassHard(T* inputs, T* outputs, size3_t shape, float freq_cutoff_1, float freq_cutoff_2, uint batches) {
        using real_t = Noa::Traits::value_type_t<T>;
        size_t elements = getElementsFFT(shape);
        uint3_t half(shape / size_t(2) + size_t(1));

        float3_t distance_sqd;
        float freq_cutoff_sqd_1 = freq_cutoff_1 * freq_cutoff_1;
        float freq_cutoff_sqd_2 = freq_cutoff_2 * freq_cutoff_2;
        float freq_sqd, filter;
        for (size_t z = 0; z < shape.z; ++z) {
            distance_sqd.z = getDistanceSquared(shape.z, half.z, z);
            for (size_t y = 0; y < shape.y; ++y) {
                distance_sqd.y = getDistanceSquared(shape.y, half.y, y);
                size_t offset = (z * shape.y + y) * half.x;
                for (size_t x = 0; x < half.x; ++x) {
                    distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
                    distance_sqd.x *= distance_sqd.x;
                    freq_sqd = Math::sum(distance_sqd);
                    filter = getHardWindow<Type::HIGHPASS>(freq_cutoff_sqd_1, freq_sqd);
                    filter *= getHardWindow<Type::LOWPASS>(freq_cutoff_sqd_2, freq_sqd);
                    for (uint batch = 0; batch < batches; ++batch)
                        outputs[batch * elements + offset + x] =
                                inputs[batch * elements + offset + x] * static_cast<real_t>(filter);
                }
            }
        }
    }

    template<Type PASS, typename T>
    void singlePassHard(T* output_filter, size3_t shape, float freq_cutoff) {
        using real_t = Noa::Traits::value_type_t<T>;
        uint3_t half(shape / size_t(2) + size_t(1));

        float3_t distance_sqd;
        float freq_cutoff_sqd = freq_cutoff * freq_cutoff;
        float filter;
        for (size_t z = 0; z < shape.z; ++z) {
            distance_sqd.z = getDistanceSquared(shape.z, half.z, z);
            for (size_t y = 0; y < shape.y; ++y) {
                distance_sqd.y = getDistanceSquared(shape.y, half.y, y);
                size_t offset = (z * shape.y + y) * half.x;
                for (size_t x = 0; x < half.x; ++x) {
                    distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
                    distance_sqd.x *= distance_sqd.x;
                    filter = getHardWindow<PASS>(freq_cutoff_sqd, Math::sum(distance_sqd));
                    output_filter[offset + x] = static_cast<real_t>(filter);
                }
            }
        }
    }

    template<typename T>
    void bandPassHard(T* output_filter, size3_t shape, float freq_cutoff_1, float freq_cutoff_2) {
        using real_t = Noa::Traits::value_type_t<T>;
        uint3_t half(shape / size_t(2) + size_t(1));

        float3_t distance_sqd;
        float freq_cutoff_sqd_1 = freq_cutoff_1 * freq_cutoff_1;
        float freq_cutoff_sqd_2 = freq_cutoff_2 * freq_cutoff_2;
        float freq_sqd, filter;
        for (size_t z = 0; z < shape.z; ++z) {
            distance_sqd.z = getDistanceSquared(shape.z, half.z, z);
            for (size_t y = 0; y < shape.y; ++y) {
                distance_sqd.y = getDistanceSquared(shape.y, half.y, y);
                size_t offset = (z * shape.y + y) * half.x;
                for (size_t x = 0; x < half.x; ++x) {
                    distance_sqd.x = static_cast<float>(x) / static_cast<float>(shape.x);
                    distance_sqd.x *= distance_sqd.x;
                    freq_sqd = Math::sum(distance_sqd);
                    filter = getHardWindow<Type::HIGHPASS>(freq_cutoff_sqd_1, freq_sqd);
                    filter *= getHardWindow<Type::LOWPASS>(freq_cutoff_sqd_2, freq_sqd);
                    output_filter[offset + x] = static_cast<real_t>(filter);
                }
            }
        }
    }
}

// Definitions & Instantiations:
namespace Noa::Fourier {
    template<typename T>
    void lowpass(T* inputs, T* outputs, size3_t shape, float freq_cutoff, float freq_width, uint batches) {
        NOA_PROFILE_FUNCTION();
        if (freq_width > 1e-8f)
            singlePassSoft<Type::LOWPASS>(inputs, outputs, shape, freq_cutoff, freq_width, batches);
        else
            singlePassHard<Type::LOWPASS>(inputs, outputs, shape, freq_cutoff, batches);
    }

    template<typename T>
    void lowpass(T* output_lowpass, size3_t shape, float freq_cutoff, float freq_width) {
        NOA_PROFILE_FUNCTION();
        if (freq_width > 1e-8f)
            singlePassSoft<Type::LOWPASS>(output_lowpass, shape, freq_cutoff, freq_width);
        else
            singlePassHard<Type::LOWPASS>(output_lowpass, shape, freq_cutoff);
    }

    template<typename T>
    void highpass(T* inputs, T* outputs, size3_t shape, float freq_cutoff, float freq_width, uint batches) {
        NOA_PROFILE_FUNCTION();
        if (freq_width > 1e-8f)
            singlePassSoft<Type::HIGHPASS>(inputs, outputs, shape, freq_cutoff, freq_width, batches);
        else
            singlePassHard<Type::HIGHPASS>(inputs, outputs, shape, freq_cutoff, batches);
    }

    template<typename T>
    void highpass(T* output_highpass, size3_t shape, float freq_cutoff, float freq_width) {
        NOA_PROFILE_FUNCTION();
        if (freq_width > 1e-8f)
            singlePassSoft<Type::HIGHPASS>(output_highpass, shape, freq_cutoff, freq_width);
        else
            singlePassHard<Type::HIGHPASS>(output_highpass, shape, freq_cutoff);
    }

    template<typename T>
    void bandpass(T* inputs, T* outputs, size3_t shape, float freq_cutoff_1, float freq_cutoff_2,
                  float freq_width_1, float freq_width_2, uint batches) {
        NOA_PROFILE_FUNCTION();
        if (freq_width_1 > 1e-8f || freq_width_2 > 1e-8f)
            bandPassSoft(inputs, outputs, shape, freq_cutoff_1, freq_cutoff_2, freq_width_1, freq_width_2, batches);
        else
            bandPassHard(inputs, outputs, shape, freq_cutoff_1, freq_cutoff_2, batches);
    }

    template<typename T>
    void bandpass(T* output_bandpass, size3_t shape, float freq_cutoff_1, float freq_cutoff_2,
                  float freq_width_1, float freq_width_2) {
        NOA_PROFILE_FUNCTION();
        if (freq_width_1 > 1e-8f || freq_width_2 > 1e-8f)
            bandPassSoft(output_bandpass, shape, freq_cutoff_1, freq_cutoff_2, freq_width_1, freq_width_2);
        else
            bandPassHard(output_bandpass, shape, freq_cutoff_1, freq_cutoff_2);
    }

    #define INSTANTIATE_FILTERS(REAL, COMPLEX)                                                      \
    template void lowpass<COMPLEX>(COMPLEX*, COMPLEX*, size3_t, float, float, uint);                \
    template void lowpass<REAL>(REAL*, REAL*, size3_t, float, float, uint);                         \
    template void lowpass<REAL>(REAL*, size3_t, float, float);                                      \
    template void highpass<COMPLEX>(COMPLEX*, COMPLEX*, size3_t, float, float, uint);               \
    template void highpass<REAL>(REAL*, REAL*, size3_t, float, float, uint);                        \
    template void highpass<REAL>(REAL*, size3_t, float, float);                                     \
    template void bandpass<COMPLEX>(COMPLEX*, COMPLEX*, size3_t, float, float, float, float, uint); \
    template void bandpass<REAL>(REAL*, REAL*, size3_t, float, float, float, float, uint);          \
    template void bandpass<REAL>(REAL*, size3_t, float, float, float, float)

    INSTANTIATE_FILTERS(float, cfloat_t);
    INSTANTIATE_FILTERS(double, cdouble_t);
}
