#include "noa/common/Assert.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/transform/fft/Shift.h"

namespace {
    using namespace ::noa;

    template<bool IS_CENTERED>
    int64_t getFrequency_(int64_t idx, int64_t proj_dim) {
        if constexpr(IS_CENTERED)
            return idx - proj_dim / 2;
        else
            return idx < (proj_dim + 1) / 2 ? idx : idx - proj_dim;
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED>
    inline size_t getOutputIndex_(size_t i_idx, [[maybe_unused]] size_t dim) {
        if constexpr (IS_SRC_CENTERED == IS_DST_CENTERED)
            return i_idx;
        else if constexpr (IS_SRC_CENTERED)
            return noa::math::iFFTShift(i_idx, dim);
        else
            return noa::math::FFTShift(i_idx, dim);
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T>
    void shift2D_(const T* input, T* output, size2_t shape, float2_t shift) {
        using real_t = traits::value_type_t<T>;

        const auto shape_y = static_cast<int64_t>(shape.y);
        const size_t half_x = shape.x / 2 + 1;
        const float2_t f_shape(shape);
        shift *= math::Constants<float>::PI2 / f_shape;

        T phase_shift;
        for (size_t i_y = 0; i_y < shape.y; ++i_y) {
            const size_t o_y = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(i_y, shape.y);
            const int64_t v = getFrequency_<IS_SRC_CENTERED>(static_cast<int64_t>(i_y), shape_y);
            const float tmp = shift.y * static_cast<float>(v);
            for (size_t x = 0; x < half_x; ++x) { // x == u
                const float factor = tmp + shift.x * static_cast<float>(x);
                math::sincos(static_cast<real_t>(-factor), &phase_shift.imag, &phase_shift.real);
                output[o_y * half_x + x] = input[i_y * half_x + x] * phase_shift;
            }
        }
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T>
    void shift3D_(const T* input, T* output, size3_t shape, float3_t shift) {
        using real_t = traits::value_type_t<T>;

        const auto shape_z = static_cast<int64_t>(shape.z);
        const auto shape_y = static_cast<int64_t>(shape.y);
        const size_t half_x = shape.x / 2 + 1;
        const float3_t f_shape(shape);
        shift *= math::Constants<float>::PI2 / f_shape;

        T phase_shift;
        for (size_t i_z = 0; i_z < shape.z; ++i_z) {
            const size_t o_z = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(i_z, shape.z);
            const int64_t w = getFrequency_<IS_SRC_CENTERED>(static_cast<int64_t>(i_z), shape_z);
            const float tmp_z = shift.z * static_cast<float>(w);
            for (size_t i_y = 0; i_y < shape.y; ++i_y) {
                const size_t o_y = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(i_y, shape.y);
                const int64_t v = getFrequency_<IS_SRC_CENTERED>(static_cast<int64_t>(i_y), shape_y);
                const float tmp_yz = tmp_z + shift.y * static_cast<float>(v);
                for (size_t x = 0; x < half_x; ++x) { // x == u
                    const float factor = tmp_yz + shift.x * static_cast<float>(x);
                    math::sincos(static_cast<real_t>(-factor), &phase_shift.imag, &phase_shift.real);
                    output[(o_z * shape.y + o_y) * half_x + x] =
                            input[(i_z * shape.y + i_y) * half_x + x] * phase_shift;
                }
            }
        }
    }
}

namespace noa::cpu::transform::fft {
    using Layout = ::noa::fft::Layout;

    template<Remap REMAP, typename T>
    void shift2D(const T* inputs, T* outputs, size2_t shape, const float2_t* shifts, size_t batches) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL) {
            static_assert(traits::always_false_v<T>);
        }
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs || IS_SRC_CENTERED == IS_DST_CENTERED);

        size_t elements = noa::elementsFFT(shape);
        for (size_t batch = 0; batch < batches; ++batch) {
            shift2D_<IS_SRC_CENTERED, IS_DST_CENTERED>(
                    inputs + elements * batch, outputs + elements * batch, shape, shifts[batch]);
        }
    }

    template<Remap REMAP, typename T>
    void shift2D(const T* inputs, T* outputs, size2_t shape, float2_t shift, size_t batches) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL) {
            static_assert(traits::always_false_v<T>);
        }
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs || IS_SRC_CENTERED == IS_DST_CENTERED);

        size_t elements = noa::elementsFFT(shape);
        for (size_t batch = 0; batch < batches; ++batch) {
            shift2D_<IS_SRC_CENTERED, IS_DST_CENTERED>(
                    inputs + elements * batch, outputs + elements * batch, shape, shift);
        }
    }

    template<Remap REMAP, typename T>
    void shift3D(const T* inputs, T* outputs, size3_t shape, const float3_t* shifts, size_t batches) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL) {
            static_assert(noa::traits::always_false_v<T>);
        }
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs || IS_SRC_CENTERED == IS_DST_CENTERED);

        size_t elements = noa::elementsFFT(shape);
        for (size_t batch = 0; batch < batches; ++batch) {
            shift3D_<IS_SRC_CENTERED, IS_DST_CENTERED>(
                    inputs + elements * batch, outputs + elements * batch, shape, shifts[batch]);
        }
    }

    template<Remap REMAP, typename T>
    void shift3D(const T* inputs, T* outputs, size3_t shape, float3_t shift, size_t batches) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL) {
            static_assert(traits::always_false_v<T>);
        }
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs || IS_SRC_CENTERED == IS_DST_CENTERED);

        size_t elements = noa::elementsFFT(shape);
        for (size_t batch = 0; batch < batches; ++batch) {
            shift3D_<IS_SRC_CENTERED, IS_DST_CENTERED>(
                    inputs + elements * batch, outputs + elements * batch, shape, shift);
        }
    }

    #define NOA_INSTANTIATE_SHIFT_(T)                                                     \
    template void shift2D<Remap::H2H>(const T*, T*, size2_t, const float2_t*, size_t);    \
    template void shift2D<Remap::H2H>(const T*, T*, size2_t, float2_t, size_t);           \
    template void shift2D<Remap::H2HC>(const T*, T*, size2_t, const float2_t*, size_t);   \
    template void shift2D<Remap::H2HC>(const T*, T*, size2_t, float2_t, size_t);          \
    template void shift2D<Remap::HC2H>(const T*, T*, size2_t, const float2_t*, size_t);   \
    template void shift2D<Remap::HC2H>(const T*, T*, size2_t, float2_t, size_t);          \
    template void shift2D<Remap::HC2HC>(const T*, T*, size2_t, const float2_t*, size_t);  \
    template void shift2D<Remap::HC2HC>(const T*, T*, size2_t, float2_t, size_t);         \
    template void shift3D<Remap::H2H>(const T*, T*, size3_t, const float3_t*, size_t);    \
    template void shift3D<Remap::H2H>(const T*, T*, size3_t, float3_t, size_t);           \
    template void shift3D<Remap::H2HC>(const T*, T*, size3_t, const float3_t*, size_t);   \
    template void shift3D<Remap::H2HC>(const T*, T*, size3_t, float3_t, size_t);          \
    template void shift3D<Remap::HC2H>(const T*, T*, size3_t, const float3_t*, size_t);   \
    template void shift3D<Remap::HC2H>(const T*, T*, size3_t, float3_t, size_t);          \
    template void shift3D<Remap::HC2HC>(const T*, T*, size3_t, const float3_t*, size_t);  \
    template void shift3D<Remap::HC2HC>(const T*, T*, size3_t, float3_t, size_t)

    NOA_INSTANTIATE_SHIFT_(cfloat_t);
    NOA_INSTANTIATE_SHIFT_(cdouble_t);
}
