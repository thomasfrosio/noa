#include "noa/common/Assert.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"

#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/transform/fft/Shift.h"

namespace {
    using namespace ::noa;

    template<bool IS_CENTERED>
    int64_t getFrequency_(int64_t idx, int64_t dim) {
        if constexpr(IS_CENTERED)
            return idx - dim / 2;
        else
            return idx < (dim + 1) / 2 ? idx : idx - dim;
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED>
    inline size_t getOutputIndex_(size_t idx, [[maybe_unused]] size_t dim) {
        if constexpr (IS_SRC_CENTERED == IS_DST_CENTERED)
            return idx;
        else if constexpr (IS_SRC_CENTERED)
            return noa::math::iFFTShift(idx, dim);
        else
            return noa::math::FFTShift(idx, dim);
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T, typename U>
    void shift2D_(const T* inputs, size2_t input_pitch, T* outputs, size2_t output_pitch, size2_t shape,
                  U shifts, size_t batches, size_t threads) {
        using real_t = traits::value_type_t<T>;

        const auto shape_y = static_cast<int64_t>(shape.y);
        const float2_t f_shape(shape);
        const float2_t pre_shift = math::Constants<float>::PI2 / f_shape;
        if constexpr (!std::is_pointer_v<U>)
            shifts *= pre_shift;

        const size_t iffset = elements(input_pitch);
        const size_t offset = elements(output_pitch);

        float factor;
        #pragma omp parallel for default(none) num_threads(threads) collapse(3) \
        shared(inputs, input_pitch, outputs, output_pitch, shape, shifts, batches, shape_y, pre_shift, iffset, offset) \
        private(factor)

        for (size_t batch = 0; batch < batches; ++batch) {
            for (size_t i_y = 0; i_y < shape.y; ++i_y) {
                for (size_t x = 0; x < shape.x / 2 + 1; ++x) { // x == u

                    const float2_t freq(x, getFrequency_<IS_SRC_CENTERED>(static_cast<int64_t>(i_y), shape_y));
                    if constexpr (std::is_pointer_v<U>)
                        factor = math::sum(freq * shifts[batch] * pre_shift);
                    else
                        factor = math::sum(freq * shifts);

                    T phase_shift;
                    math::sincos(static_cast<real_t>(-factor), &phase_shift.imag, &phase_shift.real);

                    const size_t o_y = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(i_y, shape.y);
                    outputs[offset * batch + o_y * output_pitch.x + x] =
                            inputs[iffset * batch + i_y * input_pitch.x + x] * phase_shift;
                }
            }
        }
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T, typename U>
    void shift3D_(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                  U shifts, size_t batches, size_t threads) {
        using real_t = traits::value_type_t<T>;

        const auto shape_z = static_cast<int64_t>(shape.z);
        const auto shape_y = static_cast<int64_t>(shape.y);
        const float3_t f_shape(shape);
        const float3_t pre_shift = math::Constants<float>::PI2 / f_shape;
        if constexpr (!std::is_pointer_v<U>)
            shifts *= pre_shift;

        const size_t iffset = elements(input_pitch);
        const size_t offset = elements(output_pitch);

        float factor;
        #pragma omp parallel for default(none) num_threads(threads) collapse(3) \
        shared(inputs, input_pitch, outputs, output_pitch, shape, shifts, batches, shape_y, shape_z, pre_shift, iffset, offset) \
        private(factor)

        for (size_t batch = 0; batch < batches; ++batch) {
            for (size_t i_z = 0; i_z < shape.z; ++i_z) {
                for (size_t i_y = 0; i_y < shape.y; ++i_y) {
                    for (size_t x = 0; x < shape.x / 2 + 1; ++x) { // x == u

                        const float3_t freq(x,
                                            getFrequency_<IS_SRC_CENTERED>(static_cast<int64_t>(i_y), shape_y),
                                            getFrequency_<IS_SRC_CENTERED>(static_cast<int64_t>(i_z), shape_z));
                        if constexpr (std::is_pointer_v<U>)
                            factor = math::sum(freq * shifts[batch] * pre_shift);
                        else
                            factor = math::sum(freq * shifts);

                        T phase_shift;
                        math::sincos(static_cast<real_t>(-factor), &phase_shift.imag, &phase_shift.real);

                        const size_t o_z = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(i_z, shape.z);
                        const size_t o_y = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(i_y, shape.y);
                        outputs[offset * batch + index(x, o_y, o_z, output_pitch)] =
                                inputs[iffset * batch + index(x, i_y, i_z, input_pitch)] * phase_shift;
                    }
                }
            }
        }
    }
}

namespace noa::cpu::transform::fft {
    using Layout = ::noa::fft::Layout;

    template<Remap REMAP, typename T>
    void shift2D(const T* inputs, size2_t input_pitch, T* outputs, size2_t output_pitch, size2_t shape,
                 const float2_t* shifts, size_t batches, Stream& stream) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        constexpr bool NO_REMAP = IS_SRC_CENTERED == IS_DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL) {
            static_assert(traits::always_false_v<T>);
        }
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs || NO_REMAP);

        if (NO_REMAP && !shifts)
            return memory::copy(inputs, {input_pitch, 1}, outputs, {output_pitch, 1},
                                {shape.x / 2 + 1, shape.y, 1}, batches, stream);

        const size_t threads = stream.threads();
        stream.enqueue(shift2D_<IS_SRC_CENTERED, IS_DST_CENTERED, T, const float2_t*>,
                       inputs, input_pitch, outputs, output_pitch, shape, shifts, batches, threads);
    }

    template<Remap REMAP, typename T>
    void shift2D(const T* inputs, size2_t input_pitch, T* outputs, size2_t output_pitch, size2_t shape,
                 float2_t shift, size_t batches, Stream& stream) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        constexpr bool NO_REMAP = IS_SRC_CENTERED == IS_DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL) {
            static_assert(traits::always_false_v<T>);
        }
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs || NO_REMAP);

        if (NO_REMAP && all(shift == 0))
            return memory::copy(inputs, {input_pitch, 1}, outputs, {output_pitch, 1},
                                {shape.x / 2 + 1, shape.y, 1}, batches, stream);

        const size_t threads = stream.threads();
        stream.enqueue(shift2D_<IS_SRC_CENTERED, IS_DST_CENTERED, T, float2_t>,
                       inputs, input_pitch, outputs, output_pitch, shape, shift, batches, threads);
    }

    template<Remap REMAP, typename T>
    void shift3D(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                 const float3_t* shifts, size_t batches, Stream& stream) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        constexpr bool NO_REMAP = IS_SRC_CENTERED == IS_DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL) {
            static_assert(noa::traits::always_false_v<T>);
        }
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs || NO_REMAP);

        if (NO_REMAP && !shifts)
            return memory::copy(inputs, input_pitch, outputs, output_pitch, shapeFFT(shape), batches, stream);

        const size_t threads = stream.threads();
        stream.enqueue(shift3D_<IS_SRC_CENTERED, IS_DST_CENTERED, T, const float3_t*>,
                       inputs, input_pitch, outputs, output_pitch, shape, shifts, batches, threads);
    }

    template<Remap REMAP, typename T>
    void shift3D(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                 float3_t shift, size_t batches, Stream& stream) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        constexpr bool NO_REMAP = IS_SRC_CENTERED == IS_DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL) {
            static_assert(traits::always_false_v<T>);
        }
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs || NO_REMAP);

        if (NO_REMAP && all(shift == 0))
            return memory::copy(inputs, input_pitch, outputs, output_pitch, shapeFFT(shape), batches, stream);

        const size_t threads = stream.threads();
        stream.enqueue(shift3D_<IS_SRC_CENTERED, IS_DST_CENTERED, T, float3_t>,
                       inputs, input_pitch, outputs, output_pitch, shape, shift, batches, threads);
    }

    #define NOA_INSTANTIATE_SHIFT_(T)                                                     \
    template void shift2D<Remap::H2H>(const T*, size2_t, T*, size2_t, size2_t, const float2_t*, size_t, Stream&);    \
    template void shift2D<Remap::H2H>(const T*, size2_t, T*, size2_t, size2_t, float2_t, size_t, Stream&);           \
    template void shift2D<Remap::H2HC>(const T*, size2_t, T*, size2_t, size2_t, const float2_t*, size_t, Stream&);   \
    template void shift2D<Remap::H2HC>(const T*, size2_t, T*, size2_t, size2_t, float2_t, size_t, Stream&);          \
    template void shift2D<Remap::HC2H>(const T*, size2_t, T*, size2_t, size2_t, const float2_t*, size_t, Stream&);   \
    template void shift2D<Remap::HC2H>(const T*, size2_t, T*, size2_t, size2_t, float2_t, size_t, Stream&);          \
    template void shift2D<Remap::HC2HC>(const T*, size2_t, T*, size2_t, size2_t, const float2_t*, size_t, Stream&);  \
    template void shift2D<Remap::HC2HC>(const T*, size2_t, T*, size2_t, size2_t, float2_t, size_t, Stream&);         \
    template void shift3D<Remap::H2H>(const T*, size3_t, T*, size3_t, size3_t, const float3_t*, size_t, Stream&);    \
    template void shift3D<Remap::H2H>(const T*, size3_t, T*, size3_t, size3_t, float3_t, size_t, Stream&);           \
    template void shift3D<Remap::H2HC>(const T*, size3_t, T*, size3_t, size3_t, const float3_t*, size_t, Stream&);   \
    template void shift3D<Remap::H2HC>(const T*, size3_t, T*, size3_t, size3_t, float3_t, size_t, Stream&);          \
    template void shift3D<Remap::HC2H>(const T*, size3_t, T*, size3_t, size3_t, const float3_t*, size_t, Stream&);   \
    template void shift3D<Remap::HC2H>(const T*, size3_t, T*, size3_t, size3_t, float3_t, size_t, Stream&);          \
    template void shift3D<Remap::HC2HC>(const T*, size3_t, T*, size3_t, size3_t, const float3_t*, size_t, Stream&);  \
    template void shift3D<Remap::HC2HC>(const T*, size3_t, T*, size3_t, size3_t, float3_t, size_t, Stream&)

    NOA_INSTANTIATE_SHIFT_(cfloat_t);
    NOA_INSTANTIATE_SHIFT_(cdouble_t);
}
