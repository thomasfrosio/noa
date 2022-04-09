#include "noa/common/Assert.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"

#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/Set.h"
#include "noa/cpu/fft/Remap.h"
#include "noa/cpu/geometry/fft/Shift.h"

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
    inline size_t getOutputIndex_(int64_t idx, [[maybe_unused]] int64_t dim) {
        if constexpr (IS_SRC_CENTERED == IS_DST_CENTERED)
            return static_cast<size_t>(idx);
        else if constexpr (IS_SRC_CENTERED)
            return static_cast<size_t>(noa::math::iFFTShift(idx, dim));
        else
            return static_cast<size_t>(noa::math::FFTShift(idx, dim));
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T, typename U>
    void shift2D_(const T* input, size3_t input_stride, T* output, size3_t output_stride, size3_t shape,
                  U shifts, float cutoff, size_t threads) {
        using real_t = traits::value_type_t<T>;

        const size_t batches = shape[0];
        const long2_t l_shape{shape.get() + 1};
        const float2_t pre_shift = math::Constants<float>::PI2 / float2_t{l_shape};
        if constexpr (!std::is_pointer_v<U>)
            shifts *= pre_shift;

        cutoff *= cutoff;
        const float2_t f_shape{l_shape / 2 * 2 + long2_t{l_shape == 1}}; // if odd, n-1

        #pragma omp parallel for default(none) num_threads(threads) collapse(3) \
        shared(input, input_stride, output, output_stride, shifts, cutoff, pre_shift, l_shape, f_shape, batches)

        for (size_t batch = 0; batch < batches; ++batch) {
            for (int64_t i_y = 0; i_y < l_shape[0]; ++i_y) {
                for (int64_t x = 0; x < l_shape[1] / 2 + 1; ++x) { // x == u

                    const float2_t freq{getFrequency_<IS_SRC_CENTERED>(i_y, l_shape[0]),
                                        getFrequency_<false>(x, l_shape[1])};

                    T phase_shift{1, 0};
                    const float2_t norm_freq = freq / f_shape;
                    if (math::dot(norm_freq, norm_freq) <= cutoff) {
                        float factor;
                        if constexpr (std::is_pointer_v<U>)
                            factor = math::sum(freq * shifts[batch] * pre_shift);
                        else
                            factor = math::sum(freq * shifts);
                        math::sincos(static_cast<real_t>(-factor), &phase_shift.imag, &phase_shift.real);
                    }

                    const size_t o_y = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(i_y, l_shape[0]);
                    output[indexing::at(batch, o_y, x, output_stride)] =
                            input ? input[indexing::at(batch, i_y, x, input_stride)] * phase_shift : phase_shift;
                }
            }
        }
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T, typename U>
    void shift3D_(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape,
                  U shifts, float cutoff, size_t threads) {
        using real_t = traits::value_type_t<T>;

        const size_t batches = shape[0];
        const long3_t l_shape{shape.get() + 1};
        const float3_t pre_shift = math::Constants<float>::PI2 / float3_t{l_shape};
        if constexpr (!std::is_pointer_v<U>)
            shifts *= pre_shift;

        cutoff *= cutoff;
        const float3_t f_shape{l_shape / 2 * 2 + long3_t{l_shape == 1}}; // if odd, n-1

        #pragma omp parallel for default(none) num_threads(threads) collapse(4) \
        shared(input, input_stride, output, output_stride, shifts, cutoff, l_shape, f_shape, pre_shift, batches)

        for (size_t batch = 0; batch < batches; ++batch) {
            for (int64_t i_z = 0; i_z < l_shape[0]; ++i_z) {
                for (int64_t i_y = 0; i_y < l_shape[1]; ++i_y) {
                    for (int64_t x = 0; x < l_shape[2] / 2 + 1; ++x) { // x == u

                        const float3_t freq{getFrequency_<IS_SRC_CENTERED>(i_z, l_shape[0]),
                                            getFrequency_<IS_SRC_CENTERED>(i_y, l_shape[1]),
                                            getFrequency_<false>(x, l_shape[2])};
                        T phase_shift{1, 0};
                        const float3_t norm_freq = freq / f_shape;
                        if (math::dot(norm_freq, norm_freq) <= cutoff) {
                            float factor;
                            if constexpr (std::is_pointer_v<U>)
                                factor = math::sum(freq * shifts[batch] * pre_shift);
                            else
                                factor = math::sum(freq * shifts);
                            math::sincos(static_cast<real_t>(-factor), &phase_shift.imag, &phase_shift.real);
                        }

                        const size_t o_z = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(i_z, l_shape[0]);
                        const size_t o_y = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(i_y, l_shape[1]);
                        output[indexing::at(batch, o_z, o_y, x, output_stride)] =
                                input ?
                                input[indexing::at(batch, i_z, i_y, x, input_stride)] * phase_shift : phase_shift;
                    }
                }
            }
        }
    }

    template<fft::Remap REMAP, typename T>
    void noShift_(const shared_t<T[]>& input, size4_t input_stride,
                  const shared_t<T[]>& output, size4_t output_stride,
                  size4_t shape, cpu::Stream& stream) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool NO_REMAP = (REMAP_ & fft::Layout::SRC_CENTERED) == (REMAP_ & fft::Layout::DST_CENTERED);

        if (!input) {
            cpu::memory::set(output, output_stride, shape.fft(), T(1, 0), stream);
        } else {
            if constexpr (NO_REMAP)
                cpu::memory::copy(input, input_stride, output, output_stride, shape.fft(), stream);
            else
                cpu::fft::remap(REMAP, input, input_stride, output, output_stride, shape, stream);
        }
    }
}

namespace noa::cpu::geometry::fft {
    using Layout = ::noa::fft::Layout;

    template<Remap REMAP, typename T>
    void shift2D(const shared_t<T[]>& input, size4_t input_stride,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                 const shared_t<float2_t[]>& shifts, float cutoff, Stream& stream) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL)
            static_assert(traits::always_false_v<T>);

        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output || IS_SRC_CENTERED == IS_DST_CENTERED);
        NOA_ASSERT(shape[1] == 1);

        const size3_t shape_2d{shape[0], shape[2], shape[3]};
        const size3_t i_stride{input_stride[0], input_stride[2], input_stride[3]};
        const size3_t o_stride{output_stride[0], output_stride[2], output_stride[3]};
        const size_t threads = stream.threads();
        stream.enqueue([=]() {
            shift2D_<IS_SRC_CENTERED, IS_DST_CENTERED, T, const float2_t*>(
                    input.get(), i_stride, output.get(), o_stride, shape_2d, shifts.get(), cutoff, threads);
        });
    }

    template<Remap REMAP, typename T>
    void shift2D(const shared_t<T[]>& input, size4_t input_stride,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                 float2_t shift, float cutoff, Stream& stream) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL)
                static_assert(traits::always_false_v<T>);

        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output || IS_SRC_CENTERED == IS_DST_CENTERED);
        NOA_ASSERT(shape[1] == 1);

        if (all(shift == 0))
            return noShift_<REMAP>(input, input_stride, output, output_stride, shape, stream);

        const size3_t shape_2d{shape[0], shape[2], shape[3]};
        const size3_t i_stride{input_stride[0], input_stride[2], input_stride[3]};
        const size3_t o_stride{output_stride[0], output_stride[2], output_stride[3]};
        const size_t threads = stream.threads();
        stream.enqueue([=]() {
            shift2D_<IS_SRC_CENTERED, IS_DST_CENTERED, T, float2_t>(
                    input.get(), i_stride, output.get(), o_stride, shape_2d, shift, cutoff, threads);
        });
    }

    template<Remap REMAP, typename T>
    void shift3D(const shared_t<T[]>& input, size4_t input_stride,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                 const shared_t<float3_t[]>& shifts, float cutoff, Stream& stream) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL)
                static_assert(traits::always_false_v<T>);

        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output || IS_SRC_CENTERED == IS_DST_CENTERED);

        const size_t threads = stream.threads();
        stream.enqueue([=]() {
            shift3D_<IS_SRC_CENTERED, IS_DST_CENTERED, T, const float3_t*>(
                    input.get(), input_stride, output.get(), output_stride, shape, shifts.get(), cutoff, threads);
        });
    }

    template<Remap REMAP, typename T>
    void shift3D(const shared_t<T[]>& input, size4_t input_stride,
                 const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                 float3_t shift, float cutoff, Stream& stream) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL)
                static_assert(traits::always_false_v<T>);

        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output || IS_SRC_CENTERED == IS_DST_CENTERED);

        if (all(shift == 0))
            return noShift_<REMAP>(input, input_stride, output, output_stride, shape, stream);

        const size_t threads = stream.threads();
        stream.enqueue([=]() {
            shift3D_<IS_SRC_CENTERED, IS_DST_CENTERED, T, float3_t>(
                    input.get(), input_stride, output.get(), output_stride, shape, shift, cutoff, threads);
        });
    }

    #define NOA_INSTANTIATE_SHIFT_(T)                                                                                                                           \
    template void shift2D<Remap::H2H>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<float2_t[]>&, float, Stream&);      \
    template void shift2D<Remap::H2H>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float2_t, float, Stream&);                         \
    template void shift2D<Remap::H2HC>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<float2_t[]>&, float, Stream&);     \
    template void shift2D<Remap::H2HC>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float2_t, float, Stream&);                        \
    template void shift2D<Remap::HC2H>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<float2_t[]>&, float, Stream&);     \
    template void shift2D<Remap::HC2H>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float2_t, float, Stream&);                        \
    template void shift2D<Remap::HC2HC>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<float2_t[]>&, float, Stream&);    \
    template void shift2D<Remap::HC2HC>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float2_t, float, Stream&);                       \
    template void shift3D<Remap::H2H>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<float3_t[]>&, float, Stream&);      \
    template void shift3D<Remap::H2H>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float3_t, float, Stream&);                         \
    template void shift3D<Remap::H2HC>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<float3_t[]>&, float, Stream&);     \
    template void shift3D<Remap::H2HC>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float3_t, float, Stream&);                        \
    template void shift3D<Remap::HC2H>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<float3_t[]>&, float, Stream&);     \
    template void shift3D<Remap::HC2H>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float3_t, float, Stream&);                        \
    template void shift3D<Remap::HC2HC>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<float3_t[]>&, float, Stream&);    \
    template void shift3D<Remap::HC2HC>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float3_t, float, Stream&)

    NOA_INSTANTIATE_SHIFT_(cfloat_t);
    NOA_INSTANTIATE_SHIFT_(cdouble_t);
}
