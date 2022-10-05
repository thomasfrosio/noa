#include "noa/common/Assert.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Pointers.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/signal/fft/Shift.h"

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(32, 8);

    template<bool IS_CENTERED>
    __forceinline__ __device__ int32_t getFrequency_(int32_t idx, int32_t dim) {
        if constexpr(IS_CENTERED)
            return idx - dim / 2;
        else
            return idx < (dim + 1) / 2 ? idx : idx - dim;
        return 0;
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED>
    __forceinline__ __device__ int32_t getOutputIndex_(int32_t i_idx, [[maybe_unused]] int32_t dim) {
        (void) dim;
        if constexpr (IS_SRC_CENTERED == IS_DST_CENTERED)
            return i_idx;
        else if constexpr (IS_SRC_CENTERED)
            return noa::math::iFFTShift(i_idx, dim);
        else
            return noa::math::FFTShift(i_idx, dim);
        return 0;
    }

    template<typename C, typename T>
    __forceinline__ __device__ C getPhaseShift_(T shift, T freq) {
        using real_t = traits::value_type_t<C>;
        const float factor = -math::dot(shift, freq);
        C phase_shift;
        math::sincos(static_cast<real_t>(factor), &phase_shift.imag, &phase_shift.real);
        return phase_shift;
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void shiftHalf2D_(Accessor<const T, 3, uint32_t> input,
                      Accessor<T, 3, uint32_t> output, int2_t shape) {
        const int3_t gid{blockIdx.z,
                         blockIdx.y * THREADS.y + threadIdx.y,
                         blockIdx.x * THREADS.x + threadIdx.x};
        if (gid[1] >= shape[0] || gid[2] >= shape[1] / 2 + 1)
            return;

        const int2_t freq{getFrequency_<IS_SRC_CENTERED>(gid[1], shape[0]),
                          getFrequency_<false>(gid[2], shape[1])};

        using real_t = traits::value_type_t<T>;
        const auto phase_shift = static_cast<real_t>(math::prod(1 - 2 * math::abs(freq % 2)));

        const uint32_t o_y = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid[1], shape[0]);
        output(gid[0], o_y, gid[2]) = input ? input(gid) * phase_shift : phase_shift;
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void shiftHalf3D_(Accessor<const T, 4, uint32_t> input,
                      Accessor<T, 4, uint32_t> output,
                      int3_t shape, uint32_t blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         index[0] * THREADS.y + threadIdx.y,
                         index[1] * THREADS.x + threadIdx.x};
        if (gid[2] >= shape[1] || gid[3] >= shape[2] / 2 + 1)
            return;

        const int3_t freq{getFrequency_<IS_SRC_CENTERED>(gid[1], shape[0]),
                          getFrequency_<IS_SRC_CENTERED>(gid[2], shape[1]),
                          getFrequency_<false>(gid[3], shape[2])};

        using real_t = traits::value_type_t<T>;
        const auto phase_shift = static_cast<real_t>(math::prod(1 - 2 * math::abs(freq % 2)));

        const uint32_t o_z = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid[1], shape[0]);
        const uint32_t o_y = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid[2], shape[1]);
        output(gid[0], o_z, o_y, gid[3]) = input ? input(gid) * phase_shift : phase_shift;
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void shift2D_(Accessor<const T, 3, uint32_t> input,
                  Accessor<T, 3, uint32_t> output, int2_t shape,
                  const float2_t* shifts, float cutoff_sqd, float2_t f_shape) {
        const int3_t gid{blockIdx.z,
                         blockIdx.y * THREADS.y + threadIdx.y,
                         blockIdx.x * THREADS.x + threadIdx.x};
        if (gid[1] >= shape[0] || gid[2] >= shape[1] / 2 + 1)
            return;

        float2_t shift = shifts[gid[0]];
        shift *= math::Constants<float>::PI2 / float2_t(shape);

        const float2_t freq{getFrequency_<IS_SRC_CENTERED>(gid[1], shape[0]),
                            getFrequency_<false>(gid[2], shape[1])};

        T phase_shift{1, 0};
        const float2_t norm_freq = freq / f_shape;
        if (math::dot(norm_freq, norm_freq) <= cutoff_sqd)
            phase_shift = getPhaseShift_<T>(shift, freq);

        const uint32_t o_y = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid[1], shape[0]);
        output(gid[0], o_y, gid[2]) = input ? input(gid) * phase_shift : phase_shift;
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void shift2D_single_(Accessor<const T, 3, uint32_t> input,
                         Accessor<T, 3, uint32_t> output, int2_t shape,
                         float2_t shift, float cutoff_sqd, float2_t f_shape) {
        const int3_t gid{blockIdx.z,
                         blockIdx.y * THREADS.y + threadIdx.y,
                         blockIdx.x * THREADS.x + threadIdx.x};
        if (gid[1] >= shape[0] || gid[2] >= shape[1] / 2 + 1)
            return;

        const float2_t freq{getFrequency_<IS_SRC_CENTERED>(gid[1], shape[0]),
                            getFrequency_<false>(gid[2], shape[1])};

        T phase_shift{1, 0};
        const float2_t norm_freq = freq / f_shape;
        if (math::dot(norm_freq, norm_freq) <= cutoff_sqd)
            phase_shift = getPhaseShift_<T>(shift, freq);

        const uint32_t o_y = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid[1], shape[0]);
        output(gid[0], o_y, gid[2]) = input ? input(gid) * phase_shift : phase_shift;
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void shift3D_(Accessor<const T, 4, uint32_t> input,
                  Accessor<T, 4, uint32_t> output, int3_t shape,
                  const float3_t* shifts, float cutoff_sqd, float3_t f_shape, uint32_t blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         index[0] * THREADS.y + threadIdx.y,
                         index[1] * THREADS.x + threadIdx.x};
        if (gid[2] >= shape[1] || gid[3] >= shape[2] / 2 + 1)
            return;

        float3_t shift = shifts[gid[0]];
        shift *= math::Constants<float>::PI2 / float3_t(shape);

        const float3_t freq{getFrequency_<IS_SRC_CENTERED>(gid[1], shape[0]),
                            getFrequency_<IS_SRC_CENTERED>(gid[2], shape[1]),
                            getFrequency_<false>(gid[3], shape[2])};

        T phase_shift{1, 0};
        const float3_t norm_freq = freq / f_shape;
        if (math::dot(norm_freq, norm_freq) <= cutoff_sqd)
            phase_shift = getPhaseShift_<T>(shift, freq);

        const uint32_t o_z = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid[1], shape[0]);
        const uint32_t o_y = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid[2], shape[1]);
        output(gid[0], o_z, o_y, gid[3]) = input ? input(gid) * phase_shift : phase_shift;
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void shift3D_single_(Accessor<const T, 4, uint32_t> input,
                         Accessor<T, 4, uint32_t> output, int3_t shape,
                         float3_t shift, float cutoff_sqd, float3_t f_shape, uint32_t blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         index[0] * THREADS.y + threadIdx.y,
                         index[1] * THREADS.x + threadIdx.x};
        if (gid[2] >= shape[1] || gid[3] >= shape[2] / 2 + 1)
            return;

        const float3_t freq{getFrequency_<IS_SRC_CENTERED>(gid[1], shape[0]),
                            getFrequency_<IS_SRC_CENTERED>(gid[2], shape[1]),
                            getFrequency_<false>(gid[3], shape[2])};

        T phase_shift{1, 0};
        const float3_t norm_freq = freq / f_shape;
        if (math::dot(norm_freq, norm_freq) <= cutoff_sqd)
            phase_shift = getPhaseShift_<T>(shift, freq);

        const uint32_t o_z = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid[1], shape[0]);
        const uint32_t o_y = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid[2], shape[1]);
        output(gid[0], o_z, o_y, gid[3]) = input ? input(gid) * phase_shift : phase_shift;
    }
}

namespace noa::cuda::signal::fft {
    using Layout = noa::fft::Layout;

    template<Remap REMAP, typename T, typename>
    void shift2D(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                 const shared_t<float2_t[]>& shifts, float cutoff, Stream& stream) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL)
            static_assert(traits::always_false_v<T>);
        NOA_ASSERT(input != output || IS_SRC_CENTERED == IS_DST_CENTERED);
        NOA_ASSERT(all(shape > 0) && shape[1] == 1);
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        const shared_t<float2_t[]> d_shifts = util::ensureDeviceAccess(shifts, stream, output_strides[0]);

        const auto s_shape = safe_cast<int2_t>(dim2_t(shape.get(2)));
        const float2_t f_shape(s_shape / 2 * 2 + int2_t(s_shape == 1)); // if odd, n-1
        const dim3 blocks(math::divideUp(s_shape[1] / 2 + 1, static_cast<int32_t>(THREADS.x)),
                          math::divideUp(s_shape[0], static_cast<int32_t>(THREADS.y)),
                          shape[0]);
        const LaunchConfig config{blocks, THREADS};

        const auto i_strides = safe_cast<uint3_t>(dim3_t{input_strides[0], input_strides[2], input_strides[3]});
        const auto o_strides = safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]});
        const Accessor<const T, 3, uint32_t> input_accessor(input.get(), i_strides);
        const Accessor<T, 3, uint32_t> output_accessor(output.get(), o_strides);

        stream.enqueue("signal::fft::shift2D", shift2D_<IS_SRC_CENTERED, IS_DST_CENTERED, T>, config,
                       input_accessor, output_accessor, s_shape, d_shifts.get(), cutoff * cutoff, f_shape);
        stream.attach(input, output, d_shifts);
    }

    template<Remap REMAP, typename T, typename>
    void shift2D(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                 float2_t shift, float cutoff, Stream& stream) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL)
            static_assert(traits::always_false_v<T>);
        NOA_ASSERT(input != output || IS_SRC_CENTERED == IS_DST_CENTERED);
        NOA_ASSERT(all(shape > 0) && shape[1] == 1);
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        const auto s_shape = safe_cast<int2_t>(dim2_t(shape.get(2)));
        const dim3 blocks(math::divideUp(s_shape[1] / 2 + 1, static_cast<int32_t>(THREADS.x)),
                          math::divideUp(s_shape[0], static_cast<int32_t>(THREADS.y)),
                          shape[0]);
        const LaunchConfig config{blocks, THREADS};

        const auto i_strides = safe_cast<uint3_t>(dim3_t{input_strides[0], input_strides[2], input_strides[3]});
        const auto o_strides = safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]});
        const Accessor<const T, 3, uint32_t> input_accessor(input.get(), i_strides);
        const Accessor<T, 3, uint32_t> output_accessor(output.get(), o_strides);

        if (all(math::isEqual(math::abs(shift), float2_t(s_shape) / 2)) && cutoff >= math::sqrt(0.5f)) {
            stream.enqueue("signal::fft::shift2D", shiftHalf2D_<IS_SRC_CENTERED, IS_DST_CENTERED, T>, config,
                           input_accessor, output_accessor, s_shape);
        } else {
            const float2_t f_shape(s_shape / 2 * 2 + int2_t(s_shape == 1)); // if odd, n-1
            shift *= math::Constants<float>::PI2 / float2_t(s_shape);
            stream.enqueue("signal::fft::shift2D", shift2D_single_<IS_SRC_CENTERED, IS_DST_CENTERED, T>, config,
                           input_accessor, output_accessor, s_shape, shift, cutoff * cutoff, f_shape);
        }
        stream.attach(input, output);
    }

    template<Remap REMAP, typename T, typename>
    void shift3D(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                 const shared_t<float3_t[]>& shifts, float cutoff, Stream& stream) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL)
            static_assert(traits::always_false_v<T>);
        NOA_ASSERT(input != output || IS_SRC_CENTERED == IS_DST_CENTERED);
        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        const shared_t<float3_t[]> d_shifts = util::ensureDeviceAccess(shifts, stream, output_strides[0]);

        const auto s_shape = safe_cast<int3_t>(dim3_t(shape.get(1)));
        const float3_t f_shape(s_shape / 2 * 2 + int3_t(s_shape == 1)); // if odd, n-1
        const uint32_t blocks_x = math::divideUp(s_shape[2] / 2 + 1, static_cast<int32_t>(THREADS.x));
        const uint32_t blocks_y = math::divideUp(s_shape[1], static_cast<int32_t>(THREADS.y));
        const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
        const LaunchConfig config{blocks, THREADS};

        const Accessor<const T, 4, uint32_t> input_accessor(input.get(), safe_cast<uint4_t>(input_strides));
        const Accessor<T, 4, uint32_t> output_accessor(output.get(), safe_cast<uint4_t>(output_strides));

        stream.enqueue("signal::fft::shift3D", shift3D_<IS_SRC_CENTERED, IS_DST_CENTERED, T>, config,
                       input_accessor, output_accessor, s_shape, d_shifts.get(), cutoff * cutoff, f_shape, blocks_x);
        stream.attach(input, output, d_shifts);
    }

    template<Remap REMAP, typename T, typename>
    void shift3D(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                 float3_t shift, float cutoff, Stream& stream) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL)
            static_assert(traits::always_false_v<T>);
        NOA_ASSERT(input != output || IS_SRC_CENTERED == IS_DST_CENTERED);
        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        const auto s_shape = safe_cast<int3_t>(dim3_t(shape.get(1)));
        const uint32_t blocks_x = math::divideUp(s_shape[2] / 2 + 1, static_cast<int32_t>(THREADS.x));
        const uint32_t blocks_y = math::divideUp(s_shape[1], static_cast<int32_t>(THREADS.y));
        const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
        const LaunchConfig config{blocks, THREADS};

        const Accessor<const T, 4, uint32_t> input_accessor(input.get(), safe_cast<uint4_t>(input_strides));
        const Accessor<T, 4, uint32_t> output_accessor(output.get(), safe_cast<uint4_t>(output_strides));

        if (all(math::isEqual(math::abs(shift), float3_t(s_shape) / 2)) && cutoff >= math::sqrt(0.5f)) {
            stream.enqueue("signal::fft::shift3D", shiftHalf3D_<IS_SRC_CENTERED, IS_DST_CENTERED, T>, config,
                           input_accessor, output_accessor, s_shape, blocks_x);
        } else {
            const float3_t f_shape(s_shape / 2 * 2 + int3_t(s_shape == 1)); // if odd, n-1
            shift *= math::Constants<float>::PI2 / float3_t(s_shape);
            stream.enqueue("signal::fft::shift3D", shift3D_single_<IS_SRC_CENTERED, IS_DST_CENTERED, T>, config,
                           input_accessor, output_accessor, s_shape, shift, cutoff * cutoff, f_shape, blocks_x);
        }
        stream.attach(input, output);
    }

    #define NOA_INSTANTIATE_SHIFT(T)                                                                                                                                \
    template void shift2D<Remap::H2H,T,void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float2_t[]>&, float, Stream&);      \
    template void shift2D<Remap::H2H,T,void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float, Stream&);                         \
    template void shift2D<Remap::H2HC,T,void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float2_t[]>&, float, Stream&);     \
    template void shift2D<Remap::H2HC,T,void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float, Stream&);                        \
    template void shift2D<Remap::HC2H,T,void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float2_t[]>&, float, Stream&);     \
    template void shift2D<Remap::HC2H,T,void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float, Stream&);                        \
    template void shift2D<Remap::HC2HC,T,void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float2_t[]>&, float, Stream&);    \
    template void shift2D<Remap::HC2HC,T,void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float2_t, float, Stream&);                       \
    template void shift3D<Remap::H2H,T,void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float3_t[]>&, float, Stream&);      \
    template void shift3D<Remap::H2H,T,void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float, Stream&);                         \
    template void shift3D<Remap::H2HC,T,void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float3_t[]>&, float, Stream&);     \
    template void shift3D<Remap::H2HC,T,void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float, Stream&);                        \
    template void shift3D<Remap::HC2H,T,void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float3_t[]>&, float, Stream&);     \
    template void shift3D<Remap::HC2H,T,void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float, Stream&);                        \
    template void shift3D<Remap::HC2HC,T,void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float3_t[]>&, float, Stream&);    \
    template void shift3D<Remap::HC2HC,T,void>(const shared_t<T[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, float3_t, float, Stream&)

    NOA_INSTANTIATE_SHIFT(cfloat_t);
    NOA_INSTANTIATE_SHIFT(cdouble_t);
}
