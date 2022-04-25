#include "noa/common/Assert.h"
#include "noa/common/Profiler.h"
#include "noa/common/Types.h"
#include "noa/cpu/reconstruct/ProjectBackward.h"

namespace {
    using namespace ::noa;

    inline cfloat_t getPhaseShift_(float2_t shift, float2_t freq) {
        float factor = -math::sum(shift * freq);
        cfloat_t phase_shift;
        math::sincos(factor, &phase_shift.imag, &phase_shift.real);
        return phase_shift;
    }

    // Go from the DC-centered frequency to a valid index in the output volume.
    // This operation of course depends on the volume's FFT layout.
    // The input frequency should be in-bound, i.e. -n/2 <= frequency <= (n-1)/2
    template<bool IS_VOLUME_CENTERED>
    inline int getIndex_(int frequency, int volume_dim) {
        if constexpr (IS_VOLUME_CENTERED) {
            return frequency + volume_dim / 2;
        } else {
            return frequency < 0 ? frequency + volume_dim : frequency;
        }
    }

    template<bool IS_CENTERED>
    inline int getFrequency_(int index, int shape) {
        if constexpr (IS_CENTERED) // what's the frequency v saved at index y
            return index - shape / 2;
        else
            return index < (shape + 1) / 2 ? index : index - shape;
    }

    // The gridding kernel is a tri-linear pulse.
    void setGriddingWeights(int3_t base0, float3_t freq, float o_weights[2][2][2]) {
        // So if the coordinate is centered in the bottom left corner of the cube (base0),
        // i.e. its decimal is 0, the corresponding fraction for this element should be 1.
        float3_t fraction[2];
        fraction[1] = freq - float3_t(base0);
        fraction[0] = 1.f - fraction[1];
        for (size_t w = 0; w < 2; ++w)
            for (size_t v = 0; v < 2; ++v)
                for (size_t u = 0; u < 2; ++u)
                    o_weights[w][v][u] = fraction[w][0] * fraction[v][1] * fraction[u][2];
    }

    void setBoundary(int3_t base0, int3_t shape, bool2_t* o_bound) {
        // For each voxel in the 2x2x2 cube, compute whether it is out of bound (OOB).
        // Note that for even dimensions, the negative (real) nyquist is considered OOB:
        // n = 10: -5 -4 -3 -2 -1 0 1 2 3 4    ->    allowed range: [-4, ..., 4]
        // n = 9:     -4 -3 -2 -1 0 1 2 3 4    ->    allowed range: [-4, ..., 4]
        const int3_t base1{base0 + 1};
        const int3_t idx_max = (shape - 1) / 2;

        o_bound[0][0] = base0[0] < -idx_max[0] || base0[0] > idx_max[0];
        o_bound[0][1] = base1[0] < -idx_max[0] || base1[0] > idx_max[0];

        o_bound[1][0] = base0[1] < -idx_max[1] || base0[1] > idx_max[1];
        o_bound[1][1] = base1[1] < -idx_max[1] || base1[1] > idx_max[1];

        o_bound[2][0] = base0[2] > idx_max[2];
        o_bound[2][1] = base1[2] > idx_max[2];
    }

    // Linear interpolation on the 2x2x2 cube.
    // v1 and v2 can be nullptr, freq is the frequency centered on DC (i.e. freq == 0 is the DC),
    // with negative frequencies on the left and the base is floor(freq).
    template<bool IS_VOLUME_CENTERED, typename T>
    void addByGridding_(Complex<T>* grid, size3_t grid_stride,
                        T* grid_weight, size3_t grid_weight_stride, int3_t grid_shape,
                        Complex<T> data, T weight, float3_t freq) {
        const int3_t base0{math::floor(freq)};

        float kernel[2][2][2]; // 2x2x2 tri-linear weights
        setGriddingWeights(base0, freq, kernel);

        bool2_t is_valid[3]; // left-right, for each dimension
        setBoundary(base0, grid_shape, is_valid);

        for (int w = 0; w < 2; ++w) {
            for (int v = 0; v < 2; ++v) {
                for (int u = 0; u < 2; ++u) {
                    if (is_valid[0][w] && is_valid[1][v] && is_valid[2][u]) {
                        T fraction = static_cast<T>(kernel[w][v][u]);

                        const int idx_w = getIndex_<IS_VOLUME_CENTERED>(base0[0] + w, grid_shape[0]);
                        const int idx_v = getIndex_<IS_VOLUME_CENTERED>(base0[1] + v, grid_shape[1]);
                        const int idx_u = base0[2] + u;

                        const size_t offset = at(idx_w, idx_v, idx_u, grid_stride);
                        Complex<T> i_value = data * fraction;

                        #pragma omp atomic
                        grid[offset][0] += i_value[0];

                        #pragma omp atomic
                        grid[offset][1] += i_value[1];

                        #pragma omp atomic
                        grid_weight[at(idx_w, idx_v, idx_u, grid_weight_stride)] += weight * fraction;
                    }
                }
            }
        }

        // Redundant plane: the x==0 plane (excluding the DC) is redundant, so to preserve the hermitian symmetry,
        // we must add the conjugate at (x=0, -y, -z) with the same fraction. Fortunately, if frequency A is in-bound,
        // we know for sure that frequency -A is in-bound too.
        if (base0[2] == 0 && (base0[1] != 0 || base0[0] != 0)) {
            data.imag = -data.imag;
            for (int w = 0; w < 2; ++w) {
                for (int v = 0; v < 2; ++v) {
                    if (is_valid[0][w] && is_valid[1][v]) {

                        const int64_t idx_w = getIndex_<IS_VOLUME_CENTERED>(-(base0[0] + w), grid_shape[0]);
                        const int64_t idx_v = getIndex_<IS_VOLUME_CENTERED>(-(base0[1] + v), grid_shape[1]);

                        const size_t offset = at(idx_w, idx_v, grid_stride);
                        const auto fraction = static_cast<T>(kernel[w][v][0]);
                        Complex<T> i_value = data * fraction;

                        #pragma omp atomic
                        grid[offset][0] += i_value[0];

                        #pragma omp atomic
                        grid[offset][1] += i_value[1];

                        #pragma omp atomic
                        grid_weight[at(idx_w, idx_v, grid_weight_stride)] += weight * fraction;
                    }
                }
            }
        }
    }

    // Data-driven interpolation:
    // 1) Gridding: the gridding kernel is the tri-linear pulse (i.e. triangle), so the real iFFT of the grid
    //              should be divided by the Fourier transform of that kernel, i.e. sinc^2 function.
    // 2) Normalization: the weights are inserted as well, so the grid can and should be normalized
    //                   before taking the iFFT by dividing (+ threshold) using these weights.
    template<bool IS_PROJ_CENTERED, bool IS_VOLUME_CENTERED, bool PHASE_SHIFT, typename T>
    void fourierInsert_(const Complex<T>* slice, size3_t slice_stride,
                        const T* slice_weight, size3_t slice_weight_stride, int3_t slice_shape,
                        Complex<T>* grid, size3_t grid_stride,
                        T* grid_weight, size3_t grid_weight_stride, int3_t grid_shape,
                        const float2_t* shifts, const float22_t* scales, const float33_t* rotations,
                        float cutoff, float2_t ews_radius, size_t threads) {

        const int2_t l_shape{slice_shape[2], slice_shape[3]};
        const float2_t f_slice_shape{l_shape / 2 * 2 + int2_t{l_shape == 1}};
        const float3_t f_grid_shape{grid_shape / 2 * 2 + int3_t{grid_shape == 1}};

        const float2_t pre_shift = math::Constants<float>::PI2 / float2_t{l_shape};

        // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
        // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
        const float2_t ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : float2_t{};

        cutoff = math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        #pragma omp parallel for collapse(3) default(none) num_threads(threads)     \
        shared(slice, slice_stride, slice_weight, slice_weight_stride, slice_shape, \
               grid, grid_stride, grid_weight, grid_weight_stride, grid_shape,      \
               shifts, scales, rotations, cutoff, ews_diam_inv, f_slice_shape,      \
               f_grid_shape, pre_shift)

        // The transformation goes from the projection to the volume: forward transformation.
        for (int i = 0; i < slice_shape[0]; ++i) {
            for (int y = 0; y < slice_shape[1]; ++y) {
                for (int u = 0; u < slice_shape[2] / 2 + 1; ++u) { // x == u

                    // Get the frequency.
                    const int v = getFrequency_<IS_PROJ_CENTERED>(y, slice_shape[1]);
                    const float2_t orig_freq{v, u};
                    float2_t freq_2d = orig_freq / f_slice_shape;

                    // Correct for eventual distortions.
                    freq_2d = scales[i] * orig_freq;

                    // Curve the projection to match the EWS using small angle approx (u,v are unchanged):
                    // Note that the distortions are already corrected.
                    const float z = math::sum(ews_diam_inv * freq_2d * freq_2d);
                    float3_t freq_3d{z, freq_2d[0], freq_2d[1]};

                    // The only transformation left is the rotation and the oversampling,
                    // meaning that the frequency rate will not change relative to Nyquist,
                    // i.e. we can check for the cutoff.
                    if (math::dot(freq_3d, freq_3d) > cutoff)
                        continue;

                    // Rotate and oversample. Handle the non-redundancy of the grid.
                    freq_3d = rotations[i] * freq_3d;
                    T conj = 1;
                    if (freq_3d[2] < 0) {
                        freq_3d = -freq_3d;
                        conj = -1;
                    }
                    freq_3d *= f_grid_shape;

                    // At this point, we know we are going to use the projection value and weight.
                    Complex<T> value = slice[at(i, y, u, slice_stride)];
                    value.imag *= conj;

                    // Phase shift value to apply the desired real space shift.
                    // Use the untransformed frequencies.
                    if constexpr (PHASE_SHIFT)
                        value *= Complex<T>(getPhaseShift_(shifts[i] * pre_shift, orig_freq));

                    T weight = slice_weight[at(i, y, u, slice_weight_stride)];

                    addByGridding_<IS_VOLUME_CENTERED>(
                            grid, grid_stride, grid_weight, grid_weight_stride, grid_shape,
                            value, weight, freq_3d);
                }
            }
        }
    }

    template<typename T>
    void correctSinc2_(const T* input, size4_t input_stride,
                       T* output, size4_t output_stride, size4_t shape, size_t threads) {
        const T eps = T(1e-2);
        const float3_t f_shape{shape.get() + 1};
        const float3_t half{f_shape / 2};

        #pragma omp parallel for collapse(4) num_threads(threads) default(none) \
        shared(input, input_stride, output, output_stride, shape, f_shape, half, eps)

        for (size_t i = 0; i < shape[0]; i++) {
            for (size_t j = 0; j < shape[1]; j++) {
                for (size_t k = 0; k < shape[2]; k++) {
                    for (size_t l = 0; l < shape[3]; l++) {

                        float3_t dist{j, k, l};
                        dist -= half;
                        dist /= f_shape;

                        const float radius = noa::math::sqrt(math::dot(dist, dist));
                        const float sinc = noa::math::sinc(noa::math::Constants<float>::PI * radius);
                        const T sinc2 = static_cast<T>(sinc * sinc);

                        output[at(i, j, k, l, output_stride)] =
                                input[at(i, j, k, l, input_stride)] / (sinc2 < eps ? eps : sinc2);
                    }
                }
            }
        }
    }
}

namespace noa::cpu::reconstruct::fft {
    template<Remap REMAP, typename T, typename U>
    void insert(const View<const Complex<T>>& slice, const View<const T>& slice_weight,
                const View<Complex<U>>& grid, const View<U>& grid_weight,
                const float2_t* shifts, const float22_t* scales, const float33_t* rotations,
                float cutoff, float2_t ewald_sphere_radius,
                Stream& stream) {

        using Layout = ::noa::fft::Layout;
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL)
                static_assert(traits::always_false_v<T>);

        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(all(slice.shape() == slice_weight.shape()));
        NOA_ASSERT(all(grid.shape() == grid_weight.shape()));
        NOA_ASSERT(slice.shape()[1] == 1);
        NOA_ASSERT(size2_t{slice.shape().get() + 1}.ndim() == 2);
        NOA_ASSERT(grid.shape().ndim() == 3);

        const size3_t slice_stride{slice.stride()[0], slice.stride()[2], slice.stride()[3]};
        const size3_t slice_weight_stride{slice_weight.stride()[0], slice_weight.stride()[2], slice_weight.stride()[3]};
        const size3_t grid_stride{grid.stride().get() + 1};
        const size3_t grid_weight_stride{grid_weight.stride().get() + 1};

        const int3_t slice_shape{slice.shape()[0], slice.shape()[2], slice.shape()[3]};
        const int3_t grid_shape{grid.shape().get() + 1};

        const size_t threads = stream.threads();
        stream.enqueue(shifts ?
                       fourierInsert_<IS_SRC_CENTERED, IS_DST_CENTERED, true, T> :
                       fourierInsert_<IS_SRC_CENTERED, IS_DST_CENTERED, false, T>,
                       slice.get(), slice_stride, slice_weight.get(), slice_weight_stride, slice_shape,
                       grid.get(), grid_stride, grid_weight.get(), grid_weight_stride, grid_shape,
                       shifts, scales, rotations, cutoff, ewald_sphere_radius, threads);
    }

    #define NOA_INSTANTIATE_BACKWARD_REMAP_(T, R)                                                                               \
    template void insert<R, T, T>(const View<const Complex<T>>&, const View<const T>&, const View<Complex<T>>&, const View<T>&,    \
                               const float2_t*, const float22_t*, const float33_t*, float, float2_t, Stream&)

    #define NOA_INSTANTIATE_BACKWARD_(T)                \
    NOA_INSTANTIATE_BACKWARD_REMAP_(T, noa::fft::H2H);  \
    NOA_INSTANTIATE_BACKWARD_REMAP_(T, noa::fft::H2HC); \
    NOA_INSTANTIATE_BACKWARD_REMAP_(T, noa::fft::HC2H); \
    NOA_INSTANTIATE_BACKWARD_REMAP_(T, noa::fft::HC2HC)

    NOA_INSTANTIATE_BACKWARD_(float);
    NOA_INSTANTIATE_BACKWARD_(double);

    template<typename T>
    void postCompensation(const View<const T>& input, const View<T>& output, Stream& stream) {
        NOA_ASSERT(all(input.shape() == output.shape()));

        stream.enqueue(correctSinc2_<T>, input.get(), input.stride(),
                       output.get(), output.stride(),
                       output.shape(), stream.threads());
    }

    template void postCompensation<float>(const View<const float>&, const View<float>&, Stream&);
    template void postCompensation<double>(const View<const double>&, const View<double>&, Stream&);
}
