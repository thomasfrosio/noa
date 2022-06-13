#include "noa/common/Assert.h"
#include "noa/common/Types.h"
#include "noa/cpu/geometry/Interpolator.h"
#include "noa/cpu/geometry/fft/Project.h"

namespace {
    using namespace ::noa;

    // Go from the DC-centered frequency to a valid index in the output volume.
    // This operation of course depends on the volume's FFT layout.
    // The input frequency should be in-bound, i.e. -n/2 <= frequency <= (n-1)/2
    template<bool IS_CENTERED>
    inline int getIndex_(int frequency, int volume_dim) {
        if constexpr (IS_CENTERED) {
            return frequency + volume_dim / 2;
        } else {
            return frequency < 0 ? frequency + volume_dim : frequency;
        }
    }

    template<bool IS_CENTERED>
    inline int getFrequency_(int index, int shape) {
        if constexpr (IS_CENTERED)
            return index - shape / 2;
        else
            return index < (shape + 1) / 2 ? index : index - shape;
    }

    // The gridding kernel is a tri-linear pulse.
    inline void setGriddingWeights_(int3_t base0, float3_t freq, float o_weights[2][2][2]) {
        // So if the coordinate is centered in the bottom left corner of the cube (base0),
        // i.e. its decimal is 0, the corresponding fraction for this element should be 1.
        float3_t fraction[2];
        fraction[1] = freq - float3_t{base0};
        fraction[0] = 1.f - fraction[1];
        for (size_t w = 0; w < 2; ++w)
            for (size_t v = 0; v < 2; ++v)
                for (size_t u = 0; u < 2; ++u)
                    o_weights[w][v][u] = fraction[w][0] * fraction[v][1] * fraction[u][2];
    }

    inline void setBoundary_(int3_t base0, int3_t shape, bool2_t o_bound[3]) {
        // For each voxel in the 2x2x2 cube, compute whether it is valid or out of bound (OOB).
        // Note that for even dimensions, the negative (real) nyquist is considered OOB, thereby guaranteeing
        // that if frequency A is valid, -A is valid as well.
        // n = 10: -5 -4 -3 -2 -1 0 1 2 3 4    ->    allowed range: [-4, ..., 4]
        // n = 9:     -4 -3 -2 -1 0 1 2 3 4    ->    allowed range: [-4, ..., 4]
        const int3_t base1{base0 + 1};
        const int3_t idx_max = (shape - 1) / 2;

        o_bound[0][0] = base0[0] >= -idx_max[0] && base0[0] <= idx_max[0];
        o_bound[0][1] = base1[0] >= -idx_max[0] && base1[0] <= idx_max[0];

        o_bound[1][0] = base0[1] >= -idx_max[1] && base0[1] <= idx_max[1];
        o_bound[1][1] = base1[1] >= -idx_max[1] && base1[1] <= idx_max[1];

        // For the innermost dimension, the frequency should always be >= 0.
        o_bound[2][0] = base0[2] <= idx_max[2];
        o_bound[2][1] = base1[2] <= idx_max[2];
    }

    template<typename T>
    inline void atomicAdd_(T* grid, size_t offset, T value) {
        if constexpr (traits::is_complex_v<T>) {
            #pragma omp atomic
            grid[offset][0] += value[0];
            #pragma omp atomic
            grid[offset][1] += value[1];
        } else {
            #pragma omp atomic
            grid[offset] += value;
        }
    }

    // Spread the value within a 2x2x2 centered on a particular frequency using linear interpolation.
    // "frequency" is the frequency centered on DC (i.e. 0 is the DC), with negative frequencies on the left.
    template<bool IS_DST_CENTERED, typename T>
    void addByGridding_(T* grid, size3_t grid_stride, int3_t grid_shape, T data, float3_t frequency) {
        using real_t = traits::value_type_t<T>;
        const int3_t base0{math::floor(frequency)};

        float kernel[2][2][2]; // 2x2x2 tri-linear weights
        setGriddingWeights_(base0, frequency, kernel);

        bool2_t is_valid[3]; // left-right, for each dimension
        setBoundary_(base0, grid_shape, is_valid);

        for (int w = 0; w < 2; ++w) {
            for (int v = 0; v < 2; ++v) {
                for (int u = 0; u < 2; ++u) {
                    if (is_valid[0][w] && is_valid[1][v] && is_valid[2][u]) {
                        const int idx_w = getIndex_<IS_DST_CENTERED>(base0[0] + w, grid_shape[0]);
                        const int idx_v = getIndex_<IS_DST_CENTERED>(base0[1] + v, grid_shape[1]);
                        const int idx_u = base0[2] + u;
                        const auto fraction = static_cast<real_t>(kernel[w][v][u]);
                        atomicAdd_(grid, indexing::at(idx_w, idx_v, idx_u, grid_stride), data * fraction);
                    }
                }
            }
        }

        // In the redundant X==0 ZY plane (excluding the DC) we must preserve the hermitian symmetry.
        // So if a side of this plane was modified, add the conjugate at (x=0, -y, -z) with the same fraction.
        // Fortunately, setBoundary_ guarantees that if frequency A is in-bound, frequency -A is in-bound too.
        if (base0[2] == 0 && (base0[1] != 0 || base0[0] != 0)) {
            if constexpr (traits::is_complex_v<T>)
                data.imag = -data.imag;
            for (int w = 0; w < 2; ++w) {
                for (int v = 0; v < 2; ++v) {
                    if (is_valid[0][w] && is_valid[1][v]) {
                        const int idx_w = getIndex_<IS_DST_CENTERED>(-(base0[0] + w), grid_shape[0]);
                        const int idx_v = getIndex_<IS_DST_CENTERED>(-(base0[1] + v), grid_shape[1]);
                        const auto fraction = static_cast<real_t>(kernel[w][v][0]);
                        atomicAdd_(grid, indexing::at(idx_w, idx_v, grid_stride), data * fraction);
                    }
                }
            }
        }
    }

    // Direct Fourier insertion, using data-driven interpolation:
    // 1) Gridding: the gridding kernel is the tri-linear pulse (i.e. triangle), so the real iFFT of the grid
    //              should be divided by the Fourier transform of that kernel, i.e. sinc(pi*freq)^2 function.
    // 2) Normalization: the weights are inserted as well, so the grid can and should be normalized
    //                   before taking the iFFT by dividing (+ threshold) using these weights.
    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T>
    void fourierInsert_(const T* slice, size3_t slice_stride, int3_t slice_shape,
                        T* grid, size3_t grid_stride, int3_t grid_shape,
                        const float22_t* inv_scaling_factors, const float33_t* rotations,
                        float cutoff, float2_t ews_radius, size_t threads) {
        using real_t = traits::value_type_t<T>;
        const int2_t l_shape{slice_shape.get() + 1};
        const float2_t f_slice_shape{l_shape / 2 * 2 + int2_t{l_shape == 1}};
        const float3_t f_grid_shape{grid_shape / 2 * 2 + int3_t{grid_shape == 1}};

        // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
        // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
        const float2_t ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : float2_t{};

        cutoff = math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        #pragma omp parallel for collapse(3) default(none) num_threads(threads) \
        shared(slice, slice_stride, slice_shape, grid, grid_stride, grid_shape, \
               inv_scaling_factors, rotations, cutoff, ews_diam_inv,            \
               f_slice_shape, f_grid_shape)

        for (int i = 0; i < slice_shape[0]; ++i) {
            for (int y = 0; y < slice_shape[1]; ++y) {
                for (int u = 0; u < slice_shape[2] / 2 + 1; ++u) { // x == u

                    // Get the frequency.
                    // The transformation operates on normalized frequencies.
                    // The oversampling is implicitly handled when scaling back to the grid shape.
                    const int v = getFrequency_<IS_SRC_CENTERED>(y, slice_shape[1]);
                    const float2_t orig_freq{v, u};
                    float2_t freq_2d = orig_freq / f_slice_shape;

                    // Add scaling to get the true frequencies.
                    if (inv_scaling_factors)
                        freq_2d = inv_scaling_factors[i] * freq_2d;

                    // Curve the slice to match the EWS using small angle approx (u,v are unchanged) and rotate:
                    // Note that the scaling is already applied, so the EWS is computed using the original frequencies
                    // (from the diffraction) and is spherical even under anisotropic magnification.
                    const float z = math::sum(ews_diam_inv * freq_2d * freq_2d);
                    float3_t freq_3d{z, freq_2d[0], freq_2d[1]};
                    freq_3d = rotations[i] * freq_3d;

                    // The frequency rate won't change from that point, so check for the cutoff.
                    if (math::dot(freq_3d, freq_3d) > cutoff)
                        continue;

                    // Handle the non-redundancy of the slice and grid.
                    [[maybe_unused]] real_t conj = 1;
                    if (freq_3d[2] < 0) {
                        freq_3d = -freq_3d;
                        if constexpr(traits::is_complex_v<T>)
                            conj = -1;
                    }
                    freq_3d *= f_grid_shape;

                    // At this point, we know we are going to use the slice value.
                    T value = slice[indexing::at(i, y, u, slice_stride)];
                    if constexpr(traits::is_complex_v<T>)
                        value.imag *= conj;

                    addByGridding_<IS_DST_CENTERED>(grid, grid_stride, grid_shape, value, freq_3d);
                }
            }
        }
    }

    // The exact same transformation as fourierInsert_ is applied here, but instead of inserting the transformed
    // slice(s) into the grid, the transformed slice(s) is extracted from the grid.
    template<bool IS_DST_CENTERED, typename T>
    void fourierExtract_(const T* grid, size3_t grid_stride, int3_t grid_shape,
                         T* slice, size3_t slice_stride, int3_t slice_shape,
                         const float22_t* inv_scaling_factors, const float33_t* rotations,
                         float cutoff, float2_t ews_radius, size_t threads) {
        using real_t = traits::value_type_t<T>;
        const int2_t l_shape{slice_shape.get() + 1};
        const float2_t f_slice_shape{l_shape / 2 * 2 + int2_t{l_shape == 1}};
        const float3_t f_grid_shape{grid_shape / 2 * 2 + int3_t{grid_shape == 1}};

        // Using the small-angle approximation, Z = wavelength / 2 * (X^2 + Y^2).
        // See doi:10.1016/S0304-3991(99)00120-5 for a derivation.
        const float2_t ews_diam_inv = any(ews_radius != 0) ? 1 / (2 * ews_radius) : float2_t{};

        cutoff = math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        cpu::geometry::Interpolator3D<T> interp{grid, grid_stride, size3_t{grid_shape}.fft(), T{0}};

        #pragma omp parallel for collapse(3) default(none) num_threads(threads) \
        shared(slice, slice_stride, slice_shape, grid, grid_stride, grid_shape, \
               inv_scaling_factors, rotations, cutoff, ews_diam_inv,            \
               f_slice_shape, f_grid_shape, interp)

        for (int i = 0; i < slice_shape[0]; ++i) {
            for (int y = 0; y < slice_shape[1]; ++y) {
                for (int u = 0; u < slice_shape[2] / 2 + 1; ++u) { // x == u

                    // -------------------------------- //
                    const int v = getFrequency_<IS_DST_CENTERED>(y, slice_shape[1]);
                    const float2_t orig_freq{v, u};
                    float2_t freq_2d = orig_freq / f_slice_shape;

                    if (inv_scaling_factors)
                        freq_2d = inv_scaling_factors[i] * freq_2d;

                    const float z = math::sum(ews_diam_inv * freq_2d * freq_2d);
                    float3_t freq_3d{z, freq_2d[0], freq_2d[1]};
                    freq_3d = rotations[i] * freq_3d;

                    if (math::dot(freq_3d, freq_3d) > cutoff)
                        continue;

                    [[maybe_unused]] real_t conj = 1;
                    if (freq_3d[2] < 0) {
                        freq_3d = -freq_3d;
                        if constexpr(traits::is_complex_v<T>)
                            conj = -1;
                    }
                    freq_3d[0] += 0.5f;
                    freq_3d[1] += 0.5f;
                    freq_3d *= f_grid_shape;
                    // -------------------------------- //

                    T value = interp.template get<INTERP_LINEAR, BORDER_ZERO>(freq_3d);
                    if constexpr(traits::is_complex_v<T>)
                        value.imag *= conj;

                    slice[indexing::at(i, y, u, slice_stride)] = value;
                }
            }
        }
    }

    template<bool POST_CORRECTION, typename T>
    void correctGriddingSinc2_(const T* input, size4_t input_stride,
                               T* output, size4_t output_stride, size4_t shape, size_t threads) {
        constexpr float PI = math::Constants<float>::PI;
        const int3_t l_shape{shape.get() + 1};
        const float3_t f_shape{l_shape};
        const float3_t half{f_shape / 2 * float3_t{l_shape != 1}}; // if size == 1, half should be 0

        #pragma omp parallel for collapse(4) num_threads(threads) default(none) \
        shared(input, input_stride, output, output_stride, shape, f_shape, half)

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {

                        float3_t dist{j, k, l};
                        dist -= half;
                        dist /= f_shape;

                        const float radius = math::sqrt(math::dot(dist, dist));
                        const float sinc = math::sinc(PI * radius);
                        const T sinc2 = static_cast<T>(sinc * sinc); // > 0.05

                        const size_t offset = indexing::at(i, j, k, l, input_stride);
                        output[indexing::at(i, j, k, l, output_stride)] =
                                POST_CORRECTION ? input[offset] / sinc2 : input[offset] * sinc2;
                    }
                }
            }
        }
    }
}

namespace noa::cpu::geometry::fft {
    template<Remap REMAP, typename T, typename>
    void insert3D(const shared_t<T[]>& slice, size4_t slice_stride, size4_t slice_shape,
                  const shared_t<T[]>& grid, size4_t grid_stride, size4_t grid_shape,
                  const shared_t<float22_t[]>& scaling_factors,
                  const shared_t<float33_t[]>& rotations,
                  float cutoff, float2_t ews_radius, Stream& stream) {

        using Layout = ::noa::fft::Layout;
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL)
            static_assert(traits::always_false_v<T>);

        NOA_ASSERT(slice_shape[1] == 1);
        NOA_ASSERT(grid_shape[0] == 1);

        const int3_t slice_shape_{slice_shape[0], slice_shape[2], slice_shape[3]};
        const int3_t grid_shape_{grid_shape[1], grid_shape[2], grid_shape[3]};
        const size3_t slice_stride_{slice_stride[0], slice_stride[2], slice_stride[3]};
        const size3_t grid_stride_{grid_stride[1], grid_stride[2], grid_stride[3]};

        const size_t threads = stream.threads();
        stream.enqueue([=]() {
            fourierInsert_<IS_SRC_CENTERED, IS_DST_CENTERED, T>(
                    slice.get(), slice_stride_, slice_shape_,
                    grid.get(), grid_stride_, grid_shape_,
                    scaling_factors.get(), rotations.get(),
                    cutoff, ews_radius, threads);
        });
    }

    template<Remap REMAP, typename T, typename>
    void extract3D(const shared_t<T[]>& grid, size4_t grid_stride, size4_t grid_shape,
                   const shared_t<T[]>& slice, size4_t slice_stride, size4_t slice_shape,
                   const shared_t<float22_t[]>& scaling_factors,
                   const shared_t<float33_t[]>& rotations,
                   float cutoff, float2_t ews_radius, Stream& stream) {

        using Layout = ::noa::fft::Layout;
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_NON_CENTERED ||
                      REMAP_ & Layout::SRC_FULL ||
                      REMAP_ & Layout::DST_FULL)
            static_assert(traits::always_false_v<T>);

        NOA_ASSERT(slice_shape[1] == 1);
        NOA_ASSERT(grid_shape[0] == 1);

        const int3_t slice_shape_{slice_shape[0], slice_shape[2], slice_shape[3]};
        const int3_t grid_shape_{grid_shape[1], grid_shape[2], grid_shape[3]};
        const size3_t slice_stride_{slice_stride[0], slice_stride[2], slice_stride[3]};
        const size3_t grid_stride_{grid_stride[1], grid_stride[2], grid_stride[3]};

        const size_t threads = stream.threads();
        stream.enqueue([=]() {
            fourierExtract_<IS_DST_CENTERED, T>(
                    grid.get(), grid_stride_, grid_shape_,
                    slice.get(), slice_stride_, slice_shape_,
                    scaling_factors.get(), rotations.get(),
                    cutoff, ews_radius, threads);
        });
    }

    template<typename T, typename>
    void griddingCorrection(const shared_t<T[]>& input, size4_t input_stride,
                            const shared_t<T[]>& output, size4_t output_stride,
                            size4_t shape, bool post_correction, Stream& stream) {
        const size_t threads = stream.threads();
        stream.enqueue([=]() {
            if (post_correction)
                correctGriddingSinc2_<true>(input.get(), input_stride, output.get(), output_stride, shape, threads);
            else
                correctGriddingSinc2_<false>(input.get(), input_stride, output.get(), output_stride, shape, threads);
        });
    }

    #define NOA_INSTANTIATE_INSERT_(T, R)                                                                               \
    template void insert3D<R, T, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t,  \
                                       const shared_t<float22_t[]>&, const shared_t<float33_t[]>&, float, float2_t, Stream&)

    #define NOA_INSTANTIATE_EXTRACT_(T, R)                                                                              \
    template void extract3D<R, T, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, \
                                        const shared_t<float22_t[]>&, const shared_t<float33_t[]>&, float, float2_t, Stream&)

    #define NOA_INSTANTIATE_PROJECT_(T)         \
    NOA_INSTANTIATE_INSERT_(T, Remap::H2H);     \
    NOA_INSTANTIATE_INSERT_(T, Remap::H2HC);    \
    NOA_INSTANTIATE_INSERT_(T, Remap::HC2H);    \
    NOA_INSTANTIATE_INSERT_(T, Remap::HC2HC);   \
    NOA_INSTANTIATE_EXTRACT_(T, Remap::HC2H);   \
    NOA_INSTANTIATE_EXTRACT_(T, Remap::HC2HC);  \
    template void griddingCorrection<T, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, bool, Stream&)

    NOA_INSTANTIATE_PROJECT_(float);
    NOA_INSTANTIATE_PROJECT_(double);
    NOA_INSTANTIATE_PROJECT_(cfloat_t);
    NOA_INSTANTIATE_PROJECT_(cdouble_t);
}
