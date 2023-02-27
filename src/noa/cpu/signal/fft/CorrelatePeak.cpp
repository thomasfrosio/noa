#include "noa/algorithms/signal/CorrelationPeak.hpp"
#include "noa/algorithms/Utilities.hpp"

#include "noa/core/geometry/Shape.hpp"
#include "noa/core/math/LeastSquare.hpp"
#include "noa/cpu/geometry/fft/Shape.hpp"

#include "noa/cpu/Find.hpp"
#include "noa/cpu/memory/PtrHost.hpp"
#include "noa/cpu/signal/fft/Correlate.hpp"

// TODO If centered, select subregion within xmap_ellipse_radius.

namespace {
    using namespace noa;

    template<fft::Remap REMAP, typename T>
    void enforce_max_radius_inplace_1d_(
            T* xmap, const Strides4<i64>& strides, const Shape4<i64>& shape,
            f32 xmap_ellipse_radius) {

        const i64 size = shape[3];
        const auto center = static_cast<f32>(size / 2);
        const auto edge_size = static_cast<f32>(size) * 0.05f;
        const auto line = noa::geometry::LineSmooth<T>(center, xmap_ellipse_radius, edge_size);
        const auto accessor = Accessor<T, 2, i64>(xmap, strides.filter(0, 3));

        for (i64 i = 0; i < shape[0]; ++i) {
            for (i64 l = 0; l < size; ++l) {
                const auto coords = static_cast<f32>(REMAP == fft::FC2FC ? l : noa::math::fft_shift(l, size));
                const auto mask = line(coords);
                accessor(i, l) *= mask;
            }
        }
    }

    // Fits (in the least-square sense) the peak values to a parabola to compute the vertex subpixel position and value.
    // The "xmap" (and therefore the "peak_index") can be non-centered (F2F case). The "peak_radius" defines the window
    // of the elements, around the original "peak_index" position, that should be included in the fit.
    // Returns the (fft-centered) peak position and value.
    template<noa::fft::Remap REMAP, typename Real, size_t N>
    constexpr auto subpixel_registration_parabola_1d_(
            const Real* xmap, const Strides<i64, N>& xmap_strides, const Shape<i64, N>& xmap_shape,
            Vec<i64, N> peak_index, const Vec<i64, N>& peak_radius) {
        static_assert(REMAP == fft::F2F || REMAP == fft::FC2FC);

        // Prepare buffer:
        constexpr size_t STATIC_BUFFER_SIZE = 64;
        size_t peak_window_elements{0};
        for (size_t i = 0; i < N; ++i)
            peak_window_elements += static_cast<size_t>(peak_radius[i] * 2 + 1);
        std::array<Real, STATIC_BUFFER_SIZE> static_buffer{0}; // TODO Replace with inline_vector
        std::vector<Real> dynamic_buffer;
        Real* output;
        if (peak_window_elements > STATIC_BUFFER_SIZE) {
            dynamic_buffer = std::vector<Real>(peak_window_elements, Real{0});
            output = dynamic_buffer.data();
        } else {
            output = static_buffer.data();
        }

        if constexpr (REMAP == fft::F2F) {
            Real* current_output = output;
            for (size_t dim = 0; dim < N; ++dim) {
                // Offset to peak location, except for the current dimension.
                const Real* current_xmap = xmap;
                for (size_t i = 0; i < N; ++i)
                    current_xmap += noa::indexing::at(peak_index[i], xmap_strides[i]) * (dim != i);

                // The peak window can be split across two separate quadrant.
                // Retrieve the frequency and if it is a valid frequency,
                // convert back to an index and compute the memory offset.
                const i64 dim_size = xmap_shape[dim];
                const i64 peak_frequency = noa::algorithm::index2frequency<false>(peak_index[dim], dim_size);
                for (i64 index = -peak_radius[dim]; index <= peak_radius[dim]; ++index, ++current_output) {
                    const i64 current_frequency = peak_frequency + index;
                    if (-dim_size / 2 <= current_frequency &&
                        current_frequency <= (dim_size - 1) / 2) {
                        const i64 current_index = noa::algorithm::index2frequency<false>(current_frequency, dim_size);
                        *current_output = current_xmap[noa::indexing::at(current_index, xmap_strides[dim])];
                    }
                }
            }
            // The xmap is not centered, but we want to output the centered index.
            peak_index = noa::math::fft_shift(peak_index, xmap_shape);

        } else {
            // Pre-offset to the peak location.
            // The next part will work around that location.
            xmap += noa::indexing::at(peak_index, xmap_strides);

            // Collect values in the peak window.
            // These loops can be fully unrolled.
            Real* current_output = output;
            for (size_t dim = 0; dim < N; ++dim) {
                for (i64 index = -peak_radius[dim]; index <= peak_radius[dim]; ++index, ++current_output) {
                    const i64 offset = peak_index[dim] + index;
                    if (offset >= 0 && offset < xmap_shape[dim])
                        *current_output = xmap[noa::indexing::at(index, xmap_strides[dim])];
                }
            }
        }

        // At this point, the peak window is saved in row-major order in output.
        // Also, the peak index is centered, so we can simply add the subpixel offset and compute the value.
        const auto [peak_value, peak_subpixel_coordinate] = noa::algorithm::signal::peak_parabola_1d(output, peak_radius);
        const auto peak_coordinate = peak_subpixel_coordinate + peak_index.template as<f64>();
        return std::pair{peak_coordinate.template as<f32>(), static_cast<Real>(peak_value)};
    }

    // Compute the center of mass around the peak and use it to adjust the peak coordinate.
    // TODO The peak value is NOT adjusted, and it simply returns the value at "peak_index".
    template<fft::Remap REMAP, typename Real>
    constexpr auto subpixel_registration_com_(
            const Accessor<const Real, 3, i64>& xmap, const Shape3<i64>& xmap_shape,
            Vec3<i64> peak_index, const Vec3<i64>& peak_radius) {

        // Prepare buffer:
        constexpr size_t STATIC_BUFFER_SIZE = 64;
        const auto peak_window_elements = static_cast<size_t>(noa::math::product(peak_radius * 2 + 1));
        std::array<Real, STATIC_BUFFER_SIZE> static_buffer{0}; // TODO Replace with inline_vector
        std::vector<Real> dynamic_buffer;
        Real* peak_window_values;
        if (peak_window_elements > STATIC_BUFFER_SIZE) {
            dynamic_buffer = std::vector<Real>(peak_window_elements, Real{0});
            peak_window_values = dynamic_buffer.data();
        } else {
            peak_window_values = static_buffer.data();
        }

        // Collect the elements within the peak window. Out-of-bound values are 0.
        // At the same time, find the min value.
        Real peak_window_min{0};
        if constexpr (REMAP == fft::F2F) {
            // The peak window can be split across two separate quadrant.
            // Retrieve the frequency and if it is a valid frequency,
            // convert back to an index and compute the memory offset.
            const auto frequency_min = -xmap_shape.vec() / 2;
            const auto frequency_max = (xmap_shape.vec() - 1) / 2;
            const auto peak_frequency = noa::algorithm::index2frequency<false>(peak_index, xmap_shape);

            i64 count{0};
            for (i64 j = -peak_radius[0]; j <= peak_radius[0]; ++j) {
                for (i64 k = -peak_radius[1]; k <= peak_radius[1]; ++k) {
                    for (i64 l = -peak_radius[2]; l <= peak_radius[2]; ++l, ++count) {
                        const auto relative_offset = Vec3<i64>{j, k, l};
                        const auto current_frequency = peak_frequency + relative_offset;

                        if (noa::all(frequency_min <= current_frequency && current_frequency <= frequency_max)) {
                            const auto current_index = noa::algorithm::frequency2index<false>(current_frequency, xmap_shape);
                            const auto value = xmap(current_index);
                            peak_window_values[count] = value;
                            peak_window_min = std::min(value, peak_window_min);
                        }
                    }
                }
            }
            peak_index = noa::math::fft_shift(peak_index, xmap_shape);

        } else if constexpr (REMAP == fft::FC2FC) {
            i64 count{0};
            for (i64 j = -peak_radius[0]; j <= peak_radius[0]; ++j) {
                for (i64 k = -peak_radius[1]; k <= peak_radius[1]; ++k) {
                    for (i64 l = -peak_radius[2]; l <= peak_radius[2]; ++l, ++count) {
                        const auto relative_offset = Vec3<i64>{j, k, l};
                        const auto current_indexes = peak_index + relative_offset;

                        if (noa::all(current_indexes >= 0 && current_indexes < xmap_shape.vec())) {
                            const auto value = xmap(current_indexes);
                            peak_window_values[count] = value;
                            peak_window_min = std::min(value, peak_window_min);
                        }
                    }
                }
            }
        } else {
            static_assert(noa::traits::always_false_v<Real>);
        }

        // Deal with negative values by setting the min to 0.
        // Compute the center-of-mass.
        Vec3<f64> com{0};
        f64 com_total{0};
        i64 count{0};
        for (i64 j = -peak_radius[0]; j <= peak_radius[0]; ++j) {
            for (i64 k = -peak_radius[1]; k <= peak_radius[1]; ++k) {
                for (i64 l = -peak_radius[2]; l <= peak_radius[2]; ++l, ++count) {
                    const auto value = static_cast<f64>(peak_window_values[count] - peak_window_min);
                    com += value * Vec3<f64>{j, k, l};
                    com_total += value;
                }
            }
        }
        const auto peak_coordinate = com / com_total + peak_index.as<f64>();

        // Finally, get the peak value.
        const auto peak_window_strides = Shape3<i64>(peak_radius * 2 + 1).strides();
        const auto peak_value = peak_window_values[noa::indexing::at(peak_radius, peak_window_strides)];
        return std::pair{peak_coordinate.as<f32>(), peak_value};
    }
}

namespace noa::cpu::signal::fft {
    template<Remap REMAP, typename Real, typename>
    void xpeak_1d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, Vec1<f32> xmap_ellipse_radius,
                  Vec1<f32>* output_peak_coordinates, Real* output_peak_values,
                  PeakMode peak_mode, Vec1<i64> peak_radius, i64 threads) {
        NOA_ASSERT(noa::all(shape > 0) && shape.ndim() == 1);
        NOA_ASSERT(xmap && peak_radius[0] >= 1);

        if (xmap_ellipse_radius[0] > 0)
            enforce_max_radius_inplace_1d_<REMAP>(xmap, strides, shape, xmap_ellipse_radius[0]);

        const auto peak_offsets = cpu::memory::PtrHost<i64>::alloc(shape[0]);
        noa::cpu::find_offsets(noa::first_max_t{}, xmap, strides, shape, peak_offsets.get(), true, true, threads);

        const bool is_column = shape[3] == 1;
        NOA_ASSERT(strides[3 - is_column] > 0);
        const auto batches = static_cast<size_t>(shape[0]);
        const auto stride = strides.filter(3 - is_column);
        const auto size = shape.filter(3 - is_column);
        for (size_t batch = 0; batch < batches; ++batch) {
            const auto peak_index = (Strides1<i64>{peak_offsets[batch]} / stride).vec();
            const Real* imap = xmap + strides[0] * static_cast<i64>(batch);

            switch (peak_mode) {
                case noa::signal::PeakMode::PARABOLA_1D: {
                    const auto [peak_coordinate, peak_value] = subpixel_registration_parabola_1d_<REMAP>(
                            imap, stride, size, peak_index, peak_radius);
                    if (output_peak_coordinates)
                        output_peak_coordinates[batch] = peak_coordinate;
                    if (output_peak_values)
                        output_peak_values[batch] = peak_value;
                    break;
                }
                case noa::signal::PeakMode::COM: {
                    const auto accessor = Accessor<const Real, 3, i64>(imap, Strides3<i64>{0, 0, stride[0]});
                    const auto [peak_coordinate, peak_value] = subpixel_registration_com_<REMAP>(
                            accessor,
                            Shape3<i64>{1, 1, size[0]},
                            Vec3<i64>{0, 0, peak_index[0]},
                            Vec3<i64>{0, 0, peak_radius[0]});
                    if (output_peak_coordinates)
                        output_peak_coordinates[batch] = peak_coordinate.filter(2);
                    if (output_peak_values)
                        output_peak_values[batch] = peak_value;
                    break;
                }
            }
        }
    }

    template<Remap REMAP, typename Real, typename>
    auto xpeak_1d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, Vec1<f32> xmap_ellipse_radius,
                  PeakMode peak_mode, Vec1<i64> peak_radius, i64 threads) -> std::pair<Vec1<f32>, Real> {
        NOA_CHECK(shape[0] == 1, "This overload does not support batched arrays, but got {} batches", shape[0]);

        Vec1<f32> coordinate{};
        Real value{};
        xpeak_1d<REMAP>(xmap, strides, shape, xmap_ellipse_radius,
                        &coordinate, &value, peak_mode, peak_radius, threads);
        return {coordinate, value};
    }

    template<Remap REMAP, typename Real, typename>
    void xpeak_2d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, const Vec2<f32>& xmap_ellipse_radius,
                  Vec2<f32>* output_peak_coordinates, Real* output_peak_values,
                  PeakMode peak_mode, const Vec2<i64>& peak_radius, i64 threads) {
        NOA_ASSERT(shape[1] == 1 && noa::all(shape > 0));
        NOA_ASSERT(xmap && noa::all(peak_radius >= 1));

        const auto shape_2d = shape.filter(2, 3);
        const auto strides_2d = strides.filter(2, 3);

        if (noa::any(xmap_ellipse_radius > 0)) {
            const auto center = (shape_2d / 2).vec().as<f32>();
            const auto edge_size = static_cast<f32>(noa::math::max(shape_2d)) * 0.05f;
            const auto cvalue = Real{1};
            noa::cpu::geometry::fft::ellipse<REMAP>(
                    xmap, strides, xmap, strides, shape,
                    center, xmap_ellipse_radius, edge_size,
                    Float22{}, noa::multiply_t{}, cvalue, false, threads);
        }

        const auto peak_offsets = cpu::memory::PtrHost<i64>::alloc(shape[0]);
        noa::cpu::find_offsets(noa::first_max_t{}, xmap, strides, shape, peak_offsets.get(), true, true, threads);

        for (i64 batch = 0; batch < shape[0]; ++batch) {
            const auto peak_index = noa::indexing::offset2index(peak_offsets.get()[batch], strides_2d, shape_2d);
            const Real* imap = xmap + strides[0] * batch;

            switch (peak_mode) {
                case noa::signal::PeakMode::PARABOLA_1D: {
                    const auto [peak_coordinate, peak_value] = subpixel_registration_parabola_1d_<REMAP>(
                            imap, strides_2d, shape_2d, peak_index, peak_radius);
                    if (output_peak_coordinates)
                        output_peak_coordinates[batch] = peak_coordinate;
                    if (output_peak_values)
                        output_peak_values[batch] = peak_value;
                    break;
                }
                case noa::signal::PeakMode::COM: {
                    const auto accessor = Accessor<const Real, 3, i64>(imap, strides_2d.push_front(0));
                    const auto [peak_coordinate, peak_value] = subpixel_registration_com_<REMAP>(
                            accessor, shape_2d.push_front(0), peak_index.push_front(0), peak_radius.push_front(0));
                    if (output_peak_coordinates)
                        output_peak_coordinates[batch] = peak_coordinate.pop_front();
                    if (output_peak_values)
                        output_peak_values[batch] = peak_value;
                    break;
                }
            }
        }
    }

    template<Remap REMAP, typename Real, typename>
    auto xpeak_2d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, const Vec2<f32>& xmap_ellipse_radius,
                  PeakMode peak_mode, const Vec2<i64>& peak_radius, i64 threads) -> std::pair<Vec2<f32>, Real> {
        NOA_CHECK(shape[0] == 1, "This overload does not support batched arrays, but got {} batches", shape[0]);

        Vec2<f32> coordinate{};
        Real value{};
        xpeak_2d<REMAP>(xmap, strides, shape, xmap_ellipse_radius,
                        &coordinate, &value, peak_mode, peak_radius, threads);
        return {coordinate, value};
    }

    template<Remap REMAP, typename Real, typename>
    void xpeak_3d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, const Vec3<f32>& xmap_ellipse_radius,
                  Vec3<f32>* output_peak_coordinates, Real* output_peak_values,
                  PeakMode peak_mode, const Vec3<i64>& peak_radius, i64 threads) {
        NOA_ASSERT(all(peak_radius >= 1));
        NOA_ASSERT(xmap && all(shape > 0));

        const auto shape_3d = shape.pop_front();
        const auto strides_3d = strides.pop_front();

        if (any(xmap_ellipse_radius > 0)) {
            const auto center = (shape_3d / 2).vec().as<f32>();
            const auto edge_size = static_cast<f32>(noa::math::max(shape_3d)) * 0.05f;
            const auto cvalue = Real{1};
            noa::cpu::geometry::fft::ellipse<REMAP>(
                    xmap, strides, xmap, strides, shape,
                    center, xmap_ellipse_radius, edge_size,
                    Float33{}, noa::multiply_t{}, cvalue, false, threads);
        }

        const auto peak_offsets = cpu::memory::PtrHost<i64>::alloc(shape[0]);
        noa::cpu::find_offsets(noa::first_max_t{}, xmap, strides, shape, peak_offsets.get(), true, true, threads);

        for (i64 batch = 0; batch < shape[0]; ++batch) {
            const auto peak = indexing::offset2index(peak_offsets.get()[batch], strides_3d, shape_3d);
            const Real* imap = xmap + strides[0] * batch;

            switch (peak_mode) {
                case noa::signal::PeakMode::PARABOLA_1D: {
                    const auto [peak_coordinate, peak_value] = subpixel_registration_parabola_1d_<REMAP>(
                            imap, strides_3d, shape_3d, peak, peak_radius);
                    if (output_peak_coordinates)
                        output_peak_coordinates[batch] = peak_coordinate;
                    if (output_peak_values)
                        output_peak_values[batch] = peak_value;
                    break;
                }
                case noa::signal::PeakMode::COM: {
                    const auto accessor = Accessor<const Real, 3, i64>(imap, strides_3d);
                    const auto [peak_coordinate, peak_value] = subpixel_registration_com_<REMAP>(
                            accessor, shape_3d, peak, peak_radius);
                    if (output_peak_coordinates)
                        output_peak_coordinates[batch] = peak_coordinate;
                    if (output_peak_values)
                        output_peak_values[batch] = peak_value;
                    break;
                }
            }
        }
    }

    template<Remap REMAP, typename Real, typename>
    auto xpeak_3d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, const Vec3<f32>& xmap_ellipse_radius,
                  PeakMode peak_mode, const Vec3<i64>& peak_radius, i64 threads) -> std::pair<Vec3<f32>, Real> {
        NOA_CHECK(shape[0] == 1, "This overload does not support batched arrays, but got {} batches", shape[0]);

        Vec3<f32> coordinate{};
        Real value{};
        xpeak_3d<REMAP>(xmap, strides, shape, xmap_ellipse_radius,
                        &coordinate, &value, peak_mode, peak_radius, threads);
        return {coordinate, value};
    }

    #define NOA_INSTANTIATE_XPEAK(R, T)         \
    template void xpeak_1d<R, T, void>(         \
        T*, const Strides4<i64>&,               \
        const Shape4<i64>&, Vec1<f32>,          \
        Vec1<f32>*, T*,                         \
        PeakMode, Vec1<i64>, i64);              \
    template auto xpeak_1d<R, T, void>(         \
        T*, const Strides4<i64>&,               \
        const Shape4<i64>&, Vec1<f32>,          \
        PeakMode, Vec1<i64>, i64)               \
        -> std::pair<Vec1<f32>, T>;             \
    template void xpeak_2d<R, T, void>(         \
        T*, const Strides4<i64>&,               \
        const Shape4<i64>&, const Vec2<f32>&,   \
        Vec2<f32>*, T*,                         \
        PeakMode, const Vec2<i64>&, i64);       \
    template auto xpeak_2d<R, T, void>(         \
        T*, const Strides4<i64>&,               \
        const Shape4<i64>&, const Vec2<f32>&,   \
        PeakMode, const Vec2<i64>&, i64)        \
        -> std::pair<Vec2<f32>, T>;             \
    template void xpeak_3d<R, T, void>(         \
        T*, const Strides4<i64>&,               \
        const Shape4<i64>&, const Vec3<f32>&,   \
        Vec3<f32>*, T*,                         \
        PeakMode, const Vec3<i64>&, i64);       \
    template auto xpeak_3d<R, T, void>(         \
        T*, const Strides4<i64>&,               \
        const Shape4<i64>&, const Vec3<f32>&,   \
        PeakMode, const Vec3<i64>&, i64)        \
        -> std::pair<Vec3<f32>, T>

    #define NOA_INSTANTIATE_XPEAK_ALL(T)    \
    NOA_INSTANTIATE_XPEAK(Remap::F2F, T);   \
    NOA_INSTANTIATE_XPEAK(Remap::FC2FC, T)

    NOA_INSTANTIATE_XPEAK_ALL(f32);
    NOA_INSTANTIATE_XPEAK_ALL(f64);
}
