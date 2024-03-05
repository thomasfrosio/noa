#pragma once

#include "noa/core/signal/Correlation.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/core/math/LeastSquare.hpp"

namespace noa::cpu::signal::guts {
    // Fits (in the least-square sense) the peak values to a parabola to compute the vertex subpixel position and value.
    // The "xmap" (and therefore the "peak_index") can be non-centered (F2F case). The "peak_radius" defines the window
    // of the elements, around the original "peak_index" position, that should be included in the fit.
    // Returns the (fft-centered) peak position and value.
    template<noa::fft::Remap REMAP, typename Real, size_t N>
    constexpr auto subpixel_registration_parabola_1d_(
            const Real* xmap, const Strides<i64, N>& xmap_strides, const Shape<i64, N>& xmap_shape,
            Vec<i64, N> peak_index, const Vec<i64, N>& peak_radius
    ) {
        using Remap = noa::fft::Remap;
        static_assert(REMAP == Remap::F2F or REMAP == Remap::FC2FC);

        // Prepare buffer:
        constexpr size_t STATIC_BUFFER_SIZE = 64;
        size_t peak_window_elements{0};
        for (size_t i = 0; i < N; ++i)
            peak_window_elements += static_cast<size_t>(peak_radius[i] * 2 + 1);
        std::array<Real, STATIC_BUFFER_SIZE> static_buffer{0}; // TODO Replace with flat_vector
        std::vector<Real> dynamic_buffer;
        Real* output;
        if (peak_window_elements > STATIC_BUFFER_SIZE) {
            dynamic_buffer = std::vector<Real>(peak_window_elements, Real{0});
            output = dynamic_buffer.data();
        } else {
            output = static_buffer.data();
        }

        if constexpr (REMAP == Remap::F2F) {
            Real* current_output = output;
            for (size_t dim = 0; dim < N; ++dim) {
                // Offset to peak location, except for the current dimension.
                const Real* current_xmap = xmap;
                for (size_t i = 0; i < N; ++i)
                    current_xmap += ni::offset_at(peak_index[i], xmap_strides[i]) * (dim != i);

                // The peak window can be split across two separate quadrant.
                // Retrieve the frequency and if it is a valid frequency,
                // convert back to an index and compute the memory offset.
                const i64 dim_size = xmap_shape[dim];
                const i64 peak_frequency = noa::fft::index2frequency<false>(peak_index[dim], dim_size);
                for (i64 index = -peak_radius[dim]; index <= peak_radius[dim]; ++index, ++current_output) {
                    const i64 current_frequency = peak_frequency + index;
                    if (-dim_size / 2 <= current_frequency and
                        current_frequency <= (dim_size - 1) / 2) {
                        const i64 current_index = noa::fft::frequency2index<false>(current_frequency, dim_size);
                        *current_output = current_xmap[ni::offset_at(current_index, xmap_strides[dim])];
                    }
                }
            }
            // The xmap is not centered, but we want to output the centered index.
            peak_index = noa::fft::fftshift(peak_index, xmap_shape);

        } else {
            // Pre-offset to the peak location.
            // The next part will work around that location.
            xmap += ni::offset_at(peak_index, xmap_strides);

            // Collect values in the peak window.
            // These loops can be fully unrolled.
            Real* current_output = output;
            for (size_t dim = 0; dim < N; ++dim) {
                for (i64 index = -peak_radius[dim]; index <= peak_radius[dim]; ++index, ++current_output) {
                    const i64 offset = peak_index[dim] + index;
                    if (offset >= 0 and offset < xmap_shape[dim])
                        *current_output = xmap[ni::offset_at(index, xmap_strides[dim])];
                }
            }
        }

        // At this point, the peak window is saved in row-major order in output.
        // Also, the peak index is centered, so we can simply add the subpixel offset and compute the value.
        const auto [peak_value, peak_subpixel_coordinate] = peak_parabola_1d(output, peak_radius);
        const auto peak_coordinate = peak_subpixel_coordinate + peak_index.template as<f64>();
        return Pair{peak_coordinate, static_cast<Real>(peak_value)};
    }

    // Compute the center of mass around the peak and use it to adjust the peak coordinate.
    // TODO The peak value is NOT adjusted, and it simply returns the value at "peak_index".
    template<fft::Remap REMAP, typename Real>
    constexpr auto subpixel_registration_com_(
            const Accessor<const Real, 3, i64>& xmap, const Shape3<i64>& xmap_shape,
            Vec3<i64> peak_index, const Vec3<i64>& peak_radius
    ) {
        // Prepare buffer:
        constexpr size_t STATIC_BUFFER_SIZE = 64;
        const auto peak_window_elements = static_cast<size_t>(product(peak_radius * 2 + 1));
        std::array<Real, STATIC_BUFFER_SIZE> static_buffer{0}; // TODO Replace with flat_vector
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
        if constexpr (REMAP == noa::fft::Remap::F2F) {
            // The peak window can be split across two separate quadrant.
            // Retrieve the frequency and if it is a valid frequency,
            // convert back to an index and compute the memory offset.
            const auto frequency_min = -xmap_shape.vec / 2;
            const auto frequency_max = (xmap_shape.vec - 1) / 2;
            const auto peak_frequency = noa::fft::index2frequency<false>(peak_index, xmap_shape);

            i64 count{0};
            for (i64 j = -peak_radius[0]; j <= peak_radius[0]; ++j) {
                for (i64 k = -peak_radius[1]; k <= peak_radius[1]; ++k) {
                    for (i64 l = -peak_radius[2]; l <= peak_radius[2]; ++l, ++count) {
                        const auto relative_offset = Vec{j, k, l};
                        const auto current_frequency = peak_frequency + relative_offset;

                        if (all(frequency_min <= current_frequency and current_frequency <= frequency_max)) {
                            const auto current_index = noa::fft::frequency2index<false>(current_frequency, xmap_shape);
                            const auto value = xmap(current_index);
                            peak_window_values[count] = value;
                            peak_window_min = std::min(value, peak_window_min);
                        }
                    }
                }
            }
            NOA_ASSERT(static_cast<i64>(peak_window_elements) == count);
            peak_index = noa::fft::fftshift(peak_index, xmap_shape);

        } else if constexpr (REMAP == noa::fft::Remap::FC2FC) {
            i64 count{0};
            for (i64 j = -peak_radius[0]; j <= peak_radius[0]; ++j) {
                for (i64 k = -peak_radius[1]; k <= peak_radius[1]; ++k) {
                    for (i64 l = -peak_radius[2]; l <= peak_radius[2]; ++l, ++count) {
                        const auto relative_offset = Vec{j, k, l};
                        const auto current_indexes = peak_index + relative_offset;

                        if (all(current_indexes >= 0 and current_indexes < xmap_shape.vec)) {
                            const auto value = xmap(current_indexes);
                            peak_window_values[count] = value;
                            peak_window_min = std::min(value, peak_window_min);
                        }
                    }
                }
            }
            NOA_ASSERT(static_cast<i64>(peak_window_elements) == count);
        } else {
            static_assert(nt::always_false_v<Real>);
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
                    com += value * Vec3<f64>::from_values(j, k, l);
                    com_total += value;
                }
            }
        }
        NOA_ASSERT(static_cast<i64>(peak_window_elements) == count);
        const auto peak_coordinate = com / com_total + peak_index.as<f64>();

        // Finally, get the peak value.
        const auto peak_window_strides = Shape3<i64>(peak_radius * 2 + 1).strides();
        const auto peak_value = peak_window_values[ni::offset_at(peak_radius, peak_window_strides)];
        return Pair{peak_coordinate, peak_value};
    }
}

namespace noa::cpu::signal {
    template<noa::fft::Remap REMAP, typename Real, typename Coord>
    void subpixel_registration_1d(
            Real* xmap, const Strides4<i64>& strides, const Shape4<i64>& shape,
            const i64* xmap_peak_offsets,
            Vec1<Coord>* output_peak_coordinates, Real* output_peak_values,
            noa::signal::CorrelationRegistration mode, Vec1<i64> peak_radius
    ) {
        const bool is_column = shape[3] == 1;
        const auto batches = static_cast<size_t>(shape[0]);
        const auto stride = strides.filter(3 - is_column);
        const auto size = shape.filter(3 - is_column);

        for (size_t batch = 0; batch < batches; ++batch) {
            const auto peak_indices = ni::offset2index(xmap_peak_offsets[batch], stride[0]);
            NOA_ASSERT(peak_indices[0] == batch); // peak should belong to the batch
            const auto peak_index = peak_indices.pop_front();
            const Real* imap = xmap + strides[0] * static_cast<i64>(batch);

            switch (mode) {
                case noa::signal::CorrelationRegistration::PARABOLA_1D: {
                    const auto [peak_coordinate, peak_value] = guts::subpixel_registration_parabola_1d_<REMAP>(
                            imap, stride, size, peak_index, peak_radius);
                    if (output_peak_coordinates)
                        output_peak_coordinates[batch] = peak_coordinate.template as<Coord>();
                    if (output_peak_values)
                        output_peak_values[batch] = peak_value;
                    break;
                }
                case noa::signal::CorrelationRegistration::COM: {
                    const auto accessor = Accessor<const Real, 3, i64>(imap, Strides3<i64>{0, 0, stride[0]});
                    const auto [peak_coordinate, peak_value] = subpixel_registration_com_<REMAP>(
                            accessor,
                            Shape3<i64>{1, 1, size[0]},
                            Vec3<i64>{0, 0, peak_index[0]},
                            Vec3<i64>{0, 0, peak_radius[0]});
                    if (output_peak_coordinates)
                        output_peak_coordinates[batch] = peak_coordinate.filter(2).template as<Coord>();
                    if (output_peak_values)
                        output_peak_values[batch] = peak_value;
                    break;
                }
            }
        }
    }

    template<noa::fft::Remap REMAP, typename Real, typename Coord>
    void subpixel_registration_2d(
            Real* xmap, const Strides4<i64>& strides, const Shape4<i64>& shape,
            const i64* xmap_peak_offsets,
            Vec2<Coord>* output_peak_coordinates, Real* output_peak_values,
            noa::signal::CorrelationRegistration peak_mode, const Vec2<i64>& peak_radius
    ) {
        const auto shape_2d = shape.filter(2, 3);
        const auto strides_2d = strides.filter(2, 3);

        for (i64 batch = 0; batch < shape[0]; ++batch) {
            const auto peak_index = ni::offset2index(xmap_peak_offsets[batch], strides_2d, shape_2d);
            const Real* imap = xmap + strides[0] * batch;

            switch (peak_mode) {
                case noa::signal::CorrelationRegistration::PARABOLA_1D: {
                    const auto [peak_coordinate, peak_value] = subpixel_registration_parabola_1d_<REMAP>(
                            imap, strides_2d, shape_2d, peak_index, peak_radius);
                    if (output_peak_coordinates)
                        output_peak_coordinates[batch] = peak_coordinate.template as<Coord>();
                    if (output_peak_values)
                        output_peak_values[batch] = peak_value;
                    break;
                }
                case noa::signal::CorrelationRegistration::COM: {
                    const auto accessor = Accessor<const Real, 3, i64>(imap, strides_2d.push_front(0));
                    const auto [peak_coordinate, peak_value] = subpixel_registration_com_<REMAP>(
                            accessor, shape_2d.push_front(1), peak_index.push_front(0), peak_radius.push_front(0));
                    if (output_peak_coordinates)
                        output_peak_coordinates[batch] = peak_coordinate.pop_front().template as<Coord>();
                    if (output_peak_values)
                        output_peak_values[batch] = peak_value;
                    break;
                }
            }
        }
    }

    template<noa::fft::Remap REMAP, typename Real, typename Coord>
    void subpixel_registration_3d(
            Real* xmap, const Strides4<i64>& strides, const Shape4<i64>& shape,
            const i64* xmap_peak_offsets,
            Vec3<Coord>* output_peak_coordinates, Real* output_peak_values,
            noa::signal::CorrelationRegistration peak_mode, const Vec3<i64>& peak_radius
    ) {
        const auto shape_3d = shape.pop_front();
        const auto strides_3d = strides.pop_front();

        for (i64 batch = 0; batch < shape[0]; ++batch) {
            const auto peak = ni::offset2index(xmap_peak_offsets[batch], strides_3d, shape_3d);
            const Real* imap = xmap + strides[0] * batch;

            switch (peak_mode) {
                case noa::signal::CorrelationRegistration::PARABOLA_1D: {
                    const auto [peak_coordinate, peak_value] = subpixel_registration_parabola_1d_<REMAP>(
                            imap, strides_3d, shape_3d, peak, peak_radius);
                    if (output_peak_coordinates)
                        output_peak_coordinates[batch] = peak_coordinate.template as<Coord>();
                    if (output_peak_values)
                        output_peak_values[batch] = peak_value;
                    break;
                }
                case noa::signal::CorrelationRegistration::COM: {
                    const auto accessor = Accessor<const Real, 3, i64>(imap, strides_3d);
                    const auto [peak_coordinate, peak_value] = subpixel_registration_com_<REMAP>(
                            accessor, shape_3d, peak, peak_radius);
                    if (output_peak_coordinates)
                        output_peak_coordinates[batch] = peak_coordinate.template as<Coord>();
                    if (output_peak_values)
                        output_peak_values[batch] = peak_value;
                    break;
                }
            }
        }
    }
}
