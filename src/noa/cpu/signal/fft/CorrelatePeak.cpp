#include "noa/common/signal/Shape.h"
#include "noa/common/signal/details/FourierCorrelationPeak.h"
#include "noa/common/math/LeastSquare.h"
#include "noa/common/geometry/Transform.h"

#include "noa/cpu/math/Find.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/signal/fft/Correlate.h"
#include "noa/cpu/signal/fft/Shape.h"

// TODO If centered, select subregion within xmap_ellipse_radius.

namespace {
    using namespace noa;

    template<fft::Remap REMAP, typename T>
    void enforceMaxRadiusInPlace1D_(T* xmap, dim4_t strides, dim4_t shape,
                                    float xmap_ellipse_radius, cpu::Stream& stream) {
        using real_t = traits::value_type_t<T>;

        const dim_t batches = shape[0];
        const dim_t size = shape[3];
        const auto center = static_cast<float>(size / 2);
        const auto edge_size = static_cast<float>(size) * 0.05f;
        const auto line = signal::LineSmooth<real_t>(center, xmap_ellipse_radius, edge_size);
        const Accessor<T, 2, dim_t> accessor(xmap, dim2_t{strides[0], strides[3]});

        stream.enqueue([=](){
            for (dim_t i = 0; i < batches; ++i) {
                for (dim_t l = 0; l < size; ++l) {
                    const auto coords = static_cast<float>(REMAP == fft::FC2FC ? l : math::FFTShift(l, size));
                    const auto mask = line(coords);
                    accessor(i, l) *= mask;
                }
            }
        });
    }

    // Fits (in the least-square sense) the peak values to a parabola to compute the vertex subpixel position and value.
    // The "xmap" (and therefore the "peak_index") can be non-centered (F2F case). The "peak_radius" defines the window
    // of the elements, around the original "peak_index" position, that should be included in the fit.
    // Returns the (fft-centered) peak position and value.
    template<int64_t NDIM, fft::Remap REMAP,
             typename Real, typename Int64Vector>
    constexpr auto subpixelRegistrationParabola1D_(
            const Real* xmap, Int64Vector xmap_strides, Int64Vector xmap_shape,
            Int64Vector peak_index, Int64Vector peak_radius) {
        static_assert(REMAP == fft::F2F || REMAP == fft::FC2FC);

        // Prepare buffer:
        constexpr size_t STATIC_BUFFER_SIZE = 64;
        size_t peak_window_elements{0};
        for (size_t i = 0; i < NDIM; ++i)
            peak_window_elements += static_cast<size_t>(peak_radius[i] * 2 + 1);
        std::array<Real, STATIC_BUFFER_SIZE> static_buffer{0}; // TODO Replace with inline_vector
        std::vector<Real> dynamic_buffer;
        Real* output{};
        if (peak_window_elements > STATIC_BUFFER_SIZE) {
            dynamic_buffer = std::vector<Real>(peak_window_elements, Real{0});
            output = dynamic_buffer.data();
        } else {
            output = static_buffer.data();
        }

        if constexpr (REMAP == fft::F2F) {
            Real* current_output = output;
            for (int64_t dim = 0; dim < NDIM; ++dim) {
                // Offset to peak location, except for the current dimension.
                const Real* current_xmap = xmap;
                for (int64_t i = 0; i < NDIM; ++i)
                    current_xmap += indexing::at(peak_index[i], xmap_strides[i]) * (dim != i);

                // The peak window can be split across two separate quadrant.
                // Retrieve the frequency and if it is a valid frequency,
                // convert back to an index and compute the memory offset.
                const int64_t dim_size = xmap_shape[dim];
                int64_t peak_frequency = signal::fft::details::nonCenteredIndex2Frequency(peak_index[dim], dim_size);
                for (int64_t index = -peak_radius[dim]; index <= peak_radius[dim]; ++index, ++current_output) {
                    const int64_t current_frequency = peak_frequency + index;
                    if (-dim_size / 2 <= current_frequency &&
                        current_frequency <= (dim_size - 1) / 2) {
                        const int64_t current_index =
                                signal::fft::details::frequency2NonCenteredIndex(current_frequency, dim_size);
                        *current_output = current_xmap[indexing::at(current_index, xmap_strides[dim])];
                    }
                }
            }
            // The xmap is not centered, but we want to output the centered index.
            for (int64_t dim = 0; dim < NDIM; ++dim)
                peak_index[dim] = math::FFTShift(peak_index[dim], xmap_shape[dim]);

        } else {
            // Pre-offset to the peak location.
            // The next part will work around that location.
            for (int64_t dim = 0; dim < NDIM; ++dim)
                xmap += indexing::at(peak_index[dim], xmap_strides[dim]);

            // Collect values in the peak window.
            // These loops can be fully unrolled.
            Real* current_output = output;
            for (int64_t dim = 0; dim < NDIM; ++dim) {
                for (int64_t index = -peak_radius[dim]; index <= peak_radius[dim]; ++index, ++current_output) {
                    const int64_t offset = peak_index[dim] + index;
                    if (offset >= 0 && offset < xmap_shape[dim])
                        *current_output = xmap[indexing::at(index, xmap_strides[dim])];
                }
            }
        }

        // At this point, the peak window is saved in row-major order in output.
        // Also, the peak index is centered, so we can simply add the subpixel offset and compute the value.
        using peak_coordinate_type = std::conditional_t<NDIM == 3, float3_t, float2_t>;
        peak_coordinate_type peak_coordinate{0};
        for (int64_t dim = 0; dim < NDIM; ++dim)
            peak_coordinate[dim] += static_cast<float>(peak_index[dim]);
        const double peak_value = signal::fft::details::vertexParabola1D<NDIM>(output, peak_radius, peak_coordinate);
        return std::pair{peak_coordinate, static_cast<Real>(peak_value)};
    }

    // Compute the center of mass around the peak and use it to adjust the peak coordinate.
    // TODO The peak value is NOT adjusted, and it simply returns the value at "peak_index".
    template<fft::Remap REMAP, typename Real>
    constexpr auto subpixelRegistrationCOM_(
            Accessor<const Real, 3, int64_t> xmap, long3_t xmap_shape,
            long3_t peak_index, long3_t peak_radius) {

        // Prepare buffer:
        constexpr size_t STATIC_BUFFER_SIZE = 64;
        const auto peak_window_elements = static_cast<size_t>(noa::math::prod(peak_radius * 2 + 1));
        std::array<Real, STATIC_BUFFER_SIZE> static_buffer{0}; // TODO Replace with inline_vector
        std::vector<Real> dynamic_buffer;
        Real* peak_window_values{};
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
            using namespace noa::signal::fft::details;

            // The peak window can be split across two separate quadrant.
            // Retrieve the frequency and if it is a valid frequency,
            // convert back to an index and compute the memory offset.
            const auto frequency_min = -xmap_shape / 2;
            const auto frequency_max = (xmap_shape - 1) / 2;
            long3_t peak_frequency = nonCenteredIndex2Frequency(peak_index, xmap_shape);

            int64_t count{0};
            for (int64_t j = -peak_radius[0]; j <= peak_radius[0]; ++j) {
                for (int64_t k = -peak_radius[1]; k <= peak_radius[1]; ++k) {
                    for (int64_t l = -peak_radius[2]; l <= peak_radius[2]; ++l, ++count) {
                        const auto relative_offset = long3_t{j, k, l};
                        const auto current_frequency = peak_frequency + relative_offset;

                        if (all(frequency_min <= current_frequency && current_frequency <= frequency_max)) {
                            const long3_t current_index = frequency2NonCenteredIndex(current_frequency, xmap_shape);
                            const auto value = xmap(current_index);
                            peak_window_values[count] = value;
                            peak_window_min = std::min(value, peak_window_min);
                        }
                    }
                }
            }
            peak_index = math::FFTShift(peak_index, xmap_shape);

        } else if constexpr (REMAP == fft::FC2FC) {
            int64_t count{0};
            for (int64_t j = -peak_radius[0]; j <= peak_radius[0]; ++j) {
                for (int64_t k = -peak_radius[1]; k <= peak_radius[1]; ++k) {
                    for (int64_t l = -peak_radius[2]; l <= peak_radius[2]; ++l, ++count) {
                        const auto relative_offset = long3_t{j, k, l};
                        const auto current_index = peak_index + relative_offset;

                        if (all(current_index >= 0 && current_index < xmap_shape)) {
                            const auto value = xmap(current_index);
                            peak_window_values[count] = value;
                            peak_window_min = std::min(value, peak_window_min);
                        }
                    }
                }
            }
        } else {
            static_assert(traits::always_false_v<Real>);
        }

        // Deal with negative values by setting the min to 0.
        // Compute the center-of-mass.
        double3_t com{0};
        double com_total{0};
        int64_t count{0};
        for (int64_t j = -peak_radius[0]; j <= peak_radius[0]; ++j) {
            for (int64_t k = -peak_radius[1]; k <= peak_radius[1]; ++k) {
                for (int64_t l = -peak_radius[2]; l <= peak_radius[2]; ++l, ++count) {
                    const auto value = static_cast<double>(peak_window_values[count] - peak_window_min);
                    com += value * double3_t{j, k, l};
                    com_total += value;
                }
            }
        }
        const double3_t peak_coordinate = com / com_total + static_cast<double3_t>(peak_index);

        // Finally, get the peak value.
        const auto peak_window_shape = peak_radius * 2 + 1;
        const auto peak_value = peak_window_values[noa::indexing::at(peak_radius, peak_window_shape.strides())];
        return std::pair{peak_coordinate, peak_value};
    }
}

namespace noa::cpu::signal::fft {
    template<Remap REMAP, typename Real, typename>
    void xpeak1D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float xmap_ellipse_radius,
                 const shared_t<float[]>& peak_coordinates, const shared_t<Real[]>& peak_values,
                 PeakMode peak_mode, int64_t peak_radius, Stream& stream) {
        NOA_ASSERT(all(shape > 0) && dim3_t(shape.get(1)).ndim() == 1);
        NOA_ASSERT(xmap && peak_radius >= 1);

        const dim_t threads = stream.threads();
        stream.enqueue([=]() mutable {
            Stream current_stream(Stream::CURRENT);
            current_stream.threads(threads);

            if (xmap_ellipse_radius > 0)
                enforceMaxRadiusInPlace1D_<REMAP>(xmap.get(), strides, shape, xmap_ellipse_radius, current_stream);

            const auto peak_offsets = cpu::memory::PtrHost<int64_t>::alloc(shape[0]);
            cpu::math::find(noa::math::first_max_t{}, xmap, strides, shape,
                            std::shared_ptr<int64_t[]>(xmap, peak_offsets.get()),
                            true, true, current_stream);

            const bool is_column = shape[3] == 1;
            NOA_ASSERT(strides[3 - is_column] > 0);
            const auto stride = safe_cast<int64_t>(strides[3 - is_column]);
            const auto size = safe_cast<int64_t>(shape[3 - is_column]);
            for (dim_t batch = 0; batch < shape[0]; ++batch) {
                const int64_t peak_index = peak_offsets[batch] / stride;
                const Real* imap = xmap.get() + strides[0] * batch;

                switch (peak_mode) {
                    case noa::signal::PEAK_PARABOLA_1D: {
                        // FIXME For 1D this is annoying because we don't have Vec<1,T>, yet...
                        const auto [coordinate, value] = subpixelRegistrationParabola1D_<1, REMAP>(
                                imap, long2_t(stride), long2_t(size),
                                long2_t(peak_index), long2_t(peak_radius));
                        if (peak_coordinates)
                            peak_coordinates.get()[batch] = coordinate[0];
                        if (peak_values)
                            peak_values.get()[batch] = value;
                        break;
                    }
                    case noa::signal::PEAK_COM: {
                        const auto accessor = Accessor<const Real, 3, int64_t>(imap, long3_t{0, 0, stride});
                        const auto [coordinate, value] = subpixelRegistrationCOM_<REMAP>(
                                accessor, long3_t{1, 1, size},
                                long3_t{0, 0, peak_index}, long3_t{0, 0, peak_radius});
                        if (peak_coordinates)
                            peak_coordinates.get()[batch] = static_cast<float>(coordinate[2]);
                        if (peak_values)
                            peak_values.get()[batch] = value;
                        break;
                    }
                }
            }
        });
    }

    template<Remap REMAP, typename Real, typename>
    std::pair<float, Real>
    xpeak1D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float xmap_ellipse_radius,
            PeakMode peak_mode, int64_t peak_radius, Stream& stream) {
        if (shape[0] != 1) // throw instead of assert because this could result in segfault
            NOA_THROW("This overload does not support batched arrays, but got {} batches", shape[0]);

        float coordinate{};
        Real value{};
        const shared_t<float[]> coordinate_ptr(xmap, &coordinate);
        const shared_t<Real[]> value_ptr(xmap, &value);
        xpeak1D<REMAP>(xmap, strides, shape, xmap_ellipse_radius,
                       coordinate_ptr, value_ptr, peak_mode, peak_radius, stream);
        stream.synchronize();
        return {coordinate, value};
    }

    template<Remap REMAP, typename Real, typename>
    void xpeak2D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float2_t xmap_ellipse_radius,
                 const shared_t<float2_t[]>& peak_coordinates, const shared_t<Real[]>& peak_values,
                 PeakMode peak_mode, long2_t peak_radius, Stream& stream) {
        NOA_ASSERT(shape[1] == 1 && all(shape > 0));
        NOA_ASSERT(xmap && all(peak_radius >= 1));

        const dim_t threads = stream.threads();
        stream.enqueue([=]() mutable {
            auto current_stream = cpu::Stream(Stream::CURRENT);
            current_stream.threads(threads);

            if (any(xmap_ellipse_radius > 0)) {
                const dim2_t shape_2d(shape.get(2));
                const float2_t center(shape_2d / 2);
                const float edge_size = static_cast<float>(noa::math::max(shape_2d)) * 0.05f;
                const auto cvalue =  traits::value_type_t<Real>(1);
                ellipse<REMAP>(xmap, strides, xmap, strides, shape,
                               center, xmap_ellipse_radius, edge_size,
                               float22_t{}, noa::math::multiply_t{}, cvalue, false, current_stream);
            }

            const auto peak_offsets = cpu::memory::PtrHost<int64_t>::alloc(shape[0]);
            cpu::math::find(noa::math::first_max_t{}, xmap, strides, shape,
                            std::shared_ptr<int64_t[]>(xmap, peak_offsets.get()),
                            true, true, current_stream);

            const auto shape_2d = safe_cast<long2_t>(dim2_t(shape.get(2)));
            const auto strides_2d = safe_cast<long2_t>(dim2_t(strides.get(2)));
            for (dim_t batch = 0; batch < shape[0]; ++batch) {
                const long2_t peak_index = noa::indexing::indexes(peak_offsets[batch], strides_2d, shape_2d);
                const Real* imap = xmap.get() + strides[0] * batch;

                switch (peak_mode) {
                    case noa::signal::PEAK_PARABOLA_1D: {
                        const auto [coordinate, value] = subpixelRegistrationParabola1D_<2, REMAP>(
                                imap, strides_2d, shape_2d, peak_index, peak_radius);
                        if (peak_coordinates)
                            peak_coordinates.get()[batch] = coordinate;
                        if (peak_values)
                            peak_values.get()[batch] = value;
                        break;
                    }
                    case noa::signal::PEAK_COM: {
                        const auto accessor = Accessor<const Real, 3, int64_t>(
                                imap, long3_t{0, strides_2d[0], strides_2d[1]});
                        const auto [coordinate, value] = subpixelRegistrationCOM_<REMAP>(
                                accessor, long3_t{1, shape_2d[0], shape_2d[1]},
                                long3_t{0, peak_index[0], peak_index[1]},
                                long3_t{0, peak_radius[0], peak_radius[1]});
                        if (peak_coordinates)
                            peak_coordinates.get()[batch] = float2_t{coordinate[1], coordinate[2]};
                        if (peak_values)
                            peak_values.get()[batch] = value;
                        break;
                    }
                }
            }
        });
    }

    template<Remap REMAP, typename Real, typename>
    std::pair<float2_t, Real>
    xpeak2D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float2_t xmap_ellipse_radius,
            PeakMode peak_mode, long2_t peak_radius, Stream& stream) {
        if (shape[0] != 1) // throw instead of assert because this could result in segfault
            NOA_THROW("This overload does not support batched arrays, but got {} batches", shape[0]);

        float2_t coordinate{};
        Real value{};
        const shared_t<float2_t[]> coordinate_ptr(xmap, &coordinate);
        const shared_t<Real[]> value_ptr(xmap, &value);
        xpeak2D<REMAP>(xmap, strides, shape, xmap_ellipse_radius,
                       coordinate_ptr, value_ptr, peak_mode, peak_radius, stream);
        stream.synchronize();
        return {coordinate, value};
    }

    template<Remap REMAP, typename Real, typename>
    void xpeak3D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float3_t xmap_ellipse_radius,
                 const shared_t<float3_t[]>& peak_coordinates, const shared_t<Real[]>& peak_values,
                 PeakMode peak_mode, long3_t peak_radius, Stream& stream) {
        NOA_ASSERT(all(peak_radius >= 1));
        NOA_ASSERT(xmap && all(shape > 0));

        const dim_t threads = stream.threads();
        stream.enqueue([=]() mutable {
            Stream current_stream(Stream::CURRENT);
            current_stream.threads(threads);

            if (any(xmap_ellipse_radius > 0)) {
                const dim3_t shape_3d(shape.get(1));
                const float3_t center(shape_3d / 2);
                const float edge_size = static_cast<float>(noa::math::max(shape_3d)) * 0.05f;
                const auto cvalue =  traits::value_type_t<Real>(1);
                ellipse<REMAP>(xmap, strides, xmap, strides, shape,
                               center, xmap_ellipse_radius, edge_size,
                               float33_t{}, noa::math::multiply_t{}, cvalue, false, current_stream);
            }

            const auto peak_offsets = cpu::memory::PtrHost<int64_t>::alloc(shape[0]);
            cpu::math::find(noa::math::first_max_t{}, xmap, strides, shape,
                            std::shared_ptr<int64_t[]>(xmap, peak_offsets.get()),
                            true, true, current_stream);

            const auto shape_3d = safe_cast<long3_t>(dim3_t(shape.get(1)));
            const auto strides_3d = safe_cast<long3_t>(dim3_t(strides.get(1)));
            for (dim_t batch = 0; batch < shape[0]; ++batch) {
                const long3_t peak = indexing::indexes(peak_offsets[batch], strides_3d, shape_3d);
                const Real* imap = xmap.get() + strides[0] * batch;

                switch (peak_mode) {
                    case noa::signal::PEAK_PARABOLA_1D: {
                        const auto [coordinate, value] = subpixelRegistrationParabola1D_<3, REMAP>(
                                imap, strides_3d, shape_3d, peak, peak_radius);
                        if (peak_coordinates)
                            peak_coordinates.get()[batch] = coordinate;
                        if (peak_values)
                            peak_values.get()[batch] = value;
                        break;
                    }
                    case noa::signal::PEAK_COM: {
                        const auto accessor = Accessor<const Real, 3, int64_t>(imap, strides_3d);
                        const auto [coordinate, value] = subpixelRegistrationCOM_<REMAP>(
                                accessor, shape_3d, peak, peak_radius);
                        if (peak_coordinates)
                            peak_coordinates.get()[batch] = float3_t(coordinate);
                        if (peak_values)
                            peak_values.get()[batch] = value;
                        break;
                    }
                }
            }
        });
    }

    template<Remap REMAP, typename Real, typename>
    std::pair<float3_t, Real>
    xpeak3D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float3_t xmap_ellipse_radius,
            PeakMode peak_mode, long3_t peak_radius, Stream& stream) {
        if (shape[0] != 1) // throw instead of assert because this could result in segfault
            NOA_THROW("This overload does not support batched arrays, but got {} batches", shape[0]);

        float3_t coordinate{};
        Real value{};
        const shared_t<float3_t[]> coordinate_ptr(xmap, &coordinate);
        const shared_t<Real[]> value_ptr(xmap, &value);
        xpeak3D<REMAP>(xmap, strides, shape, xmap_ellipse_radius,
                       coordinate_ptr, value_ptr, peak_mode, peak_radius, stream);
        stream.synchronize();
        return {coordinate, value};
    }

    #define NOA_INSTANTIATE_XPEAK(R, T) \
    template void xpeak1D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float, const shared_t<float[]>&, const shared_t<T[]>&, PeakMode, int64_t, Stream&);         \
    template void xpeak2D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float2_t, const shared_t<float2_t[]>&, const shared_t<T[]>&, PeakMode, long2_t, Stream&);   \
    template void xpeak3D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float3_t, const shared_t<float3_t[]>&, const shared_t<T[]>&, PeakMode, long3_t, Stream&);   \
    template std::pair<float,T> xpeak1D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float, PeakMode, int64_t, Stream&);                                           \
    template std::pair<float2_t,T> xpeak2D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float2_t, PeakMode, long2_t, Stream&);                                     \
    template std::pair<float3_t,T> xpeak3D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float3_t, PeakMode, long3_t, Stream&)

    #define NOA_INSTANTIATE_XPEAK_ALL(T)    \
    NOA_INSTANTIATE_XPEAK(Remap::F2F, T);   \
    NOA_INSTANTIATE_XPEAK(Remap::FC2FC, T)

    NOA_INSTANTIATE_XPEAK_ALL(float);
    NOA_INSTANTIATE_XPEAK_ALL(double);
}
