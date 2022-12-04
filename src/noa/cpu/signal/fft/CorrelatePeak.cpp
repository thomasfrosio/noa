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

    // Fits (in the least-square sense) the peak values to a parabola to compute the subpixel position of the vertex.
    // The "xmap" (and therefore the "peak") can be non-centered (F2F case). The "peak_radius" defines the window
    // of the elements, around the original "peak" position, that should be included in the fit.
    // Returns the centered peak position, with subpixel registration.
    template<int64_t NDIM, fft::Remap REMAP, typename Real, typename Int64Vector>
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
        // Also, the peak index is centered, so we can simply add the subpixel offset and return.

        // Add sub-pixel position.
        using real_peak_type = std::conditional_t<NDIM == 3, float3_t, float2_t>;
        real_peak_type real_peak;
        for (int64_t dim = 0; dim < NDIM; ++dim)
            real_peak[dim] = static_cast<float>(peak_index[dim]);
        signal::fft::details::addSubpixelCoordParabola1D<NDIM>(output, peak_radius, real_peak);
        return real_peak;
    }

    template<fft::Remap REMAP, typename Real>
    constexpr double3_t subpixelRegistrationCOM_(
            const Real* xmap, long3_t xmap_strides, long3_t xmap_shape,
            long3_t peak_index, long3_t peak_radius) {

        if constexpr (REMAP == fft::F2F) {
            // The peak window can be split across two separate quadrant.
            // Retrieve the frequency and if it is a valid frequency,
            // convert back to an index and compute the memory offset.
            const auto frequency_min = -xmap_shape / 2;
            const auto frequency_max = (xmap_shape - 1) / 2;
            long3_t peak_frequency = signal::fft::details::nonCenteredIndex2Frequency(peak_index, xmap_shape);

            double3_t com, com_total;
            for (int64_t j = -peak_radius[0]; j <= peak_radius[0]; ++j) {
                for (int64_t k = -peak_radius[1]; k <= peak_radius[1]; ++k) {
                    for (int64_t l = -peak_radius[2]; l <= peak_radius[2]; ++l) {
                        const auto relative_offset = long3_t{j, k, l};
                        const auto current_frequency = peak_frequency + relative_offset;
                        const auto f_relative_index = double3_t(relative_offset);

                        if (all(frequency_min <= current_frequency && current_frequency <= frequency_max)) {
                            using namespace noa::signal::fft::details;
                            const long3_t current_index = frequency2NonCenteredIndex(current_frequency, xmap_shape);
                            const auto value = static_cast<double>(xmap[indexing::at(current_index, xmap_strides)]);
                            com += value * f_relative_index;
                            com_total += value;
                        }
                    }
                }
            }
            return com / com_total + double3_t(math::FFTShift(peak_index, xmap_shape));

        } else if constexpr (REMAP == fft::FC2FC) {
            // Compute the center-of-mass.
            double3_t com, com_total;
            for (int64_t j = -peak_radius[0]; j <= peak_radius[0]; ++j) {
                for (int64_t k = -peak_radius[1]; k <= peak_radius[1]; ++k) {
                    for (int64_t l = -peak_radius[2]; l <= peak_radius[2]; ++l) {
                        const auto relative_offset = long3_t{j, k, l};
                        const auto current_index = peak_index + relative_offset;
                        const auto f_relative_index = double3_t(relative_offset);

                        if (all(current_index >= 0 && current_index < xmap_shape)) {
                            const auto value = static_cast<double>(xmap[indexing::at(current_index, xmap_strides)]);
                            com += value * f_relative_index;
                            com_total += value;
                        }
                    }
                }
            }
            return com / com_total + static_cast<double3_t>(peak_index);
        } else {
            static_assert(traits::always_false_v<Real>);
        }
    }
}

namespace noa::cpu::signal::fft {
    template<Remap REMAP, typename T, typename>
    void xpeak1D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape, float xmap_ellipse_radius,
                 const shared_t<float[]>& peak_coordinates, PeakMode peak_mode, int64_t peak_radius, Stream& stream) {
        NOA_ASSERT(dim3_t(shape.get(1)).ndim() == 1);

        const dim_t threads = stream.threads();
        stream.enqueue([=]() mutable {
            Stream current_stream(Stream::CURRENT);
            current_stream.threads(threads);

            if (xmap_ellipse_radius > 0)
                enforceMaxRadiusInPlace1D_<REMAP>(xmap.get(), strides, shape, xmap_ellipse_radius, current_stream);

            const memory::PtrHost<int64_t> peak_offsets(shape[0]);
            math::find(noa::math::first_max_t{}, xmap, strides, shape,
                       peak_offsets.share(), true, true, current_stream);

            const bool is_column = shape[3] == 1;
            NOA_ASSERT(strides[3 - is_column] > 0);
            const auto stride = safe_cast<int64_t>(strides[3 - is_column]);
            const auto size = safe_cast<int64_t>(shape[3 - is_column]);
            for (dim_t batch = 0; batch < shape[0]; ++batch) {
                const int64_t peak_index = peak_offsets[batch] / stride;
                const T* imap = xmap.get() + strides[0] * batch;

                switch (peak_mode) {
                    case noa::signal::PEAK_PARABOLA_1D: {
                        // For 1D this is annoying because we don't have Int1 or Float1...
                        peak_coordinates.get()[batch] = subpixelRegistrationParabola1D_<1, REMAP>(
                                imap, long2_t(stride), long2_t(size),
                                long2_t(peak_index), long2_t(peak_radius))[0];
                        break;
                    }
                    case noa::signal::PEAK_COM: {
                        const double3_t final_peak = subpixelRegistrationCOM_<REMAP>(
                                imap, long3_t{0, 0, stride}, long3_t{1, 1, size},
                                long3_t{0, 0, peak_index}, long3_t{0, 0, peak_radius});
                        peak_coordinates.get()[batch] = static_cast<float>(final_peak[2]);
                        break;
                    }
                }
            }
        });
    }

    template<Remap REMAP, typename T, typename>
    float xpeak1D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape, float xmap_ellipse_radius,
                  PeakMode peak_mode, int64_t peak_radius, Stream& stream) {
        if (shape[0] != 1) // throw instead of assert because this could result in segfault
            NOA_THROW("This overload does not supported batched arrays, but got {} batches", shape[0]);

        float peak{};
        const shared_t<float[]> peak_ptr(xmap, &peak);
        xpeak1D<REMAP>(xmap, strides, shape, xmap_ellipse_radius,
                       peak_ptr, peak_mode, peak_radius, stream);
        stream.synchronize();
        return peak;
    }

    template<Remap REMAP, typename T, typename>
    void xpeak2D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape, float2_t xmap_ellipse_radius,
                 const shared_t<float2_t[]>& peak_coordinates, PeakMode peak_mode, long2_t peak_radius, Stream& stream) {
        NOA_ASSERT(shape[1] == 1);
        NOA_ASSERT(all(peak_radius >= 1));

        const dim_t threads = stream.threads();
        stream.enqueue([=]() mutable {
            Stream current_stream(Stream::CURRENT);
            current_stream.threads(threads);

            if (any(xmap_ellipse_radius > 0)) {
                const dim2_t shape_2d(shape.get(2));
                const float2_t center(shape_2d / 2);
                const float edge_size = static_cast<float>(noa::math::max(shape_2d)) * 0.05f;
                const auto cvalue =  traits::value_type_t<T>(1);
                ellipse<REMAP>(xmap, strides, xmap, strides, shape,
                               center, xmap_ellipse_radius, edge_size,
                               float22_t{}, noa::math::multiply_t{}, cvalue, false, current_stream);
            }

            cpu::memory::PtrHost<int64_t> peak_offsets(shape[0]);
            math::find(noa::math::first_max_t{}, xmap, strides, shape,
                       peak_offsets.share(), true, true, current_stream);

            const auto shape_2d = safe_cast<long2_t>(dim2_t(shape.get(2)));
            const auto strides_2d = safe_cast<long2_t>(dim2_t(strides.get(2)));
            for (dim_t batch = 0; batch < shape[0]; ++batch) {
                const long2_t peak_index = indexing::indexes(peak_offsets[batch], strides_2d, shape_2d);
                const T* imap = xmap.get() + strides[0] * batch;

                switch (peak_mode) {
                    case noa::signal::PEAK_PARABOLA_1D: {
                        peak_coordinates.get()[batch] = subpixelRegistrationParabola1D_<2, REMAP>(
                                imap, strides_2d, shape_2d, peak_index, peak_radius);
                        break;
                    }
                    case noa::signal::PEAK_COM: {
                        const double3_t final_peak = subpixelRegistrationCOM_<REMAP>(
                                imap, long3_t{0, strides_2d[0], strides_2d[1]},
                                long3_t{1, shape_2d[0], shape_2d[1]}, long3_t{0, peak_index[0], peak_index[1]},
                                long3_t{0, peak_radius[0], peak_radius[1]});
                        peak_coordinates.get()[batch] = float2_t{final_peak[1], final_peak[2]};
                        break;
                    }
                }
            }
        });
    }

    template<Remap REMAP, typename T, typename>
    float2_t xpeak2D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape, float2_t xmap_ellipse_radius,
                     PeakMode peak_mode, long2_t peak_radius, Stream& stream) {
        if (shape[0] != 1) // throw instead of assert because this could result in segfault
            NOA_THROW("This overload does not supported batched arrays, but got {} batches", shape[0]);

        float2_t peak;
        const shared_t<float2_t[]> peak_ptr(xmap, &peak);
        xpeak2D<REMAP>(xmap, strides, shape, xmap_ellipse_radius,
                       peak_ptr, peak_mode, peak_radius, stream);
        stream.synchronize();
        return peak;
    }

    template<Remap REMAP, typename T, typename>
    void xpeak3D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape, float3_t xmap_ellipse_radius,
                 const shared_t<float3_t[]>& peak_coordinates, PeakMode peak_mode, long3_t peak_radius, Stream& stream) {
        NOA_ASSERT(all(peak_radius >= 1));

        const dim_t threads = stream.threads();
        stream.enqueue([=]() mutable {
            Stream current_stream(Stream::CURRENT);
            current_stream.threads(threads);

            if (any(xmap_ellipse_radius > 0)) {
                const dim3_t shape_3d(shape.get(1));
                const float3_t center(shape_3d / 2);
                const float edge_size = static_cast<float>(noa::math::max(shape_3d)) * 0.05f;
                const auto cvalue =  traits::value_type_t<T>(1);
                ellipse<REMAP>(xmap, strides, xmap, strides, shape,
                               center, xmap_ellipse_radius, edge_size,
                               float33_t{}, noa::math::multiply_t{}, cvalue, false, current_stream);
            }

            cpu::memory::PtrHost<int64_t> peak_offsets(shape[0]);
            math::find(noa::math::first_max_t{}, xmap, strides, shape,
                       peak_offsets.share(), true, true, current_stream);

            const auto shape_3d = safe_cast<long3_t>(dim3_t(shape.get(1)));
            const auto strides_3d = safe_cast<long3_t>(dim3_t(strides.get(1)));
            for (dim_t batch = 0; batch < shape[0]; ++batch) {
                const long3_t peak = indexing::indexes(peak_offsets[batch], strides_3d, shape_3d);
                const T* imap = xmap.get() + strides[0] * batch;

                switch (peak_mode) {
                    case noa::signal::PEAK_PARABOLA_1D: {
                        peak_coordinates.get()[batch] = subpixelRegistrationParabola1D_<3, REMAP>(
                                imap, strides_3d, shape_3d, peak, peak_radius);
                        break;
                    }
                    case noa::signal::PEAK_COM: {
                        const double3_t final_peak = subpixelRegistrationCOM_<REMAP>(
                                imap, strides_3d, shape_3d, peak, peak_radius);
                        peak_coordinates.get()[batch] = float3_t(final_peak);
                        break;
                    }
                }
            }
        });
    }

    template<Remap REMAP, typename T, typename>
    float3_t xpeak3D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape, float3_t xmap_ellipse_radius,
                     PeakMode peak_mode, long3_t peak_radius, Stream& stream) {
        if (shape[0] != 1) // throw instead of assert because this could result in segfault
            NOA_THROW("This overload does not supported batched arrays, but got {} batches", shape[0]);

        float3_t peak;
        const shared_t<float3_t[]> peak_ptr(xmap, &peak);
        xpeak3D<REMAP>(xmap, strides, shape, xmap_ellipse_radius,
                       peak_ptr, peak_mode, peak_radius, stream);
        stream.synchronize();
        return peak;
    }

    #define NOA_INSTANTIATE_XPEAK(R, T) \
    template void xpeak1D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float, const shared_t<float[]>&, PeakMode, int64_t, Stream&);         \
    template void xpeak2D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float2_t, const shared_t<float2_t[]>&, PeakMode, long2_t, Stream&);   \
    template void xpeak3D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float3_t, const shared_t<float3_t[]>&, PeakMode, long3_t, Stream&);   \
    template float xpeak1D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float, PeakMode, int64_t, Stream&);                                  \
    template float2_t xpeak2D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float2_t, PeakMode, long2_t, Stream&);                            \
    template float3_t xpeak3D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float3_t, PeakMode, long3_t, Stream&)

    #define NOA_INSTANTIATE_XPEAK_ALL(T)    \
    NOA_INSTANTIATE_XPEAK(Remap::F2F, T);   \
    NOA_INSTANTIATE_XPEAK(Remap::FC2FC, T)

    NOA_INSTANTIATE_XPEAK_ALL(float);
    NOA_INSTANTIATE_XPEAK_ALL(double);
}
