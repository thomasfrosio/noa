#include "noa/common/signal/Shape.h"
#include "noa/common/math/LeastSquare.h"
#include "noa/common/geometry/Transform.h"

#include "noa/cpu/math/Find.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/signal/fft/Correlate.h"
#include "noa/cpu/signal/fft/Shape.h"

namespace {
    using namespace noa;

    template<fft::Remap REMAP, typename T>
    void enforceMaxRadiusInPlace1D_(T* xmap, dim4_t strides, dim4_t shape,
                                    float ellipse_radius, cpu::Stream& stream) {
        using real_t = traits::value_type_t<T>;

        const dim_t batches = shape[0];
        const dim_t size = shape[3];
        const auto center = static_cast<float>(size / 2);
        const auto edge_size = static_cast<float>(size) * 0.05f;
        const auto line = signal::LineSmooth<real_t>(center, ellipse_radius, edge_size);
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

    // This is equivalent to math::lstsqFitQuadratic(), but slightly faster
    // (I don't think it is significantly faster though).
    template<typename T>
    inline T getParabolicVertex3Points_(T y0, T y1, T y2) noexcept {
        // From IMOD/libcfshr/filtxcorr.c::parabolicFitPosition
        const T d = 2 * (y0 + y2 - 2 * y1);
        T x = 0;
        if (math::abs(d) > math::abs(static_cast<T>(1e-2) * (y0 - y2)))
            x = (y0 - y2) / d;
        if (x > T{0.5})
            x = T{0.5};
        if (x < T{-0.5})
            x = T{-0.5};
        return x;
    }

    // Fits (in the least-square sense) the peak values to a parabola to compute the subpixel position of the vertex.
    // The "input" (and therefore the "peak") can be non-centered (F2F case). The peak vertex is always centered when
    // it is returned. The "radius" defines the window of the elements, around the original "peak" position, that
    // should be included in the fit.
    template<int RADIUS, fft::Remap REMAP, typename T>
    float addVertexOffsetParabola1D(const T* input, int64_t stride, int64_t size, int64_t peak, int64_t radius) {
        static_assert(RADIUS > 0 && RADIUS <= 50);
        static_assert(REMAP == fft::F2F || REMAP == fft::FC2FC);

        constexpr size_t ELEMENTS = static_cast<size_t>(RADIUS * 2 + 1);
        const auto elements = static_cast<size_t>(radius * 2 + 1);
        std::array<T, ELEMENTS> static_buffer{0}; // TODO Replace with inline_vector
        std::vector<T> dynamic_buffer;
        T* output;
        if (radius > RADIUS) {
            dynamic_buffer = std::vector<T>(elements, T{0});
            output = dynamic_buffer.data();
        } else {
            output = static_buffer.data();
        }

        if constexpr (REMAP == fft::F2F) {
            auto nonCenteredIndex2Frequency_ = [](int64_t index, int64_t dim_size) noexcept {
                return index < (dim_size + 1) / 2 ? index : index - dim_size;
            };

            auto frequency2NonCenteredIndex_ = [](int64_t frequency, int64_t dim_size) noexcept {
                return frequency < 0 ? dim_size + frequency : frequency;
            };

            // Here the 2 adjacent points might be in a separate quadrant.
            // Retrieve the frequency, add the offset, and if it's a valid
            // frequency, convert back to an index.
            const int64_t peak_frequency = nonCenteredIndex2Frequency_(peak, size);
            T* first = output;
            for (int64_t l = -radius; l <= radius; ++l, ++first) {
                const int64_t i_frequency = peak_frequency + l;
                if (-size / 2 <= i_frequency && i_frequency <= (size - 1) / 2) {
                    const int64_t i_index = frequency2NonCenteredIndex_(i_frequency, size);
                    *first = input[i_index * stride];
                }
            }
            // The CC map was not centered, so center it here.
            peak = math::FFTShift(peak, size);

        } else {
            input += peak * stride;
            T* first = output;
            for (int64_t l = -radius; l <= radius; ++l, ++first) {
                const int64_t il = peak + l;
                if (il >= 0 && il < size)
                    *first = input[l * stride];
            }
        }

        // Add sub-pixel position by fitting a 1D parabola to the peak and its adjacent points.
        float vertex_offset;
        if (radius == 1) {
            vertex_offset = static_cast<float>(getParabolicVertex3Points_(output[0], output[1], output[2]));
        } else {
            const auto [a, b, _] = math::lstsqFitQuadratic(output, elements);
            NOA_ASSERT(a != 0); // This can only happen if all values in output are equal.
            const auto radius_ = static_cast<double>(radius);
            vertex_offset = static_cast<float>(std::clamp(-b / (2 * a) - radius_, -radius_ + 0.5, radius_ - 0.5));
        }
        return static_cast<float>(peak) + vertex_offset;
    }

    template<fft::Remap REMAP, typename T>
    inline float subpixelRegistration1DParabola(const T* input, int64_t stride, int64_t size,
                                                int64_t peak, int64_t radius) {
        // TODO Check assembly to see if this helps and is optimized.
        switch (radius) {
            case 1:
                return addVertexOffsetParabola1D<1, REMAP>(input, stride, size, peak, 1);
            case 2:
                return addVertexOffsetParabola1D<2, REMAP>(input, stride, size, peak, 2);
            case 3:
                return addVertexOffsetParabola1D<3, REMAP>(input, stride, size, peak, 3);
            default:
                // Use dynamic buffer if buffer is more than 192 bytes (float: radius > 48, double: radius > 24).
                return addVertexOffsetParabola1D<192 / sizeof(T), REMAP>(input, stride, size, peak, radius);
        }
    }

    template<fft::Remap REMAP, typename T>
    inline float2_t subpixelRegistration2DParabola(const T* input, long2_t strides, long2_t shape,
                                                   long2_t peak, long2_t radius) {
        return {
                subpixelRegistration1DParabola<REMAP>(
                        input + peak[1] * strides[1], strides[0], shape[0], peak[0], radius[0]),
                subpixelRegistration1DParabola<REMAP>(
                        input + peak[0] * strides[0], strides[1], shape[1], peak[1], radius[1])
        };
    }

    template<fft::Remap REMAP, typename T>
    inline float3_t subpixelRegistration3DParabola(const T* input, long3_t strides, long3_t shape,
                                                   long3_t peak, long3_t radius) {
        return {
                subpixelRegistration1DParabola<REMAP>(
                        input + peak[1] * strides[1] + peak[2] * strides[2], strides[0], shape[0], peak[0], radius[0]),
                subpixelRegistration1DParabola<REMAP>(
                        input + peak[0] * strides[0] + peak[2] * strides[2], strides[1], shape[1], peak[1], radius[1]),
                subpixelRegistration1DParabola<REMAP>(
                        input + peak[0] * strides[0] + peak[1] * strides[1], strides[2], shape[2], peak[2], radius[2])
        };
    }
}

namespace noa::cpu::signal::fft {
    template<Remap REMAP, typename T, typename>
    void xpeak1D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape, const shared_t<float[]>& peaks,
                 float ellipse_radius, int64_t registration_radius, Stream& stream) {
        NOA_ASSERT(dim3_t(shape.get(1)).ndim() == 1);

        stream.enqueue([=]() mutable {
            if (ellipse_radius > 0)
                enforceMaxRadiusInPlace1D_<REMAP>(xmap.get(), strides, shape, ellipse_radius, stream);

            memory::PtrHost<int64_t> offsets(shape[0]);
            math::find(noa::math::first_max_t{}, xmap, strides, shape, offsets.share(), true, true, stream);

            const bool is_column = shape[3] == 1;
            NOA_ASSERT(strides[3 - is_column] > 0);
            const auto stride = safe_cast<int64_t>(strides[3 - is_column]);
            const auto size = safe_cast<int64_t>(shape[3 - is_column]);
            for (dim_t batch = 0; batch < shape[0]; ++batch) {
                const int64_t peak = offsets[batch] / stride;
                const T* imap = xmap.get() + strides[0] * batch;
                peaks.get()[batch] = subpixelRegistration1DParabola<REMAP>(
                        imap, stride, size, peak, registration_radius);
            }
        });
    }

    template<Remap REMAP, typename T, typename>
    float xpeak1D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape,
                  float ellipse_radius, int64_t registration_radius, Stream& stream) {
        if (shape[0] != 1) // throw instead of assert because this could result in segfault
            NOA_THROW("This overload does not supported batched arrays, but got {} batches", shape[0]);

        float peak;
        const shared_t<float[]> peak_ptr(xmap, &peak);
        xpeak1D<REMAP>(xmap, strides, shape, peak_ptr, ellipse_radius, registration_radius, stream);
        stream.synchronize();
        return peak;
    }

    template<Remap REMAP, typename T, typename>
    void xpeak2D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape, const shared_t<float2_t[]>& peaks,
                 float2_t ellipse_radius, long2_t registration_radius, Stream& stream) {
        NOA_ASSERT(shape[1] == 1);
        NOA_ASSERT(all(registration_radius >= 1));

        stream.enqueue([=]() mutable {
            if (any(ellipse_radius > 0)) {
                // TODO If centered, select subregion within ellipse_radius.

                const dim2_t shape_2d(shape.get(2));
                const float2_t center(shape_2d / 2);
                const float edge_size = static_cast<float>(noa::math::max(shape_2d)) * 0.05f;
                ellipse<REMAP>(xmap, strides, xmap, strides, shape,
                               center, ellipse_radius, edge_size, float22_t{}, false, stream);
            }

            cpu::memory::PtrHost<int64_t> offsets(shape[0]);
            math::find(noa::math::first_max_t{}, xmap, strides, shape, offsets.share(), true, true, stream);

            const auto shape_2d = safe_cast<long2_t>(dim2_t(shape.get(2)));
            const auto strides_2d = safe_cast<long2_t>(dim2_t(strides.get(2)));
            for (dim_t batch = 0; batch < shape[0]; ++batch) {
                const long2_t peak = indexing::indexes(offsets[batch], strides_2d, shape_2d);
                const T* imap = xmap.get() + strides[0] * batch;
                peaks.get()[batch] = subpixelRegistration2DParabola<REMAP>(
                        imap, strides_2d, shape_2d, peak, registration_radius);
            }
        });
    }

    template<Remap REMAP, typename T, typename>
    float2_t xpeak2D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape,
                     float2_t ellipse_radius, long2_t registration_radius, Stream& stream) {
        if (shape[0] != 1) // throw instead of assert because this could result in segfault
            NOA_THROW("This overload does not supported batched arrays, but got {} batches", shape[0]);

        float2_t peak;
        const shared_t<float2_t[]> peak_ptr(xmap, &peak);
        xpeak2D<REMAP>(xmap, strides, shape, peak_ptr, ellipse_radius, registration_radius, stream);
        stream.synchronize();
        return peak;
    }

    template<Remap REMAP, typename T, typename>
    void xpeak3D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape, const shared_t<float3_t[]>& peaks,
                 float3_t ellipse_radius, long3_t registration_radius, Stream& stream) {
        NOA_ASSERT(all(registration_radius >= 1));

        stream.enqueue([=]() mutable {
            if (any(ellipse_radius > 0)) {
                // TODO If centered, select subregion within ellipse_radius.

                const dim3_t shape_3d(shape.get(1));
                const float3_t center(shape_3d / 2);
                const float edge_size = static_cast<float>(noa::math::max(shape_3d)) * 0.05f;
                ellipse<REMAP>(xmap, strides, xmap, strides, shape,
                               center, ellipse_radius, edge_size, float33_t{}, false, stream);
            }

            cpu::memory::PtrHost<int64_t> offsets(shape[0]);
            math::find(noa::math::first_max_t{}, xmap, strides, shape, offsets.share(), true, true, stream);

            const auto shape_3d = safe_cast<long3_t>(dim3_t(shape.get(1)));
            const auto strides_3d = safe_cast<long3_t>(dim3_t(strides.get(1)));
            for (dim_t batch = 0; batch < shape[0]; ++batch) {
                const long3_t peak = indexing::indexes(offsets[batch], strides_3d, shape_3d);
                const T* imap = xmap.get() + strides[0] * batch;
                peaks.get()[batch] = subpixelRegistration3DParabola<REMAP>(
                        imap, strides_3d, shape_3d, peak, registration_radius);
            }
        });
    }

    template<Remap REMAP, typename T, typename>
    float3_t xpeak3D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape,
                     float3_t ellipse_radius, long3_t registration_radius, Stream& stream) {
        if (shape[0] != 1) // throw instead of assert because this could result in segfault
            NOA_THROW("This overload does not supported batched arrays, but got {} batches", shape[0]);

        float3_t peak;
        const shared_t<float3_t[]> peak_ptr(xmap, &peak);
        xpeak3D<REMAP>(xmap, strides, shape, peak_ptr, ellipse_radius, registration_radius, stream);
        stream.synchronize();
        return peak;
    }

    #define INSTANTIATE_XPEAK(R, T) \
    template void xpeak1D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float[]>&, float, int64_t, Stream&);         \
    template void xpeak2D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float2_t[]>&, float2_t, long2_t, Stream&);   \
    template void xpeak3D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float3_t[]>&, float3_t, long3_t, Stream&);   \
    template float xpeak1D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float, int64_t, Stream&);                                  \
    template float2_t xpeak2D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float2_t, long2_t, Stream&);                            \
    template float3_t xpeak3D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float3_t, long3_t, Stream&)

    #define INSTANTIATE_XPEAK_ALL(T)    \
    INSTANTIATE_XPEAK(Remap::F2F, T);   \
    INSTANTIATE_XPEAK(Remap::FC2FC, T)

    INSTANTIATE_XPEAK_ALL(float);
    INSTANTIATE_XPEAK_ALL(double);
}
