#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Math.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::transform::bspline::details {
    template<typename T>
    NOA_HOST void prefilter1D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                              size_t shape, size_t batches);

    template<typename T>
    NOA_HOST void prefilter2D(const T* inputs, size2_t input_pitch, T* outputs, size2_t output_pitch,
                              size2_t shape, size_t batches, size_t threads);

    template<typename T>
    NOA_HOST void prefilter3D(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                              size3_t shape, size_t batches, size_t threads);
}

namespace noa::cpu::transform::bspline {
    /// Applies a 1D prefilter to \a inputs so that the cubic B-spline values will pass through the sample data.
    /// \tparam T               float, double, cfloat_t or cdouble_t.
    /// \param[in] inputs       On the \b host.Input arrays. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Output arrays. One per batch. Can be equal to \a inputs.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param size             Size, in elements, of \a inputs and \a outputs, of one batch.
    /// \param batches          Number of batches in \a inputs and \a outputs.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \see http://www2.cs.uregina.ca/~anima/408/Notes/Interpolation/UniformBSpline.htm
    /// \see http://www.dannyruijters.nl/cubicinterpolation/
    template<typename T>
    NOA_IH void prefilter1D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                            size_t shape, size_t batches, Stream& stream) {
        stream.enqueue(details::prefilter1D<T>, inputs, input_pitch, outputs, output_pitch, shape, batches);
    }

    /// Applies a 2D prefilter to \a inputs so that the cubic B-spline values will pass through the sample data.
    /// \tparam T               float, double, cfloat_t or cdouble_t.
    /// \param[in] inputs       On the \b host.Input arrays. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Output arrays. One per batch. Can be equal to \a inputs.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium} shape of \a inputs and \a outputs, ignoring the batches.
    /// \param batches          Number of batches in \a inputs and \a outputs.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \see http://www2.cs.uregina.ca/~anima/408/Notes/Interpolation/UniformBSpline.htm
    /// \see http://www.dannyruijters.nl/cubicinterpolation/
    template<typename T>
    NOA_IH void prefilter2D(const T* inputs, size2_t input_pitch, T* outputs, size2_t output_pitch,
                            size2_t shape, size_t batches, Stream& stream) {
        const size_t threads = stream.threads();
        stream.enqueue(details::prefilter2D<T>, inputs, input_pitch, outputs, output_pitch, shape, batches, threads);
    }

    /// Applies a 3D prefilter to \a inputs so that the cubic B-spline values will pass through the sample data.
    /// \tparam T               float, double, cfloat_t or cdouble_t.
    /// \param[in] inputs       On the \b host.Input arrays. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Output arrays. One per batch. Can be equal to \a inputs.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \a inputs and \a outputs, ignoring the batches.
    /// \param batches          Number of batches in \a inputs and \a outputs.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \see http://www2.cs.uregina.ca/~anima/408/Notes/Interpolation/UniformBSpline.htm
    /// \see http://www.dannyruijters.nl/cubicinterpolation/
    template<typename T>
    NOA_IH void prefilter3D(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                              size3_t shape, size_t batches, Stream& stream) {
        const size_t threads = stream.threads();
        stream.enqueue(details::prefilter3D<T>, inputs, input_pitch, outputs, output_pitch, shape, batches, threads);
    }
}
