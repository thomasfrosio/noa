#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace Noa::CUDA::Filter {
    /// Computes the median filter using a 1D window.
    /// \tparam T               int, uint, float or double
    /// \param[in] in           Input array with data to filter. One per batch.
    /// \param in_pitch         Pitch, in elements, of \a in.
    /// \param[out] out         Output array where the filtered data is stored. One per batch.
    /// \param out_pitch        Pitch, in elements, of \a out.
    /// \param shape            Shape {fast, medium, slow} of \a in and \a out (excluding the batch), in elements.
    /// \param batches          Number of batches.
    /// \param border_mode      Border mode used for the "implicit padding". Either BORDER_ZERO, or BORDER_MIRROR.
    /// \param window           Number of elements to consider for the computation of the median.
    ///                         This corresponds to the first dimension of \a shape.
    ///                         Only odd numbers from 1 to 21 are supported. If 1, a copy is performed.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function runs asynchronously relative to the host and may return before completion.
    /// \note If \a border_mode is BORDER_MIRROR, \a shape.x should be `>= window/2 + 1`.
    /// \note \a in and \a out should not overlap.
    /// \throw If \a border_mode or \a window is not supported.
    template<typename T>
    NOA_HOST void median1(const T* in, size_t in_pitch, T* out, size_t out_pitch, size3_t shape, uint batches,
                           BorderMode border_mode, uint window, Stream& stream);

    /// Computes the median filter using a 1D window. Version for contiguous layouts.
    template<typename T>
    NOA_IH void median1(const T* in, T* out, size3_t shape, uint batches,
                         BorderMode border_mode, uint window, Stream& stream) {
        median1(in, shape.x, out, shape.x, shape, batches, border_mode, window, stream);
    }

    /// Computes the median filter using a 2D square window.
    /// \tparam T               int, uint, float or double
    /// \param[in] in           Input array with data to filter. One per batch.
    /// \param in_pitch         Pitch, in elements, of \a in.
    /// \param[out] out         Output array where the filtered data is stored. One per batch.
    /// \param out_pitch        Pitch, in elements, of \a out.
    /// \param shape            Shape {fast, medium, slow} of \a in and \a out (excluding the batch), in elements.
    /// \param batches          Number of batches.
    /// \param border_mode      Border mode used for the "implicit padding". Either BORDER_ZERO, or BORDER_MIRROR.
    /// \param window           Number of elements to consider for the computation of the median, for each dimension.
    ///                         This corresponds to the first and second dimension of \a shape.
    ///                         Only odd numbers from 1 to 11 are supported. If 1, no filter is applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function runs asynchronously relative to the host and may return before completion.
    /// \note If \a border_mode is BORDER_MIRROR, \a shape.x and \a shape.y should be `>= window/2 + 1`.
    /// \note \a in and \a out should not overlap.
    /// \throw If \a border_mode or \a window is not supported.
    template<typename T>
    NOA_HOST void median2(const T* in, size_t in_pitch, T* out, size_t out_pitch, size3_t shape, uint batches,
                           BorderMode border_mode, uint window, Stream& stream);

    /// Computes the median filter using a 2D square window. Version for contiguous layouts.
    template<typename T>
    NOA_IH void median2(const T* in, T* out, size3_t shape, uint batches,
                         BorderMode border_mode, uint window, Stream& stream) {
        median2(in, shape.x, out, shape.x, shape, batches, border_mode, window, stream);
    }

    /// Computes the median filter using a 3D cubic window.
    /// \tparam T               int, uint, float or double
    /// \param[in] in           Input array with data to filter. One per batch.
    /// \param in_pitch         Pitch, in elements, of \a in.
    /// \param[out] out         Output array where the filtered data is stored. One per batch.
    /// \param out_pitch        Pitch, in elements, of \a out.
    /// \param shape            Shape {fast, medium, slow} of \a in and \a out (excluding the batch), in elements.
    /// \param batches          Number of batches.
    /// \param border_mode      Border mode used for the "implicit padding". Either BORDER_ZERO, or BORDER_MIRROR.
    /// \param window           Number of elements to consider for the computation of the median, for each dimension.
    ///                         Only odd numbers from 1 to 5 are supported. If 1, no filter is applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function runs asynchronously relative to the host and may return before completion.
    /// \note If \a border_mode is BORDER_MIRROR, `all(shape >= window/2 + 1)`.
    /// \note \a in and \a out should not overlap.
    /// \throw If \a border_mode or \a window is not supported.
    template<typename T>
    NOA_HOST void median3(const T* in, size_t in_pitch, T* out, size_t out_pitch, size3_t shape, uint batches,
                           BorderMode border_mode, uint window, Stream& stream);

    /// Computes the median filter using a 3D cubic window. Version for contiguous layouts.
    template<typename T>
    NOA_IH void median3(const T* in, T* out, size3_t shape, uint batches,
                         BorderMode border_mode, uint window, Stream& stream) {
        median3(in, shape.x, out, shape.x, shape, batches, border_mode, window, stream);
    }
}
