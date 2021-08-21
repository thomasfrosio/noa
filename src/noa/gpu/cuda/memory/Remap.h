/// \file noa/gpu/cuda/memory/Remap.h
/// \brief Remapping functions.
/// \author Thomas - ffyr2w
/// \date 05 Jan 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

// -- Using center coordinates -- //
namespace noa::cuda::memory {
    /// Extracts from the input array one or multiple subregions at variable locations.
    /// \tparam T                       (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] input                On the \b device. Input array to use for the extraction.
    /// \param input_pitch              Pitch, in elements, of \p input.
    /// \param input_shape              Physical {fast, medium, slow} shape of \p input.
    /// \param[out] subregions          On the \b device. Output subregions.
    /// \param subregions_pitch         Pitch, in elements, of \p subregions.
    /// \param subregion_shape          Physical {fast, medium, slow} shape of one subregion.
    /// \param[in] subregion_centers    On the \b device. One per subregion.
    ///                                 Center of the subregions, corresponding to \p input_shape.
    /// \param subregion_count          Number of subregions.
    /// \param border_mode              Border mode used for OOB conditions.
    ///                                 Can be BORDER_NOTHING, BORDER_ZERO, BORDER_VALUE, BORDER_CLAMP, BORDER_MIRROR or BORDER_REFLECT.
    /// \param border_value             Constant value to use for OOB. Only used if \p border_mode is BORDER_VALUE.
    /// \param[in,out] stream           Stream on which to enqueue this function.
    ///
    /// \note  \p input and \p subregions should not overlap.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void extract(const T* input, size_t input_pitch, size3_t input_shape,
                          T* subregions, size_t subregion_pitch, size3_t subregion_shape,
                          const size3_t* subregion_centers, uint subregion_count,
                          BorderMode border_mode, T border_value, Stream& stream);

    /// Extracts from the input array one or multiple subregions at variable locations. Contiguous version.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void extract(const T* input, size3_t input_shape,
                        T* subregions, size3_t subregion_shape, const size3_t* subregion_centers, uint subregion_count,
                        BorderMode border_mode, T border_value, Stream& stream) {
        extract(input, input_shape.x, input_shape,
                subregions, subregion_shape.x, subregion_shape, subregion_centers, subregion_count,
                border_mode, border_value, stream);
    }

    /// Extracts a subregion from the input array.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void extract(const T* input, size_t input_pitch, size3_t input_shape,
                          T* subregion, size_t subregion_pitch, size3_t subregion_shape, size3_t subregion_center,
                          BorderMode border_mode, T border_value, Stream& stream);

    /// Extracts a subregion from the input array. Contiguous version.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void extract(const T* input, size3_t input_shape,
                        T* subregion, size3_t subregion_shape, size3_t subregion_center,
                        BorderMode border_mode, T border_value, Stream& stream) {
        extract(input, input_shape.x, input_shape,
                subregion, subregion_shape.x, subregion_shape, subregion_center,
                border_mode, border_value, stream);
    }

    /// Inserts into the output array one or multiple subregions (with the same shape) at variable locations.
    /// \tparam T                    (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] subregions        On the \b device. Subregion(s) to insert into \p output. One per \p subregion_count.
    /// \param subregion_pitch       Pitch of \p subregions, in elements.
    /// \param subregion_shape       Physical {fast, medium, slow} shape one subregion.
    /// \param[in] subregion_centers On the \b device. Indexes, corresponding to \p output_shape and starting from 0,
    ///                              defining the center of the subregions to insert. One per subregion.
    /// \param subregion_count       Number of subregions. Should correspond to \p subregion_centers.
    /// \param[out] output           On the \b device. Output array.
    /// \param output_pitch          Pitch of \p output, in elements.
    /// \param output_shape          Physical {fast, medium, slow} shape of \p output.
    /// \param[in,out] stream        Stream on which to enqueue this function.
    ///
    /// \note The subregions can be (partially or entirely) out of the \p output bounds.
    /// \note  \p output and \p subregions should not overlap.
    /// \note This function assumes no overlap between subregions. Overlapped elements should be considered UB.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void insert(const T* subregions, size_t subregion_pitch, size3_t subregion_shape,
                         const size3_t* subregion_centers, uint subregion_count,
                         T* output, size_t output_pitch, size3_t output_shape,
                         Stream& stream);

    /// Inserts into the output array one or multiple subregions (with the same shape) at variable locations. Contiguous version.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void insert(const T* subregions, size3_t subregion_shape,
                       const size3_t* subregion_centers, uint subregion_count,
                       T* output, size3_t output_shape,
                       Stream& stream) {
        insert(subregions, subregion_shape.x, subregion_shape, subregion_centers, subregion_count,
               output, output_shape.x, output_shape,
               stream);
    }

    /// Inserts one subregion into the output array one subregion.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void insert(const T* subregion, size_t subregion_pitch, size3_t subregion_shape, size3_t subregion_center,
                         T* output, size_t output_pitch, size3_t output_shape, Stream& stream);

    /// Inserts one subregion into the output array one subregion. Contiguous version.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void insert(const T* subregion, size3_t subregion_shape, size3_t subregion_center,
                       T* output, size3_t output_shape, Stream& stream) {
        insert(subregion, subregion_shape.x, subregion_shape, subregion_center,
               output, output_shape.x, output_shape, stream);
    }

    /// Gets the atlas layout (shape + subregion centers). This is identical to the CPU version.
    /// \param subregion_shape          Physical shape of the subregions.
    /// \param subregion_count          Number of subregions to place into the atlas.
    /// \param[out] o_subregion_centers On the \b host. Subregion centers, relative to the output atlas shape.
    ///                                 The center is defined as `N / 2`.
    /// \return                         Atlas shape.
    ///
    /// \details The shape of the atlas is not necessary a square. For instance, with 4 subregions the atlas layout
    ///          is `2x2`, but with 5 subregions is goes to `3x2` with one empty region. Subregions are ordered from
    ///          the corner left and step through the atlas in an "inverse Z" (this is when the origin is at the bottom).
    NOA_HOST size3_t getAtlasLayout(size3_t subregion_shape, uint subregion_count, size3_t* o_subregion_centers);
}

// -- Using a map (indexes) -- //
namespace noa::cuda::memory {
    /// Extracts the linear indexes where the values in \p input are larger than \p threshold.
    /// These indexes are referred to as a map.
    ///
    /// \tparam I               (u)int, (u)long, (u)long long.
    /// \tparam T               (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] input        On the \b device. Mask to read.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param input_shape      Logical {fast, medium, slow} shape of \p input.
    /// \param threshold        Threshold value. Elements greater than is value are added to the map.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream will be synchronized when the function returns.
    /// \return                 1: the map. The calling scope is the owner; use PtrDevice<size_t>::dealloc() to free it.
    ///                         2: the size of the map, in elements.
    template<typename I = size_t, typename T>
    NOA_HOST std::pair<I*, size_t> getMap(const T* input, size_t input_pitch, size3_t input_shape,
                                          T threshold, Stream& stream);

    /// Extracts the linear indexes where the values in \p input are larger than \p threshold.
    /// \note This overload is for padded layout and is otherwise identical to the overload for contiguous layout.
    template<typename I = size_t, typename T>
    NOA_HOST std::pair<I*, size_t> getMap(const T* input, size_t elements, T threshold, Stream& stream);

    /// Extracts elements from the input array(s) into the output array(s), at the indexes saved in the map.
    /// \tparam I               (u)int, (u)long, (u)long long.
    /// \tparam T               (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] sparse       On the \b device. Input arrays to extract the elements from. One per batch.
    /// \param sparse_elements  Size, in elements, of \p sparse, ignoring the batches.
    /// \param[out] dense       On the \b device. Output arrays where the extracted elements are saved in
    ///                         the same order as specified in \p map). One per batch.
    /// \param dense_elements   Size, in elements, of \p dense and \p map, ignoring the batches.
    /// \param[in] map          On the \b device. Indexes corresponding to one input array.
    ///                         The same map is applied to every batch.
    /// \param batches          Number of batches to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T, typename I>
    NOA_HOST void extract(const T* sparse, size_t sparse_elements, T* dense, size_t dense_elements,
                          const I* map, uint batches, Stream& stream);

    /// Inserts elements from the input array(s) into the output array(s) at the indexes saved in the map.
    /// \tparam I               (u)int, (u)long, (u)long long.
    /// \tparam T               (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] dense        On the \b device. Input arrays to insert in \p sparse. One per batch.
    /// \param dense_elements   Size, in elements, of \p dense and \p map, ignoring the batches.
    /// \param[out] sparse      On the \b device. Output arrays corresponding to \p map. On per batch.
    /// \param sparse_elements  Size, in elements, of \p sparse, ignoring the batches.
    /// \param[in] map          On the \b device. Indexes of \p sparse where the elements in \p dense should be inserted.
    ///                         The same map is applied to every batch.
    /// \param batches          Number of batches in \p dense and \p sparse.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T, typename I>
    NOA_HOST void insert(const T* dense, size_t dense_elements, T* sparse, size_t sparse_elements,
                         const I* map, uint batches, Stream& stream);
}
