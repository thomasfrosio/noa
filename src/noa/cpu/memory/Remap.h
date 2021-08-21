/// \file noa/cpu/memory/Remap.h
/// \brief Remapping functions.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

// -- Using center coordinates -- //
namespace noa::cpu::memory {
    /// Extracts from the input array one or multiple subregions (with the same shape) at variable locations.
    /// \tparam T                       (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] input                On the \b host. Input array to use for the extraction.
    /// \param input_shape              Physical {fast, medium, slow} shape of \p input.
    /// \param[out] subregions          On the \b host. Output subregions. One per subregion.
    /// \param subregion_shape          Physical {fast, medium, slow} shape of one subregion.
    /// \param[in] subregion_centers    On the \b host. Indexes, corresponding to \p input_shape and starting from 0,
    ///                                 defining the center of the subregions to extract. One per subregion.
    /// \param subregion_count          Number of subregions.
    /// \param border_mode              Border mode used for OOB conditions.
    ///                                 Can be BORDER_NOTHING, BORDER_ZERO, BORDER_VALUE, BORDER_CLAMP, BORDER_MIRROR or BORDER_REFLECT.
    /// \param border_value             Constant value to use for OOB. Only used if \p border_mode is BORDER_VALUE.
    ///
    /// \throw If \p border_mode is not BORDER_NOTHING, BORDER_ZERO or BORDER_VALUE.
    /// \note  \p input == \p output is not valid.
    template<typename T>
    NOA_HOST void extract(const T* input, size3_t input_shape,
                          T* subregions, size3_t subregion_shape,
                          const size3_t* subregion_centers, uint subregion_count,
                          BorderMode border_mode, T border_value);

    /// Extracts a subregion from the input array.
    template<typename T>
    NOA_IH void extract(const T* input, size3_t input_shape,
                        T* subregion, size3_t subregion_shape, size3_t subregion_center,
                        BorderMode border_mode, T border_value) {
        extract(input, input_shape, subregion, subregion_shape, &subregion_center, 1, border_mode, border_value);
    }

    /// Inserts into the output array one or multiple subregions (with the same shape) at variable locations.
    /// \tparam T                       (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] subregions           On the \b host. Subregion(s) to insert into \p output. One per subregion.
    /// \param subregion_shape          Physical {fast, medium, slow} shape one subregion.
    /// \param[in] subregion_centers    On the \b host. Indexes, corresponding to \p output_shape and starting from 0,
    ///                                 defining the center of the subregions to insert. One per subregion.
    /// \param subregion_count          Number of subregions. Should correspond to \p subregion_centers.
    /// \param[out] output              On the \b host. Output array.
    /// \param output_shape             Physical {fast, medium, slow} shape of \p output.
    ///
    /// \note The subregions can be (partially or entirely) out of the \p output bounds.
    /// \note \p subregions == \p output is not valid.
    /// \note This function assumes no overlap between subregions. Overlapped elements should be considered UB.
    template<typename T>
    NOA_HOST void insert(const T* subregions, size3_t subregion_shape,
                         const size3_t* subregion_centers, uint subregion_count,
                         T* output, size3_t output_shape);

    /// Inserts a subregion into the input array.
    template<typename T>
    NOA_IH void insert(const T* subregion, size3_t subregion_shape, size3_t subregion_center,
                       T* output, size3_t output_shape) {
        insert(subregion, subregion_shape, &subregion_center, 1, output, output_shape);
    }

    /// Inserts into the output array one or multiple subregions (with the same shape) at variable locations.
    /// \tparam T                       (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] subregions           On the \b host. Array of pointers to the subregions to insert into \p output.
    ///                                 Should be at least \p subregion_count pointers.
    /// \param subregion_shape          Physical {fast, medium, slow} shape one subregion.
    /// \param[in] subregion_centers    On the \b host. Indexes, corresponding to \p output_shape and starting from 0,
    ///                                 defining the center of the subregions to insert. One per subregion.
    /// \param subregion_count          Number of subregions. Should correspond to \p subregion_centers.
    /// \param[out] output              On the \b host. Output array.
    /// \param output_shape             Physical {fast, medium, slow} shape of \p output.
    ///
    /// \note The subregions can be (partially or entirely) out of the \p output bounds.
    /// \note \p subregions == \p output is not valid.
    /// \note This function assumes no overlap between subregions. Overlapped elements should be considered UB.
    template<typename T>
    NOA_HOST void insert(const T** subregions, size3_t subregion_shape,
                         const size3_t* subregion_centers, uint subregion_count,
                         T* output, size3_t output_shape);

    /// Gets the atlas layout (shape + subregion centers).
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
namespace noa::cpu::memory {
    /// Extracts the linear indexes where the values in \p input are larger than \p threshold.
    /// These indexes are referred to as a map.
    ///
    /// \tparam I           (u)int, (u)long, (u)long long.
    /// \tparam T           (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] input    On the \b host. Contiguous input array.
    /// \param elements     Size of \p input, in elements.
    /// \param threshold    Threshold value. Elements greater than is value are added to the map.
    /// \return             1: the map. The calling scope is the owner; use PtrHost<size_t>::dealloc() to free it.
    ///                     2: the size of the map, in elements.
    template<typename I = size_t, typename T>
    NOA_HOST std::pair<I*, size_t> getMap(const T* input, size_t elements, T threshold);

    /// Extracts the linear indexes where the values in \p input are larger than \p threshold.
    /// These indexes are referred to as a map.
    ///
    /// \tparam I           (u)int, (u)long, (u)long long.
    /// \tparam T           (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] input    On the \b host. Input array.
    /// \param pitch        Pitch, in elements, of \p input.
    /// \param shape        Logical {fast, medium, slow} shape of \p input.
    /// \param threshold    Threshold value. Elements greater than is value are added to the map.
    /// \return             1: the map. The calling scope is the owner; use PtrHost<size_t>::dealloc() to free it.
    ///                     2: the size of the map, in elements.
    template<typename I = size_t, typename T>
    NOA_HOST std::pair<I*, size_t> getMap(const T* input, size_t pitch, size3_t shape, T threshold);

    /// Extracts elements from the input array(s) into the output array(s), at the indexes saved in the map.
    /// \tparam I               (u)int, (u)long, (u)long long.
    /// \tparam T               (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] sparse       On the \b host. Input arrays to extract the elements from. One per batch.
    /// \param sparse_elements  Size, in elements, of \p sparse, ignoring the batches.
    /// \param[out] dense       On the \b host. Output arrays where the extracted elements are saved in
    ///                         the same order as specified in \p map). One per batch.
    /// \param dense_elements   Size, in elements, of \p dense and \p map, ignoring the batches.
    /// \param[in] map          On the \b host. Indexes corresponding to one input array.
    ///                         The same map is applied to every batch.
    /// \param batches          Number of batches to compute.
    template<typename I = size_t, typename T>
    NOA_HOST void extract(const T* sparse, size_t sparse_elements, T* dense, size_t dense_elements,
                          const I* map, uint batches);

    /// Inserts elements from the input array(s) into the output array(s) at the indexes saved in the map.
    /// \tparam I               (u)int, (u)long, (u)long long.
    /// \tparam T               (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] dense        On the \b host. Input arrays to insert in \p sparse. One per batch.
    /// \param dense_elements   Size, in elements, of \p dense and \p map, ignoring the batches.
    /// \param[out] sparse      On the \b host. Output arrays corresponding to \p map. On per batch.
    /// \param sparse_elements  Size, in elements, of \p sparse, ignoring the batches.
    /// \param[in] map          On the \b host. Indexes of \p sparse where the elements in \p dense should be inserted.
    ///                         The same map is applied to every batch.
    /// \param batches          Number of batches in \p dense and \p sparse.
    template<typename I = size_t, typename T>
    NOA_HOST void insert(const T* dense, size_t dense_elements, T* sparse, size_t sparse_elements,
                         const I* map, uint batches);
}
