/// \file noa/cpu/memory/Remap.h
/// \brief Remapping functions.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

namespace noa::memory {
    // -- Using center coordinates -- //

    /// Extracts from the input array one or multiple subregions (with the same shape) at variable locations.
    /// \tparam T                       (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] input                Input array to use for the extraction.
    /// \param input_shape              Physical {fast, medium, slow} shape of \a input.
    /// \param[out] subregions          Output subregions. One per subregion.
    /// \param subregion_shape          Physical {fast, medium, slow} shape of one subregion.
    /// \param[in] subregion_centers    Indexes, corresponding to \a input_shape and starting from 0, defining the
    ///                                 center of the subregions to extract. One per subregion.
    /// \param subregion_count          Number of subregions. Should correspond to \a subregion_centers.
    /// \param border_mode              Border mode applied to the elements falling out of the input bounds.
    ///                                 Should be BORDER_NOTHING, BORDER_ZERO or BORDER_VALUE.
    /// \param border_value             Border value. Only used if \a border_mode == BORDER_VALUE.
    ///
    /// \throw If \a border_mode is not BORDER_NOTHING, BORDER_ZERO or BORDER_VALUE.
    /// \note  \a input == \a output is not valid.
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
    /// \param[in] subregions           Subregion(s) to insert into \a output. One per subregion.
    /// \param subregion_shape          Physical {fast, medium, slow} shape one subregion.
    /// \param[in] subregion_centers    Indexes, corresponding to \a output_shape and starting from 0, defining the
    ///                                 center of the subregions to insert. One per subregion.
    /// \param subregion_count          Number of subregions. Should correspond to \a subregion_centers.
    /// \param[out] output              Output array.
    /// \param output_shape             Physical {fast, medium, slow} shape of \a output.
    ///
    /// \note The subregions can be (partially or entirely) out of the \a output bounds.
    /// \note \a subregions == \a output is not valid.
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
    /// \param[in] subregions           Array of pointers to the subregions to insert into \a output.
    ///                                 Should be at least \a subregion_count pointers.
    /// \param subregion_shape          Physical {fast, medium, slow} shape one subregion.
    /// \param[in] subregion_centers    Indexes, corresponding to \a output_shape and starting from 0, defining the
    ///                                 center of the subregions to insert. One per subregion.
    /// \param subregion_count          Number of subregions. Should correspond to \a subregion_centers.
    /// \param[out] output              Output array.
    /// \param output_shape             Physical {fast, medium, slow} shape of \a output.
    ///
    /// \note The subregions can be (partially or entirely) out of the \a output bounds.
    /// \note \a subregions == \a output is not valid.
    /// \note This function assumes no overlap between subregions. Overlapped elements should be considered UB.
    template<typename T>
    NOA_HOST void insert(const T** subregions, size3_t subregion_shape,
                         const size3_t* subregion_centers, uint subregion_count,
                         T* output, size3_t output_shape);

    /// Gets the atlas layout (shape + subregion centers).
    /// \param subregion_shape          Physical shape of the subregions.
    /// \param subregion_count          Number of subregions to place into the atlas.
    /// \param[out] o_subregion_centers Subregion centers, relative to the output atlas shape.
    /// \return                         Atlas shape.
    ///
    /// \details The shape of the atlas is not necessary a square. For instance, with 4 subregions the atlas layout
    ///          is `2x2`, but with 5 subregions is goes to `3x2` with one empty region. Subregions are ordered from
    ///          the corner left and step through the atlas in an "inverse Z".
    NOA_HOST size3_t getAtlasLayout(size3_t subregion_shape, uint subregion_count, size3_t* o_subregion_centers);

    // -- Using a map (indexes) -- //

    /// Extracts the linear indexes where \a mask > \a threshold. These indexes are referred to as a map.
    /// \tparam T           (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] mask     Mask to read
    /// \param elements     Size of \a mask, in elements.
    /// \param threshold    Threshold value. Elements greater than is value are added to the map.
    /// \return             1: the map. The calling scope becomes the owner of this map.
    ///                     2: the size of the map, in elements.
    template<typename T>
    NOA_HOST std::pair<size_t*, size_t> getMap(const T* mask, size_t elements, T threshold);

    /// Extracts elements from the input array(s) into the output array(s) at the indexes saved in the map.
    /// \tparam T                   (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] i_sparse         Input arrays to extract the elements from. One per batch.
    /// \param i_sparse_elements    Size, in elements, of one input array (i.e. size of one batch).
    /// \param[out] o_dense         Output arrays where the extracted elements are saved (in the same order as
    ///                             specified in \a map). One per batch.
    /// \param o_dense_elements     Size, in elements, of one output array, which is also the size of \a i_map.
    /// \param[in] i_map            Indexes corresponding to one input array. The same map is applied to every batch.
    ///                             Should be at least \a o_dense_elements elements.
    /// \param batches              Number of batches in \a i_sparse and \a o_dense.
    template<typename T>
    NOA_HOST void extract(const T* i_sparse, size_t i_sparse_elements, T* o_dense, size_t o_dense_elements,
                          const size_t* i_map, uint batches);

    /// Inserts elements from the input array(s) into the output array(s) at the indexes saved in the map.
    /// \tparam T                   (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] i_dense          Input arrays to insert. One per batch.
    /// \param i_dense_elements     Size, in elements, of one input array, which is also the size of \a map.
    /// \param[out] o_sparse        Output arrays corresponding to \a map.
    /// \param o_sparse_elements    Size, in elements, of one output array.
    /// \param[in] i_map            Indexes of \a o_sparse where the elements in \a i_dense should be inserted.
    ///                             The same map is applied to every batch.
    ///                             Should be at least \a i_dense_elements elements.
    /// \param batches              Number of batches in \a i_dense and \a o_sparse.
    template<typename T>
    NOA_HOST void insert(const T* i_dense, size_t i_dense_elements, T* o_sparse, size_t o_sparse_elements,
                         const size_t* i_map, uint batches);
}
