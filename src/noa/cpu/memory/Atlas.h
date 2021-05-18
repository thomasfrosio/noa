#pragma once

#include "noa/Definitions.h"
#include "noa/Types.h"

namespace Noa::Memory {
    /**
     * Gets the atlas layout (shape + subregion centers).
     * @param shape             Physical shape of the subregions.
     * @param count             Number of subregions to place into the atlas.
     * @param[out] o_centers    Subregion centers, relative to the output atlas shape.
     * @return                  Atlas shape.
     */
    NOA_HOST size3_t getAtlasLayout(size3_t shape, uint count, size3_t* o_centers);

    /**
     * Inserts the subregions into the atlas.
     * @tparam T                    (u)short, (u)int, (u)long, (u)long long, float, double.
     * @param[in] subregions        Arrays of pointers to the subregions to insert into @a atlas.
     *                              Should be at least @a count pointers.
     * @param subregion_shape       Physical shape of the subregions in @a subregions.
     * @param subregion_count       Number of subregions.
     * @param[out] atlas            Atlas array.
     * @param atlas_shape           Atlas shape.
     * @param[out] atlas_centers    Subregion centers, relative to @a atlas_shape.
     */
    template<typename T>
    NOA_HOST void insert(const T** subregions, size3_t subregion_shape, uint subregion_count,
                         T* atlas, size3_t atlas_shape, const size3_t* atlas_centers);

    /**
     * Inserts the subregions into the atlas. Batched version.
     * @tparam T                    (u)short, (u)int, (u)long, (u)long long, float, double.
     * @param[in] subregions        Arrays of pointers to the subregions to insert into @a atlas.
     *                              Should be at least @a count pointers.
     * @param subregion_shape       Physical shape of the subregions in @a subregions.
     * @param subregion_count       Number of subregions.
     * @param[out] atlas            Atlas array.
     * @param atlas_shape           Atlas shape.
     * @param[out] atlas_centers    Subregion centers, relative to @a atlas_shape.
     */
    template<typename T>
    NOA_IH void insert(const T* subregions, size3_t subregion_shape, uint subregion_count,
                       T* atlas, size3_t atlas_shape, const size3_t* atlas_centers) {
        std::unique_ptr<const T* []> tmp = std::make_unique<const T* []>(subregion_count);
        size_t elements = getElements(subregion_shape);
        for (uint i = 0; i < subregion_count; ++i)
            tmp[i] = subregions + i * elements;
        insert(tmp.get(), subregion_shape, subregion_count, atlas, atlas_shape, atlas_centers);
    }
}
