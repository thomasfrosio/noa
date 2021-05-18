#include "noa/Math.h"
#include "noa/cpu/memory/Atlas.h"
#include "noa/cpu/memory/Remap.h"

namespace Noa::Memory {
    size3_t getAtlasLayout(size3_t shape, uint count, size3_t* o_centers) {
        uint col = static_cast<uint>(Math::ceil(Math::sqrt(static_cast<float>(count))));
        uint row = (count + col - 1) / col;
        size3_t atlas_shape(row * shape.x, col * shape.y, shape.z);
        size3_t half = shape / size_t{2};
        for (uint y = 0; y < row; ++y) {
            for (uint x = 0; x < col; ++x) {
                uint idx = y * col + x;
                o_centers[idx] = {x * shape.x + half.x, y * shape.y + half.y, half.z};
            }
        }
        return atlas_shape;
    }

    template<typename T>
    void insert(const T** inputs, size3_t shape, uint count,
               T* atlas, size3_t atlas_shape, const size3_t* atlas_centers) {
        for (uint idx = 0; idx < count; ++idx)
            insert(inputs[idx], shape, atlas_centers[idx], atlas, atlas_shape);
    }

    #define INSTANTIATE_ATLAS(T) template void insert<T>(const T**, size3_t, uint, T*, size3_t, const size3_t*)
    INSTANTIATE_ATLAS(float);
    INSTANTIATE_ATLAS(double);
    INSTANTIATE_ATLAS(short);
    INSTANTIATE_ATLAS(int);
    INSTANTIATE_ATLAS(long);
    INSTANTIATE_ATLAS(long long);
    INSTANTIATE_ATLAS(unsigned short);
    INSTANTIATE_ATLAS(unsigned int);
    INSTANTIATE_ATLAS(unsigned long);
    INSTANTIATE_ATLAS(unsigned long long);
}
