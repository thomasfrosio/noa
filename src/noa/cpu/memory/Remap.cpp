#include "noa/common/Exception.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/Remap.h"
#include "noa/cpu/memory/Set.h"

namespace {
    using namespace noa;

    inline int3_t getCornerLeft_(int3_t subregion_shape, size3_t subregion_center) {
        return int3_t(subregion_center) - subregion_shape / 2;
    }

    inline size_t getOffset_(int3_t shape, int idx_y, int idx_z) {
        return (static_cast<size_t>(idx_z) * static_cast<size_t>(shape.y) + static_cast<size_t>(idx_y)) *
               static_cast<size_t>(shape.x);
    }

    template<typename T>
    void extractOrNothing_(const T* input, int3_t input_shape,
                           T* subregion, int3_t subregion_shape, int3_t corner_left) {
        for (int o_z = 0; o_z < subregion_shape.z; ++o_z) {
            int i_z = o_z + corner_left.z;
            if (i_z < 0 || i_z >= input_shape.z)
                continue;

            for (int o_y = 0; o_y < subregion_shape.y; ++o_y) {
                int i_y = o_y + corner_left.y;
                if (i_y < 0 || i_y >= input_shape.y)
                    continue;

                size_t i_offset = getOffset_(input_shape, i_y, i_z);
                size_t o_offset = getOffset_(subregion_shape, o_y, o_z);
                for (int o_x = 0; o_x < subregion_shape.x; ++o_x) {
                    int i_x = o_x + corner_left.x;
                    if (i_x < 0 || i_x >= input_shape.x)
                        continue;
                    subregion[o_offset + static_cast<size_t>(o_x)] = input[i_offset + static_cast<size_t>(i_x)];
                }
            }
        }
    }

    template<typename T>
    void extractOrValue_(const T* input, int3_t input_shape,
                         T* subregion, int3_t subregion_shape, int3_t corner_left, T value) {
        for (int o_z = 0; o_z < subregion_shape.z; ++o_z) {
            int i_z = o_z + corner_left.z;
            if (i_z < 0 || i_z >= input_shape.z) {
                T* start = subregion + getOffset_(subregion_shape, 0, o_z);
                cpu::memory::set(start, getElementsSlice(subregion_shape), value);
                continue;
            }
            for (int o_y = 0; o_y < subregion_shape.y; ++o_y) {
                int i_y = o_y + corner_left.y;
                if (i_y < 0 || i_y >= input_shape.y) {
                    T* start = subregion + getOffset_(subregion_shape, o_y, o_z);
                    cpu::memory::set(start, start + subregion_shape.x, value);
                    continue;
                }

                size_t i_offset = getOffset_(input_shape, i_y, i_z);
                size_t o_offset = getOffset_(subregion_shape, o_y, o_z);
                for (int o_x = 0; o_x < subregion_shape.x; ++o_x) {
                    int i_x = o_x + corner_left.x;
                    if (i_x < 0 || i_x >= input_shape.x)
                        subregion[o_offset + static_cast<size_t>(o_x)] = value;
                    else
                        subregion[o_offset + static_cast<size_t>(o_x)] = input[i_offset + static_cast<size_t>(i_x)];
                }
            }
        }
    }

    template<BorderMode MODE, typename T>
    void extract_(const T* input, int3_t input_shape,
                  T* subregion, int3_t subregion_shape, int3_t corner_left) {
        for (int o_z = 0; o_z < subregion_shape.z; ++o_z) {
            int i_z = getBorderIndex<MODE>(o_z + corner_left.z, input_shape.z);
            for (int o_y = 0; o_y < subregion_shape.y; ++o_y) {
                int i_y = getBorderIndex<MODE>(o_y + corner_left.y, input_shape.y);

                size_t i_offset = getOffset_(input_shape, i_y, i_z);
                size_t o_offset = getOffset_(subregion_shape, o_y, o_z);
                for (int o_x = 0; o_x < subregion_shape.x; ++o_x) {
                    int i_x = getBorderIndex<MODE>(o_x + corner_left.x, input_shape.x);
                    subregion[o_offset + static_cast<size_t>(o_x)] = input[i_offset + static_cast<size_t>(i_x)];
                }
            }
        }
    }

    template<typename T>
    void insert_(const T* subregion, int3_t subregion_shape, T* output, int3_t output_shape, int3_t corner_left) {
        for (int i_z = 0; i_z < subregion_shape.z; ++i_z) {
            int o_z = i_z + corner_left.z;
            if (o_z < 0 || o_z >= output_shape.z)
                continue;

            for (int i_y = 0; i_y < subregion_shape.y; ++i_y) {
                int o_y = i_y + corner_left.y;
                if (o_y < 0 || o_y >= output_shape.y)
                    continue;

                size_t i_offset = getOffset_(subregion_shape, i_y, i_z);
                size_t o_offset = getOffset_(output_shape, o_y, o_z);
                for (int i_x = 0; i_x < subregion_shape.x; ++i_x) {
                    int o_x = i_x + corner_left.x;
                    if (o_x < 0 || o_x >= output_shape.x)
                        continue;
                    output[o_offset + static_cast<size_t>(o_x)] = subregion[i_offset + static_cast<size_t>(i_x)];
                }
            }
        }
    }
}

namespace noa::cpu::memory {
    template<typename T>
    void extract(const T* input, size3_t input_shape,
                 T* subregions, size3_t subregion_shape, const size3_t* subregion_centers, uint subregion_count,
                 BorderMode border_mode, T border_value) {
        int3_t i_shape(input_shape);
        int3_t o_shape(subregion_shape);
        size_t elements = getElements(subregion_shape);

        for (uint idx = 0; idx < subregion_count; ++idx) {
            int3_t corner_left = getCornerLeft_(o_shape, subregion_centers[idx]);

            switch (border_mode) {
                case BORDER_NOTHING:
                    extractOrNothing_(input, i_shape, subregions + idx * elements, o_shape, corner_left);
                    break;
                case BORDER_ZERO:
                    extractOrValue_(input, i_shape, subregions + idx * elements, o_shape,
                                    corner_left, static_cast<T>(0));
                    break;
                case BORDER_VALUE:
                    extractOrValue_(input, i_shape, subregions + idx * elements, o_shape,
                                    corner_left, border_value);
                    break;
                case BORDER_CLAMP:
                    extract_<BORDER_CLAMP>(input, i_shape, subregions + idx * elements, o_shape, corner_left);
                    break;
                case BORDER_MIRROR:
                    extract_<BORDER_MIRROR>(input, i_shape, subregions + idx * elements, o_shape, corner_left);
                    break;
                case BORDER_REFLECT:
                    extract_<BORDER_REFLECT>(input, i_shape, subregions + idx * elements, o_shape, corner_left);
                    break;
                default:
                    NOA_THROW("Border mode {} is not supported", border_mode);
            }
        }
    }

    template<typename T>
    void insert(const T* subregions, size3_t subregion_shape, const size3_t* subregion_centers, uint subregion_count,
                T* output, size3_t output_shape) {
        int3_t i_shape(subregion_shape);
        int3_t o_shape(output_shape);
        size_t elements = getElements(subregion_shape);

        for (uint idx = 0; idx < subregion_count; ++idx) {
            int3_t corner_left = getCornerLeft_(i_shape, subregion_centers[idx]);
            insert_(subregions + idx * elements, i_shape, output, o_shape, corner_left);
        }
    }

    template<typename T>
    void insert(const T** subregions, size3_t subregion_shape, const size3_t* subregion_centers, uint subregion_count,
                T* output, size3_t output_shape) {
        int3_t i_shape(subregion_shape);
        int3_t o_shape(output_shape);

        for (uint idx = 0; idx < subregion_count; ++idx) {
            int3_t corner_left = getCornerLeft_(i_shape, subregion_centers[idx]);
            insert_(subregions[idx], i_shape, output, o_shape, corner_left);
        }
    }

    // TODO When noa::Vector<> will be created, use and return it directly...
    template<typename I, typename T>
    std::pair<I*, size_t> getMap(const T* input, size_t elements, T threshold) {
        std::vector<I> tmp_map;
        tmp_map.reserve(1000);
        for (size_t idx = 0; idx < elements; ++idx)
            if (input[idx] > threshold)
                tmp_map.emplace_back(static_cast<I>(idx));
        PtrHost<I> map(tmp_map.size()); // we cannot release std::vector...
        copy(tmp_map.data(), map.get(), tmp_map.size());
        return {map.release(), tmp_map.size()};
    }

    template<typename I, typename T>
    std::pair<I*, size_t> getMap(const T* input, size_t pitch, size3_t shape, T threshold) {
        std::vector<I> tmp_map;
        tmp_map.reserve(1000);
        for (size_t z = 0; z < shape.z; ++z) {
            size_t tmp = z * shape.y * pitch;
            for (size_t y = 0; y < shape.y; ++y) {
                size_t offset = tmp + y * pitch;
                for (size_t x = 0; x < shape.x; ++x) {
                    size_t idx = offset + x;
                    if (input[idx] > threshold)
                        tmp_map.emplace_back(static_cast<I>(idx));
                }
            }
        }
        PtrHost<I> map(tmp_map.size());
        copy(tmp_map.data(), map.get(), tmp_map.size());
        return {map.release(), tmp_map.size()};
    }

    template<typename I, typename T>
    void extract(const T* i_sparse, size_t i_sparse_elements, T* o_dense, size_t o_dense_elements,
                 const I* i_map, uint batches) {
        for (uint batch = 0; batch < batches; ++batch) {
            const T* input = i_sparse + batch * i_sparse_elements;
            T* output = o_dense + batch * o_dense_elements;
            for (size_t idx = 0; idx < o_dense_elements; ++idx)
                output[idx] = input[i_map[idx]];
        }
    }

    template<typename I, typename T>
    void insert(const T* i_dense, size_t i_dense_elements, T* o_sparse, size_t o_sparse_elements,
                const I* i_map, uint batches) {
        for (uint batch = 0; batch < batches; ++batch) {
            const T* input = i_dense + batch * i_dense_elements;
            T* output = o_sparse + batch * o_sparse_elements;
            for (size_t idx = 0; idx < i_dense_elements; ++idx)
                output[i_map[idx]] = input[idx];
        }
    }

    #define NOA_INSTANTIATE_EXTRACT_INSERT_(T)                                                      \
    template void extract<T>(const T*, size3_t, T*, size3_t, const size3_t*, uint, BorderMode, T);  \
    template void insert<T>(const T*, size3_t, const size3_t*, uint, T*, size3_t);                  \
    template void insert<T>(const T**, size3_t, const size3_t*, uint, T*, size3_t)

    NOA_INSTANTIATE_EXTRACT_INSERT_(short);
    NOA_INSTANTIATE_EXTRACT_INSERT_(int);
    NOA_INSTANTIATE_EXTRACT_INSERT_(long);
    NOA_INSTANTIATE_EXTRACT_INSERT_(long long);
    NOA_INSTANTIATE_EXTRACT_INSERT_(unsigned short);
    NOA_INSTANTIATE_EXTRACT_INSERT_(unsigned int);
    NOA_INSTANTIATE_EXTRACT_INSERT_(unsigned long);
    NOA_INSTANTIATE_EXTRACT_INSERT_(unsigned long long);
    NOA_INSTANTIATE_EXTRACT_INSERT_(float);
    NOA_INSTANTIATE_EXTRACT_INSERT_(double);

    #define NOA_INSTANTIATE_MAP1_(I, T)                                         \
    template std::pair<I*, size_t> getMap<I, T>(const T*, size_t, T);           \
    template std::pair<I*, size_t> getMap<I, T>(const T*, size_t, size3_t, T);  \
    template void extract<I, T>(const T*, size_t, T*, size_t, const I*, uint);  \
    template void insert<I, T>(const T*, size_t, T*, size_t, const I*, uint)

    #define NOA_INSTANTIATE_MAP_(T)             \
    NOA_INSTANTIATE_MAP1_(int, T);              \
    NOA_INSTANTIATE_MAP1_(long, T);             \
    NOA_INSTANTIATE_MAP1_(long long, T);        \
    NOA_INSTANTIATE_MAP1_(unsigned int, T);     \
    NOA_INSTANTIATE_MAP1_(unsigned long, T);    \
    NOA_INSTANTIATE_MAP1_(unsigned long long, T)

    NOA_INSTANTIATE_MAP_(short);
    NOA_INSTANTIATE_MAP_(int);
    NOA_INSTANTIATE_MAP_(long);
    NOA_INSTANTIATE_MAP_(long long);
    NOA_INSTANTIATE_MAP_(unsigned short);
    NOA_INSTANTIATE_MAP_(unsigned int);
    NOA_INSTANTIATE_MAP_(unsigned long);
    NOA_INSTANTIATE_MAP_(unsigned long long);
    NOA_INSTANTIATE_MAP_(float);
    NOA_INSTANTIATE_MAP_(double);

    size3_t getAtlasLayout(size3_t subregion_shape, uint subregion_count, size3_t* o_subregion_centers) {
        uint col = static_cast<uint>(math::ceil(math::sqrt(static_cast<float>(subregion_count))));
        uint row = (subregion_count + col - 1) / col;
        size3_t atlas_shape(col * subregion_shape.x, row * subregion_shape.y, subregion_shape.z);
        size3_t half = subregion_shape / size_t{2};
        for (uint y = 0; y < row; ++y) {
            for (uint x = 0; x < col; ++x) {
                uint idx = y * col + x;
                if (idx >= subregion_count)
                    break;
                o_subregion_centers[idx] = {x * subregion_shape.x + half.x,
                                            y * subregion_shape.y + half.y,
                                            half.z};
            }
        }
        return atlas_shape;
    }
}
