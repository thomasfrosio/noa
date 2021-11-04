#include "noa/common/Assert.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/Index.h"
#include "noa/cpu/memory/Set.h"

namespace {
    using namespace noa;

    inline int3_t getCornerLeft_(int3_t subregion_shape, size3_t subregion_center) {
        return int3_t(subregion_center) - subregion_shape / 2;
    }

    template<typename T>
    void extractOrNothing_(const T* input, int3_t input_shape,
                           T* subregion, int3_t subregion_shape, int3_t corner_left) {
        for (int o_z = 0; o_z < subregion_shape.z; ++o_z) {
            const int i_z = o_z + corner_left.z;
            if (i_z < 0 || i_z >= input_shape.z)
                continue;

            for (int o_y = 0; o_y < subregion_shape.y; ++o_y) {
                const int i_y = o_y + corner_left.y;
                if (i_y < 0 || i_y >= input_shape.y)
                    continue;

                const size_t i_offset = index(i_y, i_z, input_shape);
                const size_t o_offset = index(o_y, o_z, subregion_shape);
                for (int o_x = 0; o_x < subregion_shape.x; ++o_x) {
                    const int i_x = o_x + corner_left.x;
                    if (i_x < 0 || i_x >= input_shape.x)
                        continue;
                    subregion[o_offset + static_cast<size_t>(o_x)] =
                            input[i_offset + static_cast<size_t>(i_x)];
                }
            }
        }
    }

    template<typename T>
    void extractOrValue_(const T* input, int3_t input_shape,
                         T* subregion, int3_t subregion_shape, int3_t corner_left, T value) {
        for (int o_z = 0; o_z < subregion_shape.z; ++o_z) {
            const int i_z = o_z + corner_left.z;
            if (i_z < 0 || i_z >= input_shape.z) {
                T* start = subregion + index(0, o_z, subregion_shape);
                cpu::memory::set(start, elementsSlice(size3_t{subregion_shape}), value);
                continue;
            }
            for (int o_y = 0; o_y < subregion_shape.y; ++o_y) {
                const int i_y = o_y + corner_left.y;
                if (i_y < 0 || i_y >= input_shape.y) {
                    T* start = subregion + index(o_y, o_z, subregion_shape);
                    cpu::memory::set(start, start + subregion_shape.x, value);
                    continue;
                }

                const size_t i_offset = index(i_y, i_z, input_shape);
                const size_t o_offset = index(o_y, o_z, subregion_shape);
                for (int o_x = 0; o_x < subregion_shape.x; ++o_x) {
                    const int i_x = o_x + corner_left.x;
                    if (i_x < 0 || i_x >= input_shape.x)
                        subregion[o_offset + static_cast<size_t>(o_x)] = value;
                    else
                        subregion[o_offset + static_cast<size_t>(o_x)] =
                                input[i_offset + static_cast<size_t>(i_x)];
                }
            }
        }
    }

    template<BorderMode MODE, typename T>
    void extract_(const T* input, int3_t input_shape,
                  T* subregion, int3_t subregion_shape, int3_t corner_left) {
        for (int o_z = 0; o_z < subregion_shape.z; ++o_z) {
            const int i_z = getBorderIndex<MODE>(o_z + corner_left.z, input_shape.z);
            for (int o_y = 0; o_y < subregion_shape.y; ++o_y) {
                const int i_y = getBorderIndex<MODE>(o_y + corner_left.y, input_shape.y);

                const size_t i_offset = index(i_y, i_z, input_shape);
                const size_t o_offset = index(o_y, o_z, subregion_shape);
                for (int o_x = 0; o_x < subregion_shape.x; ++o_x) {
                    const int i_x = getBorderIndex<MODE>(o_x + corner_left.x, input_shape.x);
                    subregion[o_offset + static_cast<size_t>(o_x)] =
                            input[i_offset + static_cast<size_t>(i_x)];
                }
            }
        }
    }

    template<typename T>
    void insert_(const T* subregion, int3_t subregion_shape,
                 T* output, int3_t output_shape, int3_t corner_left) {
        for (int i_z = 0; i_z < subregion_shape.z; ++i_z) {
            const int o_z = i_z + corner_left.z;
            if (o_z < 0 || o_z >= output_shape.z)
                continue;

            for (int i_y = 0; i_y < subregion_shape.y; ++i_y) {
                const int o_y = i_y + corner_left.y;
                if (o_y < 0 || o_y >= output_shape.y)
                    continue;

                const size_t i_offset = index(i_y, i_z, subregion_shape);
                const size_t o_offset = index(o_y, o_z, output_shape);
                for (int i_x = 0; i_x < subregion_shape.x; ++i_x) {
                    const int o_x = i_x + corner_left.x;
                    if (o_x < 0 || o_x >= output_shape.x)
                        continue;
                    output[o_offset + static_cast<size_t>(o_x)] =
                            subregion[i_offset + static_cast<size_t>(i_x)];
                }
            }
        }
    }
}

namespace noa::cpu::memory {
    template<typename T>
    void extract(const T* input, size3_t input_shape,
                 T* subregions, size3_t subregion_shape, const size3_t* subregion_centers, size_t subregion_count,
                 BorderMode border_mode, T border_value) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != subregions);

        const int3_t i_shape(input_shape);
        const int3_t o_shape(subregion_shape);
        const size_t elements = noa::elements(subregion_shape);

        for (size_t idx = 0; idx < subregion_count; ++idx) {
            const int3_t corner_left = getCornerLeft_(o_shape, subregion_centers[idx]);

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
    void insert(const T* subregions, size3_t subregion_shape, const size3_t* subregion_centers, size_t subregion_count,
                T* output, size3_t output_shape) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(output != subregions);

        const int3_t i_shape(subregion_shape);
        const int3_t o_shape(output_shape);
        const size_t elements = noa::elements(subregion_shape);

        for (size_t idx = 0; idx < subregion_count; ++idx) {
            const int3_t corner_left = getCornerLeft_(i_shape, subregion_centers[idx]);
            insert_(subregions + idx * elements, i_shape, output, o_shape, corner_left);
        }
    }

    template<typename T>
    void insert(const T** subregions, size3_t subregion_shape, const size3_t* subregion_centers, size_t subregion_count,
                T* output, size3_t output_shape) {
        NOA_PROFILE_FUNCTION();
        const int3_t i_shape(subregion_shape);
        const int3_t o_shape(output_shape);

        for (size_t idx = 0; idx < subregion_count; ++idx) {
            const int3_t corner_left = getCornerLeft_(i_shape, subregion_centers[idx]);
            insert_(subregions[idx], i_shape, output, o_shape, corner_left);
        }
    }

    // TODO When noa::Vector<> will be created, use and return it directly...
    template<typename I, typename T>
    std::pair<I*, size_t> where(const T* input, size_t elements, T threshold) {
        NOA_PROFILE_FUNCTION();
        std::vector<I> tmp_seq;
        tmp_seq.reserve(1000);
        for (size_t idx = 0; idx < elements; ++idx)
            if (input[idx] > threshold)
                tmp_seq.emplace_back(static_cast<I>(idx));
        PtrHost<I> seq(tmp_seq.size()); // we cannot release std::vector...
        copy(tmp_seq.data(), seq.get(), tmp_seq.size());
        return {seq.release(), tmp_seq.size()};
    }

    template<typename I, typename T>
    std::pair<I*, size_t> where(const T* input, size_t pitch, size3_t shape, size_t batches, T threshold) {
        NOA_PROFILE_FUNCTION();
        std::vector<I> tmp_seq;
        tmp_seq.reserve(1000);
        for (size_t batch = 0; batch < batches; ++batch) {
            const size_t o_b = batch * rows(shape) * pitch;
            for (size_t z = 0; z < shape.z; ++z) {
                const size_t o_z = o_b + z * shape.y * pitch;
                for (size_t y = 0; y < shape.y; ++y) {
                    const size_t o_y = o_z + y * pitch;
                    for (size_t x = 0; x < shape.x; ++x) {
                        const size_t idx = o_y + x;
                        if (input[idx] > threshold)
                            tmp_seq.emplace_back(static_cast<I>(idx));
                    }
                }
            }
        }
        PtrHost<I> seq(tmp_seq.size());
        copy(tmp_seq.data(), seq.get(), tmp_seq.size());
        return {seq.release(), tmp_seq.size()};
    }

    template<typename I, typename T>
    void extract(const T* sparse, size_t sparse_elements, T* dense, size_t dense_elements,
                 const I* sequence, size_t batches) {
        NOA_PROFILE_FUNCTION();
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = sparse + batch * sparse_elements;
            T* output = dense + batch * dense_elements;
            for (size_t idx = 0; idx < dense_elements; ++idx)
                output[idx] = input[sequence[idx]];
        }
    }

    template<typename I, typename T>
    void insert(const T* dense, size_t dense_elements, T* sparse, size_t sparse_elements,
                const I* sequence, size_t batches) {
        NOA_PROFILE_FUNCTION();
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = dense + batch * dense_elements;
            T* output = sparse + batch * sparse_elements;
            for (size_t idx = 0; idx < dense_elements; ++idx)
                output[sequence[idx]] = input[idx];
        }
    }

    #define NOA_INSTANTIATE_EXTRACT_INSERT_(T)                                                          \
    template void extract<T>(const T*, size3_t, T*, size3_t, const size3_t*, size_t, BorderMode, T);    \
    template void insert<T>(const T*, size3_t, const size3_t*, size_t, T*, size3_t);                    \
    template void insert<T>(const T**, size3_t, const size3_t*, size_t, T*, size3_t)

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

    #define NOA_INSTANTIATE_MAP1_(I, T)                                                 \
    template std::pair<I*, size_t> where<I, T>(const T*, size_t, T);                    \
    template std::pair<I*, size_t> where<I, T>(const T*, size_t, size3_t, size_t, T);   \
    template void extract<I, T>(const T*, size_t, T*, size_t, const I*, size_t);        \
    template void insert<I, T>(const T*, size_t, T*, size_t, const I*, size_t)

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

    size3_t atlasLayout(size3_t subregion_shape, size_t subregion_count, size3_t* o_subregion_centers) {
        const auto col = static_cast<size_t>(math::ceil(math::sqrt(static_cast<float>(subregion_count))));
        const size_t row = (subregion_count + col - 1) / col;
        const size3_t atlas_shape(col * subregion_shape.x, row * subregion_shape.y, subregion_shape.z);
        const size3_t half = subregion_shape / size_t{2};
        for (size_t y = 0; y < row; ++y) {
            for (size_t x = 0; x < col; ++x) {
                const size_t idx = y * col + x;
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
