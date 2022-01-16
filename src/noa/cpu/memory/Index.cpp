#include "noa/common/Assert.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"

#include "noa/cpu/memory/Index.h"
#include "noa/cpu/memory/Set.h"

namespace {
    using namespace noa;

    template<typename T>
    void extractOrNothing_(const T* inputs, size3_t input_pitch, size3_t input_shape,
                           T* subregions, size3_t subregion_pitch, size3_t subregion_shape,
                           const int3_t* origins, size_t batches) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != subregions);

        const int3_t i_shape(input_shape);
        const int3_t o_shape(subregion_shape);
        const size_t iffset = elements(input_pitch);
        const size_t offset = elements(subregion_pitch);

        for (size_t batch = 0; batch < batches; ++batch) {
            const int3_t corner_left = origins[batch];
            const T* input = inputs + batch * iffset;
            T* subregion = subregions + batch * offset;

            for (int o_z = 0; o_z < o_shape.z; ++o_z) {
                const int i_z = o_z + corner_left.z;
                if (i_z < 0 || i_z >= i_shape.z)
                    continue;

                for (int o_y = 0; o_y < o_shape.y; ++o_y) {
                    const int i_y = o_y + corner_left.y;
                    if (i_y < 0 || i_y >= i_shape.y)
                        continue;

                    const size_t i_offset = index(i_y, i_z, input_pitch);
                    const size_t o_offset = index(o_y, o_z, subregion_pitch);
                    for (int o_x = 0; o_x < o_shape.x; ++o_x) {
                        const int i_x = o_x + corner_left.x;
                        if (i_x < 0 || i_x >= i_shape.x)
                            continue;
                        subregion[o_offset + static_cast<size_t>(o_x)] =
                                input[i_offset + static_cast<size_t>(i_x)];
                    }
                }
            }
        }
    }

    template<typename T>
    void extractOrValue_(const T* inputs, size3_t input_pitch, size3_t input_shape,
                         T* subregions, size3_t subregion_pitch, size3_t subregion_shape,
                         const int3_t* origins, size_t batches, T value) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != subregions);

        const int3_t i_shape(input_shape);
        const int3_t o_shape(subregion_shape);
        const size_t iffset = elements(input_pitch);
        const size_t offset = elements(subregion_pitch);

        for (size_t batch = 0; batch < batches; ++batch) {
            const int3_t corner_left = origins[batch];
            const T* input = inputs + batch * iffset;
            T* subregion = subregions + batch * offset;

            for (int o_z = 0; o_z < o_shape.z; ++o_z) {
                const int i_z = o_z + corner_left.z;
                if (i_z < 0 || i_z >= i_shape.z) {
                    T* start = subregion + index(0, o_z, subregion_pitch);
                    cpu::memory::set(start, subregion_pitch, {subregion_shape.x, subregion_shape.y, 1}, 1, value);
                    continue;
                }
                for (int o_y = 0; o_y < o_shape.y; ++o_y) {
                    const int i_y = o_y + corner_left.y;
                    if (i_y < 0 || i_y >= i_shape.y) {
                        T* start = subregion + index(o_y, o_z, subregion_pitch);
                        cpu::memory::set(start, start + subregion_shape.x, value);
                        continue;
                    }

                    const size_t i_offset = index(i_y, i_z, input_pitch);
                    const size_t o_offset = index(o_y, o_z, subregion_pitch);
                    for (int o_x = 0; o_x < o_shape.x; ++o_x) {
                        const int i_x = o_x + corner_left.x;
                        if (i_x < 0 || i_x >= i_shape.x)
                            subregion[o_offset + static_cast<size_t>(o_x)] = value;
                        else
                            subregion[o_offset + static_cast<size_t>(o_x)] =
                                    input[i_offset + static_cast<size_t>(i_x)];
                    }
                }
            }
        }
    }

    template<BorderMode MODE, typename T>
    void extract_(const T* inputs, size3_t input_pitch, size3_t input_shape,
                  T* subregions, size3_t subregion_pitch, size3_t subregion_shape,
                  const int3_t* origins, size_t batches) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != subregions);

        const int3_t i_shape(input_shape);
        const int3_t o_shape(subregion_shape);
        const size_t iffset = elements(input_pitch);
        const size_t offset = elements(subregion_pitch);

        for (size_t batch = 0; batch < batches; ++batch) {
            const int3_t corner_left = origins[batch];
            const T* input = inputs + batch * iffset;
            T* subregion = subregions + batch * offset;

            for (int o_z = 0; o_z < o_shape.z; ++o_z) {
                const int i_z = getBorderIndex<MODE>(o_z + corner_left.z, i_shape.z);
                for (int o_y = 0; o_y < o_shape.y; ++o_y) {
                    const int i_y = getBorderIndex<MODE>(o_y + corner_left.y, i_shape.y);
                    for (int o_x = 0; o_x < o_shape.x; ++o_x) {
                        const int i_x = getBorderIndex<MODE>(o_x + corner_left.x, i_shape.x);
                        subregion[index(o_x, o_y, o_z, subregion_pitch)] = input[index(i_x, i_y, i_z, input_pitch)];
                    }
                }
            }
        }
    }

    template<typename T>
    void insert_(const T* subregions, size3_t subregion_pitch, size3_t subregion_shape,
                 T* outputs, size3_t output_pitch, size3_t output_shape, const int3_t* origins, size_t batches) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(outputs != subregions);

        const int3_t i_shape(subregion_shape);
        const int3_t o_shape(output_shape);
        const size_t iffset = elements(subregion_pitch);
        const size_t offset = elements(output_pitch);

        for (size_t batch = 0; batch < batches; ++batch) {
            const int3_t corner_left = origins[batch];
            const T* subregion = subregions + batch * iffset;
            T* output = outputs + batch * offset;

            for (int i_z = 0; i_z < i_shape.z; ++i_z) {
                const int o_z = i_z + corner_left.z;
                if (o_z < 0 || o_z >= o_shape.z)
                    continue;

                for (int i_y = 0; i_y < i_shape.y; ++i_y) {
                    const int o_y = i_y + corner_left.y;
                    if (o_y < 0 || o_y >= o_shape.y)
                        continue;

                    const size_t i_offset = index(i_y, i_z, subregion_pitch);
                    const size_t o_offset = index(o_y, o_z, output_pitch);
                    for (int i_x = 0; i_x < i_shape.x; ++i_x) {
                        const int o_x = i_x + corner_left.x;
                        if (o_x < 0 || o_x >= o_shape.x)
                            continue;
                        output[o_offset + static_cast<size_t>(o_x)] =
                                subregion[i_offset + static_cast<size_t>(i_x)];
                    }
                }
            }
        }
    }
}

namespace noa::cpu::memory {
    template<typename T>
    void extract(const T* inputs, size3_t input_pitch, size3_t input_shape,
                 T* subregions, size3_t subregion_pitch, size3_t subregion_shape,
                 const int3_t* origins, size_t batches, BorderMode border_mode,
                 T border_value, Stream& stream) {
        NOA_PROFILE_FUNCTION();

        switch (border_mode) {
            case BORDER_NOTHING:
                return stream.enqueue(extractOrNothing_<T>, inputs, input_pitch, input_shape,
                                      subregions, subregion_pitch, subregion_shape,
                                      origins, batches);
            case BORDER_ZERO:
                return stream.enqueue(extractOrValue_<T>, inputs, input_pitch, input_shape,
                                      subregions, subregion_pitch, subregion_shape,
                                      origins, batches, static_cast<T>(0));
            case BORDER_VALUE:
                return stream.enqueue(extractOrValue_<T>, inputs, input_pitch, input_shape,
                                      subregions, subregion_pitch, subregion_shape,
                                      origins, batches, border_value);
            case BORDER_CLAMP:
                return stream.enqueue(extract_<BORDER_CLAMP, T>, inputs, input_pitch, input_shape,
                                      subregions, subregion_pitch, subregion_shape,
                                      origins, batches);
            case BORDER_MIRROR:
                return stream.enqueue(extract_<BORDER_MIRROR, T>, inputs, input_pitch, input_shape,
                                      subregions, subregion_pitch, subregion_shape,
                                      origins, batches);
            case BORDER_REFLECT:
                return stream.enqueue(extract_<BORDER_REFLECT, T>, inputs, input_pitch, input_shape,
                                      subregions, subregion_pitch, subregion_shape,
                                      origins, batches);
            default:
                NOA_THROW("{} is not supported", border_mode);
        }
    }

    template<typename T>
    void insert(const T* subregions, size3_t subregion_pitch, size3_t subregion_shape,
                T* outputs, size3_t output_pitch, size3_t output_shape,
                const int3_t* origins, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();

        stream.enqueue(insert_<T>, subregions, subregion_pitch, subregion_shape,
                       outputs, output_pitch, output_shape, origins, batches);
    }

    #define NOA_INSTANTIATE_EXTRACT_INSERT_(T)                                                                                  \
    template void extract<T>(const T*, size3_t, size3_t, T*, size3_t, size3_t, const int3_t*, size_t, BorderMode, T, Stream&);  \
    template void insert<T>(const T*, size3_t, size3_t, T*, size3_t, size3_t, const int3_t*, size_t, Stream&)

    NOA_INSTANTIATE_EXTRACT_INSERT_(char);
    NOA_INSTANTIATE_EXTRACT_INSERT_(short);
    NOA_INSTANTIATE_EXTRACT_INSERT_(int);
    NOA_INSTANTIATE_EXTRACT_INSERT_(long);
    NOA_INSTANTIATE_EXTRACT_INSERT_(long long);
    NOA_INSTANTIATE_EXTRACT_INSERT_(unsigned char);
    NOA_INSTANTIATE_EXTRACT_INSERT_(unsigned short);
    NOA_INSTANTIATE_EXTRACT_INSERT_(unsigned int);
    NOA_INSTANTIATE_EXTRACT_INSERT_(unsigned long);
    NOA_INSTANTIATE_EXTRACT_INSERT_(unsigned long long);
    NOA_INSTANTIATE_EXTRACT_INSERT_(half_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(float);
    NOA_INSTANTIATE_EXTRACT_INSERT_(double);
    NOA_INSTANTIATE_EXTRACT_INSERT_(chalf_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(cfloat_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(cdouble_t);
}
