#include "noa/common/Assert.h"
#include "noa/common/Exception.h"
#include "noa/common/Profiler.h"

#include "noa/cpu/memory/Index.h"
#include "noa/cpu/memory/Set.h"

// TODO Add OpenMP on the batch loop? One subregion per thread?

namespace {
    using namespace noa;

    template<typename T>
    void extractOrNothing_(const T* input, size4_t input_stride, size4_t input_shape,
                           T* subregions, size4_t subregion_stride, size4_t subregion_shape,
                           const int4_t* origins) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != subregions);

        const int4_t i_shape(input_shape);
        const int4_t o_shape(subregion_shape);

        for (int batch = 0; batch < o_shape[0]; ++batch) {
            const int4_t corner_left = origins[batch];
            // The outermost dimension of subregions is used as batch.
            // We don't use it to index the input.
            const int ii = corner_left[0];
            if (ii < 0 || ii >= i_shape[0])
                continue;

            for (int oj = 0; oj < o_shape[1]; ++oj) {
                for (int ok = 0; ok < o_shape[2]; ++ok) {
                    for (int ol = 0; ol < o_shape[3]; ++ol) {

                        const int ij = oj + corner_left[1];
                        const int ik = ok + corner_left[2];
                        const int il = ol + corner_left[3];
                        if (ij < 0 || ij >= i_shape[1] ||
                            ik < 0 || ik >= i_shape[2] ||
                            il < 0 || il >= i_shape[3])
                            continue;

                        subregions[at(batch, oj, ok, ol, subregion_stride)] =
                                input[at(ii, ij, ik, il, input_stride)];
                    }
                }
            }
        }
    }

    template<typename T>
    void extractOrValue_(const T* input, size4_t input_stride, size4_t input_shape,
                         T* subregions, size4_t subregion_stride, size4_t subregion_shape,
                         const int4_t* origins, T value) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != subregions);

        const int4_t i_shape(input_shape);
        const int4_t o_shape(subregion_shape);

        for (size_t batch = 0; batch < subregion_shape[0]; ++batch) {
            const int4_t corner_left = origins[batch];
            // The outermost dimension of subregions is used as batch.
            // We don't use it to index the input.
            const int ii = corner_left[0];
            if (ii < 0 || ii >= i_shape[0]) {
                const size4_t one_subregion{1, subregion_shape[1], subregion_shape[2], subregion_shape[3]};
                cpu::memory::set(subregions + subregion_stride[0] * batch, subregion_stride, one_subregion, value);
                continue;
            }

            for (int oj = 0; oj < o_shape[1]; ++oj) {
                for (int ok = 0; ok < o_shape[2]; ++ok) {
                    for (int ol = 0; ol < o_shape[3]; ++ol) {

                        const int ij = oj + corner_left[1];
                        const int ik = ok + corner_left[2];
                        const int il = ol + corner_left[3];
                        const bool valid = ij < 0 || ij >= i_shape[1] ||
                                           ik < 0 || ik >= i_shape[2] ||
                                           il < 0 || il >= i_shape[3];

                        subregions[at(batch, oj, ok, ol, subregion_stride)] = valid ?
                                                                              input[at(ii, ij, ik, il, input_stride)]
                                                                                    : value;
                    }
                }
            }
        }
    }

    template<BorderMode MODE, typename T>
    void extract_(const T* input, size4_t input_stride, size4_t input_shape,
                  T* subregions, size4_t subregion_stride, size4_t subregion_shape,
                  const int4_t* origins) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != subregions);

        const int4_t i_shape(input_shape);
        const int4_t o_shape(subregion_shape);

        for (size_t batch = 0; batch < subregion_shape[0]; ++batch) {
            const int4_t corner_left = origins[batch];
            // The outermost dimension of subregions is used as batch.
            // We don't use it to index the input.
            const int ii = getBorderIndex<MODE>(corner_left[0], i_shape[0]);

            for (int oj = 0; oj < o_shape[1]; ++oj) {
                for (int ok = 0; ok < o_shape[2]; ++ok) {
                    for (int ol = 0; ol < o_shape[3]; ++ol) {

                        const int ij = getBorderIndex<MODE>(oj + corner_left[1], i_shape[1]);
                        const int ik = getBorderIndex<MODE>(ok + corner_left[2], i_shape[2]);
                        const int il = getBorderIndex<MODE>(ol + corner_left[3], i_shape[3]);
                        subregions[at(batch, oj, ok, ol, subregion_stride)] = input[at(ii, ij, ik, il, input_stride)];
                    }
                }
            }
        }
    }

    template<typename T>
    void insert_(const T* subregions, size4_t subregion_stride, size4_t subregion_shape,
                 T* output, size4_t output_stride, size4_t output_shape, const int4_t* origins) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(output != subregions);

        const int4_t i_shape(subregion_shape);
        const int4_t o_shape(output_shape);

        for (size_t batch = 0; batch < subregion_shape[0]; ++batch) {
            const int4_t corner_left = origins[batch];
            const int oi = corner_left[0];
            if (oi < 0 || oi >= o_shape[0])
                continue;

            for (int ij = 0; ij < i_shape[1]; ++ij) {
                for (int ik = 0; ik < i_shape[2]; ++ik) {
                    for (int il = 0; il < i_shape[3]; ++il) {

                        const int oj = ij + corner_left[1];
                        const int ok = ik + corner_left[2];
                        const int ol = il + corner_left[3];
                        if (oj < 0 || oj >= o_shape[1] ||
                            ok < 0 || ok >= o_shape[2] ||
                            ol < 0 || ol >= o_shape[3])
                            continue;

                        output[at(oi, oj, ok, ol, output_stride)] =
                                subregions[at(batch, ij, ik, il, subregion_stride)];
                    }
                }
            }
        }
    }
}

namespace noa::cpu::memory {
    template<typename T>
    void extract(const T* input, size4_t input_stride, size4_t input_shape,
                 T* subregions, size4_t subregion_stride, size4_t subregion_shape,
                 const int4_t* origins, BorderMode border_mode, T border_value, Stream& stream) {
        NOA_PROFILE_FUNCTION();

        switch (border_mode) {
            case BORDER_NOTHING:
                return stream.enqueue(extractOrNothing_<T>, input, input_stride, input_shape,
                                      subregions, subregion_stride, subregion_shape, origins);
            case BORDER_ZERO:
                return stream.enqueue(extractOrValue_<T>, input, input_stride, input_shape,
                                      subregions, subregion_stride, subregion_shape, origins, static_cast<T>(0));
            case BORDER_VALUE:
                return stream.enqueue(extractOrValue_<T>, input, input_stride, input_shape,
                                      subregions, subregion_stride, subregion_shape, origins, border_value);
            case BORDER_CLAMP:
                return stream.enqueue(extract_<BORDER_CLAMP, T>, input, input_stride, input_shape,
                                      subregions, subregion_stride, subregion_shape, origins);
            case BORDER_MIRROR:
                return stream.enqueue(extract_<BORDER_MIRROR, T>, input, input_stride, input_shape,
                                      subregions, subregion_stride, subregion_shape, origins);
            case BORDER_REFLECT:
                return stream.enqueue(extract_<BORDER_REFLECT, T>, input, input_stride, input_shape,
                                      subregions, subregion_stride, subregion_shape, origins);
            default:
                NOA_THROW("{} is not supported", border_mode);
        }
    }

    template<typename T>
    void insert(const T* subregions, size4_t subregion_stride, size4_t subregion_shape,
                T* outputs, size4_t output_stride, size4_t output_shape,
                const int4_t* origins, Stream& stream) {
        NOA_PROFILE_FUNCTION();

        stream.enqueue(insert_<T>, subregions, subregion_stride, subregion_shape,
                       outputs, output_stride, output_shape, origins);
    }

    #define NOA_INSTANTIATE_EXTRACT_INSERT_(T)                                                                          \
    template void extract<T>(const T*, size4_t, size4_t, T*, size4_t, size4_t, const int4_t*, BorderMode, T, Stream&);  \
    template void insert<T>(const T*, size4_t, size4_t, T*, size4_t, size4_t, const int4_t*, Stream&)

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
