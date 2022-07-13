#include "noa/common/Assert.h"
#include "noa/common/Exception.h"

#include "noa/cpu/memory/Index.h"
#include "noa/cpu/memory/Set.h"

// TODO Add OpenMP on the batch loop? One subregion per thread?

namespace {
    using namespace noa;

    template<typename T>
    void extractOrNothing_(const T* input, size4_t input_stride, size4_t input_shape,
                           T* subregions, size4_t subregion_stride, size4_t subregion_shape,
                           const int4_t* origins) {
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

                        subregions[indexing::at(batch, oj, ok, ol, subregion_stride)] =
                                input[indexing::at(ii, ij, ik, il, input_stride)];
                    }
                }
            }
        }
    }

    template<typename T>
    void extractOrValue_(const T* input, size4_t input_stride, size4_t input_shape,
                         T* subregions, size4_t subregion_stride, size4_t subregion_shape,
                         const int4_t* origins, T value) {
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
                        const bool valid = ij >= 0 && ij < i_shape[1] &&
                                           ik >= 0 && ik < i_shape[2] &&
                                           il >= 0 && il < i_shape[3];

                        subregions[indexing::at(batch, oj, ok, ol, subregion_stride)] =
                                valid ? input[indexing::at(ii, ij, ik, il, input_stride)] : value;
                    }
                }
            }
        }
    }

    template<BorderMode MODE, typename T>
    void extract_(const T* input, size4_t input_stride, size4_t input_shape,
                  T* subregions, size4_t subregion_stride, size4_t subregion_shape,
                  const int4_t* origins) {
        NOA_ASSERT(input != subregions);
        const int4_t i_shape(input_shape);
        const int4_t o_shape(subregion_shape);

        for (size_t batch = 0; batch < subregion_shape[0]; ++batch) {
            const int4_t corner_left = origins[batch];
            // The outermost dimension of subregions is used as batch.
            // We don't use it to index the input.
            const int ii = indexing::at<MODE>(corner_left[0], i_shape[0]);

            for (int oj = 0; oj < o_shape[1]; ++oj) {
                for (int ok = 0; ok < o_shape[2]; ++ok) {
                    for (int ol = 0; ol < o_shape[3]; ++ol) {

                        const int ij = indexing::at<MODE>(oj + corner_left[1], i_shape[1]);
                        const int ik = indexing::at<MODE>(ok + corner_left[2], i_shape[2]);
                        const int il = indexing::at<MODE>(ol + corner_left[3], i_shape[3]);
                        subregions[indexing::at(batch, oj, ok, ol, subregion_stride)] =
                                input[indexing::at(ii, ij, ik, il, input_stride)];
                    }
                }
            }
        }
    }

    template<typename T>
    void insert_(const T* subregions, size4_t subregion_stride, size4_t subregion_shape,
                 T* output, size4_t output_stride, size4_t output_shape, const int4_t* origins) {
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

                        output[indexing::at(oi, oj, ok, ol, output_stride)] =
                                subregions[indexing::at(batch, ij, ik, il, subregion_stride)];
                    }
                }
            }
        }
    }
}

namespace noa::cpu::memory {
    template<typename T, typename>
    void extract(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                 const shared_t<T[]>& subregions, size4_t subregion_stride, size4_t subregion_shape,
                 const shared_t<int4_t[]>& origins, BorderMode border_mode, T border_value,
                 Stream& stream) {
        stream.enqueue([=]() {
            switch (border_mode) {
                case BORDER_NOTHING:
                    return extractOrNothing_(input.get(), input_stride, input_shape,
                                             subregions.get(), subregion_stride, subregion_shape,
                                             origins.get());
                case BORDER_ZERO:
                    return extractOrValue_(input.get(), input_stride, input_shape,
                                           subregions.get(), subregion_stride, subregion_shape,
                                           origins.get(), static_cast<T>(0));
                case BORDER_VALUE:
                    return extractOrValue_(input.get(), input_stride, input_shape,
                                           subregions.get(), subregion_stride, subregion_shape,
                                           origins.get(), border_value);
                case BORDER_CLAMP:
                    return extract_<BORDER_CLAMP>(input.get(), input_stride, input_shape,
                                                  subregions.get(), subregion_stride, subregion_shape,
                                                  origins.get());
                case BORDER_MIRROR:
                    return extract_<BORDER_MIRROR>(input.get(), input_stride, input_shape,
                                                   subregions.get(), subregion_stride, subregion_shape,
                                                   origins.get());
                case BORDER_REFLECT:
                    return extract_<BORDER_REFLECT>(input.get(), input_stride, input_shape,
                                                    subregions.get(), subregion_stride, subregion_shape,
                                                    origins.get());
                default:
                    NOA_THROW("{} is not supported", border_mode);
            }
        });
    }

    template<typename T, typename>
    void insert(const shared_t<T[]>& subregions, size4_t subregion_stride, size4_t subregion_shape,
                const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                const shared_t<int4_t[]>& origins, Stream& stream) {
        stream.enqueue([=]() {
            insert_<T>(subregions.get(), subregion_stride, subregion_shape,
                       output.get(), output_stride, output_shape, origins.get());
        });
    }

    #define NOA_INSTANTIATE_EXTRACT_INSERT_(T)                                                                                                                          \
    template void extract<T, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<int4_t[]>&, BorderMode, T, Stream&);  \
    template void insert<T, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<int4_t[]>&, Stream&)

    NOA_INSTANTIATE_EXTRACT_INSERT_(bool);
    NOA_INSTANTIATE_EXTRACT_INSERT_(int8_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(int16_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(int32_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(int64_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(uint8_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(uint16_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(uint32_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(uint64_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(half_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(float);
    NOA_INSTANTIATE_EXTRACT_INSERT_(double);
    NOA_INSTANTIATE_EXTRACT_INSERT_(chalf_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(cfloat_t);
    NOA_INSTANTIATE_EXTRACT_INSERT_(cdouble_t);
}
