#include "noa/common/Assert.h"
#include "noa/common/Exception.h"

#include "noa/cpu/memory/Index.h"
#include "noa/cpu/memory/Set.h"

namespace {
    using namespace noa;

    template<typename T>
    void extractOrNothing_(AccessorRestrict<const T, 4, dim_t> input, dim4_t input_shape,
                           AccessorRestrict<T, 4, dim_t> subregions, dim4_t subregion_shape,
                           const int4_t* origins, int4_t order, dim_t threads) {
        const int4_t i_shape(input_shape);
        const int4_t o_shape(subregion_shape);

        [[maybe_unused]] const dim_t elements_per_subregion = dim3_t(subregion_shape.get(1)).elements();
        #pragma omp parallel for if(elements_per_subregion > 16384) num_threads(threads) default(none) \
        shared(input, input_shape, subregions, origins, order, i_shape, o_shape)

        for (int32_t batch = 0; batch < o_shape[0]; ++batch) {
            const int4_t corner_left = indexing::reorder(origins[batch], order);
            // The outermost dimension of subregions is used as batch.
            // We don't use it to index the input.
            const int32_t ii = corner_left[0];
            if (ii < 0 || ii >= i_shape[0])
                continue;

            const auto subregion = subregions[batch];
            for (int32_t oj = 0; oj < o_shape[1]; ++oj) {
                for (int32_t ok = 0; ok < o_shape[2]; ++ok) {
                    for (int32_t ol = 0; ol < o_shape[3]; ++ol) {

                        const int32_t ij = oj + corner_left[1];
                        const int32_t ik = ok + corner_left[2];
                        const int32_t il = ol + corner_left[3];
                        if (ij < 0 || ij >= i_shape[1] ||
                            ik < 0 || ik >= i_shape[2] ||
                            il < 0 || il >= i_shape[3])
                            continue;

                        subregion(oj, ok, ol) = input(ii, ij, ik, il);
                    }
                }
            }
        }
    }

    template<typename T>
    void extractOrValue_(AccessorRestrict<const T, 4, dim_t> input, dim4_t input_shape,
                         AccessorRestrict<T, 4, dim_t> subregions, dim4_t subregion_shape,
                         const int4_t* origins, T value, int4_t order, dim_t threads) {
        const int4_t i_shape(input_shape);
        const int4_t o_shape(subregion_shape);

        [[maybe_unused]] const dim_t elements_per_subregion = dim3_t(subregion_shape.get(1)).elements();
        #pragma omp parallel for if(elements_per_subregion > 16384) num_threads(threads) default(none) \
        shared(input, input_shape, subregions, subregion_shape, origins, value, order, i_shape, o_shape)

        for (dim_t batch = 0; batch < subregion_shape[0]; ++batch) {
            const int4_t corner_left = indexing::reorder(origins[batch], order);
            const auto subregion = subregions[batch];

            // The outermost dimension of subregions is used as batch.
            // We don't use it to index the input.
            const int32_t ii = corner_left[0];
            if (ii < 0 || ii >= i_shape[0]) {
                const dim4_t one_subregion{1, subregion_shape[1], subregion_shape[2], subregion_shape[3]};
                cpu::memory::set(subregion.get(), dim4_t(subregions.strides()), one_subregion, value);
                continue;
            }

            for (int32_t oj = 0; oj < o_shape[1]; ++oj) {
                for (int32_t ok = 0; ok < o_shape[2]; ++ok) {
                    for (int32_t ol = 0; ol < o_shape[3]; ++ol) {

                        const int32_t ij = oj + corner_left[1];
                        const int32_t ik = ok + corner_left[2];
                        const int32_t il = ol + corner_left[3];
                        const bool valid = ij >= 0 && ij < i_shape[1] &&
                                           ik >= 0 && ik < i_shape[2] &&
                                           il >= 0 && il < i_shape[3];

                        subregion(oj, ok, ol) = valid ? input(ii, ij, ik, il) : value;
                    }
                }
            }
        }
    }

    template<BorderMode MODE, typename T>
    void extract_(AccessorRestrict<const T, 4, dim_t> input, dim4_t input_shape,
                  AccessorRestrict<T, 4, dim_t> subregions, dim4_t subregion_shape,
                  const int4_t* origins, int4_t order, dim_t threads) {
        const int4_t i_shape(input_shape);
        const int4_t o_shape(subregion_shape);

        [[maybe_unused]] const dim_t elements_per_subregion = dim3_t(subregion_shape.get(1)).elements();
        #pragma omp parallel for if(elements_per_subregion > 16384) num_threads(threads) default(none) \
        shared(input, input_shape, subregions, subregion_shape, origins, order, i_shape, o_shape)

        for (dim_t batch = 0; batch < subregion_shape[0]; ++batch) {
            const int4_t corner_left = indexing::reorder(origins[batch], order);
            // The outermost dimension of subregions is used as batch.
            // We don't use it to index the input.
            const int32_t ii = indexing::at<MODE>(corner_left[0], i_shape[0]);

            for (int32_t oj = 0; oj < o_shape[1]; ++oj) {
                for (int32_t ok = 0; ok < o_shape[2]; ++ok) {
                    for (int32_t ol = 0; ol < o_shape[3]; ++ol) {

                        const int32_t ij = indexing::at<MODE>(oj + corner_left[1], i_shape[1]);
                        const int32_t ik = indexing::at<MODE>(ok + corner_left[2], i_shape[2]);
                        const int32_t il = indexing::at<MODE>(ol + corner_left[3], i_shape[3]);
                        subregions(batch, oj, ok, ol) = input(ii, ij, ik, il);
                    }
                }
            }
        }
    }

    template<typename T>
    void insert_(AccessorRestrict<const T, 4, dim_t> subregions, dim4_t subregion_shape,
                 AccessorRestrict<T, 4, dim_t> output, dim4_t output_shape, const int4_t* origins,
                 int4_t order, dim_t threads) {
        const int4_t i_shape(subregion_shape);
        const int4_t o_shape(output_shape);

        [[maybe_unused]] const dim_t elements_per_subregion = dim3_t(subregion_shape.get(1)).elements();
        #pragma omp parallel for if(elements_per_subregion > 16384) num_threads(threads) default(none) \
        shared(subregions, subregion_shape, output, origins, order, i_shape, o_shape)

        for (dim_t batch = 0; batch < subregion_shape[0]; ++batch) {
            const int4_t corner_left = indexing::reorder(origins[batch], order);
            const int32_t oi = corner_left[0];
            if (oi < 0 || oi >= o_shape[0])
                continue;

            const auto subregion = subregions[batch];
            for (int32_t ij = 0; ij < i_shape[1]; ++ij) {
                for (int32_t ik = 0; ik < i_shape[2]; ++ik) {
                    for (int32_t il = 0; il < i_shape[3]; ++il) {

                        const int32_t oj = ij + corner_left[1];
                        const int32_t ok = ik + corner_left[2];
                        const int32_t ol = il + corner_left[3];
                        if (oj < 0 || oj >= o_shape[1] ||
                            ok < 0 || ok >= o_shape[2] ||
                            ol < 0 || ol >= o_shape[3])
                            continue;

                        // We assume no overlap in the output between subregions.
                        output(oi, oj, ok, ol) = subregion(ij, ik, il);
                    }
                }
            }
        }
    }
}

namespace noa::cpu::memory {
    template<typename T, typename>
    void extract(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                 const shared_t<T[]>& subregions, dim4_t subregion_strides, dim4_t subregion_shape,
                 const shared_t<int4_t[]>& origins, BorderMode border_mode, T border_value,
                 Stream& stream) {
        const dim_t threads = stream.threads();
        stream.enqueue([=]() mutable {
            // Reorder the DHW dimensions to the rightmost order.
            // We'll have to reorder the origins similarly later.
            const dim3_t order_3d = indexing::order(dim3_t(subregion_strides.get(1)),
                                                    dim3_t(subregion_shape.get(1))) + 1;
            const int4_t order(0, order_3d[0], order_3d[1], order_3d[2]);
            input_strides = indexing::reorder(input_strides, order);
            input_shape = indexing::reorder(input_shape, order);
            subregion_strides = indexing::reorder(subregion_strides, order);
            subregion_shape = indexing::reorder(subregion_shape, order);

            switch (border_mode) {
                case BORDER_NOTHING:
                    return extractOrNothing_<T>({input.get(), input_strides}, input_shape,
                                                {subregions.get(), subregion_strides}, subregion_shape,
                                                origins.get(), order, threads);
                case BORDER_ZERO:
                    return extractOrValue_({input.get(), input_strides}, input_shape,
                                           {subregions.get(), subregion_strides}, subregion_shape,
                                           origins.get(), T{0}, order, threads);
                case BORDER_VALUE:
                    return extractOrValue_({input.get(), input_strides}, input_shape,
                                           {subregions.get(), subregion_strides}, subregion_shape,
                                           origins.get(), border_value, order, threads);
                case BORDER_CLAMP:
                    return extract_<BORDER_CLAMP, T>({input.get(), input_strides}, input_shape,
                                                     {subregions.get(), subregion_strides}, subregion_shape,
                                                     origins.get(), order, threads);
                case BORDER_MIRROR:
                    return extract_<BORDER_MIRROR, T>({input.get(), input_strides}, input_shape,
                                                      {subregions.get(), subregion_strides}, subregion_shape,
                                                      origins.get(), order, threads);
                case BORDER_REFLECT:
                    return extract_<BORDER_REFLECT, T>({input.get(), input_strides}, input_shape,
                                                       {subregions.get(), subregion_strides}, subregion_shape,
                                                       origins.get(), order, threads);
                default:
                    NOA_THROW("{} is not supported", border_mode);
            }
        });
    }

    template<typename T, typename>
    void insert(const shared_t<T[]>& subregions, dim4_t subregion_strides, dim4_t subregion_shape,
                const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                const shared_t<int4_t[]>& origins, Stream& stream) {
        const dim_t threads = stream.threads();
        stream.enqueue([=]() mutable {
            // Reorder the DHW dimensions to the rightmost order.
            // We'll have to reorder the origins similarly later.
            const dim3_t order_3d = indexing::order(dim3_t(subregion_strides.get(1)),
                                                    dim3_t(subregion_shape.get(1))) + 1;
            const int4_t order(0, order_3d[0], order_3d[1], order_3d[2]);
            output_strides = indexing::reorder(output_strides, order);
            output_shape = indexing::reorder(output_shape, order);
            subregion_strides = indexing::reorder(subregion_strides, order);
            subregion_shape = indexing::reorder(subregion_shape, order);

            insert_<T>({subregions.get(), subregion_strides}, subregion_shape,
                       {output.get(), output_strides}, output_shape, origins.get(), order, threads);
        });
    }

    #define NOA_INSTANTIATE_EXTRACT_INSERT_(T)                                                                                                                      \
    template void extract<T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<int4_t[]>&, BorderMode, T, Stream&);  \
    template void insert<T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<int4_t[]>&, Stream&)

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
