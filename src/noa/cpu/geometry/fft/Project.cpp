#include "noa/common/Assert.h"
#include "noa/common/Types.h"
#include "noa/common/geometry/Interpolator.h"
#include "noa/common/geometry/InterpolatorValue.h"
#include "noa/common/geometry/details/FourierProjections.h"

#include "noa/cpu/geometry/fft/Project.h"
#include "noa/cpu/utils/Iwise.h"

namespace {
    using namespace ::noa;

    template<bool ALLOW_NULL, typename MatrixWrapper>
    auto matrixOrRawConstPtr_(const MatrixWrapper& matrix) {
        if constexpr (traits::is_floatXX_v<MatrixWrapper>) {
            return MatrixWrapper(matrix);
        } else {
            using clean_t = traits::remove_ref_cv_t<MatrixWrapper>;
            using raw_const_ptr_t = const traits::element_type_t<clean_t>*;
            NOA_ASSERT(ALLOW_NULL || matrix.get());
            return static_cast<raw_const_ptr_t>(matrix.get());
        }
    }
}

namespace noa::cpu::geometry::fft {
    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void insert3D(const shared_t<Value[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                  const shared_t<Value[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                  const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
                  float cutoff, dim4_t target_shape, float2_t ews_radius, Stream& stream) {

        const dim3_t slice_strides_3d{slice_strides[0], slice_strides[2], slice_strides[3]};
        const dim3_t grid_strides_3d{grid_strides[1], grid_strides[2], grid_strides[3]};
        const dim_t threads = stream.threads();

        stream.enqueue([=]() {
            const auto slice_accessor = AccessorRestrict<const Value, 3, dim_t>(slice.get(), slice_strides_3d);
            const auto grid_accessor = AccessorRestrict<Value, 3, dim_t>(grid.get(), grid_strides_3d);
            const auto iwise_shape = long3_t{slice_shape[0], slice_shape[2], slice_shape[3]}.fft();

            const auto inv_scaling_matrices_ = matrixOrRawConstPtr_<false>(inv_scaling_matrices);
            const auto fwd_rotation_matrices_ = matrixOrRawConstPtr_<true>(fwd_rotation_matrices);

            const auto apply_ews = any(ews_radius != 0);
            const bool apply_scale = inv_scaling_matrices != Scale{};

            using namespace noa::geometry::fft::details;
            if (apply_ews || apply_scale) {
                const auto functor = fourierInsertionByGridding<REMAP, int64_t>(
                        slice_accessor, slice_shape, grid_accessor, grid_shape,
                        inv_scaling_matrices_, fwd_rotation_matrices_,
                        cutoff, target_shape, ews_radius);
                utils::iwise3D(iwise_shape, functor, threads);
            } else {
                const auto functor = fourierInsertionByGridding<REMAP, int64_t>(
                        slice_accessor, slice_shape, grid_accessor, grid_shape,
                        empty_t{}, fwd_rotation_matrices_,
                        cutoff, target_shape, empty_t{});
                utils::iwise3D(iwise_shape, functor, threads);
            }
        });
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void insert3D(Value slice, dim4_t slice_shape,
                  const shared_t<Value[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                  const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
                  float cutoff, dim4_t target_shape, float2_t ews_radius, Stream& stream) {

        const dim3_t grid_strides_3d{grid_strides[1], grid_strides[2], grid_strides[3]};
        const dim_t threads = stream.threads();

        stream.enqueue([=]() {
            const auto grid_accessor = AccessorRestrict<Value, 3, dim_t>(grid.get(), grid_strides_3d);
            const auto iwise_shape = long3_t{slice_shape[0], slice_shape[2], slice_shape[3]}.fft();

            const auto inv_scaling_matrices_ = matrixOrRawConstPtr_<false>(inv_scaling_matrices);
            const auto fwd_rotation_matrices_ = matrixOrRawConstPtr_<true>(fwd_rotation_matrices);

            const auto apply_ews = any(ews_radius != 0);
            const bool apply_scale = inv_scaling_matrices != Scale{};

            using namespace noa::geometry::fft::details;
            if (apply_ews || apply_scale) {
                const auto functor = fourierInsertionByGridding<REMAP, int64_t>(
                        slice, slice_shape, grid_accessor, grid_shape,
                        inv_scaling_matrices_, fwd_rotation_matrices_,
                        cutoff, target_shape, ews_radius);
                utils::iwise3D(iwise_shape, functor, threads);
            } else {
                const auto functor = fourierInsertionByGridding<REMAP, int64_t>(
                        slice, slice_shape, grid_accessor, grid_shape,
                        empty_t{}, fwd_rotation_matrices_,
                        cutoff, target_shape, empty_t{});
                utils::iwise3D(iwise_shape, functor, threads);
            }
        });
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void insert3D(const shared_t<Value[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                  const shared_t<Value[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                  const Scale& fwd_scaling_matrices, const Rotate& inv_rotation_matrices,
                  float cutoff, dim4_t target_shape, float2_t ews_radius,
                  float slice_z_radius, Stream& stream) {

        const auto slice_shape_2d = safe_cast<long2_t>(dim2_t{slice_shape.get(2)});
        const dim3_t slice_strides_3d{slice_strides[0], slice_strides[2], slice_strides[3]};
        const dim3_t grid_strides_3d{grid_strides[1], grid_strides[2], grid_strides[3]};
        const dim_t threads = stream.threads();

        stream.enqueue([=]() {
            const auto slice_accessor = AccessorRestrict<const Value, 3, dim_t>(slice.get(), slice_strides_3d);
            const auto grid_accessor = AccessorRestrict<Value, 3, dim_t>(grid.get(), grid_strides_3d);
            const auto iwise_shape = long3_t{grid_shape.get(1)}.fft();
            const auto slice_interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_LINEAR>(
                    slice_accessor, slice_shape_2d.fft(), Value{0});

            const auto apply_ews = any(ews_radius != 0);
            const bool apply_scale = fwd_scaling_matrices != Scale{};

            const auto fwd_scaling_matrices_ = matrixOrRawConstPtr_<false>(fwd_scaling_matrices);
            const auto inv_rotation_matrices_ = matrixOrRawConstPtr_<true>(inv_rotation_matrices);

            using namespace noa::geometry::fft::details;
            if (apply_ews || apply_scale) {
                const auto functor = fourierInsertionExplicitThickness<REMAP, int64_t>(
                        slice_interpolator, slice_shape, grid_accessor, grid_shape,
                        fwd_scaling_matrices_, inv_rotation_matrices_,
                        cutoff, target_shape, ews_radius, slice_z_radius);
                utils::iwise3D(iwise_shape, functor, threads);
            } else {
                const auto functor = fourierInsertionExplicitThickness<REMAP, int64_t>(
                        slice_interpolator, slice_shape, grid_accessor, grid_shape,
                        empty_t{}, inv_rotation_matrices_,
                        cutoff, target_shape, empty_t{}, slice_z_radius);
                utils::iwise3D(iwise_shape, functor, threads);
            }
        });
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void insert3D(Value slice, dim4_t slice_shape,
                  const shared_t<Value[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                  const Scale& fwd_scaling_matrices, const Rotate& inv_rotation_matrices,
                  float cutoff, dim4_t target_shape, float2_t ews_radius,
                  float slice_z_radius, Stream& stream) {

        const auto slice_shape_2d = safe_cast<long2_t>(dim2_t{slice_shape.get(2)});
        const dim3_t grid_strides_3d{grid_strides[1], grid_strides[2], grid_strides[3]};
        const dim_t threads = stream.threads();

        stream.enqueue([=]() {
            const auto grid_accessor = AccessorRestrict<Value, 3, dim_t>(grid.get(), grid_strides_3d);
            const auto iwise_shape = long3_t{grid_shape.get(1)}.fft();
            const auto slice_interpolator = noa::geometry::interpolatorValue2D<BORDER_ZERO, INTERP_LINEAR>(
                slice, slice_shape_2d.fft(), Value{0});

            const auto apply_ews = any(ews_radius != 0);
            const bool apply_scale = fwd_scaling_matrices != Scale{};

            const auto fwd_scaling_matrices_ = matrixOrRawConstPtr_<false>(fwd_scaling_matrices);
            const auto inv_rotation_matrices_ = matrixOrRawConstPtr_<true>(inv_rotation_matrices);

            using namespace noa::geometry::fft::details;
            if (apply_ews || apply_scale) {
                const auto functor = fourierInsertionExplicitThickness<REMAP, int64_t>(
                        slice_interpolator, slice_shape, grid_accessor, grid_shape,
                        fwd_scaling_matrices_, inv_rotation_matrices_,
                        cutoff, target_shape, ews_radius, slice_z_radius);
                utils::iwise3D(iwise_shape, functor, threads);
            } else {
                const auto functor = fourierInsertionExplicitThickness<REMAP, int64_t>(
                        slice_interpolator, slice_shape, grid_accessor, grid_shape,
                        empty_t{}, inv_rotation_matrices_,
                        cutoff, target_shape, empty_t{}, slice_z_radius);
                utils::iwise3D(iwise_shape, functor, threads);
            }
        });
    }

    template<Remap REMAP, typename Value, typename Scale, typename Rotate, typename>
    void extract3D(const shared_t<Value[]>& grid, dim4_t grid_strides, dim4_t grid_shape,
                   const shared_t<Value[]>& slice, dim4_t slice_strides, dim4_t slice_shape,
                   const Scale& inv_scaling_matrices, const Rotate& fwd_rotation_matrices,
                   float cutoff, dim4_t target_shape, float2_t ews_radius, Stream& stream) {

        const long3_t slice_shape_3d{slice_shape[0], slice_shape[2], slice_shape[3]};
        const dim3_t slice_strides_3d{slice_strides[0], slice_strides[2], slice_strides[3]};
        const long3_t grid_shape_3d{grid_shape[1], grid_shape[2], grid_shape[3]};
        const dim3_t grid_strides_3d{grid_strides[1], grid_strides[2], grid_strides[3]};
        const dim_t threads = stream.threads();

        stream.enqueue([=]() {
            const auto slice_accessor = AccessorRestrict<Value, 3, dim_t>(slice.get(), slice_strides_3d);
            const auto grid_accessor = AccessorRestrict<const Value, 3, dim_t>(grid.get(), grid_strides_3d);
            const auto iwise_shape = slice_shape_3d.fft();
            const auto grid_interpolator = noa::geometry::interpolator3D<BORDER_ZERO, INTERP_LINEAR>(
                    grid_accessor, grid_shape_3d.fft(), Value{0});

            const auto inv_scaling_matrices_ = matrixOrRawConstPtr_<false>(inv_scaling_matrices);
            const auto fwd_rotation_matrices_ = matrixOrRawConstPtr_<true>(fwd_rotation_matrices);

            const auto apply_ews = any(ews_radius != 0);
            const bool apply_scale = inv_scaling_matrices != Scale{};

            using namespace noa::geometry::fft::details;
            if (apply_ews || apply_scale) {
                const auto functor = fourierExtraction<REMAP, int64_t>(
                        grid_interpolator, grid_shape, slice_accessor, slice_shape,
                        inv_scaling_matrices_, fwd_rotation_matrices_,
                        cutoff, target_shape, ews_radius);
                utils::iwise3D(iwise_shape, functor, threads);
            } else {
                const auto functor = fourierExtraction<REMAP, int64_t>(
                        grid_interpolator, grid_shape, slice_accessor, slice_shape,
                        empty_t{}, fwd_rotation_matrices_,
                        cutoff, target_shape, empty_t{});
                utils::iwise3D(iwise_shape, functor, threads);
            }
        });
    }

    template<Remap REMAP, typename Value, typename Scale0, typename Scale1, typename Rotate0, typename Rotate1, typename>
    void extract3D(const shared_t<Value[]>& input_slice, dim4_t input_slice_strides, dim4_t input_slice_shape,
                   const shared_t<Value[]>& output_slice, dim4_t output_slice_strides, dim4_t output_slice_shape,
                   const Scale0& insert_fwd_scaling_matrices, const Rotate0& insert_inv_rotation_matrices,
                   const Scale1& extract_inv_scaling_matrices, const Rotate1& extract_fwd_rotation_matrices,
                   float cutoff, float2_t ews_radius, float slice_z_radius, Stream& stream) {

        const dim3_t input_slice_strides_2d{input_slice_strides[0], input_slice_strides[2], input_slice_strides[3]};
        const dim3_t output_slice_strides_2d{output_slice_strides[0], output_slice_strides[2], output_slice_strides[3]};
        const dim3_t output_slice_shape_2d{output_slice_shape[0], output_slice_shape[2], output_slice_shape[3]};
        const auto iwise_shape = safe_cast<long3_t>(output_slice_shape_2d).fft();

        const dim_t threads = stream.threads();
        stream.enqueue([=]() {
            const auto input_slice_accessor = AccessorRestrict<const Value, 3, dim_t>(input_slice.get(), input_slice_strides_2d);
            const auto output_slice_accessor = AccessorRestrict<Value, 3, dim_t>(output_slice.get(), output_slice_strides_2d);
            const auto input_slice_interpolator = noa::geometry::interpolator2D<BORDER_ZERO, INTERP_LINEAR>(
                    input_slice_accessor, safe_cast<long2_t>(dim2_t(input_slice_shape.get(2))).fft(), Value{0});

            const auto apply_ews = any(ews_radius != 0);
            const bool apply_scale = insert_fwd_scaling_matrices != Scale0{};

            // The transformation for the insertion needs to be inverted.
            const auto insert_fwd_scaling_matrices_ = matrixOrRawConstPtr_<false>(insert_fwd_scaling_matrices);
            const auto insert_inv_rotation_matrices_ = matrixOrRawConstPtr_<true>(insert_inv_rotation_matrices);
            const auto extract_inv_scaling_matrices_ = matrixOrRawConstPtr_<false>(extract_inv_scaling_matrices);
            const auto extract_fwd_rotation_matrices_ = matrixOrRawConstPtr_<true>(extract_fwd_rotation_matrices);

            using namespace noa::geometry::fft::details;
            if (apply_ews || apply_scale) {
                const auto functor = fourierInsertExtraction<REMAP, int64_t>(
                        input_slice_interpolator, input_slice_shape,
                        output_slice_accessor, output_slice_shape,
                        insert_fwd_scaling_matrices_, insert_inv_rotation_matrices_,
                        extract_inv_scaling_matrices_, extract_fwd_rotation_matrices_,
                        cutoff, ews_radius, slice_z_radius);
                utils::iwise3D(iwise_shape, functor, threads);
            } else {
                const auto functor = fourierInsertExtraction<REMAP, int64_t>(
                        input_slice_interpolator, input_slice_shape,
                        output_slice_accessor, output_slice_shape,
                        empty_t{}, insert_inv_rotation_matrices_,
                        extract_inv_scaling_matrices_, extract_fwd_rotation_matrices_,
                        cutoff, empty_t{}, slice_z_radius);
                utils::iwise3D(iwise_shape, functor, threads);
            }
        });
    }

    template<Remap REMAP, typename Value, typename Scale0, typename Scale1, typename Rotate0, typename Rotate1, typename>
    void extract3D(Value input_slice, dim4_t input_slice_shape,
                   const shared_t<Value[]>& output_slice, dim4_t output_slice_strides, dim4_t output_slice_shape,
                   const Scale0& insert_fwd_scaling_matrices, const Rotate0& insert_inv_rotation_matrices,
                   const Scale1& extract_inv_scaling_matrices, const Rotate1& extract_fwd_rotation_matrices,
                   float cutoff, float2_t ews_radius, float slice_z_radius, Stream& stream) {

        const dim3_t output_slice_strides_2d{output_slice_strides[0], output_slice_strides[2], output_slice_strides[3]};
        const dim3_t output_slice_shape_2d{output_slice_shape[0], output_slice_shape[2], output_slice_shape[3]};
        const auto iwise_shape = safe_cast<long3_t>(output_slice_shape_2d).fft();

        const dim_t threads = stream.threads();
        stream.enqueue([=]() {
            const auto output_slice_accessor = AccessorRestrict<Value, 3, dim_t>(output_slice.get(), output_slice_strides_2d);
            const auto input_slice_interpolator = noa::geometry::interpolatorValue2D<BORDER_ZERO, INTERP_LINEAR>(
                    input_slice, safe_cast<long2_t>(dim2_t(input_slice_shape.get(2))).fft(), Value{0});

            const auto apply_ews = any(ews_radius != 0);
            const bool apply_scale = insert_fwd_scaling_matrices != Scale0{};

            // The transformation for the insertion needs to be inverted.
            const auto insert_fwd_scaling_matrices_ = matrixOrRawConstPtr_<false>(insert_fwd_scaling_matrices);
            const auto insert_inv_rotation_matrices_ = matrixOrRawConstPtr_<true>(insert_inv_rotation_matrices);
            const auto extract_inv_scaling_matrices_ = matrixOrRawConstPtr_<false>(extract_inv_scaling_matrices);
            const auto extract_fwd_rotation_matrices_ = matrixOrRawConstPtr_<true>(extract_fwd_rotation_matrices);

            using namespace noa::geometry::fft::details;
            if (apply_ews || apply_scale) {
                const auto functor = fourierInsertExtraction<REMAP, int64_t>(
                        input_slice_interpolator, input_slice_shape,
                        output_slice_accessor, output_slice_shape,
                        insert_fwd_scaling_matrices_, insert_inv_rotation_matrices_,
                        extract_inv_scaling_matrices_, extract_fwd_rotation_matrices_,
                        cutoff, ews_radius, slice_z_radius);
                utils::iwise3D(iwise_shape, functor, threads);
            } else {
                const auto functor = fourierInsertExtraction<REMAP, int64_t>(
                        input_slice_interpolator, input_slice_shape,
                        output_slice_accessor, output_slice_shape,
                        empty_t{}, insert_inv_rotation_matrices_,
                        extract_inv_scaling_matrices_, extract_fwd_rotation_matrices_,
                        cutoff, empty_t{}, slice_z_radius);
                utils::iwise3D(iwise_shape, functor, threads);
            }
        });
    }

    template<typename Value, typename>
    void griddingCorrection(const shared_t<Value[]>& input, dim4_t input_strides,
                            const shared_t<Value[]>& output, dim4_t output_strides,
                            dim4_t shape, bool post_correction, Stream& stream) {
        NOA_ASSERT(input && input && all(shape > 0));

        const dim_t threads = stream.threads();
        stream.enqueue([=]() {
            const auto input_accessor = Accessor<const Value, 4, dim_t>(input.get(), input_strides);
            const auto output_accessor = Accessor<Value, 4, dim_t>(output.get(), output_strides);

            if (post_correction) {
                const auto functor = noa::geometry::fft::details::griddingCorrection<true>(
                        input_accessor, output_accessor, shape);
                utils::iwise4D(shape, functor, threads);
            } else {
                const auto functor = noa::geometry::fft::details::griddingCorrection<false>(
                        input_accessor, output_accessor, shape);
                utils::iwise4D(shape, functor, threads);
            }
        });
    }
    template void griddingCorrection<float, void>(const shared_t<float[]>&, dim4_t, const shared_t<float[]>&, dim4_t, dim4_t, bool, Stream&);
    template void griddingCorrection<double, void>(const shared_t<double[]>&, dim4_t, const shared_t<double[]>&, dim4_t, dim4_t, bool, Stream&);

    #define NOA_INSTANTIATE_INSERT_(T, REMAP, S, R)             \
    template void insert3D<REMAP, T, S, R, void>(               \
        const shared_t<T[]>&, dim4_t, dim4_t,                   \
        const shared_t<T[]>&, dim4_t, dim4_t,                   \
        const S&, const R&, float, dim4_t, float2_t, Stream&);  \
    template void insert3D<REMAP, T, S, R, void>(               \
        T, dim4_t,                                              \
        const shared_t<T[]>&, dim4_t, dim4_t,                   \
        const S&, const R&, float, dim4_t, float2_t, Stream&)

    #define NOA_INSTANTIATE_INSERT_THICK_(T, REMAP, S, R)               \
    template void insert3D<REMAP, T, S, R, void>(                       \
        const shared_t<T[]>&, dim4_t, dim4_t,                           \
        const shared_t<T[]>&, dim4_t, dim4_t,                           \
        const S&, const R&, float, dim4_t, float2_t, float, Stream&);   \
    template void insert3D<REMAP, T, S, R, void>(                       \
        T, dim4_t,                                                      \
        const shared_t<T[]>&, dim4_t, dim4_t,                           \
        const S&, const R&, float, dim4_t, float2_t, float, Stream&)

    #define NOA_INSTANTIATE_EXTRACT_(T, REMAP, S, R)    \
    template void extract3D<REMAP, T, S, R, void>(      \
        const shared_t<T[]>&, dim4_t, dim4_t,           \
        const shared_t<T[]>&, dim4_t, dim4_t,           \
        const S&, const R&, float, dim4_t, float2_t, Stream&)

    #define NOA_INSTANTIATE_INSERT_EXTRACT_(T, REMAP, S0, S1, R0, R1)                   \
    template void extract3D<REMAP, T, S0, S1, R0, R1, void>(                            \
        const shared_t<T[]>&, dim4_t, dim4_t,                                           \
        const shared_t<T[]>&, dim4_t, dim4_t,                                           \
        const S0&, const R0&, const S1&, const R1&, float, float2_t, float, Stream&);   \
    template void extract3D<REMAP, T, S0, S1, R0, R1, void>(                            \
        T, dim4_t,                                                                      \
        const shared_t<T[]>&, dim4_t, dim4_t,                                           \
        const S0&, const R0&, const S1&, const R1&, float, float2_t, float, Stream&)

    #define NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, S, R)      \
    NOA_INSTANTIATE_INSERT_(T, Remap::H2H, S, R);           \
    NOA_INSTANTIATE_INSERT_(T, Remap::H2HC, S, R);          \
    NOA_INSTANTIATE_INSERT_(T, Remap::HC2H, S, R);          \
    NOA_INSTANTIATE_INSERT_(T, Remap::HC2HC, S, R);         \
    NOA_INSTANTIATE_INSERT_THICK_(T, Remap::HC2H, S, R);    \
    NOA_INSTANTIATE_INSERT_THICK_(T, Remap::HC2HC, S, R);   \
    NOA_INSTANTIATE_EXTRACT_(T, Remap::HC2H, S, R);         \
    NOA_INSTANTIATE_EXTRACT_(T, Remap::HC2HC, S, R)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, S0, S1, R0, R1)  \
    NOA_INSTANTIATE_INSERT_EXTRACT_(T, Remap::HC2H, S0, S1, R0, R1);    \
    NOA_INSTANTIATE_INSERT_EXTRACT_(T, Remap::HC2HC, S0, S1, R0, R1)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, R0, R1)                              \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, float22_t, float22_t, R0, R1);               \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, shared_t<float22_t[]>, float22_t, R0, R1);   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, float22_t, shared_t<float22_t[]>, R0, R1);   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_REMAP(T, shared_t<float22_t[]>, shared_t<float22_t[]>, R0, R1)

    #define NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(T)                             \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, float33_t, float33_t);               \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, shared_t<float33_t[]>, float33_t);   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, float33_t, shared_t<float33_t[]>);   \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_SCALE(T, shared_t<float33_t[]>, shared_t<float33_t[]>)

    #define NOA_INSTANTIATE_PROJECT_ALL_(T)                                             \
    NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, float22_t, float33_t);                         \
    NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, shared_t<float22_t[]>, float33_t);             \
    NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, float22_t, shared_t<float33_t[]>);             \
    NOA_INSTANTIATE_PROJECT_ALL_REMAP(T, shared_t<float22_t[]>, shared_t<float33_t[]>); \
    NOA_INSTANTIATE_PROJECT_MERGE_ALL_ROTATE(T);                                        \

    NOA_INSTANTIATE_PROJECT_ALL_(float);
    NOA_INSTANTIATE_PROJECT_ALL_(double);
    NOA_INSTANTIATE_PROJECT_ALL_(cfloat_t);
    NOA_INSTANTIATE_PROJECT_ALL_(cdouble_t);
}
