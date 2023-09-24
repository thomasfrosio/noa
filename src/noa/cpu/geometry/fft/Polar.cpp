#include "noa/algorithms/geometry/PolarTransformRFFT.hpp"
#include "noa/algorithms/geometry/RotationalAverage.hpp"
#include "noa/core/geometry/Interpolator.hpp"
#include "noa/core/signal/fft/CTF.hpp"
#include "noa/cpu/geometry/fft/Polar.hpp"
#include "noa/cpu/memory/AllocatorHeap.hpp"
#include "noa/cpu/memory/Set.hpp"
#include "noa/cpu/utils/Iwise.hpp"
#include "noa/cpu/utils/EwiseBinary.hpp"

namespace noa::cpu::geometry::fft {
    template<noa::fft::Remap, typename Input, typename Output, typename>
    void cartesian2polar(
            const Input* cartesian, Strides4<i64> cartesian_strides, Shape4<i64> cartesian_shape,
            Output* polar, const Strides4<i64>& polar_strides, const Shape4<i64>& polar_shape,
            const Vec2<f32>& frequency_range, bool frequency_range_endpoint,
            const Vec2<f32>& angle_range, bool angle_range_endpoint,
            InterpMode interp_mode, i64 threads) {
        NOA_ASSERT(cartesian && polar && static_cast<const void*>(cartesian) != static_cast<void*>(polar) &&
                   noa::all(cartesian_shape > 0) && noa::all(polar_shape > 0));
        NOA_ASSERT(cartesian_shape[1] == 1 && polar_shape[1] == 1);
        NOA_ASSERT(cartesian_shape[0] == 1 || cartesian_shape[0] == polar_shape[0]);

        // Broadcast the input to every output batch.
        if (cartesian_shape[0] == 1)
            cartesian_strides[0] = 0;
        else if (cartesian_strides[0] == 0)
            cartesian_shape[0] = 1;

        const auto iwise_shape = polar_shape.filter(0,2,3);
        const auto cartesian_shape_2d = cartesian_shape.filter(2,3);
        const auto cartesian_accessor = AccessorRestrict<const Input, 3, i64>(cartesian, cartesian_strides.filter(0, 2, 3));
        const auto polar_accessor = AccessorRestrict<Output, 3, i64>(polar, polar_strides.filter(0, 2, 3));

        switch (interp_mode) {
            case InterpMode::NEAREST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::NEAREST>(
                        cartesian_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft<i64>(
                        interpolator, cartesian_shape, polar_accessor, polar_shape,
                        frequency_range, frequency_range_endpoint,
                        angle_range, angle_range_endpoint);
                return cpu::utils::iwise_3d(iwise_shape, kernel, threads);
            }
            case InterpMode::LINEAR: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::LINEAR>(
                        cartesian_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft<i64>(
                        interpolator, cartesian_shape, polar_accessor, polar_shape,
                        frequency_range, frequency_range_endpoint,
                        angle_range, angle_range_endpoint);
                return cpu::utils::iwise_3d(iwise_shape, kernel, threads);
            }
            case InterpMode::COSINE: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::COSINE>(
                        cartesian_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft<i64>(
                        interpolator, cartesian_shape, polar_accessor, polar_shape,
                        frequency_range, frequency_range_endpoint,
                        angle_range, angle_range_endpoint);
                return cpu::utils::iwise_3d(iwise_shape, kernel, threads);
            }
            case InterpMode::LINEAR_FAST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::LINEAR_FAST>(
                        cartesian_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft<i64>(
                        interpolator, cartesian_shape, polar_accessor, polar_shape,
                        frequency_range, frequency_range_endpoint,
                        angle_range, angle_range_endpoint);
                return cpu::utils::iwise_3d(iwise_shape, kernel, threads);
            }
            case InterpMode::COSINE_FAST: {
                const auto interpolator = noa::geometry::interpolator_2d<BorderMode::ZERO, InterpMode::COSINE_FAST>(
                        cartesian_accessor, cartesian_shape_2d);
                const auto kernel = noa::algorithm::geometry::cartesian2polar_rfft<i64>(
                        interpolator, cartesian_shape, polar_accessor, polar_shape,
                        frequency_range, frequency_range_endpoint,
                        angle_range, angle_range_endpoint);
                return cpu::utils::iwise_3d(iwise_shape, kernel, threads);
            }
            case InterpMode::CUBIC:
            case InterpMode::CUBIC_BSPLINE:
            case InterpMode::CUBIC_BSPLINE_FAST:
                NOA_THROW("{} is not supported", interp_mode);
        }
    }

    template<noa::fft::Remap REMAP, typename Input, typename Ctf, typename Output, typename Weight, typename>
    void rotational_average(
            const Input* input, Strides4<i64> input_strides, Shape4<i64> input_shape, const Ctf& input_ctf,
            Output* output, i64 output_batch_stride, Weight* weight, i64 weight_batch_stride, i64 n_shells,
            const Vec2<f32>& frequency_range, bool frequency_range_endpoint, bool average, i64 threads
    ) {
        const auto n_batches = input_shape[0];
        const auto shell_shape = Shape4<i64>{n_batches, 1, 1, n_shells};
        const auto output_strides = Strides4<i64>{output_batch_stride, 0, 0, 1};
        auto weight_strides = Strides4<i64>{weight_batch_stride, 0, 0, 1};

        // Output must be zeroed.
        noa::cpu::memory::set(output, output_strides, shell_shape, Output{0}, /*threads=*/ 1);

        // When computing the average, the weights must be valid.
        using unique_t = typename noa::cpu::memory::AllocatorHeap<Weight>::calloc_unique_type;
        unique_t weight_buffer;
        Weight* weight_ptr = weight;
        if (average) {
            if (weight_ptr == nullptr) {
                weight_buffer = noa::cpu::memory::AllocatorHeap<Weight>::calloc(n_batches * n_shells);
                weight_ptr = weight_buffer.get();
                weight_strides[0] = n_shells; // contiguous batches
            } else {
                noa::cpu::memory::set(weight_ptr, weight_strides, shell_shape, Weight{0}, /*threads=*/ 1);
            }
        }

        constexpr bool IS_HALF = static_cast<u8>(REMAP) & noa::fft::Layout::SRC_HALF;
        if (input_shape.ndim() == 2) {
            // Reorder HW-dimensions to rightmost.
            const auto order = noa::indexing::order(input_strides.filter(2, 3), input_shape.filter(2, 3));
            if (noa::any(order != Vec2<i64>{0, 1})) {
                std::swap(input_strides[2], input_strides[3]);
                std::swap(input_shape[2], input_shape[3]);
            }

            const auto kernel = noa::algorithm::geometry::rotational_average_2d<REMAP>(
                    input, input_strides, input_shape, input_ctf,
                    output, output_strides[0], weight_ptr, weight_strides[0], n_shells,
                    frequency_range, frequency_range_endpoint);

            auto iwise_shape = input_shape.filter(0, 2, 3);
            if constexpr (IS_HALF)
                iwise_shape = iwise_shape.rfft();
            noa::cpu::utils::iwise_3d(iwise_shape, kernel, threads);

        } else {
            // Reorder BHW dimensions to rightmost.
            const auto order = noa::indexing::order(input_strides.pop_front(), input_shape.pop_front());
            if (noa::any(order != Vec3<i64>{0, 1, 2})) {
                const auto order_3d = (order + 1).push_front(0);
                input_strides = noa::indexing::reorder(input_strides, order_3d);
                input_shape = noa::indexing::reorder(input_shape, order_3d);
            }

            const auto kernel = noa::algorithm::geometry::rotational_average_3d<REMAP>(
                    input, input_strides, input_shape,
                    output, output_strides[0], weight_ptr, weight_strides[0], n_shells,
                    frequency_range, frequency_range_endpoint);

            noa::cpu::utils::iwise_4d(IS_HALF ? input_shape.rfft() : input_shape, kernel, threads);
        }

        if (average) {
            // Some shells can be 0, so use divide_safe_t.
            noa::cpu::utils::ewise_binary(
                    output, output_strides,
                    weight_ptr, weight_strides,
                    output, output_strides, shell_shape,
                    noa::divide_safe_t{}, threads);
        }
    }

    #define NOA_INSTANTIATE_POLAR(Input, Output)                                \
    template void cartesian2polar<noa::fft::Remap::HC2FC, Input, Output, void>( \
        const Input*, Strides4<i64>, Shape4<i64>,                               \
        Output*, const Strides4<i64>&, const Shape4<i64>&,                      \
        const Vec2<f32>&, bool, const Vec2<f32>&, bool, InterpMode, i64)

//    NOA_INSTANTIATE_POLAR(f32, f32);
//    NOA_INSTANTIATE_POLAR(f64, f64);
//    NOA_INSTANTIATE_POLAR(c32, c32);
//    NOA_INSTANTIATE_POLAR(c64, c64);
//    NOA_INSTANTIATE_POLAR(c32, f32);
//    NOA_INSTANTIATE_POLAR(c64, f64);

    #define NOA_INSTANTIATE_ROTATIONAL_AVERAGE(Remap, Input, Ctf, Output, Weight)   \
    template void rotational_average<Remap, Input, Ctf, Output, Weight, void>(      \
            const Input*, Strides4<i64>, Shape4<i64>, Ctf const&,                   \
            Output*, i64, Weight*, i64, i64,                                        \
            const Vec2<f32>&, bool, bool, i64)

    #define NOA_INSTANTIATE_ROTATIONAL_AVERAGE_CTF(Remap, Input, Output, Weight)                                \
    NOA_INSTANTIATE_ROTATIONAL_AVERAGE(Remap, Input, Empty, Output, Weight);                                    \
    NOA_INSTANTIATE_ROTATIONAL_AVERAGE(Remap, Input, signal::fft::CTFAnisotropic<f64>, Output, Weight);         \
    NOA_INSTANTIATE_ROTATIONAL_AVERAGE(Remap, Input, const signal::fft::CTFAnisotropic<f64>*, Output, Weight)

//    NOA_INSTANTIATE_ROTATIONAL_AVERAGE(Remap, Input, signal::fft::CTFAnisotropic<f32>, Output, Weight);
//    NOA_INSTANTIATE_ROTATIONAL_AVERAGE(Remap, Input, const signal::fft::CTFAnisotropic<f32>*, Output, Weight);

    #define NOA_INSTANTIATE_ROTATIONAL_AVERAGE_REMAP(Input, Output, Weight)                 \
    NOA_INSTANTIATE_ROTATIONAL_AVERAGE_CTF(noa::fft::Remap::H2H, Input, Output, Weight);
//    NOA_INSTANTIATE_ROTATIONAL_AVERAGE_CTF(noa::fft::Remap::HC2H, Input, Output, Weight);
//    NOA_INSTANTIATE_ROTATIONAL_AVERAGE_CTF(noa::fft::Remap::F2H, Input, Output, Weight);
//    NOA_INSTANTIATE_ROTATIONAL_AVERAGE_CTF(noa::fft::Remap::FC2H, Input, Output, Weight)

    NOA_INSTANTIATE_ROTATIONAL_AVERAGE_REMAP(f32, f32, f32);
//    NOA_INSTANTIATE_ROTATIONAL_AVERAGE_REMAP(f64, f64, f64);
    NOA_INSTANTIATE_ROTATIONAL_AVERAGE_REMAP(c32, c32, f32);
//    NOA_INSTANTIATE_ROTATIONAL_AVERAGE_REMAP(c64, c64, f64);
    NOA_INSTANTIATE_ROTATIONAL_AVERAGE_REMAP(c32, f32, f32);
//    NOA_INSTANTIATE_ROTATIONAL_AVERAGE_REMAP(c64, f64, f64);
}
