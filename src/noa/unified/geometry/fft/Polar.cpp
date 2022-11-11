#include "noa/unified/geometry/fft/Polar.h"

#include "noa/cpu/geometry/fft/Polar.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/fft/Polar.h"
#endif

namespace {
    using namespace ::noa;

    template<typename ArrayOrTexture, typename Value>
    void c2pCheckParameters_(const ArrayOrTexture& input, dim4_t input_shape, const Array<Value>& output) {
        const char* FUNC_NAME = "cartesian";
        NOA_CHECK_FUNC(FUNC_NAME, !input.empty() && !output.empty(), "Empty array detected");
        NOA_CHECK_FUNC(FUNC_NAME, input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                       "The number of batches in the input array ({}) is not compatible with the number of "
                       "batches in the output array ({})", input.shape()[0], output.shape()[0]);
        NOA_CHECK_FUNC(FUNC_NAME,
                       input.shape()[3] == input_shape[3] / 2 + 1 &&
                       input.shape()[2] == input_shape[2] &&
                       input.shape()[1] == input_shape[1],
                       "The non-redundant FFT with shape {} doesn't match the logical shape {}",
                       input.shape(), input_shape);
        NOA_CHECK_FUNC(FUNC_NAME, input.shape()[1] == 1 && output.shape()[1] == 1,
                       "3D arrays are not supported");

        if (input.device().cpu()) {
            NOA_CHECK_FUNC(FUNC_NAME, input.device() == output.device(),
                           "The input and output arrays must be on the same device, "
                           "but got input:{} and output:{}", input.device(), output.device());
            if constexpr (std::is_same_v<ArrayOrTexture, Array<Value>>) {
                NOA_CHECK_FUNC(FUNC_NAME, !indexing::isOverlap(input, output),
                               "Input and output arrays should not overlap");
            }
        } else {
            if constexpr (std::is_same_v<ArrayOrTexture, Array<Value>>) {
                if (input.device().cpu())
                    Stream::current(Device(Device::CPU)).synchronize();
            } else {
                NOA_CHECK_FUNC(FUNC_NAME, input.device() == output.device(),
                               "The input texture and output array must be on the same device, "
                               "but got input:{} and output:{}", input.device(), output.device());
            }
        }
    }
}

namespace noa::geometry::fft {
    template<Remap REMAP, typename Value, typename>
    void cartesian2polar(const Array<Value>& cartesian, dim4_t cartesian_shape,
                         const Array<Value>& polar,
                         float2_t frequency_range, float2_t angle_range,
                         bool log, InterpMode interp) {
        c2pCheckParameters_(cartesian, cartesian_shape, polar);

        const Device device = polar.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::fft::cartesian2polar<REMAP>(
                    cartesian.share(), cartesian.strides(), cartesian_shape,
                    polar.share(), polar.strides(), polar.shape(),
                    frequency_range, angle_range, log, interp, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<Value, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                cuda::geometry::fft::cartesian2polar<REMAP>(
                        cartesian.share(), cartesian.strides(), cartesian_shape,
                        polar.share(), polar.strides(), polar.shape(),
                        frequency_range, angle_range, log, interp, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<Remap REMAP, typename Value, typename>
    void cartesian2polar(const Texture<Value>& cartesian, dim4_t cartesian_shape, const Array<Value>& polar,
                         float2_t frequency_range, float2_t angle_range,
                         bool log) {
        const Device device = polar.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            const cpu::Texture<Value>& texture = cartesian.cpu();
            const Array<Value> tmp(texture.ptr, cartesian.shape(), texture.strides, cartesian.options());
            cartesian2polar<REMAP>(tmp, cartesian_shape, polar,
                                   frequency_range, angle_range, log, cartesian.interp());
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<Value, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported by this function");
            } else {
                c2pCheckParameters_(cartesian, cartesian_shape, polar);
                const cuda::Texture<Value>& texture = cartesian.cuda();
                cuda::geometry::fft::cartesian2polar(
                        texture.array, texture.texture, cartesian.interp(), cartesian_shape,
                        polar.share(), polar.strides(), polar.shape(),
                        frequency_range, angle_range, log, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    #define NOA_INSTANTIATE_UNIFIED_GEOMETRY_C2P_(T)                                                                                        \
    template void cartesian2polar<Remap::HC2FC, T, void>(const Array<T>&, dim4_t, const Array<T>&, float2_t, float2_t, bool, InterpMode);   \
    template void cartesian2polar<Remap::HC2FC, T, void>(const Texture<T>&, dim4_t, const Array<T>&, float2_t, float2_t, bool)

    NOA_INSTANTIATE_UNIFIED_GEOMETRY_C2P_(float);
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_C2P_(double);
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_C2P_(cfloat_t);
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_C2P_(cdouble_t);
}
