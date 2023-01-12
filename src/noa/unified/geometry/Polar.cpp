#include "noa/unified/geometry/Polar.h"

#include "noa/cpu/geometry/Polar.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/geometry/Polar.h"
#endif

namespace {
    using namespace ::noa;

    enum class PolarDirection { C2P, P2C };

    template<PolarDirection DIRECTION, typename ArrayOrTexture, typename Value>
    void polarCheckParameters_(const ArrayOrTexture& input, const Array<Value>& output) {
        const char* FUNC_NAME = DIRECTION == PolarDirection::C2P ? "cartesian2polar" : "polar2cartesian";

        NOA_CHECK_FUNC(FUNC_NAME, !input.empty() && !output.empty(), "Empty array detected");
        NOA_CHECK_FUNC(FUNC_NAME, input.shape()[0] == 1 || input.shape()[0] == output.shape()[0],
                       "The number of batches in the input array ({}) is not compatible with the number of "
                       "batches in the output array ({})", input.shape()[0], output.shape()[0]);
        NOA_CHECK_FUNC(FUNC_NAME, input.shape()[1] == 1 && output.shape()[1] == 1, "3D arrays are not supported");

        NOA_CHECK(input.device() == output.device(),
                  "The input and output arrays must be on the same device, "
                  "but got input:{} and output:{}", input.device(), output.device());

        if constexpr (std::is_same_v<ArrayOrTexture, Array<Value>>) {
            NOA_CHECK(!indexing::isOverlap(input, output),
                      "Input and output arrays should not overlap");
            NOA_CHECK_FUNC(FUNC_NAME, indexing::areElementsUnique(output.strides(), output.shape()),
                           "The elements in the output should not overlap in memory, "
                           "otherwise a data-race might occur. Got output strides:{} and shape:{}",
                           output.strides(), output.shape());
        }
    }
}

namespace noa::geometry {
    template<typename Value, typename>
    void cartesian2polar(const Array<Value>& cartesian, const Array<Value>& polar,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interpolation_mode, bool prefilter) {
        polarCheckParameters_<PolarDirection::C2P>(cartesian, polar);

        const Device device = polar.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::cartesian2polar(
                    cartesian.share(), cartesian.strides(), cartesian.shape(),
                    polar.share(), polar.strides(), polar.shape(),
                    cartesian_center, radius_range, angle_range, log,
                    interpolation_mode, prefilter, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::geometry::cartesian2polar(
                    cartesian.share(), cartesian.strides(), cartesian.shape(),
                    polar.share(), polar.strides(), polar.shape(),
                    cartesian_center, radius_range, angle_range, log,
                    interpolation_mode, prefilter, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value, typename>
    void cartesian2polar(const Texture<Value>& cartesian, const Array<Value>& polar,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log) {
        const Device device = cartesian.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            const cpu::Texture<Value>& texture = cartesian.cpu();
            const Array<Value> tmp(texture.ptr, cartesian.shape(), texture.strides, cartesian.options());
            cartesian2polar(tmp, polar, cartesian_center, radius_range, angle_range, log, cartesian.interp(), false);
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<Value, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                polarCheckParameters_<PolarDirection::C2P>(cartesian, polar);
                const cuda::Texture<Value>& texture = cartesian.cuda();
                cuda::geometry::cartesian2polar(
                        texture.array, texture.texture, cartesian.interp(), cartesian.shape(),
                        polar.share(), polar.strides(), polar.shape(),
                        cartesian_center, radius_range, angle_range,
                        log, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value, typename>
    void polar2cartesian(const Array<Value>& polar, const Array<Value>& cartesian,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log, InterpMode interpolation_mode, bool prefilter) {
        polarCheckParameters_<PolarDirection::P2C>(cartesian, polar);

        const Device device = cartesian.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::geometry::cartesian2polar(
                    polar.share(), polar.strides(), polar.shape(),
                    cartesian.share(), cartesian.strides(), cartesian.shape(),
                    cartesian_center, radius_range, angle_range, log,
                    interpolation_mode, prefilter, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::geometry::cartesian2polar(
                    polar.share(), polar.strides(), polar.shape(),
                    cartesian.share(), cartesian.strides(), cartesian.shape(),
                    cartesian_center, radius_range, angle_range, log,
                    interpolation_mode, prefilter, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    template<typename Value, typename>
    void polar2cartesian(const Texture<Value>& polar, const Array<Value>& cartesian,
                         float2_t cartesian_center, float2_t radius_range, float2_t angle_range,
                         bool log) {
        const Device device = cartesian.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            const cpu::Texture<Value>& texture = polar.cpu();
            const Array<Value> tmp(texture.ptr, polar.shape(), texture.strides, polar.options());
            polar2cartesian(tmp, cartesian, cartesian_center, radius_range, angle_range, log, polar.interp(), false);
        } else {
            #ifdef NOA_ENABLE_CUDA
            if constexpr (!traits::is_any_v<Value, float, cfloat_t>) {
                NOA_THROW("In the CUDA backend, double-precision floating-points are not supported");
            } else {
                polarCheckParameters_<PolarDirection::P2C>(polar, cartesian);
                const cuda::Texture<Value>& texture = polar.cuda();
                cuda::geometry::polar2cartesian(
                        texture.array, texture.texture, polar.interp(), polar.shape(),
                        cartesian.share(), cartesian.strides(), cartesian.shape(),
                        cartesian_center, radius_range, angle_range,
                        log, stream.cuda());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    #define NOA_INSTANTIATE_UNIFIED_GEOMETRY_POLAR_(T) \
    template void cartesian2polar<T, void>(const Array<T>&, const Array<T>&, float2_t, float2_t, float2_t, bool, InterpMode, bool); \
    template void cartesian2polar<T, void>(const Texture<T>&, const Array<T>&, float2_t, float2_t, float2_t, bool);                 \
    template void polar2cartesian<T, void>(const Array<T>&, const Array<T>&, float2_t, float2_t, float2_t, bool, InterpMode, bool); \
    template void polar2cartesian<T, void>(const Texture<T>&, const Array<T>&, float2_t, float2_t, float2_t, bool)

    NOA_INSTANTIATE_UNIFIED_GEOMETRY_POLAR_(float);
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_POLAR_(double);
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_POLAR_(cfloat_t);
    NOA_INSTANTIATE_UNIFIED_GEOMETRY_POLAR_(cdouble_t);
}
