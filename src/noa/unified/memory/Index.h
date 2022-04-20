#pragma once

#include "noa/cpu/memory/Index.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Index.h"
#endif

#include "noa/unified/Array.h"

// TODO Add sequences

namespace noa::memory {
    /// Extracts one or multiple ND (1 <= N <= 3) subregions at various locations in the input array.
    /// \tparam T                   Any data type.
    /// \param[in] input            Input array to extract from.
    /// \param[out] subregions      Output subregion(s).
    /// \param[in] origins          Rightmost indexes, defining the origin where to extract subregions from \p input.
    ///                             Should be a row vector with a set of 4 indexes per subregion. The outermost
    ///                             dimension of \p subregions is the batch dimension and sets the number of subregions
    ///                             to extract. While usually within the input frame, subregions can be (partially)
    ///                             out-of-bound.
    /// \param border_mode          Border mode used for out-of-bound conditions.
    ///                             Can be BORDER_{NOTHING|ZERO|VALUE|CLAMP|MIRROR|REFLECT}.
    /// \param border_value         Constant value to use for out-of-bound conditions.
    ///                             Only used if \p border_mode is BORDER_VALUE.
    /// \note \p input and \p subregions should not overlap.
    template<typename T>
    void extract(const Array<T>& input, const Array<T>& subregions, const Array<int4_t>& origins,
                 BorderMode border_mode = BORDER_ZERO, T border_value = T(0)) {
        NOA_CHECK(subregions.device() == input.device(),
                  "The input and subregion arrays must be on the same device, but got input:{} and subregion:{}",
                  subregions.device(), input.device());
        NOA_CHECK(origins.device().cpu() || origins.device() == subregions.device(),
                  "The indexes should be on the same device as the other arrays");
        NOA_CHECK(origins.shape().ndim() == 1 && origins.shape()[3] == subregions.shape()[0],
                  "The indexes should be specified as a row vector of shape {} but got {}",
                  int4_t{1, 1, 1, subregions.shape()[0]}, origins.shape());
        NOA_CHECK(subregions.get() != input.get(), "The subregion(s) and the output arrays should not overlap");

        const Device device(subregions.device());
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::extract<T>(input.share(), input.stride(), input.shape(),
                                    subregions.share(), subregions.stride(), subregions.shape(),
                                    origins.share(), border_mode, border_value, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::extract<T>(input.share(), input.stride(), input.shape(),
                                     subregions.share(), subregions.stride(), subregions.shape(),
                                     origins.share(), border_mode, border_value, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Inserts into the output array one or multiple ND (1 <= N <= 3) subregions at various locations.
    /// \tparam T                   Any data type.
    /// \param[in] subregions       Subregion(s) to insert into \p output.
    /// \param[out] output          Output array.
    /// \param[in] origins          Rightmost indexes, defining the origin where to insert subregions into \p output.
    ///                             Should be a row vector with a set of 4 indexes per subregion. The outermost
    ///                             dimension of \p subregion_shape is the batch dimension and sets the number of
    ///                             subregions to insert. Thus, subregions can be up to 3 dimensions. While usually
    ///                             within the output frame, subregions can be (partially) out-of-bound. However,
    ///                             this function assumes no overlap between subregions. There's no guarantee on the
    ///                             order of insertion.
    template<typename T>
    void insert(const Array<T>& subregions, const Array<T>& output, const Array<int4_t>& origins) {
        NOA_CHECK(subregions.device() == output.device(),
                  "The output and subregion arrays must be on the same device, but got output:{} and subregion:{}",
                  subregions.device(), output.device());
        NOA_CHECK(origins.device().cpu() || origins.device() == subregions.device(),
                  "The indexes should be on the same device as the other arrays");
        NOA_CHECK(origins.shape().ndim() == 1 && origins.shape()[3] == subregions.shape()[0],
                  "The indexes should be specified as a row vector of shape {} but got {}",
                  int4_t{1, 1, 1, subregions.shape()[0]}, origins.shape());
        NOA_CHECK(subregions.get() != output.get(), "The subregion(s) and the output arrays should not overlap");

        const Device device(subregions.device());
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::insert<T>(subregions.share(), subregions.stride(), subregions.shape(),
                                   output.share(), output.stride(), output.shape(),
                                   origins.share(), stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::insert<T>(subregions.share(), subregions.stride(), subregions.shape(),
                                    output.share(), output.stride(), output.shape(),
                                    origins.share(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Gets the atlas layout (shape + subregion origins).
    /// \param subregion_shape          Rightmost shape of the subregion(s).
    ///                                 The outermost dimension is the number of subregion(s) to place into the atlas.
    /// \param[out] origins             Subregion origin(s), relative to the atlas shape.
    /// \return                         Atlas shape.
    ///
    /// \note The shape of the atlas is not necessary a square. For instance, with 4 subregions the atlas layout
    ///       is `2x2`, but with 5 subregions is goes to `3x2` with one empty region. Subregions are in row-major order.
    /// \note The origin is always 0 for the two outermost dimensions. The function is effectively un-batching the
    ///       2D/3D subregions into a 2D/3D atlas.
    NOA_IH size4_t atlasLayout(size4_t subregion_shape, int4_t* origins) {
        return cpu::memory::atlasLayout(subregion_shape, origins);
    }
}
