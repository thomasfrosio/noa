#pragma once

#include "noa/cpu/memory/Subregion.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Subregion.hpp"
#endif

#include "noa/unified/Array.hpp"
#include "noa/unified/Indexing.hpp"

namespace noa::memory {
    /// Extracts one or multiple {1|2|3}d subregions at various locations in the input array.
    /// \param[in] input        Input array to extract from.
    /// \param[out] subregions  Output subregion(s).
    /// \param[in] origins      Contiguous vector with the BDHW indexes of the subregions to extract. There should be
    ///                         one Vec4 per \p subregion batch. These indexes define the origin where to extract
    ///                         subregions from \p input. While usually within the input frame, subregions can
    ///                         be (partially) out-of-bound.
    /// \param border_mode      Border mode used for out-of-bound conditions.
    ///                         Can be BorderMode::{NOTHING|ZERO|VALUE|CLAMP|MIRROR|REFLECT}.
    /// \param border_value     Constant value to use for out-of-bound conditions.
    ///                         Only used if \p border_mode is BorderMode::VALUE.
    /// \note \p input and \p subregions should not overlap.
    template<typename Input, typename Subregion, typename Origin,
             typename Value = nt::mutable_value_type_t<Input>, typename = std::enable_if_t<
             nt::are_varray_of_restricted_numeric_v<Input, Subregion> &&
             nt::is_varray_v<Origin> &&
             nt::are_almost_same_value_type_v<Input, Subregion> &&
             nt::is_almost_same_v<nt::value_type_t<Input>, Value> &&
             nt::is_almost_any_v<nt::value_type_t<Origin>, Vec4<i32>, Vec4<i64>>>>
    void extract_subregions(
            const Input& input,
            const Subregion& subregions,
            const Origin& origins,
            BorderMode border_mode = BorderMode::ZERO,
            Value border_value = Value{0}
    ) {
        NOA_CHECK(!input.is_empty() && !subregions.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(input, subregions),
                  "The input and subregion(s) arrays should not overlap");
        NOA_CHECK(noa::indexing::is_contiguous_vector(origins) && origins.elements() == subregions.shape()[0],
                  "The indexes should be a contiguous vector of {} elements but got shape {} and strides {}",
                  subregions.shape()[0], origins.shape(), origins.strides());

        const Device device = subregions.device();
        NOA_CHECK(device == input.device() && device == origins.device(),
                  "The input and output arrays must be on the same device, but got input:{}, origins:{}, subregions:{}",
                  input.device(), origins.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::memory::extract_subregions(
                        input.get(), input.strides(), input.shape(),
                        subregions.get(), subregions.strides(), subregions.shape(),
                        origins.get(), border_mode, border_value, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::memory::extract_subregions(
                    input.get(), input.strides(), input.shape(),
                    subregions.get(), subregions.strides(), subregions.shape(),
                    origins.get(), border_mode, border_value, cuda_stream);
            cuda_stream.enqueue_attach(input, subregions, origins);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Inserts into the output array one or multiple {1|2|3}d subregions at various locations.
    /// \tparam T               Any data type.
    /// \param[in] subregions   Subregion(s) to insert into \p output.
    /// \param[out] output      Output array.
    /// \param[in] origins      Contiguous vector with the BDHW indexes of the subregions to insert into \p output.
    ///                         There should be one Vec4 per \p subregion batch. While usually within the output frame,
    ///                         subregions can be (partially) out-of-bound. This function assumes no overlap between
    ///                         subregions and an overlap may trigger a data race.
    template<typename Subregion, typename Output, typename Origin, typename = std::enable_if_t<
             nt::are_varray_of_restricted_numeric_v<Output, Subregion> &&
             nt::is_varray_v<Origin> &&
             nt::are_almost_same_value_type_v<Output, Subregion> &&
             nt::is_almost_any_v<nt::value_type_t<Origin>, Vec4<i32>, Vec4<i64>>>>
    void insert_subregions(
            const Subregion& subregions,
            const Output& output,
            const Origin& origins
    ) {
        NOA_CHECK(!output.is_empty() && !subregions.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(output, subregions),
                  "The subregion(s) and output arrays should not overlap");
        NOA_CHECK(noa::indexing::is_contiguous_vector(origins) && origins.elements() == subregions.shape()[0],
                  "The indexes should be a contiguous vector of {} elements but got shape {} and strides {}",
                  subregions.shape()[0], origins.shape(), origins.strides());

        const Device device = output.device();
        NOA_CHECK(device == subregions.device() && device == origins.device(),
                  "The input and output arrays must be on the same device, but got output:{}, origins:{}, subregions:{}",
                  device, origins.device(), subregions.device());

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::memory::insert_subregions(
                        subregions.get(), subregions.strides(), subregions.shape(),
                        output.get(), output.strides(), output.shape(),
                        origins.get(), threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::memory::insert_subregions(
                    subregions.get(), subregions.strides(), subregions.shape(),
                    output.get(), output.strides(), output.shape(),
                    origins.get(), cuda_stream);
            cuda_stream.enqueue_attach(output, subregions, origins);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Given a set of subregions with the same 2D/3D shape, compute an atlas layout (atlas shape + subregion origins).
    /// \details The atlas places the 2D/3D subregions in the same height-width plane. The depth of the atlas is the
    ///          same as the subregions'. Note that the shape of the atlas is not necessary a square. For instance,
    ///          with 4 subregions the atlas layout is 2x2, but with 5 subregions it goes to `3x2` with one empty
    ///          region.
    /// \param subregion_shape  BDHW shape of the subregion(s).
    ///                         The batch dimension is the number of subregion(s) to place into the atlas.
    /// \param[out] origins     Subregion origin(s), relative to the atlas shape.
    ///                         This function is effectively un-batching the 2D/3D subregions into a 2D/3D atlas.
    ///                         As such, the batch and depth dimensions are always set to 0, hence the possibility
    ///                         to return only the height and width dimension.
    /// \return                 Atlas shape.
    template<typename Origins, typename = std::enable_if_t<
             nt::is_varray_v<Origins> &&
             (nt::is_int2_v<nt::value_type_t<Origins>> ||
              nt::is_int4_v<nt::value_type_t<Origins>>)>>
    Shape4<i64> atlas_layout(const Shape4<i64>& subregion_shape, const Origins& output_origins) {
        NOA_ASSERT(noa::all(subregion_shape > 0));
        NOA_CHECK(!output_origins.is_empty(), "Empty array detected");
        NOA_CHECK(output_origins.device().is_cpu() &&
                  noa::indexing::is_contiguous_vector(output_origins) &&
                  output_origins.elements() == subregion_shape[0],
                  "The output subregion origins should be a CPU contiguous vector,"
                  "but got device={}, shape={} and strides={}",
                  output_origins.device(), output_origins.shape(), output_origins.strides());

        const auto columns = static_cast<i64>(
                noa::math::ceil(noa::math::sqrt(static_cast<f32>(subregion_shape[0]))));
        const i64 rows = (subregion_shape[0] + columns - 1) / columns;
        const auto atlas_shape = Shape4<i64>{
                1,
                subregion_shape[1],
                rows * subregion_shape[2],
                columns * subregion_shape[3]};

        output_origins.eval();
        const auto origins_1d = output_origins.accessor_contiguous_1d();
        for (i64 y = 0; y < rows; ++y) {
            for (i64 x = 0; x < columns; ++x) {
                const i64 idx = y * columns + x;
                if (idx >= subregion_shape[0])
                    break;
                if constexpr (nt::is_int4_v<typename Origins::value_type>)
                    origins_1d[idx] = {0, 0, y * subregion_shape[2], x * subregion_shape[3]};
                else
                    origins_1d[idx] = {y * subregion_shape[2], x * subregion_shape[3]};
            }
        }
        return atlas_shape;
    }

    template<typename Int = i64, size_t N = 4, typename = std::enable_if_t<
             nt::is_any_v<Int, i32, i64> && (N == 2 || N == 4)>>
    auto atlas_layout(const Shape4<i64>& subregion_shape) -> std::pair<Shape4<i64>, Array<Vec<Int, N>>> {
        Array<Vec<Int, N>> output_subregion_origins(subregion_shape.batch());
        return {atlas_layout(subregion_shape, output_subregion_origins), output_subregion_origins};
    }
}
