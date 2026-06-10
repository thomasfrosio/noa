#pragma once

#include "noa/runtime/Array.hpp"
#include "noa/runtime/Iwise.hpp"

namespace noa::details {
    template<
        Border MODE,
        nt::integer Index, usize N,
        nt::readable_nd<N> Input,
        nt::writable_nd<N + 1> Subregions,
        nt::readable_nd<1> Origins>
    class ExtractSubregion {
    public:
        using input_type = std::remove_const_t<Input>;
        using subregions_type = std::remove_const_t<Subregions>;
        using subregions_value_type = nt::value_type_t<subregions_type>;
        using index_type = std::remove_const_t<Index>;

        using origins_type = std::remove_const_t<Origins>;
        using origins_value_type = nt::mutable_value_type_t<origins_type>;
        static_assert(nt::vec_integer_size<origins_value_type, N>);

        using shape_type = Shape<index_type, N>;
        using subregions_value_or_empty_type = std::conditional_t<MODE == Border::VALUE, subregions_value_type, Empty>;

    public:
        constexpr ExtractSubregion(
            const input_type& input,
            const subregions_type& subregions,
            const origins_type& origins,
            const shape_type& input_shape,
            subregions_value_type cvalue
        ) :
            m_input(input),
            m_subregions(subregions),
            m_origins(origins),
            m_input_shape(input_shape)
        {
            if constexpr (not nt::empty<subregions_value_or_empty_type>)
                m_cvalue = cvalue;
            else
                (void) cvalue;
        }

        template<nt::same_as<index_type>... T>
        constexpr void operator()(index_type b, T... output_indices) const {
            const auto corner_left = m_origins[b].template as<index_type>();
            const auto input_indices = Vec{output_indices...} + corner_left;

            if constexpr (MODE == Border::NOTHING) {
                if (is_inbound(m_input_shape, input_indices))
                    m_subregions(b, output_indices...) =
                        cast_or_abs_squared<subregions_value_type>(m_input(input_indices));
            } else if constexpr (MODE == Border::ZERO) {
                m_subregions(b, output_indices...) =
                    is_inbound(m_input_shape, input_indices) ?
                        cast_or_abs_squared<subregions_value_type>(m_input(input_indices)) :
                        subregions_value_type{};
            } else if constexpr (MODE == Border::VALUE) {
                m_subregions(b, output_indices...) =
                    is_inbound(m_input_shape, input_indices) ?
                        cast_or_abs_squared<subregions_value_type>(m_input(input_indices)) :
                        m_cvalue;
            } else {
                const auto bounded_indices = index_at<MODE>(input_indices, m_input_shape);
                m_subregions(b, output_indices...) =
                    cast_or_abs_squared<subregions_value_type>(m_input(bounded_indices));
            }
        }

    private:
        input_type m_input;
        subregions_type m_subregions;
        origins_type m_origins;
        shape_type m_input_shape;
        NOA_NO_UNIQUE_ADDRESS subregions_value_or_empty_type m_cvalue;
    };

    template<
        nt::integer Index, usize N,
        nt::readable_nd<N + 1> Subregions,
        nt::writable_nd<N> Output,
        nt::readable_nd<1> Origins>
    class InsertSubregion {
    public:
        using output_type = std::remove_const_t<Output>;
        using output_value_type = nt::value_type_t<output_type>;
        using subregions_type = std::remove_const_t<Subregions>;
        using index_type = std::remove_const_t<Index>;
        using shape_type = Shape<Index, N>;

        using origins_type = std::remove_const_t<Origins>;
        using origins_value_type = nt::mutable_value_type_t<origins_type>;
        static_assert(nt::vec_integer_size<origins_value_type, N>);

    public:
        constexpr InsertSubregion(
            const subregions_type& subregions,
            const output_type& output,
            const origins_type& origins,
            const shape_type& output_shape
        ) :
            m_output(output),
            m_subregions(subregions),
            m_origins(origins),
            m_output_shape(output_shape) {}

        template<nt::same_as<index_type>... T>
        constexpr void operator()(index_type b, T... input_indices) const {
            const auto corner_left = m_origins[b].template as<index_type>();
            const auto output_indices = Vec{input_indices...} + corner_left;

            // SAFETY: We assume no overlap in the output between subregions. If overlap, this can data race.
            if (is_inbound(m_output_shape, output_indices))
                m_output(output_indices) = cast_or_abs_squared<output_value_type>(m_subregions(b, input_indices...));
        }

    private:
        output_type m_output;
        subregions_type m_subregions;
        origins_type m_origins;
        shape_type m_output_shape;
        origins_type m_order;
    };

}

namespace noa {
    /// Extracts one or multiple ND subregions at various locations in the input array.
    /// \param[in] input:
    ///     (...n) Input array to extract from.
    /// \param[out] subregions:
    ///     (b,...n) Output batched subregion(s).
    ///     Input values are cast to the subregion value type and it should not overlap with the input.
    /// \param[in] origins:
    ///     (b) Contiguous vector with the (n)d-offsets of the subregions to extract.
    ///     There should be one set of offsets per subregion batch.
    ///     These offsets define the origin where to extract subregions from input.
    ///     While usually within the input frame, subregions can be (partially) out-of-bound.
    /// \param border_mode:
    ///     Border mode used for out-of-bound conditions.
    ///     Can be Border::{NOTHING|ZERO|VALUE|CLAMP|MIRROR|REFLECT}.
    /// \param border_value:
    ///     Constant value to use for out-of-bound conditions.
    ///     Only used if border_mode is Border::VALUE.
    /// \note This function assumes the rightmost order as the fastest.
    template<nt::readable_array_decay Input,
             nt::writable_array_decay Subregion,
             nt::readable_array_decay Origin>
    requires (nt::array_decay_with_compatible_or_spectrum_types<Input, Subregion> and
              (nt::array_size_v<Input> + 1 == nt::array_size_v<Subregion>) and
              nt::array_size_v<Origin> == 1 and
              nt::vec_integer_size<nt::value_type_t<Origin>, nt::array_size_v<Input>>)
    void extract_subregions(
        Input&& input,
        Subregion&& subregions,
        Origin&& origins,
        Border border_mode = Border::ZERO,
        nt::value_type_t<Subregion> border_value = {}
    ) {
        check(nd::are_arrays_valid(input, subregions), "Empty array detected");
        check(not are_overlapped(input, subregions),
              "The input and subregion(s) arrays should not overlap");
        check(origins.is_contiguous() and origins.n_elements() == subregions.shape()[0],
              "The origin should be a contiguous vector of {} elements but got shape={} and strides={}",
              subregions.shape()[0], origins.shape(), origins.strides());

        const Device device = subregions.device();
        check(device == input.device() and device == origins.device(),
              "The input and output arrays must be on the same device, but got input:device={}, origins:device={}, subregions:device={}",
              input.device(), origins.device(), device);

        constexpr auto N = nt::array_size_v<Input>;
        using input_accessor_t = AccessorRestrict<nt::const_value_type_t<Input>, N, isize>;
        using subregion_accessor_t = AccessorRestrict<nt::value_type_t<Subregion>, N + 1, isize>;
        using origin_accessor_t = AccessorRestrictContiguous<nt::const_value_type_t<Origin>, 1, isize>;
        const auto input_accessor = input_accessor_t(input.data(), input.strides());
        const auto subregion_accessor = subregion_accessor_t(subregions.data(), subregions.strides());
        const auto origin_accessor = origin_accessor_t(origins.data());

        switch (border_mode) {
            #define NOA_GENERATE_SUBREGION_(border)                                                                             \
            case border: {                                                                                                      \
                auto op = nd::ExtractSubregion<border, isize, N, input_accessor_t, subregion_accessor_t, origin_accessor_t>(    \
                        input_accessor, subregion_accessor, origin_accessor, input.shape(), border_value);                      \
                return iwise(subregions.shape(), device, std::move(op),                                                         \
                             std::forward<Input>(input),                                                                        \
                             std::forward<Subregion>(subregions),                                                               \
                             std::forward<Origin>(origins));                                                                    \
            }
            NOA_GENERATE_SUBREGION_(Border::NOTHING)
            NOA_GENERATE_SUBREGION_(Border::ZERO)
            NOA_GENERATE_SUBREGION_(Border::VALUE)
            NOA_GENERATE_SUBREGION_(Border::CLAMP)
            NOA_GENERATE_SUBREGION_(Border::MIRROR)
            NOA_GENERATE_SUBREGION_(Border::PERIODIC)
            NOA_GENERATE_SUBREGION_(Border::REFLECT)

            default:
                panic("{} not supported", border_mode);
        }
    }

    /// Inserts into the output array one or multiple ND subregions at various locations.
    /// \param[in] subregions:
    ///     (b,...n) Batched subregions to insert into the output.
    /// \param[out] output:
    ///     (...n) Output array.
    /// \param[in] origins
    ///     (b) Contiguous vector with the (n)d-offsets of the subregions to insert into the output.
    ///     There should be one offset per subregion batch.
    ///     While usually within the output frame, subregions can be (partially) out-of-bound.
    ///     This function assumes no overlap between subregions and an overlap may trigger a data race.
    /// \note This function assumes the rightmost order as the fastest.
    template<nt::readable_array_decay Subregion,
             nt::writable_array_decay Output,
             nt::readable_array_decay Origin>
    requires (nt::array_decay_with_compatible_or_spectrum_types<Subregion, Output> and
              (nt::array_size_v<Output> + 1 == nt::array_size_v<Subregion>) and
              nt::array_size_v<Origin> == 1 and
              nt::vec_integer_size<nt::value_type_t<Origin>, nt::array_size_v<Output>>)
    void insert_subregions(
        Subregion&& subregions,
        Output&& output,
        Origin&& origins
    ) {
        check(nd::are_arrays_valid(output, subregions), "Empty array detected");
        check(not are_overlapped(output, subregions),
              "The subregion(s) and output arrays should not overlap");
        check(origins.is_contiguous() and origins.n_elements() == subregions.shape()[0],
              "The origins should be a contiguous vector of {} elements but got shape={} and strides={}",
              subregions.shape()[0], origins.shape(), origins.strides());

        const Device device = output.device();
        check(device == subregions.device() and device == origins.device(),
              "The input and output arrays must be on the same device, but got output:device={}, origins:device={}, subregions:device={}",
              device, origins.device(), subregions.device());

        constexpr auto N = nt::array_size_v<Output>;
        using subregion_accessor_t = AccessorRestrict<nt::const_value_type_t<Subregion>, N + 1, isize>;
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, N, isize>;
        using origin_accessor_t = AccessorRestrictContiguous<nt::const_value_type_t<Origin>, 1, isize>;
        const auto subregion_accessor = subregion_accessor_t(subregions.data(), subregions.strides());
        const auto output_accessor = output_accessor_t(output.data(), output.strides());
        const auto origin_accessor = origin_accessor_t(origins.data());

        auto op = nd::InsertSubregion<isize, N, subregion_accessor_t, output_accessor_t, origin_accessor_t>(
            subregion_accessor, output_accessor, origin_accessor, output.shape());
        iwise(subregions.shape(), device, std::move(op),
              std::forward<Subregion>(subregions),
              std::forward<Output>(output),
              std::forward<Origin>(origins));
    }

    /// Given a set of ND subregions (N >= 2), compute an atlas layout (atlas shape + subregion origins).
    /// \details
    ///     The atlas is a 2d grid containing the subregions, so its HW is a multiple of the HW of the subregion.
    ///     The outer dimensions of the atlas (if any, e.g. the depth in 3d case) is the same as the subregion.
    ///     Note that the shape of the atlas is not necessary a square. For instance, with four subregions, the
    ///     atlas layout is 2x2, but with five subregions it goes to `3x2` with one empty region.
    /// \param subregion_shape:
    ///     Shape of a single subregion.
    /// \param[out] output_origins
    ///     Subregion origin(s), relative to the atlas shape.
    ///     Its size specifies the number of subregions to map onto the 2d grid.
    /// \return Atlas shape.
    template<usize N, nt::vec_integer_size<N> T, ArrayOwnership O> requires (N >= 2)
    auto atlas_layout(const Shape<isize, N>& subregion_shape, const Array<T, 1, O>& output_origins) -> Shape<isize, N> {
        NOA_ASSERT(not subregion_shape.is_empty());

        check(not output_origins.is_empty(), "Empty array detected");
        check(output_origins.is_dereferenceable() and output_origins.is_contiguous(),
              "The output subregion origins should be a CPU-dereferenceable contiguous vector");

        const auto n_subregions = output_origins.n_elements();
        const auto columns = static_cast<isize>(ceil(sqrt(static_cast<f32>(n_subregions))));
        const auto rows = (n_subregions + columns - 1) / columns;
        auto atlas_shape = subregion_shape;
        atlas_shape[N - 2] = rows * subregion_shape[N - 2];
        atlas_shape[N - 1] = columns * subregion_shape[N - 1];

        output_origins.eval();
        const auto span = output_origins.span_1d();
        for (isize y{}; y < rows; ++y) {
            for (isize x{}; x < columns; ++x) {
                const isize idx = y * columns + x;
                if (idx >= subregion_shape[0])
                    break;
                auto origin = T{};
                origin[N - 2] = y * subregion_shape[N - 2];
                origin[N - 1] = x * subregion_shape[N - 1];
                span[idx] = origin;
            }
        }
        return atlas_shape;
    }

    /// Given a set of subregions, compute an atlas layout (atlas shape + subregion origins).
    /// \example
    /// \code
    /// // Constructs an atlas of subregions.
    /// auto batched_subregions = Array<f32, N + 1>{...};
    /// auto subregion_shape = batched_subregions.shape().pop_front(); // Shape<isize, N>
    /// auto n_subregions = batched_subregions.shape()[0];
    /// auto [atlas_shape, atlas_origins] = noa::atlas_layout<i32>(subregion_shape, n_subregions);
    /// auto atlas = noa::zeros<f32>(atlas_shape);
    /// noa::insert_subregions(batched_subregions, atlas, atlas_origins);
    /// \endcode
    template<nt::integer T = isize, usize N> requires (N >= 2)
    auto atlas_layout(const Shape<isize, N>& subregion_shape, isize n_subregions) {
        auto output_subregion_origins = Array<Vec<T, N>, 1>(n_subregions);
        auto atlas_shape = atlas_layout(subregion_shape, output_subregion_origins);
        return Pair{atlas_shape, std::move(output_subregion_origins)};
    }
}
