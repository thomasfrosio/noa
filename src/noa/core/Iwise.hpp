#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/indexing/Offset.hpp"
#include "noa/core/indexing/Subregion.hpp"

namespace noa {
    /// Defines evenly spaced values within a given interval.
    template<nt::numeric Value>
    struct Arange {
        using value_type = Value;
        using real_type = nt::value_type_t<value_type>;

        Value start{0};
        Value step{1};

        [[nodiscard]] NOA_HD constexpr Value operator()(nt::integer auto index) const noexcept {
            return start + static_cast<real_type>(index) * step;
        }
    };

    /// Defines evenly spaced numbers over a specified interval.
    template<nt::numeric T>
    struct Linspace {
        using value_type = T;
        using real_type = nt::value_type_t<value_type>;

        value_type start;
        value_type stop;
        bool endpoint{true};

        template<nt::integer I>
        struct Op {
            using value_type = T;
            using index_type = I;

            value_type start;
            value_type step;
            value_type stop;
            index_type index_end;
            bool endpoint;

            [[nodiscard]] NOA_HD constexpr value_type operator()(index_type i) const noexcept {
                return endpoint and i == index_end ? stop : start + static_cast<real_type>(i) * step;
            }
        };

        template<nt::integer I>
        auto for_size(const I& size) -> Op<I> requires nt::scalar<value_type> {
            Op<I> op;
            op.start = start;
            op.stop = stop;
            op.index_end = max(I{}, size - 1);
            op.endpoint = endpoint;

            const auto count = size - static_cast<I>(endpoint);
            const auto delta = stop - start;
            op.step = delta / static_cast<value_type>(count);
            return op;
        }

        template<nt::integer I>
        auto for_size(const I& size) -> Op<I> requires nt::complex<value_type> {
            auto real = Linspace<real_type>{start.real, stop.real, endpoint}.for_size(size);
            auto imag = Linspace<real_type>{start.imag, stop.imag, endpoint}.for_size(size);
            return Op{
                .start = {real.start, imag.start},
                .step = {real.step, imag.step},
                .stop = {real.stop, imag.stop},
                .index_end = real.index_end,
                .endpoint = real.endpoint
            };
        }
    };
}

namespace noa::guts {
    /// Arange index-wise operator for nd ranges.
    template<size_t N, nt::writable_nd<N> T, nt::integer I, typename R>
    class IwiseRange {
    public:
        using output_type = T;
        using index_type = I;
        using range_type = R;
        using value_type = nt::value_type_t<output_type>;
        using shape_type = Shape<index_type, N>;

        static constexpr bool CONTIGUOUS_1D = N == 1 and nt::marked_contiguous<T>;
        using strides_type = std::conditional_t<CONTIGUOUS_1D, Empty, Strides<index_type, N>>;

    public:
        constexpr IwiseRange(
            const output_type& accessor,
            const shape_type& shape,
            const range_type& range
        ) requires (not CONTIGUOUS_1D) :
            m_output(accessor),
            m_contiguous_strides(shape.strides()),
            m_range{range} {}

        constexpr IwiseRange(
            const output_type& accessor,
            const shape_type&, // for CTAD
            const range_type& range
        ) requires CONTIGUOUS_1D :
            m_output(accessor),
            m_range{range} {}

        template<typename... U> requires nt::iwise_core_indexing<N, index_type, U...>
        NOA_HD constexpr void operator()(U... indices) const {
            if constexpr (CONTIGUOUS_1D)
                m_output(indices...) = static_cast<value_type>(m_range(indices...));
            else
                m_output(indices...) = static_cast<value_type>(m_range(ni::offset_at(m_contiguous_strides, indices...)));
        }

    private:
        output_type m_output;
        NOA_NO_UNIQUE_ADDRESS strides_type m_contiguous_strides;
        range_type m_range;
    };

    template<size_t N, nt::writable_nd<N> T, nt::integer I>
    class Iota {
    public:
        using output_type = T;
        using index_type = I;
        using indices_type = Vec<I, N>;
        using shape_type = Shape<I, N>;
        using value_type = nt::value_type_t<output_type>;

        static constexpr bool CONTIGUOUS_1D = N == 1 and nt::marked_contiguous<T>;
        using strides_type = std::conditional_t<CONTIGUOUS_1D, Empty, Strides<index_type, N>>;

    public:
        constexpr Iota(
            const output_type& accessor,
            const shape_type& shape,
            const indices_type& tile
        ) requires (not CONTIGUOUS_1D) :
            m_output(accessor),
            m_tile(tile),
            m_contiguous_strides(shape.strides()) {}

        constexpr Iota(
            const output_type& accessor,
            const shape_type&, // for CTAD
            const indices_type& tile
        ) requires CONTIGUOUS_1D :
            m_output(accessor),
            m_tile(tile) {}

        NOA_HD constexpr void operator()(const indices_type& indices) const {
            if constexpr (CONTIGUOUS_1D) {
                m_output(indices) = static_cast<value_type>(indices % m_tile);
            } else {
                const auto iota = ni::offset_at(m_contiguous_strides, indices % m_tile);
                m_output(indices) = static_cast<value_type>(iota);
            }
        }

    private:
        output_type m_output;
        indices_type m_tile;
        NOA_NO_UNIQUE_ADDRESS strides_type m_contiguous_strides;
    };

    /// Extract subregions from one or multiple arrays.
    /// \details Subregions are defined by their 3d shape and their 2d (hw) or 4d (batch + dhw) origins.
    ///          If the subregion falls (even partially) out of the input bounds, the border mode is used
    ///          to handle that case.
    /// \note The origins dimensions might not correspond to the input/subregion dimensions because of the
    ///       rearranging before the index-wise transformation. Thus, this operator keeps the dimension "order"
    ///       and rearranges the origin on-the-fly (instead of allocating a new "origins" vector).
    template<Border MODE,
             nt::integer Index,
             nt::vec_integer_size<2, 4> Origins,
             nt::readable_nd<4> InputAccessor,
             nt::writable_nd<4> SubregionAccessor>
    class ExtractSubregion {
    public:
        using input_accessor_type = std::remove_const_t<InputAccessor>;
        using subregion_accessor_type = std::remove_const_t<SubregionAccessor>;
        using subregion_value_type = nt::value_type_t<subregion_accessor_type>;
        using index_type = std::remove_const_t<Index>;

        using origins_type = std::remove_const_t<Origins>;
        using origins_pointer_type = const origins_type*;

        using index4_type = Vec4<index_type>;
        using shape4_type = Shape4<index_type>;
        using subregion_value_or_empty_type = std::conditional_t<MODE == Border::VALUE, subregion_value_type, Empty>;

    public:
        constexpr ExtractSubregion(
            const input_accessor_type& input_accessor,
            const subregion_accessor_type& subregion_accessor,
            const shape4_type& input_shape,
            origins_pointer_type origins,
            subregion_value_type cvalue,
            const origins_type& order
        ) :
            m_input(input_accessor),
            m_subregions(subregion_accessor),
            m_subregion_origins(origins),
            m_input_shape(input_shape),
            m_order(order)
        {
            if constexpr (not nt::empty<subregion_value_or_empty_type>)
                m_cvalue = cvalue;
            else
                (void) cvalue;
        }

        constexpr void operator()(const index4_type& output_indices) const {
            // TODO For CUDA, the origins could copied to constant memory.
            //      Although these can be loaded in a single vectorized instruction.
            const auto corner_left = m_subregion_origins[output_indices[0]].reorder(m_order).template as<index_type>();

            index4_type input_indices;
            if constexpr (origins_type::SIZE == 2) {
                input_indices = {
                        0,
                        output_indices[1],
                        output_indices[2] + corner_left[0],
                        output_indices[3] + corner_left[1],
                };
            } else if constexpr (origins_type::SIZE == 4) {
                input_indices = {
                        corner_left[0],
                        output_indices[1] + corner_left[1],
                        output_indices[2] + corner_left[2],
                        output_indices[3] + corner_left[3],
                };
            } else {
                static_assert(nt::always_false<>);
            }

            if constexpr (MODE == Border::NOTHING) {
                if (ni::is_inbounds(m_input_shape, input_indices))
                    m_subregions(output_indices) = cast_or_abs_squared<subregion_value_type>(m_input(input_indices));

            } else if constexpr (MODE == Border::ZERO) {
                m_subregions(output_indices) = ni::is_inbounds(m_input_shape, input_indices) ?
                                               cast_or_abs_squared<subregion_value_type>(m_input(input_indices)) :
                                               subregion_value_type{};

            } else if constexpr (MODE == Border::VALUE) {
                m_subregions(output_indices) = ni::is_inbounds(m_input_shape, input_indices) ?
                                               cast_or_abs_squared<subregion_value_type>(m_input(input_indices)) :
                                               m_cvalue;

            } else {
                const index4_type bounded_indices = ni::index_at<MODE>(input_indices, m_input_shape);
                m_subregions(output_indices) = cast_or_abs_squared<subregion_value_type>(m_input(bounded_indices));
            }
        }

    private:
        input_accessor_type m_input;
        subregion_accessor_type m_subregions;
        origins_pointer_type m_subregion_origins;
        shape4_type m_input_shape;
        origins_type m_order;
        NOA_NO_UNIQUE_ADDRESS subregion_value_or_empty_type m_cvalue;
    };

    /// Insert subregions into one or multiple arrays.
    /// \details This works as expected and is similar to ExtractSubregion. Subregions can be (even partially) out
    ///          of the output bounds. The only catch here is that overlapped subregions are not explicitly supported
    ///          since it is not clear what we want in these cases (add?), so for now, just ignore it out.
    template<nt::integer Index,
             nt::vec_integer_size<2, 4> Origins,
             nt::readable_nd<4> SubregionAccessor,
             nt::writable_nd<4> OutputAccessor>
    class InsertSubregion {
    public:
        using output_accessor_type = std::remove_const_t<OutputAccessor>;
        using output_value_type = nt::value_type_t<output_accessor_type>;
        using subregion_accessor_type = std::remove_const_t<SubregionAccessor>;
        using index_type = std::remove_const_t<Index>;

        using origins_type = std::remove_const_t<Origins>;
        using origins_pointer_type = const origins_type*;

        using index4_type = Vec4<Index>;
        using shape4_type = Shape4<Index>;

    public:
        constexpr InsertSubregion(
            const subregion_accessor_type& subregion_accessor,
            const output_accessor_type& output_accessor,
            const shape4_type& output_shape,
            origins_pointer_type origins,
            const origins_type& order
        ) :
            m_output(output_accessor),
            m_subregions(subregion_accessor),
            m_subregion_origins(origins),
            m_output_shape(output_shape),
            m_order(order) {}

        constexpr void operator()(const index4_type& input_indices) const {
            // TODO For CUDA, the origins could copied to constant memory.
            //      Although these can be loaded in a single vectorized instruction.
            const auto corner_left = m_subregion_origins[input_indices[0]].reorder(m_order).template as<index_type>();

            index4_type output_indices;
            if constexpr (origins_type::SIZE == 2) {
                output_indices = {
                        0,
                        input_indices[1],
                        input_indices[2] + corner_left[0],
                        input_indices[3] + corner_left[1],
                };
            } else if constexpr (origins_type::SIZE == 4) {
                output_indices = {
                        corner_left[0],
                        input_indices[1] + corner_left[1],
                        input_indices[2] + corner_left[2],
                        input_indices[3] + corner_left[3],
                };
            } else {
                static_assert(nt::always_false<>);
            }

            // We assume no overlap in the output between subregions.
            if (ni::is_inbounds(m_output_shape, output_indices))
                m_output(output_indices) = cast_or_abs_squared<output_value_type>(m_subregions(input_indices));
        }

    private:
        output_accessor_type m_output;
        subregion_accessor_type m_subregions;
        origins_pointer_type m_subregion_origins;
        shape4_type m_output_shape;
        origins_type m_order;
    };

    /// 4d index-wise operator to resize an array, out of place.
    /// \details The "border_left" and "border_right", specify the number of elements to crop (negative value)
    ///          or pad (positive value) on the left or right side of each dimension. Padded elements are handled
    ///          according to the Border. The input and output arrays should not overlap.
    template<Border MODE,
             nt::sinteger Index,
             nt::readable_nd<4> Input,
             nt::writable_nd<4> Output>
    class Resize {
    public:
        static_assert(MODE != Border::NOTHING);

        using input_accessor_type = Input;
        using output_accessor_type = Output;
        using output_value_type = nt::value_type_t<output_accessor_type>;
        using index_type = Index;
        using indices_type = Vec<index_type, 4>;
        using shape_type = Shape<index_type, 4>;
        using output_value_or_empty_type = std::conditional_t<MODE == Border::VALUE, output_value_type, Empty>;

        static constexpr bool IS_BOUNDLESS = MODE != Border::VALUE and MODE != Border::ZERO;
        using index4_or_empty_type = std::conditional_t<IS_BOUNDLESS, Empty, indices_type>;

    public:
        constexpr Resize(
            const input_accessor_type& input_accessor,
            const output_accessor_type& output_accessor,
            const shape_type& input_shape,
            const shape_type& output_shape,
            const indices_type& border_left,
            const indices_type& border_right,
            output_value_type cvalue
        ) :
            m_input(input_accessor),
            m_output(output_accessor),
            m_input_shape(input_shape),
            m_crop_left(min(border_left, index_type{}) * -1),
            m_pad_left(max(border_left, index_type{}))
        {
            if constexpr (MODE == Border::VALUE or MODE == Border::ZERO) {
                const auto pad_right = max(border_right, index_type{});
                m_right = output_shape.vec - pad_right;
            } else {
                (void) border_right;
                (void) output_shape;
            }

            if constexpr (MODE == Border::VALUE)
                m_cvalue = cvalue;
            else
                (void) cvalue;
        }

        constexpr void operator()(const indices_type& output_indices) const {
            const auto input_indices = output_indices - m_pad_left + m_crop_left;

            if constexpr (MODE == Border::VALUE or MODE == Border::ZERO) {
                const auto is_within_input = [](auto i, auto l, auto r) { return i >= l and i < r; };
                if constexpr (MODE == Border::VALUE) {
                    m_output(output_indices) =
                            vall(is_within_input, output_indices, m_pad_left, m_right) ?
                            cast_or_abs_squared<output_value_type>(m_input(input_indices)) : m_cvalue;
                } else {
                    m_output(output_indices) =
                            vall(is_within_input, output_indices, m_pad_left, m_right) ?
                            cast_or_abs_squared<output_value_type>(m_input(input_indices)) : output_value_type{};
                }
            } else { // CLAMP or PERIODIC or MIRROR or REFLECT
                const indices_type indices_bounded = ni::index_at<MODE>(input_indices, m_input_shape);
                m_output(output_indices) = cast_or_abs_squared<output_value_type>(m_input(indices_bounded));
            }
        }

    private:
        input_accessor_type m_input;
        output_accessor_type m_output;
        shape_type m_input_shape;
        indices_type m_crop_left;
        indices_type m_pad_left;
        NOA_NO_UNIQUE_ADDRESS index4_or_empty_type m_right;
        NOA_NO_UNIQUE_ADDRESS output_value_or_empty_type m_cvalue;
    };

    /// Computes the common subregions between the input and output.
    /// These can then be used to copy the input subregion into the output subregion.
    [[nodiscard]] constexpr auto extract_common_subregion(
            const Shape4<i64>& input_shape, const Shape4<i64>& output_shape,
            const Vec4<i64>& border_left, const Vec4<i64>& border_right
    ) noexcept -> Pair<ni::Subregion<4, ni::Slice, ni::Slice, ni::Slice, ni::Slice>,
                       ni::Subregion<4, ni::Slice, ni::Slice, ni::Slice, ni::Slice>> {
        // Exclude the regions in the input that don't end up in the output.
        const auto crop_left = min(border_left, 0) * -1;
        const auto crop_right = min(border_right, 0) * -1;
        const auto cropped_input = ni::make_subregion<4>(
                    ni::Slice{crop_left[0], input_shape[0] - crop_right[0]},
                    ni::Slice{crop_left[1], input_shape[1] - crop_right[1]},
                    ni::Slice{crop_left[2], input_shape[2] - crop_right[2]},
                    ni::Slice{crop_left[3], input_shape[3] - crop_right[3]});

        // Exclude the regions in the output that are not from the input.
        const auto pad_left = max(border_left, 0);
        const auto pad_right = max(border_right, 0);
        const auto cropped_output = ni::make_subregion<4>(
                    ni::Slice{pad_left[0], output_shape[0] - pad_right[0]},
                    ni::Slice{pad_left[1], output_shape[1] - pad_right[1]},
                    ni::Slice{pad_left[2], output_shape[2] - pad_right[2]},
                    ni::Slice{pad_left[3], output_shape[3] - pad_right[3]});

        // One can now copy cropped_input -> cropped_output.
        return {cropped_input, cropped_output};
    }
}
