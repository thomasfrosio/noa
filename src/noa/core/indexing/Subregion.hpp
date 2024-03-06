#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Exception.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/utils/ClampCast.hpp"
#include "noa/core/utils/SafeCast.hpp"

#ifdef NOA_IS_OFFLINE
namespace noa::indexing {
    /// Ellipsis or "..." operator, which selects the full extent of the remaining outermost dimension(s).
    struct Ellipsis {};

    /// Selects the entire the dimension.
    struct FullExtent {};

    /// Slice operator.
    /// Negative indexes are valid and starts from the end like in python.
    /// Indexes will be clamped to the dimension size.
    /// The step must be non-zero positive (negative strides are not supported).
    struct Slice {
        template<typename T = int64_t, typename U = int64_t, typename V = int64_t>
        constexpr explicit Slice(
                T start_ = 0,
                U end_ = std::numeric_limits<int64_t>::max(),
                V step_ = V{1})
                : start(static_cast<int64_t>(start_)),
                  end(static_cast<int64_t>(end_)),
                  step(static_cast<int64_t>(step_)) {}

        int64_t start{};
        int64_t end{};
        int64_t step{};
    };

    template<typename U>
    static constexpr bool is_subregion_indexer_v = std::bool_constant<
            nt::is_int_v<U> ||
            nt::is_almost_same_v<U, FullExtent> ||
            nt::is_almost_same_v<U, Slice>
    >::value;
    template<typename... Ts> using are_subregion_indexers = nt::bool_and<is_subregion_indexer_v<Ts>...>;
    template<typename... Ts> static constexpr bool are_subregion_indexers_v = are_subregion_indexers<Ts...>::value;

    /// Subregion object, i.e. a capture of the indexing object for each dimension.
    template<typename B = FullExtent, typename D = FullExtent, typename H = FullExtent, typename W = FullExtent,
             typename = std::enable_if_t<are_subregion_indexers_v<B, D, H, W>>>
    struct Subregion {
        B batch{};
        D depth{};
        H height{};
        W width{};

    public:
        constexpr explicit Subregion(B batch_ = {}, D depth_ = {}, H height_ = {}, W width_ = {})
                : batch(batch_), depth(depth_), height(height_), width(width_) {}

        constexpr explicit Subregion(Ellipsis)                       : Subregion(FullExtent{}, FullExtent{}, FullExtent{}, FullExtent{}) {}
        constexpr Subregion(Ellipsis, W width_)                      : Subregion(FullExtent{}, FullExtent{}, FullExtent{}, width_) {}
        constexpr Subregion(Ellipsis, H height_, W width_)           : Subregion(FullExtent{}, FullExtent{}, height_, width_) {}
        constexpr Subregion(Ellipsis, D depth_, H height_, W width_) : Subregion(FullExtent{}, depth_, height_, width_) {}
    };

    /// Utility to create indexing subregions.
    /// Dimensions can be extracted using either:
    /// -   A single index value: This is bound-checked. Negative values are allowed.
    /// -   FullExtent: Selects the entire dimension.
    /// -   Slice: Slice operator. Slices are clamped to the dimension size. Negative values are allowed.
    /// -   Ellipsis: Fills all unspecified dimensions with FullExtent.
    struct SubregionIndexer {
    public:
        using index_type = int64_t;
        using offset_type = int64_t;
        using shape_type = Shape4<index_type>;
        using strides_type = Strides4<offset_type>;

    public:
        shape_type shape{};
        strides_type strides{};
        offset_type offset{0};

    public:
        constexpr SubregionIndexer() = default;

        template<typename T, typename U, typename V = int64_t>
        constexpr SubregionIndexer(
                const Shape4<T>& start_shape,
                const Strides4<U>& start_strides,
                V start_offset = V{0}
        ) noexcept
                : shape(start_shape.template as_safe<index_type>()),
                  strides(start_strides.template as_safe<offset_type>()),
                  offset(start_offset) {}

        template<typename... Ts>
        [[nodiscard]] constexpr SubregionIndexer extract_subregion(Ts&&... indexes) const {
            Subregion subregion(std::forward<Ts>(indexes)...);
            SubregionIndexer out{};
            extract_dim_(subregion.batch,  0, shape[0], strides[0], out.shape[0], out.strides[0], out.offset);
            extract_dim_(subregion.depth,  1, shape[1], strides[1], out.shape[1], out.strides[1], out.offset);
            extract_dim_(subregion.height, 2, shape[2], strides[2], out.shape[2], out.strides[2], out.offset);
            extract_dim_(subregion.width,  3, shape[3], strides[3], out.shape[3], out.strides[3], out.offset);
            return out;
        }

    private:
        // Compute the new size, strides and offset, for one dimension, given an indexing mode (integral, slice or full).
        template<typename IndexMode>
        static constexpr void extract_dim_(
                IndexMode idx_mode, int64_t dim,
                int64_t old_size, int64_t old_strides,
                int64_t& new_size, int64_t& new_strides, int64_t& new_offset
        ) {
            if constexpr (nt::is_int_v<IndexMode>) {
                auto index = clamp_cast<int64_t>(idx_mode);
                check(!(index < -old_size || index >= old_size),
                      "Index {} is out of range for a size of {} at dimension {}",
                      index, old_size, dim);

                if (index < 0)
                    index += old_size;
                new_strides = old_strides; // or 0
                new_size = 1;
                new_offset += old_strides * index;

            } else if constexpr(std::is_same_v<FullExtent, IndexMode>) {
                new_strides = old_strides;
                new_size = old_size;
                new_offset += 0;
                (void) idx_mode;
                (void) dim;

            } else if constexpr(std::is_same_v<Slice, IndexMode>) {
                check(idx_mode.step > 0, "Slice step must be positive, got {}", idx_mode.step);

                if (idx_mode.start < 0)
                    idx_mode.start += old_size;
                if (idx_mode.end < 0)
                    idx_mode.end += old_size;

                idx_mode.start = clamp(idx_mode.start, int64_t{0}, old_size);
                idx_mode.end = clamp(idx_mode.end, idx_mode.start, old_size);

                new_size = divide_up(idx_mode.end - idx_mode.start, idx_mode.step);
                new_strides = old_strides * idx_mode.step;
                new_offset += idx_mode.start * old_strides;
                (void) dim;
            } else {
                static_assert(nt::always_false_v<IndexMode>);
            }
        }
    };
}
#endif
