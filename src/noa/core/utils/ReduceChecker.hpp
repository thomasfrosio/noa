#pragma once

#include "noa/core/Traits.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/types/Tuple.hpp"

namespace noa {
    ///
    /// \tparam N
    /// \tparam Op
    /// \tparam Index
    /// \tparam Reduced
    /// \tparam Output
    template<size_t N, typename Op, typename Index, typename Reduced, typename Output>
    requires std::is_integral_v<Index> && nt::are_tuple_v<Reduced, Output>
    struct ReduceIwiseChecker {
        static constexpr size_t N_DIMENSIONS = N;
        using index_type = Index;

        static constexpr auto has_init() -> bool { return m_has_init.first; }
        static constexpr auto is_init_packed() -> bool { return m_has_init.second; }
        static constexpr auto has_join() -> bool { return m_has_join; }
        static constexpr auto has_final() -> bool { return m_has_final.first; }
        static constexpr auto is_final_defaulted() -> bool { return m_has_final.second; }
        static constexpr auto is_valid() -> bool { return has_init() && has_join() && has_final(); }

    private:
        using reduced_element_list = std::remove_reference_t<Reduced>::element_list;
        using output_element_list = std::remove_reference_t<Output>::element_list;
        static constexpr Pair m_has_init = check_init_(nt::repeat<Index, N, nt::TypeList>{}, reduced_element_list{});
        static constexpr bool m_has_join = check_join_(reduced_element_list{});
        static constexpr Pair m_has_final = check_final_(reduced_element_list{}, output_element_list{});

        template<typename... Indices, typename... Accessors>
        static constexpr Pair<bool, bool> check_init_(nt::TypeList<Indices...>, nt::TypeList<Accessors...>) noexcept {
            constexpr bool has_init_packed = requires (Op op, Vec<Index, N> indices, Accessors::value_type... values) {
                op.init(indices, values...);
            };
            constexpr bool has_init_unpacked = requires (Op op, Indices... indices, Accessors::value_type... values) {
                op.init(indices..., values...);
            };
            return {has_init_unpacked || has_init_packed, has_init_packed};
        }

        template<typename... Accessors>
        static constexpr bool check_join_(nt::TypeList<Accessors...>) noexcept {
            constexpr bool valid = requires (Op op, Accessors::value_type... values) {
                op.join(values..., values...);
            };
            return valid;
        }

        template<typename... R, typename... O>
        static constexpr auto check_final_(nt::TypeList<R...>, nt::TypeList<O...>) noexcept {
            constexpr bool valid = requires (Op op, R::value_type... reduced, O::value_type... outputs) {
                op.join(reduced..., outputs...);
            };
            constexpr bool is_defaulted = valid ? false :
            requires (R::value_type... reduced, O::value_type... outputs) {
                ((outputs = static_cast<O::value_type>(reduced)), ...);
            };
            return Pair{valid || is_defaulted, is_defaulted};
        }
    };

    template<typename Op, typename Input, typename Reduced, typename Output>
    requires nt::is_tuple_v<Input> && nt::are_tuple_v<Reduced, Output>
    struct ReduceEwiseChecker {
        static constexpr auto has_init() -> bool { return m_has_init; }
        static constexpr auto has_join() -> bool { return m_has_join; }
        static constexpr auto has_final() -> bool { return m_has_final.first; }
        static constexpr auto is_final_defaulted() -> bool { return m_has_final.second; }
        static constexpr auto is_valid() -> bool { return has_init() && has_join() && has_final(); }

    private:
        using input_element_list = std::remove_reference_t<Input>::element_list;
        using reduced_element_list = std::remove_reference_t<Reduced>::element_list;
        using output_element_list = std::remove_reference_t<Output>::element_list;
        static constexpr bool m_has_init = check_init_(input_element_list{}, reduced_element_list{});
        static constexpr bool m_has_join = check_join_(reduced_element_list{});
        static constexpr Pair m_has_final = check_final_(reduced_element_list{}, output_element_list{});

        template<typename... I, typename... R>
        static constexpr bool check_init_(nt::TypeList<I...>, nt::TypeList<R...>) noexcept {
            return requires (Op op, I::value_type... inputs, R::value_type... values) {
                op.init(inputs..., values...);
            };
        }

        template<typename... Accessors>
        static constexpr bool check_join_(nt::TypeList<Accessors...>) noexcept {
            return requires (Op op, Accessors::value_type... values) {
                op.join(values..., values...);
            };
        }

        template<typename... R, typename... O>
        static constexpr auto check_final_(nt::TypeList<R...>, nt::TypeList<O...>) noexcept {
            constexpr bool valid = requires (Op op, R::value_type... reduced, O::value_type... outputs) {
                op.join(reduced..., outputs...);
            };
            constexpr bool is_defaulted = valid ? false :
            requires (R::value_type... reduced, O::value_type... outputs) {
                ((outputs = static_cast<O::value_type>(reduced)), ...);
            };
            return Pair{valid || is_defaulted, is_defaulted};
        }
    };
}
