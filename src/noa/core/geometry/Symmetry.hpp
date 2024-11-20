#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/utils/Strings.hpp"
#include "noa/core/geometry/Transform.hpp"
#include "noa/core/types/Mat.hpp"
#include "noa/core/types/Span.hpp"

#ifdef NOA_IS_OFFLINE
#include <optional>

namespace noa::geometry {
    /// Symmetry code.
    struct SymmetryCode {
        i32 order{};
        char type{};

        [[nodiscard]] static auto from_string(std::string_view symmetry) -> std::optional<SymmetryCode> {
            symmetry = ns::trim(symmetry);
            if (symmetry.empty())
                return std::nullopt;

            SymmetryCode out{};
            out.type = static_cast<char>(std::toupper(static_cast<unsigned char>(symmetry[0])));

            if (symmetry.size() > 1) {
                auto opt = ns::parse<i32>(std::string(symmetry, 1, symmetry.length())); // offset by 1
                if (not opt)
                    return std::nullopt;
                out.order = opt.value();
            } else {
                out.order = 0;
            }
            return out;
        }

        [[nodiscard]] std::string to_string() const {
            if (order)
                return fmt::format("{}{}", type, order);
            return {type};
        }
    };
}

namespace noa::geometry::guts {
    /// Sets the {2|3}d rotation matrices for the CX symmetry.
    /// \param[out] matrices Rotation matrices, excluding the identity. The order X is matrices.size() + 1.
    template<typename T, typename I, StridesTraits S> requires (nt::mat22<T> or nt::mat33<T>)
    constexpr void set_cx_symmetry_matrices(Span<T, 1, I, S> matrices) {
        using value_t = nt::value_type_t<T>;
        i64 order = matrices.ssize() + 1;
        const auto angle = Constant<f64>::PI * 2 / static_cast<f64>(order);
        for (i64 i = 1; i < order; ++i) { // skip the identity
            const auto i_angle = static_cast<f64>(i) * angle;
            if constexpr (nt::mat22<T>)
                matrices[i - 1] = rotate(i_angle).as<value_t>();
            else
                matrices[i - 1] = rotate_z(i_angle).as<value_t>();
        }
    }
}
#endif
