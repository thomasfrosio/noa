#pragma once

#include "noa/core/Traits.hpp"
#include "noa/core/types/Tuple.hpp"

namespace noa::guts {
    template<typename... T>
    struct AdaptorZip {
        static constexpr bool ZIP = true;
        Tuple<T...> tuple;
    };

    template<typename... T>
    struct AdaptorUnzip {
        static constexpr bool ZIP = false;
        Tuple<T...> tuple;
    };

    NOA_GENERATE_PROCLAIM_FULL(adaptor);
    template<typename... T> struct proclaim_is_adaptor<AdaptorZip<T...>> : std::true_type {};
    template<typename... T> struct proclaim_is_adaptor<AdaptorUnzip<T...>> : std::true_type {};
}

namespace noa {
    /// Core functions utility used to zip arguments.
    /// - Wrapped arguments are zipped into a Tuple, which is then passed to the operator.
    ///   The core functions handle the details of this operation, the adaptor is just a thin wrapper
    ///   used to store and forward the arguments to these core functions.
    /// - Temporaries (rvalue references) are moved (if their type is movable, otherwise they are copied) and
    ///   stored by value. Lvalue references are stored as such, i.e. no move/copy are involved.
    ///   See forward_as_final_tuple for more details.
    template<typename... T>
    constexpr auto zip(T&&... a) noexcept {
        return ng::AdaptorZip{.tuple=forward_as_final_tuple(std::forward<T>(a)...)};
    }

    /// Core functions utility used to wrap arguments.
    /// - Wrapped arguments are passed directly to the operator.
    ///   The core functions handle the details of this operation, the adaptor is just a thin wrapper
    ///   used to store and forward the arguments to these core functions.
    /// - Temporaries (rvalue references) are moved (if their type is movable, otherwise they are copied) and
    ///   stored by value. Lvalue references are stored as such, i.e. no move/copy are involved.
    ///   See forward_as_final_tuple for more details.
    template<typename... T>
    constexpr auto wrap(T&&... a) noexcept {
        return ng::AdaptorUnzip{.tuple=forward_as_final_tuple(std::forward<T>(a)...)};
    }
}
