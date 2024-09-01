#pragma once

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

    template<typename> struct proclaim_is_adaptor : std::false_type {};
    template<typename... T> struct proclaim_is_adaptor<AdaptorZip<T...>> : std::true_type {};
    template<typename... T> struct proclaim_is_adaptor<AdaptorUnzip<T...>> : std::true_type {};

    template<typename T> using is_adaptor = proclaim_is_adaptor<T>;
    template<typename T> constexpr bool is_adaptor_v = is_adaptor<T>::value;
    template<typename... T> using are_adaptor = std::conjunction<is_adaptor<T>...>;
    template<typename... T> constexpr bool are_adaptor_v = are_adaptor<T...>::value;
    template<typename... T> concept adaptor = are_adaptor_v<T...>;
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
