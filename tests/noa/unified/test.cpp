#include <iostream>
#include <noa/unified/Array.hpp>
#include <noa/core/Enums.hpp>

using namespace noa::types;

namespace test1 {
    void test() {
        using T = noa::AccessorRestrictContiguous<int, 2, int>;
        static_assert(noa::traits::atomic_addable_nd<T, 2>);
        static_assert(noa::traits::readable_nd<T, 2, 3>);
        static_assert(T::SIZE == 2);
        static_assert(noa::traits::pointer<decltype(std::declval<const T&>().get())>);
        static_assert(noa::traits::same_as<
            decltype(std::declval<const T&>().offset_pointer(std::declval<const T&>().get(), 0, 1)),
            decltype(std::declval<const T&>().get())>);
        static_assert(noa::traits::same_as<
            decltype(std::declval<const T&>().offset_pointer(std::declval<const T&>().get(), Vec<i32, 2>{})),
            decltype(std::declval<const T&>().get())>);

        Vec<i64, 4> a{};
        vany(noa::NotEqual{}, a, Vec{1, 2, 3, 4});

        check(true, "{}", noa::Remap::H2F);
    }
}
