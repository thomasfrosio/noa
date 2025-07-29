#include <noa/core/Traits.hpp>
#include <noa/core/types/Complex.hpp>

#include "Catch.hpp"

namespace { // largest_type_t
    using namespace noa::types;
    namespace nt = noa::traits;

    using a1 = std::array<i8, 1>;
    using a2 = std::array<i8, 2>;
    using a3 = std::array<i8, 3>;
    using a4 = std::array<i8, 4>;
    using a5 = std::array<i8, 5>;
    using a6 = std::array<i8, 6>;

    static_assert(std::same_as<nt::largest_type_t<f32, f64>, f64>);
    static_assert(std::same_as<nt::largest_type_t<a1, a2, a3, a4, a5, a6>, a6>);
    static_assert(std::same_as<nt::largest_type_t<a1>, a1>);
    static_assert(std::same_as<nt::largest_type_t<f32, i32>, f32>); // first one is returned

    static_assert(std::same_as<i32, nt::double_precision_t<i32>>);
    static_assert(std::same_as<f64, nt::double_precision_t<f32>>);
    static_assert(std::same_as<c64, nt::double_precision_t<c32>>);
    static_assert(std::same_as<f64, nt::double_precision_t<f16>>);
}
