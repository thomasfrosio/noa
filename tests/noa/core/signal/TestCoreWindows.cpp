#include <noa/core/signal/Windows.hpp>
#include <catch2/catch.hpp>

#include "Utils.hpp"

TEST_CASE("core::signal:: windows", "[noa][core]") {
    using namespace noa::types;
    {
        std::array<f64, 20> window{};
        noa::signal::window_blackman(window.data(), 20, {.normalize = false, .half_window = false});
        constexpr std::array window_blackman_numpy{
            -1.38777878e-17, 1.02226199e-02, 4.50685843e-02, 1.14390287e-01,
            2.26899356e-01, 3.82380768e-01, 5.66665187e-01, 7.52034438e-01,
            9.03492728e-01, 9.88846031e-01, 9.88846031e-01, 9.03492728e-01,
            7.52034438e-01, 5.66665187e-01, 3.82380768e-01, 2.26899356e-01,
            1.14390287e-01, 4.50685843e-02, 1.02226199e-02, -1.38777878e-17
        };
        REQUIRE(test::allclose_abs(window.data(), window_blackman_numpy.data(), 20, 1e-8));

        noa::signal::window_blackman(window.data(), 20, {.normalize = false, .half_window = true});
        REQUIRE(test::allclose_abs(window.data(), window_blackman_numpy.data() + 10, 10, 1e-8));
    }
    {
        std::array<f64, 21> window{};
        noa::signal::window_blackman(window.data(), 21, {.normalize = false, .half_window = false});
        constexpr std::array window_blackman_numpy{
            -1.38777878e-17, 9.19310140e-03, 4.02128624e-02, 1.01386014e-01,
            2.00770143e-01, 3.40000000e-01, 5.09787138e-01, 6.89171267e-01,
            8.49229857e-01, 9.60249618e-01, 1.00000000e+00, 9.60249618e-01,
            8.49229857e-01, 6.89171267e-01, 5.09787138e-01, 3.40000000e-01,
            2.00770143e-01, 1.01386014e-01, 4.02128624e-02, 9.19310140e-03,
            -1.38777878e-17
        };
        REQUIRE(test::allclose_abs(window.data(), window_blackman_numpy.data(), 21, 1e-8));

        noa::signal::window_blackman(window.data(), 21, {.normalize = false, .half_window = true});
        REQUIRE(test::allclose_abs(window.data(), window_blackman_numpy.data() + 10, 11, 1e-8));
    }
    {
        std::array<f64, 20> window{};
        noa::signal::window_sinc(window.data(), 20, 1., {.normalize = false, .half_window = false});
        constexpr std::array window_sinc_numpy{
            -0.0335063, 0.03744822, -0.04244132, 0.04897075, -0.05787452,
            0.07073553, -0.09094568, 0.12732395, -0.21220659, 0.63661977,
            0.63661977, -0.21220659, 0.12732395, -0.09094568, 0.07073553,
            -0.05787452, 0.04897075, -0.04244132, 0.03744822, -0.0335063
        };
        REQUIRE(test::allclose_abs(window.data(), window_sinc_numpy.data(), 20, 1e-8));

        noa::signal::window_sinc(window.data(), 20, 1., {.normalize = false, .half_window = true});
        REQUIRE(test::allclose_abs(window.data(), window_sinc_numpy.data() + 10, 10, 1e-8));
    }
    {
        std::array<f64, 21> window{};
        noa::signal::window_sinc(window.data(), 21, 0.33, {.normalize = false, .half_window = false});
        constexpr std::array window_sinc_numpy{
            -0.02575181, 0.0033284, 0.03600192, 0.0376097, -0.00333114,
            -0.05672324, -0.06718948, 0.00333279, 0.13946854, 0.2739827,
            0.33, 0.2739827, 0.13946854, 0.00333279, -0.06718948,
            -0.05672324, -0.00333114, 0.0376097, 0.03600192, 0.0033284,
            -0.02575181
        };
        REQUIRE(test::allclose_abs(window.data(), window_sinc_numpy.data(), 21, 1e-8));

        noa::signal::window_sinc(window.data(), 21, 0.33, {.normalize = false, .half_window = true});
        REQUIRE(test::allclose_abs(window.data(), window_sinc_numpy.data() + 10, 11, 1e-8));
    }
}
