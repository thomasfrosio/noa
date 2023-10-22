#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"

namespace noa {
    struct distribution_t {};
    struct uniform_t : public distribution_t {};
    struct normal_t : public distribution_t {};
    struct log_normal_t : public distribution_t {};
    struct poisson_t : public distribution_t {};

    // The "xoshiro256** 1.0" generator.
    // Based on the C version by David Blackman and Sebastiano Vigna (2018),
    // https://prng.di.unimi.it/xoshiro256starstar.c
    struct xoshiro256ss {
        uint64_t s[4]{};

        static constexpr uint64_t splitmix64(uint64_t& x) {
            uint64_t z = (x += 0x9e3779b97f4a7c15uLL);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9uLL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebuLL;
            return z ^ (z >> 31);
        }

        constexpr xoshiro256ss() = default;

        constexpr explicit xoshiro256ss(uint64_t seed) {
            s[0] = splitmix64(seed);
            s[1] = splitmix64(seed);
            s[2] = splitmix64(seed);
            s[3] = splitmix64(seed);
        }

        using result_type = uint64_t;
        static constexpr uint64_t min() { return 0; }
        static constexpr uint64_t max() { return uint64_t(-1); }

        static constexpr uint64_t rotl(uint64_t x, int k) {
            return (x << k) | (x >> (64 - k));
        }

        constexpr uint64_t operator()() {
            const uint64_t result = rotl(s[1] * 5, 7) * 9;
            const uint64_t t = s[1] << 17;
            s[2] ^= s[0];
            s[3] ^= s[1];
            s[1] ^= s[2];
            s[0] ^= s[3];
            s[2] ^= t;
            s[3] = rotl(s[3], 45);
            return result;
        }
    };
}
