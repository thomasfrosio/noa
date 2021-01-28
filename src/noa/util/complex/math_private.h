#pragma once

#include <cstdint>

#include "noa/Define.h"

namespace Noa::Math::Details::Complex {
    typedef union {
        float value;
        uint32_t word;
    } ieee_float_shape_type;

    NOA_DH inline void get_float_word(uint32_t& i, float d) {
        ieee_float_shape_type gf_u;
        gf_u.value = (d);
        (i) = gf_u.word;
    }

    NOA_DH inline void get_float_word(int32_t& i, float d) {
        ieee_float_shape_type gf_u;
        gf_u.value = (d);
        (i) = static_cast<int32_t>(gf_u.word);
    }

    NOA_DH inline void set_float_word(float& d, uint32_t i) {
        ieee_float_shape_type sf_u;
        sf_u.word = (i);
        (d) = sf_u.value;
    }

    // Assumes little endian ordering
    typedef union {
        double value;
        struct {
            uint32_t lsw;
            uint32_t msw;
        } parts;
        struct {
            uint64_t w;
        } xparts;
    } ieee_double_shape_type;

    NOA_DH inline void get_high_word(uint32_t& i, double d) {
        ieee_double_shape_type gh_u;
        gh_u.value = (d);
        (i) = gh_u.parts.msw;
    }

    /* Set the more significant 32 bits of a double from an int.  */
    NOA_DH inline void set_high_word(double& d, uint32_t v) {
        ieee_double_shape_type sh_u;
        sh_u.value = (d);
        sh_u.parts.msw = (v);
        (d) = sh_u.value;
    }

    NOA_DH inline void insert_words(double& d, uint32_t ix0, uint32_t ix1) {
        ieee_double_shape_type iw_u;
        iw_u.parts.msw = (ix0);
        iw_u.parts.lsw = (ix1);
        (d) = iw_u.value;
    }

    /* Get two 32 bit ints from a double.  */
    NOA_DH inline void extract_words(uint32_t& ix0, uint32_t& ix1, double d) {
        ieee_double_shape_type ew_u;
        ew_u.value = (d);
        (ix0) = ew_u.parts.msw;
        (ix1) = ew_u.parts.lsw;
    }

/* Get two 32 bit ints from a double.  */
    NOA_DH inline void extract_words(int32_t& ix0, int32_t& ix1, double d) {
        ieee_double_shape_type ew_u;
        ew_u.value = (d);
        (ix0) = static_cast<int32_t>(ew_u.parts.msw);
        (ix1) = static_cast<int32_t>(ew_u.parts.lsw);
    }

    template<typename T>
    inline NOA_DH T infinity();

    template<>
    inline NOA_DH float infinity<float>() {
        float res;
        set_float_word(res, 0x7f800000);
        return res;
    }

    template<>
    inline NOA_DH double infinity<double>() {
        double res;
        insert_words(res, 0x7ff00000, 0);
        return res;
    }
}
