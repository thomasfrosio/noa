#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include "noa/core/Enums.hpp"

namespace noa::fft {
    struct RemapInterface {
        Remap remap;

        constexpr /*implicit*/ RemapInterface(Remap remap_) : remap(remap_) {}

        template<size_t N>
        constexpr /*implicit*/ RemapInterface(const char (& name)[N]) {
            std::string_view name_(name);
            if (name_ == "h2h" or name_ == "H2H") {
                remap = Remap::H2H;
            } else if (name_ == "h2hc" or name_ == "H2HC") {
                remap = Remap::H2HC;
            } else if (name_ == "h2f" or name_ == "H2F") {
                remap = Remap::H2F;
            } else if (name_ == "h2fc" or name_ == "H2FC") {
                remap = Remap::H2FC;

            } else if (name_ == "hc2h" or name_ == "HC2H") {
                remap = Remap::HC2H;
            } else if (name_ == "hc2hc" or name_ == "HC2HC") {
                remap = Remap::HC2HC;
            } else if (name_ == "hc2h" or name_ == "HC2F") {
                remap = Remap::HC2F;
            } else if (name_ == "hc2h" or name_ == "HC2FC") {
                remap = Remap::HC2FC;

            } else if (name_ == "f2h" or name_ == "F2H") {
                remap = Remap::F2H;
            } else if (name_ == "f2hc" or name_ == "F2HC") {
                remap = Remap::F2HC;
            } else if (name_ == "f2f" or name_ == "F2F") {
                remap = Remap::F2F;
            } else if (name_ == "f2fc" or name_ == "F2FC") {
                remap = Remap::F2FC;

            } else if (name_ == "fc2h" or name_ == "FC2H") {
                remap = Remap::FC2H;
            } else if (name_ == "fc2hc" or name_ == "FC2HC") {
                remap = Remap::FC2HC;
            } else if (name_ == "fc2f" or name_ == "FC2F") {
                remap = Remap::FC2F;
            } else if (name_ == "fc2fc" or name_ == "FC2FC") {
                remap = Remap::FC2FC;

            } else {
                remap = Remap::NONE;
            }
        }
    };
}
#endif
