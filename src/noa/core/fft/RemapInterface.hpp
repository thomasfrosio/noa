#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Enums.hpp"

#ifdef NOA_IS_OFFLINE
#include <string_view>
#endif

namespace noa::fft {
    class RemapInterface {
    public:
        using underlying_type = std::underlying_type_t<Remap>;
        using Layout = noa::fft::Layout;

        Remap remap;

    public:
        constexpr /*implicit*/ RemapInterface(Remap remap_) : remap(remap_) {}

#ifdef NOA_IS_OFFLINE
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
#endif

        [[nodiscard]] constexpr bool is_any(auto... remaps) const noexcept {
            return ((remap == remaps) or ...);
        }

        [[nodiscard]] constexpr bool is_hx2xx()  const noexcept { return layout_() & Layout::SRC_HALF; }
        [[nodiscard]] constexpr bool is_fx2xx()  const noexcept { return layout_() & Layout::SRC_FULL; }
        [[nodiscard]] constexpr bool is_xx2hx()  const noexcept { return layout_() & Layout::DST_HALF; }
        [[nodiscard]] constexpr bool is_xx2fx()  const noexcept { return layout_() & Layout::DST_FULL; }

        [[nodiscard]] constexpr bool is_xc2xx()  const noexcept { return layout_() & Layout::SRC_CENTERED; }
        [[nodiscard]] constexpr bool is_xx2xc()  const noexcept { return layout_() & Layout::DST_CENTERED; }

        [[nodiscard]] constexpr bool is_hc2xx() const noexcept { return (layout_() & Layout::SRC_HALF) and (layout_() & Layout::SRC_CENTERED); }
        [[nodiscard]] constexpr bool is_h2xx()  const noexcept { return (layout_() & Layout::SRC_HALF) and (layout_() & Layout::SRC_NON_CENTERED); }
        [[nodiscard]] constexpr bool is_fc2xx() const noexcept { return (layout_() & Layout::SRC_FULL) and (layout_() & Layout::SRC_CENTERED); }
        [[nodiscard]] constexpr bool is_f2xx()  const noexcept { return (layout_() & Layout::SRC_FULL) and (layout_() & Layout::SRC_NON_CENTERED); }
        [[nodiscard]] constexpr bool is_xx2hc() const noexcept { return (layout_() & Layout::DST_HALF) and (layout_() & Layout::DST_CENTERED); }
        [[nodiscard]] constexpr bool is_xx2h()  const noexcept { return (layout_() & Layout::DST_HALF) and (layout_() & Layout::DST_NON_CENTERED); }
        [[nodiscard]] constexpr bool is_xx2fc() const noexcept { return (layout_() & Layout::DST_FULL) and (layout_() & Layout::DST_CENTERED); }
        [[nodiscard]] constexpr bool is_xx2f()  const noexcept { return (layout_() & Layout::DST_FULL) and (layout_() & Layout::DST_NON_CENTERED); }

    private:
        [[nodiscard]] constexpr underlying_type layout_() const noexcept {
            return static_cast<underlying_type>(remap);
        }
    };
}

#ifdef NOA_IS_OFFLINE
namespace noa::fft {
    inline std::ostream& operator<<(std::ostream& os, RemapInterface remap) {
        return os << remap.remap;
    }
}
namespace fmt {
    template<> struct formatter<noa::fft::RemapInterface> : ostream_formatter {};
}
#endif
