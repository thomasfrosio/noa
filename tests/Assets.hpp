#pragma once

#include <noa/base/Vec.hpp>
#include <noa/runtime/core/Shape.hpp>
#include <noa/io/Encoding.hpp>
#include <noa/xform/core/Interp.hpp>
#include <noa/fft/core/Layout.hpp>

#if defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif

#include <yaml-cpp/yaml.h>

#if defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic pop
#endif

namespace YAML {
    template<typename T, size_t N>
    struct convert<noa::Shape<T, N>> {
        static Node encode(const noa::Shape<T, N>& rhs) {
            Node node;
            for (auto e: rhs)
                node.push_back(e);
            node.SetStyle(EmitterStyle::Flow);
            return node;
        }

        static bool decode(const Node& node, noa::Shape<T, N>& rhs) {
            if (not node.IsSequence() or node.size() != N)
                return false;
            for (size_t i = 0; i < N; ++i)
                rhs[i] = node[i].as<T>();
            return true;
        }
    };

    template<typename T, size_t N>
    struct convert<noa::Strides<T, N>> {
        static Node encode(const noa::Strides<T, N>& rhs) {
            Node node;
            for (auto e: rhs)
                node.push_back(e);
            node.SetStyle(EmitterStyle::Flow);
            return node;
        }

        static bool decode(const Node& node, noa::Strides<T, N>& rhs) {
            if (not node.IsSequence() or node.size() != N)
                return false;
            for (size_t i = 0; i < N; ++i)
                rhs[i] = node[i].as<T>();
            return true;
        }
    };

    template<typename T, size_t N>
    struct convert<noa::Vec<T, N>> {
        static Node encode(const noa::Vec<T, N>& rhs) {
            Node node;
            for (auto e: rhs)
                node.push_back(e);
            node.SetStyle(EmitterStyle::Flow);
            return node;
        }

        static bool decode(const Node& node, noa::Vec<T, N>& rhs) {
            if (not node.IsSequence() or node.size() != N)
                return false;
            for (size_t i = 0; i < N; ++i)
                rhs[i] = node[i].as<T>();
            return true;
        }
    };

    template<>
    struct convert<noa::Path> {
        static Node encode(const noa::Path& rhs) {
            return convert<std::string>::encode(rhs.string());
        }

        static bool decode(const Node& node, noa::Path& rhs) {
            std::string str;
            bool status = convert<std::string>::decode(node, str);
            rhs = str;
            return status;
        }
    };

    template<>
    struct convert<noa::Border> {
        static Node encode(const noa::Border& rhs) {
            std::ostringstream stream;
            stream << rhs;
            return convert<std::string>::encode(stream.str());
        }

        static bool decode(const Node& node, noa::Border& rhs) {
            if (!node.IsScalar())
                return false;
            const std::string& buffer = node.Scalar();

            if (buffer == "BORDER_NOTHING")
                rhs = noa::Border::NOTHING;
            else if (buffer == "BORDER_ZERO")
                rhs = noa::Border::ZERO;
            else if (buffer == "BORDER_VALUE")
                rhs = noa::Border::VALUE;
            else if (buffer == "BORDER_CLAMP")
                rhs = noa::Border::CLAMP;
            else if (buffer == "BORDER_REFLECT")
                rhs = noa::Border::REFLECT;
            else if (buffer == "BORDER_MIRROR")
                rhs = noa::Border::MIRROR;
            else if (buffer == "BORDER_PERIODIC")
                rhs = noa::Border::PERIODIC;
            else
                return false;
            return true;
        }
    };

    template<>
    struct convert<noa::xform::Interp> {
        static Node encode(const noa::xform::Interp& rhs) {
            std::ostringstream stream;
            stream << rhs;
            return convert<std::string>::encode(stream.str());
        }

        static bool decode(const Node& node, noa::xform::Interp& rhs) {
            if (not node.IsScalar())
                return false;
            const std::string& buffer = node.Scalar();

            if (buffer == "INTERP_NEAREST")
                rhs = noa::xform::Interp::NEAREST;
            else if (buffer == "INTERP_NEAREST_FAST")
                rhs = noa::xform::Interp::NEAREST_FAST;
            else if (buffer == "INTERP_LINEAR")
                rhs = noa::xform::Interp::LINEAR;
            else if (buffer == "INTERP_LINEAR_FAST")
                rhs = noa::xform::Interp::LINEAR_FAST;
            else if (buffer == "INTERP_CUBIC")
                rhs = noa::xform::Interp::CUBIC;
            else if (buffer == "INTERP_CUBIC_FAST")
                rhs = noa::xform::Interp::CUBIC_FAST;
            else if (buffer == "INTERP_CUBIC_BSPLINE")
                rhs = noa::xform::Interp::CUBIC_BSPLINE;
            else if (buffer == "INTERP_CUBIC_BSPLINE_FAST")
                rhs = noa::xform::Interp::CUBIC_BSPLINE_FAST;
            else if (buffer == "INTERP_LANCZOS4")
                rhs = noa::xform::Interp::LANCZOS4;
            else if (buffer == "INTERP_LANCZOS4_FAST")
                rhs = noa::xform::Interp::LANCZOS4_FAST;
            else if (buffer == "INTERP_LANCZOS6")
                rhs = noa::xform::Interp::LANCZOS6;
            else if (buffer == "INTERP_LANCZOS6_FAST")
                rhs = noa::xform::Interp::LANCZOS6_FAST;
            else if (buffer == "INTERP_LANCZOS8")
                rhs = noa::xform::Interp::LANCZOS8;
            else if (buffer == "INTERP_LANCZOS8_FAST")
                rhs = noa::xform::Interp::LANCZOS8_FAST;
            else
                return false;
            return true;
        }
    };

    template<>
    struct convert<noa::fft::Layout> {
        static Node encode(const noa::fft::Layout& rhs) {
            std::ostringstream stream;
            stream << rhs;
            return convert<std::string>::encode(stream.str());
        }

        static bool decode(const Node& node, noa::fft::Layout& rhs) {
            if (not node.IsScalar())
                return false;
            const std::string& buffer = node.Scalar();

            if (buffer == "H2H")
                rhs = noa::fft::Layout::H2H;
            else if (buffer == "HC2HC")
                rhs = noa::fft::Layout::HC2HC;
            else if (buffer == "H2HC")
                rhs = noa::fft::Layout::H2HC;
            else if (buffer == "HC2H")
                rhs = noa::fft::Layout::HC2H;
            else if (buffer == "H2F")
                rhs = noa::fft::Layout::H2F;
            else if (buffer == "F2H")
                rhs = noa::fft::Layout::F2H;
            else if (buffer == "F2FC")
                rhs = noa::fft::Layout::F2FC;
            else if (buffer == "FC2F")
                rhs = noa::fft::Layout::FC2F;
            else if (buffer == "HC2F")
                rhs = noa::fft::Layout::HC2F;
            else if (buffer == "F2HC")
                rhs = noa::fft::Layout::F2HC;
            else if (buffer == "H2FC")
                rhs = noa::fft::Layout::H2FC;
            else if (buffer == "FC2H")
                rhs = noa::fft::Layout::FC2H;
            else if (buffer == "F2F")
                rhs = noa::fft::Layout::F2F;
            else if (buffer == "FC2FC")
                rhs = noa::fft::Layout::FC2FC;
            else
                return false;
            return true;
        }
    };
}
