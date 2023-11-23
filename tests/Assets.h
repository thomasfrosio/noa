#pragma once

#include <noa/core/Types.hpp>
#include <ostream>

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
            if (!node.IsSequence() || node.size() != N)
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
            if (!node.IsSequence() || node.size() != N)
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
            if (!node.IsSequence() || node.size() != N)
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
    struct convert<noa::Interp> {
        static Node encode(const noa::Interp& rhs) {
            std::ostringstream stream;
            stream << rhs;
            return convert<std::string>::encode(stream.str());
        }

        static bool decode(const Node& node, noa::Interp& rhs) {
            if (!node.IsScalar())
                return false;
            const std::string& buffer = node.Scalar();

            using namespace ::noa;
            if (buffer == "INTERP_NEAREST")
                rhs = Interp::NEAREST;
            else if (buffer == "INTERP_LINEAR")
                rhs = Interp::LINEAR;
            else if (buffer == "INTERP_COSINE")
                rhs = Interp::COSINE;
            else if (buffer == "INTERP_CUBIC")
                rhs = Interp::CUBIC;
            else if (buffer == "INTERP_CUBIC_BSPLINE")
                rhs = Interp::CUBIC_BSPLINE;
            else if (buffer == "INTERP_LINEAR_FAST")
                rhs = Interp::LINEAR_FAST;
            else if (buffer == "INTERP_COSINE_FAST")
                rhs = Interp::COSINE_FAST;
            else if (buffer == "INTERP_CUBIC_BSPLINE_FAST")
                rhs = Interp::CUBIC_BSPLINE_FAST;
            else
                return false;
            return true;
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

            using namespace ::noa;
            if (buffer == "BORDER_NOTHING")
                rhs = Border::NOTHING;
            else if (buffer == "BORDER_ZERO")
                rhs = Border::ZERO;
            else if (buffer == "BORDER_VALUE")
                rhs = Border::VALUE;
            else if (buffer == "BORDER_CLAMP")
                rhs = Border::CLAMP;
            else if (buffer == "BORDER_REFLECT")
                rhs = Border::REFLECT;
            else if (buffer == "BORDER_MIRROR")
                rhs = Border::MIRROR;
            else if (buffer == "BORDER_PERIODIC")
                rhs = Border::PERIODIC;
            else
                return false;
            return true;
        }
    };

    template<>
    struct convert<noa::fft::Remap> {
        static Node encode(const noa::fft::Remap& rhs) {
            std::ostringstream stream;
            stream << rhs;
            return convert<std::string>::encode(stream.str());
        }

        static bool decode(const Node& node, noa::fft::Remap& rhs) {
            if (!node.IsScalar())
                return false;
            const std::string& buffer = node.Scalar();

            using namespace ::noa;
            if (buffer == "H2H")
                rhs = fft::H2H;
            else if (buffer == "HC2HC")
                rhs = fft::HC2HC;
            else if (buffer == "H2HC")
                rhs = fft::H2HC;
            else if (buffer == "HC2H")
                rhs = fft::HC2H;
            else if (buffer == "H2F")
                rhs = fft::H2F;
            else if (buffer == "F2H")
                rhs = fft::F2H;
            else if (buffer == "F2FC")
                rhs = fft::F2FC;
            else if (buffer == "FC2F")
                rhs = fft::FC2F;
            else if (buffer == "HC2F")
                rhs = fft::HC2F;
            else if (buffer == "F2HC")
                rhs = fft::F2HC;
            else if (buffer == "H2FC")
                rhs = fft::H2FC;
            else if (buffer == "FC2H")
                rhs = fft::FC2H;
            else if (buffer == "F2F")
                rhs = fft::F2F;
            else if (buffer == "FC2FC")
                rhs = fft::FC2FC;
            else
                return false;
            return true;
        }
    };
}
