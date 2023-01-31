#pragma once

#include <noa/common/Types.h>
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
            if (!node.IsSequence() || node.size() != 4)
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
            if (!node.IsSequence() || node.size() != 4)
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
            if (!node.IsSequence() || node.size() != 4)
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
    struct convert<noa::InterpMode> {
        static Node encode(const noa::InterpMode& rhs) {
            std::ostringstream stream;
            stream << rhs;
            return convert<std::string>::encode(stream.str());
        }

        static bool decode(const Node& node, noa::InterpMode& rhs) {
            if (!node.IsScalar())
                return false;
            const std::string& buffer = node.Scalar();

            using namespace ::noa;
            if (buffer == "INTERP_NEAREST")
                rhs = InterpMode::NEAREST;
            else if (buffer == "INTERP_LINEAR")
                rhs = InterpMode::LINEAR;
            else if (buffer == "INTERP_COSINE")
                rhs = InterpMode::COSINE;
            else if (buffer == "INTERP_CUBIC")
                rhs = InterpMode::CUBIC;
            else if (buffer == "INTERP_CUBIC_BSPLINE")
                rhs = InterpMode::CUBIC_BSPLINE;
            else if (buffer == "INTERP_LINEAR_FAST")
                rhs = InterpMode::LINEAR_FAST;
            else if (buffer == "INTERP_COSINE_FAST")
                rhs = InterpMode::COSINE_FAST;
            else if (buffer == "INTERP_CUBIC_BSPLINE_FAST")
                rhs = InterpMode::CUBIC_BSPLINE_FAST;
            else
                return false;
            return true;
        }
    };

    template<>
    struct convert<noa::BorderMode> {
        static Node encode(const noa::BorderMode& rhs) {
            std::ostringstream stream;
            stream << rhs;
            return convert<std::string>::encode(stream.str());
        }

        static bool decode(const Node& node, noa::BorderMode& rhs) {
            if (!node.IsScalar())
                return false;
            const std::string& buffer = node.Scalar();

            using namespace ::noa;
            if (buffer == "BORDER_NOTHING")
                rhs = BorderMode::NOTHING;
            else if (buffer == "BORDER_ZERO")
                rhs = BorderMode::ZERO;
            else if (buffer == "BORDER_VALUE")
                rhs = BorderMode::VALUE;
            else if (buffer == "BORDER_CLAMP")
                rhs = BorderMode::CLAMP;
            else if (buffer == "BORDER_REFLECT")
                rhs = BorderMode::REFLECT;
            else if (buffer == "BORDER_MIRROR")
                rhs = BorderMode::MIRROR;
            else if (buffer == "BORDER_PERIODIC")
                rhs = BorderMode::PERIODIC;
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
