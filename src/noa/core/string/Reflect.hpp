#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/string/Format.hpp"

#if defined(NOA_IS_OFFLINE)
#include <string>
#include <string_view>
#include <memory>
#include <vector>
#include <cxxabi.h> // abi::__cxa_demangle

namespace noa {
    // Forward declaration.
    template<typename T>
    inline std::string reflect(const T& value);
}

namespace noa::guts {
    template<typename T>
    inline std::string value_string(const T& x) {
        return std::to_string(x);
    }

    template<>
    inline std::string value_string<bool>(const bool& x) {
        return x ? "true" : "false";
    }

    struct Demangler {
        // Returns the demangled name corresponding to the given typeinfo structure.
        static std::string demangle(const std::type_info& typeinfo) {
            // Implementation from nvrtc (using raw pointers) or jitify (using unique_ptr).
            // MSVC version was removed.
            const char* mangled_name = typeinfo.name();
            size_t bufsize = 0;
            char* buf = nullptr;
            int status;
            auto demangled_ptr = std::unique_ptr<char, void (*)(void*)>(
                    abi::__cxa_demangle(mangled_name, buf, &bufsize, &status), std::free);
            switch (status) {
                case 0:
                    return demangled_ptr.get(); // demangled successfully
                case -2:
                    return mangled_name; // interpret as plain unmangled name
                case -1:
                    [[fallthrough]]; // memory allocation failure
                case -3:
                    [[fallthrough]]; // invalid argument
                default:
                    return {};
            }
        }

        template<typename>
        struct TypeWrapper_ {};

        // Returns the demangled name of the given type.
        template<typename T>
        static inline std::string demangle() {
            // typeid discards cv qualifiers on value-types, so wrap the type in another type
            // to preserve cv-qualifiers, then strip off the wrapper from the resulting string.
            std::string wrapped_name = demangle(typeid(TypeWrapper_<T>));
            const std::string wrapper_class_name = "TypeWrapper_<";
            size_t start = wrapped_name.find(wrapper_class_name);
            if (start == std::string::npos)
                return {}; // unexpected error
            start += wrapper_class_name.size();
            return wrapped_name.substr(start, wrapped_name.size() - (start + 1));
        }
    };

    template<typename T>
    struct ReflectType {
        const std::string& operator()() const {
            // Storing this statically means it is cached after the first call.
            static const std::string type_name = Demangler::demangle<T>();
            return type_name;
        }
    };
}

namespace noa {
    /// Generate a code-string for a type, such as: \c reflect<float>()->"float".
    template<typename T>
    inline std::string reflect() {
        return guts::ReflectType<T>{}();
    }

    /// Generate a code-string for a value, such as \c reflect(3.14f)->"(float)3.14".
    template<typename T>
    inline std::string reflect(const T& value) {
        return fmt::format("({}){}", reflect<T>(), value);
    }

    /// Generate a code-string for a generic non-type template argument, such as \c reflect<int,7>()->"(int)7".
    template<typename T, T Value>
    inline std::string reflect() {
        return fmt::format("({}){}", reflect<T>(), Value);
    }

    /// Create an Instance object that contains a const reference to the
    /// value.  We use this to wrap abstract objects from which we want to extract
    /// their type at runtime (e.g., derived type). This is used to facilitate
    /// templating on derived type when all we know at compile time is abstract type.
    template<typename T>
    struct Instance {
        const T& value;
        explicit Instance(const T& value_arg) : value(value_arg) {}
    };

    /// Create an Instance object from which we can extract the value's run-time type.
    /// \param value The const value to be captured.
    template<typename T>
    inline Instance<T const> instance_of(T const& value) {
        return Instance<T const>(value);
    }

    /// Generate a code-string for a type wrapped as an Instance instance.
    /// \c reflect(Instance<float>(3.1f))->"float"
    /// or more simply when passed to a instance_of helper
    /// \creflect(instance_of(3.1f))->"float"
    /// This is specifically for the case where we want to extract the run-time
    /// type, i.e., derived type, of an object pointer.
    template <typename T>
    inline std::string reflect(const Instance<T>& value) {
        return guts::Demangler::demangle(typeid(value.value));
    }

    /// Use an existing code string as-is.
    inline std::string reflect(const std::string& s) { return s; }
    inline const char* reflect(const char* s) { return s; }
    inline std::string_view reflect(std::string_view s) { return s; }

    /// Convenience class for generating code-strings for template instantiations.
    class Template {
        std::string m_name;

    public:
        /// Construct the class.
        explicit Template(std::string_view name) : m_name(name) {}

    public:
        /// Generate a code-string for a template instantiation.
        static inline std::string reflect_template(const std::vector<std::string>& args) {
            return fmt::format("< {} >", fmt::join(args, ","));
        }

        /// Generate a code-string for a template instantiation.
        template <typename... Ts>
        static inline std::string reflect_template() {
            return reflect_template({reflect<Ts>()...});
        }

        /// Generate a code-string for a template instantiation.
        template <typename... Args>
        static inline std::string reflect_template(const Args&... args) {
            return reflect_template({reflect(args)...});
        }

    public:
        /// Generate a code-string for an instantiation of the template
        [[nodiscard]] std::string instantiate(const std::vector<std::string>& template_args = {}) const {
            return m_name + reflect_template(template_args);
        }

        /// Generate a code-string for an instantiation of the template
        template<typename... TemplateArgs>
        [[nodiscard]] std::string instantiate() const {
            return m_name + reflect_template<TemplateArgs...>();
        }

        /// Generate a code-string for an instantiation of the template
        template<typename... TemplateArgs>
        [[nodiscard]] std::string instantiate(const TemplateArgs& ... targs) const {
            return m_name + reflect_template(targs...);
        }
    };
}
#endif
