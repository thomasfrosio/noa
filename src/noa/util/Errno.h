/**
 * @file Errno.h
 * @brief Type-safe error numbers used throughout the @a Noa namespace.
 * @author Thomas - ffyr2w
 * @date 05 Jan 2021
 */
 #pragma once

#include "noa/util/Flag.h"


namespace Noa {
    /** Error numbers. Often exchanged as Flag<Errno>. These are NOT bitset. */
    enum class Errno {
        good = 0, // this one should not change !

        fail,
        invalid_argument,
        invalid_size,
        invalid_data,
        invalid_state,
        out_of_range,
        not_supported,

        // I/O and streams
        fail_close,
        fail_open,
        fail_read,
        fail_write,

        // OS
        out_of_memory,
        fail_os,
    };


    /**
     * Template specialization of Flag.
     * Flag<Errno> is NOT a bitset and should not be used as such. Only one error number can be
     * used at the same time (like ERRNO).
     */
    template<>
    class NOA_API Flag<Errno> {
        using int_t = std::underlying_type_t<Errno>;
        int_t err{0};

    public:
        inline constexpr Flag() = default;

        /** Implicitly converting enum to Flag<enum>. */
        inline constexpr Flag(Errno v) noexcept : err(static_cast<int_t>(v)) {}


        [[nodiscard]] inline constexpr explicit operator bool() const noexcept {
            return err;
        }

        [[nodiscard]] inline constexpr bool operator==(Flag<Errno> rhs) const noexcept {
            return err == rhs.err;
        }

        [[nodiscard]] inline constexpr bool operator&&(Flag<Errno> rhs) const noexcept {
            return err && rhs.err;
        }

        [[nodiscard]] inline constexpr bool operator||(Flag<Errno> rhs) const noexcept {
            return err || rhs.err;
        }

        /** Update if and only if there is no error. */
        inline constexpr Flag<Errno> update(Flag<Errno> candidate) noexcept {
            if (candidate && !err)
                err = candidate.err;
            return *this;
        }

        /** Converts the error number in a human-readable string. */
        [[nodiscard]] std::string toString() const noexcept {
            switch (static_cast<Errno>(err)) {
                case Errno::fail:
                    return "Errno::fail";
                case Errno::invalid_argument:
                    return "Errno::invalid_argument";
                case Errno::invalid_size:
                    return "Errno::invalid_size";
                case Errno::invalid_data:
                    return "Errno::invalid_data";
                case Errno::invalid_state:
                    return "Errno::invalid_state";
                case Errno::out_of_range:
                    return "Errno::out_of_range";
                case Errno::not_supported:
                    return "Errno::not_supported";
                case Errno::fail_close:
                    return "Errno::fail_close";
                case Errno::fail_open:
                    return "Errno::fail_open";
                case Errno::fail_read:
                    return "Errno::fail_read";
                case Errno::fail_write:
                    return "Errno::fail_write";
                case Errno::out_of_memory:
                    return "Errno::out_of_memory";
                case Errno::fail_os:
                    return "Errno::fail_os";
                default:
                    return fmt::format("{}:{}:{}: DEV: Errno \"{}\" was not added",
                                       __FILE__, __FUNCTION__, __LINE__);
            }
        }
    };
}
