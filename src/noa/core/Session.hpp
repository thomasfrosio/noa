#pragma once

#include "noa/Version.hpp"
#include "noa/core/Logger.hpp"
#include "noa/core/string/Format.hpp"

namespace noa {
    /// Global session. There should only be one session at a given time.
    /// When using the library, there should always be an active session.
    class Session {
    public:
        /// Creates a new session.
        /// \param name         Name of the session.
        /// \param filename     Filename of the sessions log file.
        ///                     If it is an empty string, the logger only logs in the console.
        /// \param verbosity    Verbosity of the console. The log file, if any, is always set to the maximum verbosity.
        /// \param threads      The maximum number of internal threads used during a session.
        ///                     If 0, retrieve value from environmental variable NOA_THREADS or OMP_NUM_THREADS.
        ///                     If these variables are empty or not defined, try to deduce the number of available
        ///                     threads and use this number instead.
        /// \note The logger is always accessible, and its settings and its sinks can be replaced at any time.
        Session(std::string_view name,
                std::string_view filename,
                Logger::Level verbosity = Logger::BASIC,
                size_t threads = 0) {
            logger = Logger(name, filename, verbosity);
            Session::threads(threads);
        }

        /// Sets the maximum number of internal threads used by a session.
        /// \param threads  Maximum number of threads.
        ///                 If 0, retrieve value from environmental variable NOA_THREADS or OMP_NUM_THREADS.
        ///                 If these variables are empty or not defined, try to deduce the number of available
        ///                 threads and use this number instead.
        /// \note This is the maximum number of internal threads. Users can of course create additional threads
        ///       using tools from the library, e.g. ThreadPool or Stream.
        static void threads(size_t threads);

        /// Returns the maximum number of internal threads.
        [[nodiscard]] static size_t threads() noexcept {
            return m_threads;
        }

    public:
        static std::string version() { return NOA_VERSION; }
        static std::string url() { return NOA_URL; }

    public:
        /// Logger used by all functions in the library.
        static Logger logger;

    private:
        static size_t m_threads;
    };
}
