/// \file noa/common/Profiler.h
/// \brief The profiler used throughout the project.
/// \author Thomas - ffyr2w
/// \date 23 March 2021

#pragma once

//#define NOA_PROFILE
#ifdef NOA_PROFILE

#include <fstream>
#include <string>
#include <mutex>
#include <memory>
#include <chrono>
#include <thread>

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Types.h"
#include "noa/common/files/TextFile.h"
#include "noa/common/string/Format.h"

namespace noa {
    /// These fields describe a "complete event" (ph: "X").
    /// \a name:             The name of the event, usually the function name.
    /// \a category:         The event categories. This is a comma separated list of categories for the event.
    ///                      This is mostly used to distinguish between backend, i.e. "cpu", "cuda,stream1", etc.
    /// \a start:            Tracing clock timestamp of the event in microseconds.
    /// \a elapsed_time:     Tracing clock duration of complete events in microseconds.
    /// \a thread_id:        The thread ID for the thread that output this event.
    struct DurationEvent {
        std::string name;
        std::chrono::duration<double, std::micro> start;
        std::chrono::microseconds elapsed_time;
        std::thread::id thread_id;
    };

    /// This profiler writes profiling results (aka @a DurationEvent) to its attached file in the Google "Trace Event
    /// Format". This is a singleton and is usually used via the macros defined at the end of this file. Different
    /// timers might use this profiler.
    /// To visualize the event-based JSON file, one can use the tracing from chrome browsers (i.e. chrome://tracing) or
    /// other flame graph visualization tools that support this format (e.g. https://github.com/jlfwong/speedscope).
    class Profiler {
    private: // member variables
        TextFile<std::ofstream> m_file;
        std::mutex m_mutex{};

    public: // static public function
        /// Retrieves the singleton. Instantiate it if it is the first time.
        static Profiler& get() {
            static Profiler instance;
            return instance;
        }

    public:
        /// \param file_path
        /// \note If there is already a current session, then close it before beginning new one. Subsequent profiling
        ///       output meant for the original session will end up in the newly opened session instead.
        void begin(const path_t& file_path) {
            std::lock_guard lock(m_mutex);
            if (m_file.isOpen())
                endSession_();
            try {
                m_file.open(file_path, io::WRITE);
                writeHeader_();
            } catch (...) {
                NOA_THROW("Failed to open the result file");
            }
        }

        void write(const DurationEvent& event) {
            std::string json;
            json.reserve(256);

            json += ",\n{";
            json += string::format(
                    R"("name":"{}","cat":"function","dur":{},"ph":"X","pid":0,"tid":{},"ts":{})",
                    event.name, event.elapsed_time.count(), event.thread_id, event.start.count());
            json += "}";

            std::lock_guard lock(m_mutex);
            m_file.write(json);
        }

        void end() {
            std::lock_guard lock(m_mutex);
            endSession_();
        };

    private:
        // By default, displayTimeUnit is in ms.
        void writeHeader_() {
            m_file.write(R"({"otherData": {},"traceEvents":[{})");
        }

        void writeFooter_() {
            m_file.write("\n]}");
        }

        // NOT thread safe. Acquire the lock before calling this function.
        void endSession_() {
            if (m_file) {
                writeFooter_();
                m_file.close();
            }
        }
    };

    /// The default timer for the Profiler.
    class ProfilerTimer {
    private:
        const char* m_name;
        std::chrono::time_point<std::chrono::steady_clock> m_start_time_point;
        bool m_stopped;
    public:
        explicit ProfilerTimer(const char* name)
                : m_name(name), m_stopped(false) {
            m_start_time_point = std::chrono::steady_clock::now();
        };

        void stop() {
            auto end_time_point = std::chrono::steady_clock::now();
            auto high_res_start = std::chrono::duration<double, std::micro>{m_start_time_point.time_since_epoch()};
            auto elapsed_time =
                    std::chrono::time_point_cast<std::chrono::microseconds>(end_time_point).time_since_epoch() -
                    std::chrono::time_point_cast<std::chrono::microseconds>(m_start_time_point).time_since_epoch();

            Profiler::get().write({m_name, high_res_start, elapsed_time, std::this_thread::get_id()});
            m_stopped = true;
        }

        ~ProfilerTimer() {
            if (!m_stopped)
                stop();
        }
    };
}

#if defined(__GNUC__) || (defined(__MWERKS__) && (__MWERKS__ >= 0x3000)) || (defined(__ICC) && (__ICC >= 600)) || defined(__ghs__)
#define NOA_PROFILE_FUNC_SIG __PRETTY_FUNCTION__
#elif defined(__DMC__) && (__DMC__ >= 0x810)
#define NOA_PROFILE_FUNC_SIG __PRETTY_FUNCTION__
#elif (defined(__FUNCSIG__) || (_MSC_VER))
#define NOA_PROFILE_FUNC_SIG __FUNCSIG__
#elif (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 600)) || (defined(__IBMCPP__) && (__IBMCPP__ >= 500))
#define NOA_PROFILE_FUNC_SIG __FUNCTION__
#elif defined(__BORLANDC__) && (__BORLANDC__ >= 0x550)
#define NOA_PROFILE_FUNC_SIG __FUNC__
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)
#define NOA_PROFILE_FUNC_SIG __func__
#elif defined(__cplusplus) && (__cplusplus >= 201103)
#define NOA_PROFILE_FUNC_SIG __func__
#else
#define NOA_PROFILE_FUNC_SIG "NOA_FUNC_SIG unknown!"
#endif

#define NOA_PROFILE_BEGIN_SESSION(filepath) ::noa::Profiler::get().begin(filepath)
#define NOA_PROFILE_END_SESSION() ::noa::Profiler::get().end()

#define NOA_PROFILE_SCOPE_PRIVATE_2(name, line) ::noa::ProfilerTimer profile_timer_##line(name)
#define NOA_PROFILE_SCOPE_PRIVATE_1(name, line) NOA_PROFILE_SCOPE_PRIVATE_2(name, line)
#define NOA_PROFILE_SCOPE(name) NOA_PROFILE_SCOPE_PRIVATE_1(name, __LINE__)
#define NOA_PROFILE_FUNCTION() NOA_PROFILE_SCOPE(NOA_PROFILE_FUNC_SIG)

#else
#define NOA_PROFILE_BEGIN_SESSION(filepath)
#define NOA_PROFILE_END_SESSION()
#define NOA_PROFILE_SCOPE(name)
#define NOA_PROFILE_FUNCTION()
#endif // NOA_PROFILE
