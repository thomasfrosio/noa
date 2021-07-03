#pragma once

#include <noa/Session.h>
#include <noa/common/string/Format.h>
#include <noa/gpu/cuda/util/Event.h>

#include "../tests/Helpers.h" // gives access to the Test namespace

namespace benchmark {
    inline std::string formatPath(const char* file, int line) {
        namespace fs = std::filesystem;
        size_t idx = std::string(file).rfind(std::string("benchmarks") + fs::path::preferred_separator);
        return noa::string::format("{}:{}",
                                   idx == std::string::npos ? fs::path(file).filename().string() : file + idx,
                                   line);
    }

    template<typename... Args>
    void logHeader(const char* file, int line, Args&& ... args) {
        std::string header = noa::string::format(" {}: {} ", formatPath(file, line), noa::string::format(args...));
        noa::Session::logger.warn("{:*^120}", header);
    }
}

namespace benchmark::cuda {
    class Timer {
    private:
        noa::cuda::Stream* m_stream;
        noa::cuda::Event m_start, m_end;
        std::string m_header;
    public:
        template<typename... Args>
        explicit Timer(noa::cuda::Stream& stream, Args&& ... args)
                : m_stream(&stream), m_start(m_stream->device()), m_header("    - Timer: ") {
            m_header += noa::string::format(args...);
            noa::cuda::Event::record(*m_stream, m_start);
        }

        ~Timer() {
            noa::cuda::Event::record(*m_stream, m_end);
            noa::cuda::Event::synchronize(m_end);
            noa::Session::logger.trace("{:<100}: took {}ms", m_header, noa::cuda::Event::elapsedTime(m_start, m_end));
        }
    };
}

#define NOA_BENCHMARK_HEADER(...) benchmark::logHeader(__FILE__, __LINE__, __VA_ARGS__)

#define NOA_BENCHMARK_CUDA_SCOPE_PRIVATE_2(stream, line, ...) ::benchmark::cuda::Timer timer_##line(stream, __VA_ARGS__)
#define NOA_BENCHMARK_CUDA_SCOPE_PRIVATE_1(stream, line, ...) NOA_BENCHMARK_CUDA_SCOPE_PRIVATE_2(stream, line, __VA_ARGS__)
#define NOA_BENCHMARK_CUDA_SCOPE(stream, ...) NOA_BENCHMARK_CUDA_SCOPE_PRIVATE_1(stream, __LINE__, __VA_ARGS__)
