#pragma once

#include <noa/Session.h>
#include <noa/util/string/Format.h>

#include "../tests/Helpers.h" // gives access to the Test namespace

namespace Benchmark {
    inline std::string formatPath(const char* file, int line) {
        namespace fs = std::filesystem;
        size_t idx = std::string(file).rfind(std::string("benchmarks") + fs::path::preferred_separator);
        return Noa::String::format("{}:{}",
                                   idx == std::string::npos ? fs::path(file).filename().string() : file + idx,
                                   line);
    }

    template<typename... Args>
    void logHeader(const char* file, int line, Args&& ... args) {
        std::string header = Noa::String::format(" {}: {} ", formatPath(file, line), Noa::String::format(args...));
        Noa::Session::logger.warn("{:*^120}", header);
    }
}

namespace Benchmark::CUDA {
    class Timer {
    private:
        Noa::CUDA::Stream* m_stream;
        Noa::CUDA::Event m_start, m_end;
        std::string m_header;
    public:
        template<typename... Args>
        explicit Timer(Noa::CUDA::Stream& stream, Args&& ... args)
                : m_stream(&stream), m_start(m_stream->device()), m_header("    - Timer: ") {
            m_header += Noa::String::format(args...);
            Noa::CUDA::Event::record(*m_stream, m_start);
        }

        ~Timer() {
            Noa::CUDA::Event::record(*m_stream, m_end);
            Noa::CUDA::Event::synchronize(m_end);
            Noa::Session::logger.trace("{:<100}: took {}ms", m_header, Noa::CUDA::Event::elapsedTime(m_start, m_end));
        }
    };
}

#define NOA_BENCHMARK_HEADER(...) Benchmark::logHeader(__FILE__, __LINE__, __VA_ARGS__)

#define NOA_BENCHMARK_CUDA_SCOPE_PRIVATE_2(stream, line, ...) ::Benchmark::CUDA::Timer timer_##line(stream, __VA_ARGS__)
#define NOA_BENCHMARK_CUDA_SCOPE_PRIVATE_1(stream, line, ...) NOA_BENCHMARK_CUDA_SCOPE_PRIVATE_2(stream, line, __VA_ARGS__)
#define NOA_BENCHMARK_CUDA_SCOPE(stream, ...) NOA_BENCHMARK_CUDA_SCOPE_PRIVATE_1(stream, __LINE__, __VA_ARGS__)
