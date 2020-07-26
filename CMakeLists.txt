cmake_minimum_required(VERSION 3.16)
project(Noa)

set(CMAKE_CXX_STANDARD 17)

# spdlog header files.
set(SPDLOG_FILES
        src/vendor/spdlog/include/spdlog/spdlog.h
        src/vendor/spdlog/include/spdlog/fmt/ostr.h
        src/vendor/spdlog/include/spdlog/sinks/stdout_color_sinks.h
        src/vendor/spdlog/include/spdlog/sinks/basic_file_sink.h)

add_executable(noa
        ${SPDLOG_FILES}
        src/noa/program/EntryPoint.cpp
        src/noa/include/Parser.h
        src/noa/include/String.h
        src/noa/include/Assert.h
        src/noa/include/Noa.h
        src/noa/include/Traits.h
        src/noa/include/Helper.h
        src/noa/include/Log.h
        )

target_include_directories(noa PRIVATE src/vendor/spdlog/include)
