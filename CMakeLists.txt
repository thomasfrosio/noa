cmake_minimum_required(VERSION 3.16)
project(Noa)

set(CMAKE_CXX_STANDARD 17)

add_executable(noa
        src/noa/program/EntryPoint.cpp
        src/noa/include/Parser.h
        src/noa/include/String.h
        src/noa/include/Assert.h
        src/noa/include/Core.h
        src/noa/include/Traits.h
        src/noa/include/Helper.h
        src/noa/include/Log.h
        src/noa/include/InputManager.h)

target_include_directories(noa PRIVATE src/vendor/spdlog/include)
target_precompile_headers(noa
        PRIVATE
        src/noa/include/Core.h
        )
