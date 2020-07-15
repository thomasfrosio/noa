cmake_minimum_required(VERSION 3.16)
project(Noa)

set(CMAKE_CXX_STANDARD 20)

add_executable(Noa
        src/Noa/core/EntryPoint.cpp
        src/Noa/core/Parser.cpp
        src/Noa/core/Parser.h
        src/Noa/core/Options.cpp
        src/Noa/core/Options.h
        src/Noa/program/fft.cpp
        src/Noa/program/fft.h src/Noa/core/String.cpp src/Noa/core/String.h)
