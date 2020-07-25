cmake_minimum_required(VERSION 3.16)
project(Noa)

set(CMAKE_CXX_STANDARD 17)

add_executable(Noa
        src/Noa/program/EntryPoint.cpp
        src/Noa/include/Parser.h
        src/Noa/include/String.h
        src/Noa/include/Assert.h
        src/Noa/include/Noa.h
        src/Noa/include/Traits.h
        src/Noa/include/Helper.h)
