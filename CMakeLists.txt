cmake_minimum_required(VERSION 3.16)
project(Noa)

set(CMAKE_CXX_STANDARD 20)

add_executable(Noa src/Noa/EntryPoint.cpp src/Noa/InputManager.cpp src/Noa/InputManager.h)
