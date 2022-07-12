message(STATUS "Eigen3: fetching static dependency...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

set(Eigen3_REPOSITORY https://gitlab.com/libeigen/eigen.git)
set(Eigen3_TAG 3.4.0)

message(STATUS "Repository: ${Eigen3_REPOSITORY}")
message(STATUS "Git tag: ${Eigen3_TAG}")

include(FetchContent)
FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG master
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE)
set(EIGEN_BUILD_DOC OFF CACHE INTERNAL "")
set(EIGEN_BUILD_PKGCONFIG OFF CACHE INTERNAL "")
set(EIGEN_BUILD_TESTING OFF CACHE INTERNAL "")
set(EIGEN_BUILD_BTL OFF CACHE INTERNAL "")
set(BUILD_TESTING_OLD ${BUILD_TESTING})
set(BUILD_TESTING OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(Eigen3)
set(BUILD_TESTING ${BUILD_TESTING} CACHE BOOL "Build tests" FORCE)

message(STATUS "New imported target available: Eigen3::Eigen")

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "Eigen3: fetching static dependency... done")
