message(STATUS "Eigen3: fetching static dependency...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

if (TARGET Eigen3::Eigen)
    message(STATUS "Target already exists: Eigen3::Eigen")
else ()
    set(Eigen3_REPOSITORY https://gitlab.com/libeigen/eigen.git)
    set(Eigen3_TAG e67c494cba7180066e73b9f6234d0b2129f1cdf5)

    message(STATUS "Repository: ${Eigen3_REPOSITORY}")
    message(STATUS "Git tag: ${Eigen3_TAG}")

    include(FetchContent)
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY ${Eigen3_REPOSITORY}
        GIT_TAG ${Eigen3_TAG}
#        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    # Eigen 3.4 uses CMP0077 OLD (ignore set), so use option to force it from here.
    option(EIGEN_BUILD_BLAS "Toggles the building of the Eigen Blas library" OFF)
    option(EIGEN_BUILD_LAPACK "Toggles the building of the included Eigen LAPACK library" OFF)
    option(EIGEN_BUILD_TESTING "Enable creation of Eigen tests." OFF)
    option(EIGEN_BUILD_PKGCONFIG "Build pkg-config .pc file for Eigen" OFF)
    option(EIGEN_BUILD_DOC "Enable creation of Eigen documentation" OFF)
    FetchContent_MakeAvailable(Eigen)

    message(STATUS "New imported target available: Eigen3::Eigen")
endif ()

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "Eigen3: fetching static dependency... done")
