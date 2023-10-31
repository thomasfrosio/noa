message(STATUS "cuda-rtc: fetching dependency...")
list(APPEND CMAKE_MESSAGE_INDENT "   ")

if (TARGET cuda_rtc::jitify2)
    message(STATUS "Target already exists: cuda_rtc::jitify2")
else ()
    set(cuda_rtc_REPOSITORY https://github.com/thomasfrosio/cuda-rtc.git)
    set(cuda_rtc_TAG master)

    message(STATUS "Repository: ${cuda_rtc_REPOSITORY}")
    message(STATUS "Git tag: ${cuda_rtc_TAG}")

    include(FetchContent)
    FetchContent_Declare(cuda_rtc
        GIT_REPOSITORY ${cuda_rtc_REPOSITORY}
        GIT_TAG ${cuda_rtc_TAG}
    )
    FetchContent_MakeAvailable(cuda_rtc)

    message(STATUS "New imported target available: cuda_rtc::jitify2, cuda_rtc::preprocess")
endif ()

list(POP_BACK CMAKE_MESSAGE_INDENT)
message(STATUS "cuda-rtc: fetching dependency... done")
