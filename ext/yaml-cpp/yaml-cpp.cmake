include(FetchContent)
message(STATUS "Fetching static dependency: yaml-cpp")
FetchContent_Declare(yaml-cpp GIT_REPOSITORY https://github.com/ffyr2w/yaml-cpp)
FetchContent_MakeAvailable(yaml-cpp)
