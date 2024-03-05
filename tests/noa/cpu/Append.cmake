if (NOT NOA_ENABLE_CPU)
    return()
endif ()

set(TEST_CPU_SOURCES
    noa/cpu/TestCPUIwise.cpp
    noa/cpu/TestCPUEwise.cpp
    noa/cpu/TestCPUReduceIwise.cpp
    noa/cpu/TestCPUReduceEwise.cpp
    noa/cpu/TestCPUReduceAxesEwise.cpp
    noa/cpu/TestCPUReduceAxesIwise.cpp
    noa/cpu/TestCPUDevice.cpp
    noa/cpu/TestCPUStream.cpp
    )

set(TEST_SOURCES ${TEST_SOURCES} ${TEST_CPU_SOURCES})
