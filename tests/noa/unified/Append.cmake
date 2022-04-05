if (NOT NOA_ENABLE_UNIFIED)
    return()
endif()

set(TEST_UNIFIED_SOURCES
        noa/unified/TestUnifiedArray.cpp
        noa/unified/TestUnifiedDevice.cpp
        noa/unified/TestUnifiedStream.cpp
        )

set(TEST_SOURCES ${TEST_SOURCES} ${TEST_UNIFIED_SOURCES})
