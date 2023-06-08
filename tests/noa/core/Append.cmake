set(TEST_COMMON_SOURCES
        noa/core/TestCommonOS.cpp

        noa/core/utils/TestCommonClampCast.cpp
        noa/core/utils/TestCommonIndexing.cpp
        noa/core/utils/TestCommonIRange.cpp
        noa/core/utils/TestCommonSafeCast.cpp

        noa/core/string/TestCommonString.cpp

        noa/core/traits/TestCommonTraitsVectors.cpp

        noa/core/types/TestCommonAccessor.cpp
        noa/core/types/TestCommonComplex.cpp
        noa/core/types/TestCommonHalf.cpp
        noa/core/types/TestCommonMatrices.cpp
        noa/core/types/TestCommonVectors.cpp
        noa/core/types/TestCoreSpan.cpp

        noa/core/geometry/TestCommonGeometryEuler.cpp
        noa/core/geometry/TestCommonInterpolator.cpp

        noa/core/io/TestCommonIO.cpp
        noa/core/io/TestMRCFile.cpp
        noa/core/io/TestTextFile.cpp

        )

set(TEST_SOURCES ${TEST_SOURCES} ${TEST_COMMON_SOURCES})
