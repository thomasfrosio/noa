set(TEST_COMMON_SOURCES
        noa/common/TestCommonIndexing.cpp
        noa/common/TestCommonIRange.cpp
        noa/common/TestCommonOS.cpp
        noa/common/TestCommonView.cpp

        noa/common/geometry/TestCommonGeometryEuler.cpp

        noa/common/io/TestCommonIO.cpp
        noa/common/io/TestImageFile.cpp
        noa/common/io/TestTextFile.cpp

        noa/common/string/TestCommonString.cpp

        noa/common/traits/TestCommonTraits.cpp
        noa/common/traits/TestCommonTraitsVectors.cpp

        noa/common/types/TestCommonClampCast.cpp
        noa/common/types/TestCommonComplex.cpp
        noa/common/types/TestCommonHalf.cpp
        noa/common/types/TestCommonMatrices.cpp
        noa/common/types/TestCommonVectors.cpp

        )

set(TEST_SOURCES ${TEST_SOURCES} ${TEST_COMMON_SOURCES})
