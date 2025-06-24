set(TEST_COMMON_SOURCES
    noa/core/TestCoreAllclose.cpp
    noa/core/TestCoreRemap.cpp
    noa/core/TestCoreInterpolation.cpp
    noa/core/TestCoreTraits.cpp

    noa/core/geometry/TestCoreEuler.cpp
    noa/core/geometry/TestCoreQuaternion.cpp
    noa/core/indexing/TestCoreIndexing.cpp
    noa/core/signal/TestCoreWindows.cpp

    noa/core/io/TestCoreIO.cpp
    noa/core/io/TestCoreMRCFile.cpp
    noa/core/io/TestCoreTiffFile.cpp
    noa/core/io/TestCoreBinaryFile.cpp
    noa/core/io/TestCoreOS.cpp
    noa/core/io/TestCoreTextFile.cpp

    noa/core/types/TestCoreAccessor.cpp
    noa/core/types/TestCoreComplex.cpp
    noa/core/types/TestCoreHalf.cpp
    noa/core/types/TestCoreMat.cpp
    noa/core/types/TestCoreSpan.cpp
    noa/core/types/TestCoreTuple.cpp
    noa/core/types/TestCoreVec.cpp
    noa/core/types/TestCoreVecTraits.cpp

    noa/core/utils/TestCoreClampCast.cpp
    noa/core/utils/TestCoreIRange.cpp
    noa/core/utils/TestCoreSafeCast.cpp
    noa/core/utils/TestCoreInterfaces.cpp
    noa/core/utils/TestCoreString.cpp
    noa/core/utils/TestCoreZip.cpp

    )

set(TEST_SOURCES ${TEST_SOURCES} ${TEST_COMMON_SOURCES})
