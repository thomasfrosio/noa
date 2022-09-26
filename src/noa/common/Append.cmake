# Included files for noa/common:

set(NOA_COMMON_HEADERS
        common/Assert.h
        common/Definitions.h
        common/Exception.h
        common/Functors.h
        common/Indexing.h
        common/Logger.h
        common/OS.h
        common/Session.h
        common/Threadpool.h

        common/Math.h
        common/math/Comparison.h
        common/math/Constant.h
        common/math/Generic.h

        common/utils/Any.h
        common/utils/Irange.h
        common/utils/Profiler.h
        common/utils/Sort.h
        common/utils/Timer.h

        common/geometry/Euler.cpp
        common/geometry/Euler.h
        common/geometry/Polar.h
        common/geometry/Symmetry.h
        common/geometry/Transform.h

        common/io/BinaryFile.h
        common/io/ImageFile.h
        common/io/IO.h
        common/io/IO.inl
        common/io/MRCFile.h
        common/io/Stats.h
        common/io/TextFile.h
        common/io/TextFile.inl
        common/io/TIFFFile.h

        common/signal/Windows.h

        common/string/Format.h
        common/string/Parse.h
        common/string/Parse.inl
        common/string/Split.h

        common/Traits.h
        common/traits/ArrayTypes.h
        common/traits/BaseTypes.h
        common/traits/STLContainers.h
        common/traits/Utilities.h

        common/Types.h
        common/types/Accessor.h
        common/types/Bool2.h
        common/types/Bool3.h
        common/types/Bool4.h
        common/types/ClampCast.h
        common/types/Complex.h
        common/types/Constants.h
        common/types/Float2.h
        common/types/Float3.h
        common/types/Float4.h
        common/types/Half.h
        common/types/Int2.h
        common/types/Int3.h
        common/types/Int4.h
        common/types/Mat22.h
        common/types/Mat23.h
        common/types/Mat33.h
        common/types/Mat34.h
        common/types/Mat44.h
        common/types/Pair.h
        common/types/SafeCast.h
        common/types/View.h
        )

set(NOA_COMMON_SOURCES
        common/Exception.cpp
        common/Logger.cpp
        common/Session.cpp

        common/geometry/Symmetry.cpp

        common/io/IO.cpp
        common/io/BinaryFile.cpp
        common/io/MRCFile.cpp
        common/io/TIFFFile.cpp

        common/types/Constants.cpp
        )

set(NOA_HEADERS ${NOA_HEADERS} ${NOA_COMMON_HEADERS})
set(NOA_SOURCES ${NOA_SOURCES} ${NOA_COMMON_SOURCES})
