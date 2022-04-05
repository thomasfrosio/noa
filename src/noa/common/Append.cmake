# Included files for noa/common:

set(NOA_COMMON_HEADERS
        common/Assert.h
        common/Definitions.h
        common/Exception.h
        common/Functors.h
        common/Indexing.h
        common/Irange.h
        common/Logger.h
        common/Math.h
        common/OS.h
        common/Profiler.h
        common/Threadpool.h
        common/Timer.h

        common/geometry/Euler.cpp
        common/geometry/Euler.h
        common/geometry/Symmetry.h
        common/geometry/Transform.h

        common/io/BinaryFile.h
        common/io/header/Header.h
        common/io/header/MRCHeader.h
        common/io/header/TIFFHeader.h
        common/io/ImageFile.h
        common/io/ImageFile.inl
        common/io/IO.h
        common/io/IO.inl
        common/io/Stats.h
        common/io/TextFile.h
        common/io/TextFile.inl

        common/string/Format.h
        common/string/Parse.h
        common/string/Split.h

        common/Traits.h
        common/traits/BaseTypes.h
        common/traits/STLContainers.h

        common/Types.h
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
        common/types/View.h
        )

set(NOA_COMMON_SOURCES
        common/Exception.cpp
        common/Logger.cpp

        common/geometry/Symmetry.cpp
        common/io/BinaryFile.cpp
        common/io/header/MRCHeader.cpp
        common/io/header/TIFFHeader.cpp
        common/io/IO.cpp
        common/string/Parse.cpp
        common/types/Constants.cpp
        )

set(NOA_HEADERS ${NOA_HEADERS} ${NOA_COMMON_HEADERS})
set(NOA_SOURCES ${NOA_SOURCES} ${NOA_COMMON_SOURCES})