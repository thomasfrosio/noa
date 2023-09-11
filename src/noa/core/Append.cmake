# Included files for noa/core:

set(NOA_CORE_HEADERS
    core/Assert.hpp
    core/Definitions.hpp
    core/Exception.hpp
    core/OS.hpp

    core/Math.hpp
    core/math/Comparison.hpp
    core/math/Constant.hpp
    core/math/Distribution.hpp
    core/math/Enums.hpp
    core/math/Generic.hpp
    core/math/LeastSquare.hpp
    core/math/Range.hpp

    core/utils/Atomic.hpp
    core/utils/ClampCast.hpp
    core/utils/Indexing.hpp
    core/utils/Irange.hpp
    core/utils/Misc.hpp
    core/utils/SafeCast.hpp
    core/utils/Sort.hpp
    core/utils/Threadpool.hpp
    core/utils/Timer.hpp

    core/string/Format.hpp
    core/string/Parse.hpp
    core/string/Parse.inl
    core/string/Split.hpp

    core/Traits.hpp
    core/traits/Accessor.hpp
    core/traits/CTF.hpp
    core/traits/Matrix.hpp
    core/traits/Numerics.hpp
    core/traits/Shape.hpp
    core/traits/String.hpp
    core/traits/Utilities.hpp
    core/traits/VArray.hpp
    core/traits/Vec.hpp

    core/Types.hpp
    core/types/Accessor.hpp
    core/types/Complex.hpp
    core/types/Functors.hpp
    core/types/Half.hpp
    core/types/Mat.hpp
    core/types/Mat22.hpp
    core/types/Mat23.hpp
    core/types/Mat33.hpp
    core/types/Mat34.hpp
    core/types/Mat44.hpp
    core/types/Pair.hpp
    core/types/Shape.hpp
    core/types/Span.hpp
    core/types/Vec.hpp

    core/geometry/Enums.hpp
    core/geometry/Euler.hpp
    core/geometry/Interpolate.hpp
    core/geometry/Interpolator.hpp
    core/geometry/InterpolatorValue.hpp
    core/geometry/Polar.hpp
    core/geometry/Shape.hpp
    core/geometry/Symmetry.hpp
    core/geometry/Transform.hpp
    core/geometry/Quaternion.hpp

    core/signal/fft/CTF.hpp
    core/signal/Enums.hpp
    core/signal/Windows.hpp

    core/io/BinaryFile.hpp
    core/io/ImageFile.hpp
    core/io/IO.hpp
    core/io/IO.inl
    core/io/MRCFile.hpp
    core/io/Stats.hpp
    core/io/TextFile.hpp
    core/io/TIFFFile.hpp

    core/fft/Frequency.hpp
    core/fft/Enums.hpp

    )

set(NOA_CORE_SOURCES
    core/Exception.cpp

    core/geometry/Euler.cpp
    core/geometry/Symmetry.cpp

    core/io/IO.cpp
    core/io/BinaryFile.cpp
    core/io/MRCFile.cpp
    core/io/TIFFFile.cpp
    )

set(NOA_HEADERS ${NOA_HEADERS} ${NOA_CORE_HEADERS})
set(NOA_SOURCES ${NOA_SOURCES} ${NOA_CORE_SOURCES})
