set(NOA_CORE_HEADERS
    core/Config.hpp
    core/Exception.hpp
    core/Enums.hpp
    core/Traits.hpp

    core/Math.hpp
    core/math/Comparison.hpp
    core/math/Constant.hpp
    core/math/Distribution.hpp
    core/math/Generic.hpp
    core/math/LeastSquare.hpp
    core/math/Ops.hpp
    core/math/ReduceOps.hpp

    core/Indexing.hpp
    core/indexing/Offset.hpp
    core/indexing/Layout.hpp
    core/indexing/Subregion.hpp

    core/utils/Atomic.hpp
    core/utils/ClampCast.hpp
    core/utils/Irange.hpp
    core/utils/Misc.hpp
    core/utils/SafeCast.hpp
    core/utils/Sort.hpp
    core/utils/Threadpool.hpp
    core/utils/Timer.hpp

    core/string/Format.hpp
    core/string/Parse.hpp
    core/string/Reflect.hpp
    core/string/Split.hpp

    core/Types.hpp
    core/types/Accessor.hpp
    core/types/Complex.hpp
    core/types/Half.hpp
    core/types/Mat.hpp
    core/types/Mat22.hpp
    core/types/Mat23.hpp
    core/types/Mat33.hpp
    core/types/Mat34.hpp
    core/types/Mat44.hpp
    core/types/Shape.hpp
    core/types/Span.hpp
    core/types/Tuple.hpp
    core/types/Vec.hpp

    core/geometry/Euler.hpp
    core/geometry/Interpolate.hpp
    core/geometry/Interpolator.hpp
    core/geometry/Polar.hpp
    core/geometry/Prefilter.hpp
    core/geometry/Quaternion.hpp
    core/geometry/Shape.hpp
    core/geometry/Symmetry.hpp
    core/geometry/Transform.hpp

    core/signal/fft/CTF.hpp
    core/signal/Windows.hpp

    core/io/BinaryFile.hpp
    core/io/ImageFile.hpp
    core/io/IO.hpp
    core/io/MRCFile.hpp
    core/io/OS.hpp
    core/io/Stats.hpp
    core/io/TextFile.hpp
    core/io/TIFFFile.hpp

    core/fft/Frequency.hpp
    )

set(NOA_CORE_SOURCES
    core/Exception.cpp
    core/geometry/Euler.cpp
    core/io/IO.cpp
    core/io/BinaryFile.cpp
    core/io/MRCFile.cpp
    core/io/TIFFFile.cpp
    )
