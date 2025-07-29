set(NOA_CORE_HEADERS
    core/Config.hpp
    core/Enums.hpp
    core/Ewise.hpp
    core/Error.hpp
    core/Interfaces.hpp
    core/Interpolation.hpp
    core/Iwise.hpp
    core/Namespace.hpp
    core/Reduce.hpp
    core/Traits.hpp

    core/fft/Frequency.hpp

    core/indexing/Bounds.hpp
    core/indexing/Layout.hpp
    core/indexing/Offset.hpp
    core/indexing/Subregion.hpp

    core/io/BinaryFile.hpp
    core/io/Encoding.hpp
    core/io/ImageFile.hpp
    core/io/IO.hpp
    core/io/TextFile.hpp

    core/math/Comparison.hpp
    core/math/Constant.hpp
    core/math/Distribution.hpp
    core/math/Generic.hpp
    core/math/LeastSquare.hpp

    core/types/Accessor.hpp
    core/types/Complex.hpp
    core/types/Half.hpp
    core/types/Mat.hpp
    core/types/Pair.hpp
    core/types/Shape.hpp
    core/types/Span.hpp
    core/types/Tuple.hpp
    core/types/Vec.hpp

    core/utils/Adaptor.hpp
    core/utils/Atomic.hpp
    core/utils/BatchedParameter.hpp
    core/utils/ClampCast.hpp
    core/utils/Irange.hpp
    core/utils/Misc.hpp
    core/utils/SafeCast.hpp
    core/utils/ShareHandles.hpp
    core/utils/Sort.hpp
    core/utils/Threadpool.hpp
    core/utils/Timer.hpp
    core/utils/Strings.hpp
    core/utils/Zip.hpp

    core/signal/CTF.hpp
    core/signal/Windows.hpp

    core/geometry/DrawShape.hpp
    core/geometry/Euler.hpp
    core/geometry/Polar.hpp
    core/geometry/Quaternion.hpp
    core/geometry/Symmetry.hpp
    core/geometry/Transform.hpp
    )

set(NOA_CORE_SOURCES
    core/Enums.cpp
    core/Error.cpp
    core/geometry/Euler.cpp
    core/io/BinaryFile.cpp
    core/io/Encoding.cpp
    core/io/ImageFile.cpp
    )
