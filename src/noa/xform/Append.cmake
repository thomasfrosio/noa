list(APPEND NOA_HEADERS
    noa/xform/core/CubicBSplinePrefilter.hpp
    noa/xform/core/Draw.hpp
    noa/xform/core/Euler.hpp
    noa/xform/core/Interp.hpp
    noa/xform/core/Interpolation.hpp
    noa/xform/core/Polar.hpp
    noa/xform/core/Quaternion.hpp
    noa/xform/core/Symmetry.hpp
    noa/xform/core/Transform.hpp

    noa/xform/CubicBSplinePrefilter.hpp
    noa/xform/Draw.hpp
    noa/xform/FourierProject.hpp
    noa/xform/PolarTransform.hpp
    noa/xform/PolarTransformSpectrum.hpp
    noa/xform/Project.hpp
    noa/xform/RotationalAverage.hpp
    noa/xform/Symmetry.hpp
    noa/xform/Texture.hpp
    noa/xform/Transform.hpp
    noa/xform/TransformSpectrum.hpp
)

list(APPEND NOA_SOURCES
    noa/xform/core/Euler.cpp
)

if (NOA_ENABLE_CPU)
    list(APPEND NOA_HEADERS
        noa/xform/cpu/CubicBSplinePrefilter.hpp
    )
endif ()

if (NOA_ENABLE_CUDA)
    list(APPEND NOA_HEADERS
        noa/xform/cuda/Allocators.hpp
        noa/xform/cuda/Texture.cuh
        noa/xform/cuda/CubicBSplinePrefilter.cuh
    )
endif ()
