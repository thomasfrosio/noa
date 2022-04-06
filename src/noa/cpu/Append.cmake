# Included files for noa/cpu:
if (NOT NOA_ENABLE_CPU)
    return()
endif ()

set(NOA_CPU_HEADERS
        # noa::cpu
        cpu/Device.h
        cpu/Event.h
        cpu/Stream.h

        # # noa::cpu::fft
        # cpu/fft/Filters.h
        # cpu/fft/Plan.h
        # cpu/fft/Remap.h
        # cpu/fft/Resize.h
        # cpu/fft/Transforms.h

        # noa::cpu::math
        cpu/math/Complex.h
        cpu/math/Ewise.h
        cpu/math/Ewise.inl
        cpu/math/Find.h
        cpu/math/Reduce.h
        cpu/math/Reduce.inl

        # noa::cpu::filter
        cpu/filter/Convolve.h
        cpu/filter/Median.h
        cpu/filter/Shape.h

        # noa::cpu::memory
        cpu/memory/Arange.h
        cpu/memory/Cast.h
        cpu/memory/Copy.h
        cpu/memory/Index.h
        cpu/memory/Index.inl
        cpu/memory/Linspace.h
        cpu/memory/PtrHost.h
        cpu/memory/Resize.h
        cpu/memory/Set.h
        cpu/memory/Transpose.h

        # # noa::cpu::geometry
        # cpu/geometry/Interpolate.h
        # cpu/geometry/Interpolator.h
        # cpu/geometry/Prefilter.h
        # cpu/geometry/Rotate.h
        # cpu/geometry/Scale.h
        # cpu/geometry/Shift.h
        # cpu/geometry/Symmetry.h
        # cpu/geometry/Transform.h
        #
        # # noa::cpu::geometry::fft
        # cpu/geometry/fft/Shift.h
        # cpu/geometry/fft/Symmetry.h
        # cpu/geometry/fft/Transform.h
        #
        # # noa::cpu::reconstruct
        # cpu/reconstruct/ProjectBackward.h
        # cpu/reconstruct/ProjectForward.h


        )

set(NOA_CPU_SOURCES
        cpu/Device.cpp

        # # noa::cpu::fft
        # cpu/fft/Filters.cpp
        # cpu/fft/Plan.cpp
        # cpu/fft/Remap.cpp
        # cpu/fft/Resize.cpp

        # noa::cpu::math
        cpu/math/Find.cpp
        cpu/math/Reduce.cpp

        # noa::cpu::filter
        cpu/filter/Convolve.cpp
        cpu/filter/Median.cpp
        cpu/filter/ShapeCylinder.cpp
        cpu/filter/ShapeRectangle.cpp
        cpu/filter/ShapeSphere.cpp

        # noa::cpu::memory
        cpu/memory/Index.cpp
        cpu/memory/Resize.cpp
        cpu/memory/Transpose.cpp

        # # noa::cpu::geometry
        # cpu/geometry/Prefilter.cpp
        # cpu/geometry/Shift.cpp
        # cpu/geometry/Symmetry.cpp
        # cpu/geometry/Transform.cpp
        # cpu/geometry/TransformSymmetry.cpp
        #
        # # noa::cpu::geometry::fft
        # cpu/geometry/fft/Shift.cpp
        # cpu/geometry/fft/Transform.cpp
        # cpu/geometry/fft/TransformSymmetry.cpp
        #
        # # noa::cpu::reconstruct
        # cpu/reconstruct/ProjectBackward.cpp
        # cpu/reconstruct/ProjectForward.cpp

        )

set(NOA_HEADERS ${NOA_HEADERS} ${NOA_CPU_HEADERS})
set(NOA_SOURCES ${NOA_SOURCES} ${NOA_CPU_SOURCES})
