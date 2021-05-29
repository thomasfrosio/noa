# ---------------------------------------------------------------------------------------
# The library source and header files
# ---------------------------------------------------------------------------------------
set(NOA_SOURCES
        Logger.cpp
        Session.cpp

        # IO
        io/files/MRCFile.cpp
        io/files/ImageFile.cpp
        io/Inputs.cpp
        io/IO.cpp

        util/string/Convert.cpp

        )

set(NOA_HEADERS
        Angles.h
        Assert.h
        Definitions.h
        Environment.h
        Exception.h
        Logger.h
        Math.h
        Profiler.h
        Session.h
        Types.h

        # IO
        io/files/ImageFile.h
        io/files/MRCFile.h
        io/files/TextFile.h
        io/Inputs.h
        io/IO.h
        io/OS.h

        # Utilities
        util/Complex.h
        util/FloatX.h
        util/Float2.h
        util/Float3.h
        util/Float4.h
        util/BoolX.h
        util/Bool2.h
        util/Bool3.h
        util/Bool4.h
        util/IntX.h
        util/Int2.h
        util/Int3.h
        util/Int4.h
        util/Mat2.h
        util/Mat3.h
        util/Mat4.h
        util/MatX.h
        util/Traits.h
        util/traits/BaseTypes.h
        util/traits/Containers.h
        util/Sizes.h
        util/Stats.h
        util/String.h
        util/string/Convert.h
        util/string/Format.h
        util/string/Parse.h

        )

set(NOA_CPU_SOURCES
        cpu/fourier/Plan.cpp
        cpu/fourier/Filters.cpp
        cpu/fourier/Remap.cpp
        cpu/fourier/Resize.cpp

        cpu/math/Reductions.cpp
        cpu/math/Indexes.cpp
        cpu/math/ArithmeticsComposite.cpp
        cpu/math/Arithmetics.cpp

        cpu/masks/Sphere.cpp
        cpu/masks/Rectangle.cpp
        cpu/masks/Cylinder.cpp

        cpu/memory/Remap.cpp
        cpu/memory/Resize.cpp

        )

set(NOA_CPU_HEADERS
        # Fourier
        cpu/fourier/Transforms.h
        cpu/fourier/Plan.h
        cpu/fourier/Resize.h
        cpu/fourier/Remap.h
        cpu/fourier/Filters.h

        # Maths
        cpu/math/Arithmetics.h
        cpu/math/ArithmeticsComposite.h
        cpu/math/Booleans.h
        cpu/math/Generics.h
        cpu/math/Indexes.h
        cpu/math/Reductions.h

        # Masks
        cpu/masks/Sphere.h
        cpu/masks/Rectangle.h
        cpu/masks/Cylinder.h

        # Others
        cpu/Threadpool.h

        cpu/memory/Copy.h
        cpu/memory/PtrHost.h
        cpu/memory/Remap.h
        cpu/memory/Resize.h
        cpu/memory/Set.h

        )

set(NOA_CUDA_SOURCES
        gpu/cuda/fourier/Filters.cu
        gpu/cuda/fourier/Remap.cu
        gpu/cuda/fourier/Resize.cu

        gpu/cuda/math/Arithmetics.cu
        gpu/cuda/math/ArithmeticsComposite.cu
        gpu/cuda/math/Booleans.cu
        gpu/cuda/math/Generics.cu
        gpu/cuda/math/Indexes.cu
        gpu/cuda/math/reductions/Min_Max_SumMean.cu
        gpu/cuda/math/reductions/MinMaxSumMean.cu
        gpu/cuda/math/reductions/VarianceStddev.cu
        gpu/cuda/math/reductions/ReduceAdd.cu
        gpu/cuda/math/reductions/MinMax.cu

        gpu/cuda/masks/Sphere.cu
        gpu/cuda/masks/Rectangle.cu
        gpu/cuda/masks/Cylinder.cu

        gpu/cuda/memory/Remap.cu
        gpu/cuda/memory/Resize.cu
        gpu/cuda/memory/Set.cu

        )

set(NOA_CUDA_HEADERS
        gpu/cuda/Exception.h
        gpu/cuda/Types.h

        gpu/cuda/memory/Copy.h
        gpu/cuda/memory/PtrDevice.h
        gpu/cuda/memory/PtrPinned.h
        gpu/cuda/memory/PtrDevicePadded.h
        gpu/cuda/memory/PtrArray.h
        gpu/cuda/memory/PtrTexture.h

        # Fourier
        gpu/cuda/fourier/Exception.h
        gpu/cuda/fourier/Filters.h
        gpu/cuda/fourier/Plan.h
        gpu/cuda/fourier/Transforms.h
        gpu/cuda/fourier/Resize.h
        gpu/cuda/fourier/Remap.h

        # Math
        gpu/cuda/math/Arithmetics.h
        gpu/cuda/math/ArithmeticsComposite.h
        gpu/cuda/math/Booleans.h
        gpu/cuda/math/Generics.h
        gpu/cuda/math/Indexes.h
        gpu/cuda/math/Reductions.h

        # Mask
        gpu/cuda/masks/Sphere.h
        gpu/cuda/masks/Cylinder.h
        gpu/cuda/masks/Rectangle.h

        # Memory
        gpu/cuda/memory/Shared.h
        gpu/cuda/memory/Remap.h
        gpu/cuda/memory/Resize.h
        gpu/cuda/memory/Set.h

        # Utilities
        gpu/cuda/util/Device.h
        gpu/cuda/util/Event.h
        gpu/cuda/util/Stream.h
        gpu/cuda/util/Sizes.h

        )

include(${PROJECT_SOURCE_DIR}/cmake/ConfigLibrary.cmake)
