#include "noa/common/Assert.h"
#include "noa/common/Math.h"

#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/memory/Index.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/util/Pointers.h"

namespace {
    using namespace noa;
    constexpr uint BLOCK_SIZE = 256;
    constexpr dim3 BLOCK_SIZE_2D(32, BLOCK_SIZE / 32);
    constexpr uint ELEMENTS_PER_THREAD = 4;
    constexpr dim3 BLOCK_WORK_SIZE_2D(BLOCK_SIZE_2D.x * ELEMENTS_PER_THREAD, BLOCK_SIZE_2D.y);

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void extractOrNothing_(const T* __restrict__ input, uint4_t input_strides, int4_t input_shape,
                           T* __restrict__ subregions, uint4_t subregion_strides, int2_t subregion_shape,
                           const int4_t* __restrict__ origins, int4_t order, uint blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        if (gid[2] >= subregion_shape[0])
            return;

        const int4_t origin = indexing::reorder(origins[gid[0]], order); // TODO constant memory?
        const int ii = origin[0];
        const int ij = origin[1] + gid[1];
        const int ik = origin[2] + gid[2];
        if (ii < 0 || ii >= input_shape[0] ||
            ij < 0 || ij >= input_shape[1] ||
            ik < 0 || ik >= input_shape[2])
            return;

        input += indexing::at(ii, ij, ik, input_strides);
        subregions += indexing::at(gid[0], gid[1], gid[2], subregion_strides);

        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            const int ol = gid[3] + static_cast<int>(BLOCK_SIZE_2D.x) * i;
            const int il = origin[3] + ol;
            if (ol < subregion_shape[1] && il >= 0 && il < input_shape[3])
                subregions[ol * subregion_strides[3]] = input[il * input_strides[3]];
        }
    }

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void extractOrValue_(const T* __restrict__ input, uint4_t input_strides, int4_t input_shape,
                         T* __restrict__ subregions, uint4_t subregion_strides, int2_t subregion_shape,
                         const int4_t* __restrict__ origins, T value, int4_t order, uint blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        if (gid[2] >= subregion_shape[0])
            return;

        const int4_t origin = indexing::reorder(origins[gid[0]], order); // TODO constant memory?
        const int ii = origin[0];
        const int ij = origin[1] + gid[1];
        const int ik = origin[2] + gid[2];
        const bool is_in = ii >= 0 && ii < input_shape[0] &&
                           ij >= 0 && ij < input_shape[1] &&
                           ik >= 0 && ik < input_shape[2];

        subregions += indexing::at(gid[0], gid[1], gid[2], subregion_strides);
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            const int ol = gid[3] + static_cast<int>(BLOCK_SIZE_2D.x) * i;
            if (ol >= subregion_shape[1])
                return;

            const int il = origin[3] + ol;
            if (is_in && il >= 0 && il < input_shape[3])
                subregions[ol * subregion_strides[3]] = input[indexing::at(ii, ij, ik, il, input_strides)];
            else
                subregions[ol * subregion_strides[3]] = value;
        }
    }

    template<BorderMode MODE, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void extract_(const T* __restrict__ input, uint4_t input_strides, int4_t input_shape,
                  T* __restrict__ subregions, uint4_t subregion_strides, int2_t subregion_shape,
                  const int4_t* __restrict__ origins, int4_t order, uint blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        if (gid[2] >= subregion_shape[0])
            return;

        const int4_t origin = indexing::reorder(origins[gid[0]], order); // TODO constant memory?
        const int ii = indexing::at<MODE>(origin[0], input_shape[0]);
        const int ij = indexing::at<MODE>(origin[1] + gid[1], input_shape[1]);
        const int ik = indexing::at<MODE>(origin[2] + gid[2], input_shape[2]);

        input += indexing::at(ii, ij, ik, input_strides);
        subregions += indexing::at(gid[0], gid[1], gid[2], subregion_strides);

        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            const int ol = gid[2] + static_cast<int>(BLOCK_SIZE_2D.x) * i;
            const int il = indexing::at<MODE>(origin[3] + ol, input_shape[3]);
            if (ol < subregion_shape[1])
                subregions[ol * subregion_strides[3]] = input[il * input_strides[3]];
        }
    }

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void insert_(const T* __restrict__ subregions, uint4_t subregion_strides, int2_t subregion_shape,
                 T* __restrict__ output, uint4_t output_strides, int4_t output_shape,
                 const int4_t* __restrict__ origins, int4_t order, uint blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        if (gid[2] >= subregion_shape[0])
            return;

        const int4_t origin = indexing::reorder(origins[gid[0]], order); // TODO constant memory?
        const int oi = origin[0];
        const int oj = origin[1] + gid[1];
        const int ok = origin[2] + gid[2];
        if (oi < 0 || oi >= output_shape[0] ||
            oj < 0 || oj >= output_shape[1] ||
            ok < 0 || ok >= output_shape[2])
            return;

        output += indexing::at(oi, oj, ok, output_strides);
        subregions += indexing::at(gid[0], gid[1], gid[2], subregion_strides);

        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            const int il = gid[3] + static_cast<int>(BLOCK_SIZE_2D.x) * i;
            const int ol = origin[3] + il;
            if (il < subregion_shape[1] && ol >= 0 && ol < output_shape[3])
                output[ol * output_strides[3]] = subregions[il * subregion_strides[3]];
        }
    }
}

namespace noa::cuda::memory {
    template<typename T, typename>
    void extract(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                 const shared_t<T[]>& subregions, size4_t subregion_strides, size4_t subregion_shape,
                 const shared_t<int4_t[]>& origins, BorderMode border_mode, T border_value, Stream& stream) {
        NOA_ASSERT(input != subregions);

        // Reorder the DHW dimensions to the rightmost order.
        // We'll have to reorder the origins similarly later.
        const size3_t order_3d = indexing::order(size3_t(subregion_strides.get(1)),
                                                 size3_t(subregion_shape.get(1))) + 1;
        const int4_t order(0, order_3d[0], order_3d[1], order_3d[2]);
        input_strides = indexing::reorder(input_strides, order);
        input_shape = indexing::reorder(input_shape, order);
        subregion_strides = indexing::reorder(subregion_strides, order);
        subregion_shape = indexing::reorder(subregion_shape, order);

        const shared_t<int4_t[]> d_origins = util::ensureDeviceAccess(origins, stream, subregion_shape[0]);
        const int4_t i_shape(input_shape);
        const int2_t o_shape(subregion_shape.get(2));

        const uint blocks_x = math::divideUp(static_cast<uint>(o_shape[1]), BLOCK_WORK_SIZE_2D.x);
        const uint blocks_y = math::divideUp(static_cast<uint>(o_shape[0]), BLOCK_WORK_SIZE_2D.y);
        const dim3 blocks(blocks_x * blocks_y, subregion_shape[1], subregion_shape[0]);
        switch (border_mode) {
            case BORDER_NOTHING:
                stream.enqueue("memory::extractOrNothing", extractOrNothing_<T>, {blocks, BLOCK_SIZE_2D},
                               input.get(), uint4_t(input_strides), i_shape,
                               subregions.get(), uint4_t(subregion_strides),
                               o_shape, d_origins.get(), order, blocks_x);
                break;
            case BORDER_ZERO:
                stream.enqueue("memory::extractOrValue", extractOrValue_<T>, {blocks, BLOCK_SIZE_2D},
                               input.get(), uint4_t(input_strides), i_shape,
                               subregions.get(), uint4_t(subregion_strides),
                               o_shape, d_origins.get(), static_cast<T>(0), order, blocks_x);
                break;
            case BORDER_VALUE:
                stream.enqueue("memory::extractOrValue", extractOrValue_<T>, {blocks, BLOCK_SIZE_2D},
                               input.get(), uint4_t(input_strides), i_shape,
                               subregions.get(), uint4_t(subregion_strides),
                               o_shape, d_origins.get(), border_value, order, blocks_x);
                break;
            case BORDER_CLAMP:
                stream.enqueue("memory::extract<CLAMP>", extract_<BORDER_CLAMP, T>, {blocks, BLOCK_SIZE_2D},
                               input.get(), uint4_t(input_strides), i_shape,
                               subregions.get(), uint4_t(subregion_strides),
                               o_shape, d_origins.get(), order, blocks_x);
                break;
            case BORDER_MIRROR:
                stream.enqueue("memory::extract<MIRROR>", extract_<BORDER_MIRROR, T>, {blocks, BLOCK_SIZE_2D},
                               input.get(), uint4_t(input_strides), i_shape,
                               subregions.get(), uint4_t(subregion_strides),
                               o_shape, d_origins.get(), order, blocks_x);
                break;
            case BORDER_REFLECT:
                stream.enqueue("memory::extract<REFLECT>", extract_<BORDER_REFLECT, T>, {blocks, BLOCK_SIZE_2D},
                               input.get(), uint4_t(input_strides), i_shape,
                               subregions.get(), uint4_t(subregion_strides),
                               o_shape, d_origins.get(), order, blocks_x);
                break;
            default:
                NOA_THROW("Border mode {} is not supported", border_mode);
        }
        stream.attach(input, subregions, d_origins);
    }

    template<typename T, typename>
    void insert(const shared_t<T[]>& subregions, size4_t subregion_strides, size4_t subregion_shape,
                const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape,
                const shared_t<int4_t[]>& origins, Stream& stream) {
        NOA_ASSERT(subregions != output);

        // Reorder the DHW dimensions to the rightmost order.
        // We'll have to reorder the origins similarly later.
        const size3_t order_3d = indexing::order(size3_t(subregion_strides.get(1)),
                                                 size3_t(subregion_shape.get(1))) + 1;
        const int4_t order(0, order_3d[0], order_3d[1], order_3d[2]);
        output_strides = indexing::reorder(output_strides, order);
        output_shape = indexing::reorder(output_shape, order);
        subregion_strides = indexing::reorder(subregion_strides, order);
        subregion_shape = indexing::reorder(subregion_shape, order);

        const shared_t<int4_t[]> d_origins = util::ensureDeviceAccess(origins, stream, subregion_shape[0]);
        const int2_t i_shape(subregion_shape.get(2));
        const uint blocks_x = math::divideUp(static_cast<uint>(i_shape[1]), BLOCK_WORK_SIZE_2D.x);
        const uint blocks_y = math::divideUp(static_cast<uint>(i_shape[0]), BLOCK_WORK_SIZE_2D.y);
        const dim3 blocks(blocks_x * blocks_y, subregion_shape[1], subregion_shape[0]);
        stream.enqueue("memory::insert", insert_<T>, {blocks, BLOCK_SIZE_2D},
                       subregions.get(), uint4_t(subregion_strides), i_shape,
                       output.get(), uint4_t(output_strides), int4_t{output_shape}, d_origins.get(),
                       order, blocks_x);
        stream.attach(subregions, output, d_origins);
    }

    #define INSTANTIATE_EXTRACT_INSERT_(T)                                                                                                                              \
    template void extract<T, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<int4_t[]>&, BorderMode, T, Stream&);  \
    template void insert<T, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<int4_t[]>&, Stream&)

    INSTANTIATE_EXTRACT_INSERT_(bool);
    INSTANTIATE_EXTRACT_INSERT_(int8_t);
    INSTANTIATE_EXTRACT_INSERT_(int16_t);
    INSTANTIATE_EXTRACT_INSERT_(int32_t);
    INSTANTIATE_EXTRACT_INSERT_(int64_t);
    INSTANTIATE_EXTRACT_INSERT_(uint8_t);
    INSTANTIATE_EXTRACT_INSERT_(uint16_t);
    INSTANTIATE_EXTRACT_INSERT_(uint32_t);
    INSTANTIATE_EXTRACT_INSERT_(uint64_t);
    INSTANTIATE_EXTRACT_INSERT_(half_t);
    INSTANTIATE_EXTRACT_INSERT_(float);
    INSTANTIATE_EXTRACT_INSERT_(double);
    INSTANTIATE_EXTRACT_INSERT_(chalf_t);
    INSTANTIATE_EXTRACT_INSERT_(cfloat_t);
    INSTANTIATE_EXTRACT_INSERT_(cdouble_t);
}
