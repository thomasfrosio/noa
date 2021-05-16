#include "noa/cpu/memory/PtrHost.h"
#include "noa/cpu/memory/Remap.h"

#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/Remap.h"

namespace {
    using namespace Noa;

    NOA_FHD int3_t getCornerLeft_(int3_t subregion_shape, size3_t subregion_center) {
        return int3_t(subregion_center) - subregion_shape / 2;
    }

    NOA_FD size_t getOffset_(int3_t shape, size_t pitch, int idx_y, int idx_z) {
        return (static_cast<size_t>(idx_z) * static_cast<size_t>(shape.y) + static_cast<size_t>(idx_y)) * pitch;
    }

    template<typename T>
    __global__ void extractOrNothing_(const T* input, size_t input_pitch, int3_t input_shape,
                                      T* subregions, size_t subregion_pitch, size_t subregion_elements,
                                      int3_t subregion_shape, const size3_t* subregion_centers) {
        uint batch = blockIdx.z;
        int3_t corner_left = getCornerLeft_(subregion_shape, subregion_centers[batch]);

        uint o_y = blockIdx.x;
        uint o_z = blockIdx.y;
        int i_y = corner_left.y + static_cast<int>(o_y);
        int i_z = corner_left.z + static_cast<int>(o_z);
        if (i_z < 0 || i_z >= input_shape.z || i_y < 0 || i_y >= input_shape.y)
            return;

        input += getOffset_(input_shape, input_pitch, i_y, i_z);
        subregions += getOffset_(subregion_shape, subregion_pitch, o_y, o_z) + batch * subregion_elements;
        for (uint o_x = threadIdx.x; o_x < subregion_shape.x; o_x += blockDim.x) {
            int i_x = corner_left.x + static_cast<int>(o_x);
            if (i_x < 0 || i_x >= input_shape.x)
                continue;
            subregions[o_x] = input[i_x];
        }
    }

    template<typename T>
    __global__ void extractOrNothing_(const T* input, size_t input_pitch, int3_t input_shape,
                                      T* subregion, size_t subregion_pitch, int3_t subregion_shape,
                                      int3_t corner_left) {
        uint o_y = blockIdx.x;
        uint o_z = blockIdx.y;
        int i_y = corner_left.y + static_cast<int>(o_y);
        int i_z = corner_left.z + static_cast<int>(o_z);
        if (i_z < 0 || i_z >= input_shape.z || i_y < 0 || i_y >= input_shape.y)
            return;

        input += getOffset_(input_shape, input_pitch, i_y, i_z);
        subregion += getOffset_(subregion_shape, subregion_pitch, o_y, o_z);
        for (uint o_x = threadIdx.x; o_x < subregion_shape.x; o_x += blockDim.x) {
            int i_x = corner_left.x + static_cast<int>(o_x);
            if (i_x < 0 || i_x >= input_shape.x)
                continue;
            subregion[o_x] = input[i_x];
        }
    }

    template<typename T>
    __global__ void extractOrValue_(const T* input, size_t input_pitch, int3_t input_shape,
                                    T* subregions, size_t subregion_pitch, size_t subregion_elements,
                                    int3_t subregion_shape, const size3_t* subregion_centers, T value) {
        uint batch = blockIdx.z;
        int3_t corner_left = getCornerLeft_(subregion_shape, subregion_centers[batch]);

        uint o_y = blockIdx.x;
        uint o_z = blockIdx.y;
        int i_y = corner_left.y + static_cast<int>(o_y);
        int i_z = corner_left.z + static_cast<int>(o_z);
        bool is_out = i_z < 0 || i_z >= input_shape.z || i_y < 0 || i_y >= input_shape.y;

        subregions += getOffset_(subregion_shape, subregion_pitch, o_y, o_z) + batch * subregion_elements;
        for (uint o_x = threadIdx.x; o_x < subregion_shape.x; o_x += blockDim.x) {
            int i_x = corner_left.x + static_cast<int>(o_x);
            if (is_out || i_x < 0 || i_x >= input_shape.x)
                subregions[o_x] = value;
            else
                subregions[o_x] = input[getOffset_(input_shape, input_pitch, i_y, i_z) + i_x];
        }
    }

    template<typename T>
    __global__ void extractOrValue_(const T* input, size_t input_pitch, int3_t input_shape,
                                    T* subregion, size_t subregion_pitch,
                                    int3_t subregion_shape, int3_t corner_left, T value) {
        uint o_y = blockIdx.x;
        uint o_z = blockIdx.y;
        int i_y = corner_left.y + static_cast<int>(o_y);
        int i_z = corner_left.z + static_cast<int>(o_z);
        bool is_out = i_z < 0 || i_z >= input_shape.z || i_y < 0 || i_y >= input_shape.y;

        subregion += getOffset_(subregion_shape, subregion_pitch, o_y, o_z);
        for (uint o_x = threadIdx.x; o_x < subregion_shape.x; o_x += blockDim.x) {
            int i_x = corner_left.x + static_cast<int>(o_x);
            if (is_out || i_x < 0 || i_x >= input_shape.x)
                subregion[o_x] = value;
            else
                subregion[o_x] = input[getOffset_(input_shape, input_pitch, i_y, i_z) + i_x];
        }
    }

    template<typename T>
    __global__ void insert_(const T* subregion, size_t subregion_pitch, int3_t subregion_shape, int3_t corner_left,
                            T* output, size_t output_pitch, int3_t output_shape) {
        uint i_y = blockIdx.x;
        uint i_z = blockIdx.y;
        int o_y = corner_left.y + static_cast<int>(i_y);
        int o_z = corner_left.z + static_cast<int>(i_z);
        if (o_z < 0 || o_z >= output_shape.z || o_y < 0 || o_y >= output_shape.y)
            return;

        output += getOffset_(output_shape, output_pitch, o_y, o_z);
        subregion += getOffset_(subregion_shape, subregion_pitch, i_y, i_z);
        for (uint i_x = threadIdx.x; i_x < subregion_shape.x; i_x += blockDim.x) {
            int o_x = corner_left.x + static_cast<int>(i_x);
            if (o_x < 0 || o_x >= output_shape.x)
                continue;
            output[o_x] = subregion[i_x];
        }
    }

    template<typename T>
    __global__ void extractMap_(const T* i_sparse, size_t i_sparse_elements, T* o_dense, size_t o_dense_elements,
                                const size_t* i_map, uint batches) {
        for (uint batch = 0; batch < batches; ++batch) {
            const T* input = i_sparse + batch * i_sparse_elements;
            T* output = o_dense + batch * o_dense_elements;
            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                 idx < o_dense_elements;
                 idx += blockDim.x * gridDim.x)
                output[idx] = input[i_map[idx]];
        }
    }

    template<typename T>
    __global__ void insertMap_(const T* i_dense, size_t i_dense_elements, T* o_sparse, size_t o_sparse_elements,
                               const size_t* map, uint batches) {
        for (uint batch = 0; batch < batches; ++batch) {
            const T* input = i_dense + batch * i_dense_elements;
            T* output = o_sparse + batch * o_sparse_elements;
            for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
                 idx < i_dense_elements;
                 idx += blockDim.x * gridDim.x)
                output[map[idx]] = input[idx];
        }
    }
}

namespace Noa::CUDA::Memory {
    template<typename T>
    void extract(const T* input, size_t input_pitch, size3_t input_shape,
                 T* subregions, size_t subregion_pitch, size3_t subregion_shape, const size3_t* subregion_centers,
                 uint subregion_count, BorderMode border_mode, T border_value, Stream& stream) {
        int3_t i_shape(input_shape);
        int3_t o_shape(subregion_shape);
        size_t o_elements = subregion_pitch * getRows(subregion_shape);

        if (border_mode == BORDER_ZERO)
            border_value = 0;
        else if (border_mode != BORDER_VALUE && border_mode != BORDER_NOTHING)
            NOA_THROW("BorderMode not supported. Should be {}, {} or {}, got {}",
                      BORDER_NOTHING, BORDER_ZERO, BORDER_VALUE, border_mode);

        uint threads = Math::min(256U, Math::nextMultipleOf(static_cast<uint>(subregion_shape.x), 32U));
        dim3 blocks(o_shape.y, o_shape.z, subregion_count);
        if (border_mode == BORDER_NOTHING) {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            extractOrNothing_,
                            input, input_pitch, i_shape,
                            subregions, subregion_pitch, o_elements, o_shape, subregion_centers);
        } else {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            extractOrValue_,
                            input, input_pitch, i_shape,
                            subregions, subregion_pitch, o_elements, o_shape, subregion_centers, border_value);
        }
    }

    template<typename T>
    void extract(const T* input, size_t input_pitch, size3_t input_shape,
                 T* subregion, size_t subregion_pitch, size3_t subregion_shape, size3_t subregion_center,
                 BorderMode border_mode, T border_value, Stream& stream) {
        int3_t i_shape(input_shape);
        int3_t o_shape(subregion_shape);
        int3_t corner_left = getCornerLeft_(o_shape, subregion_center);

        if (border_mode == BORDER_ZERO)
            border_value = 0;
        else if (border_mode != BORDER_VALUE && border_mode != BORDER_NOTHING)
            NOA_THROW("BorderMode not supported. Should be {}, {} or {}, got {}",
                      BORDER_NOTHING, BORDER_ZERO, BORDER_VALUE, border_mode);

        uint threads = Math::min(256U, Math::nextMultipleOf(static_cast<uint>(subregion_shape.x), 32U));
        dim3 blocks(o_shape.y, o_shape.z, 1);
        if (border_mode == BORDER_NOTHING) {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            extractOrNothing_,
                            input, input_pitch, i_shape,
                            subregion, subregion_pitch, o_shape, corner_left);
        } else {
            NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                            extractOrValue_,
                            input, input_pitch, i_shape,
                            subregion, subregion_pitch, o_shape, corner_left, border_value);
        }
    }

    template<typename T>
    void insert(const T* subregion, size_t subregion_pitch, size3_t subregion_shape, size3_t subregion_center,
                T* output, size_t output_pitch, size3_t output_shape, Stream& stream) {
        int3_t i_shape(subregion_shape);
        int3_t o_shape(output_shape);
        int3_t corner_left = getCornerLeft_(i_shape, subregion_center);
        uint threads = Math::min(256U, Math::nextMultipleOf(static_cast<uint>(i_shape.x), 32U));
        dim3 blocks(i_shape.y, i_shape.z, 1);
        NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                        insert_,
                        subregion, subregion_pitch, i_shape, corner_left, output, output_pitch, o_shape);
    }

    template<typename T>
    std::pair<size_t*, size_t> getMap(const T* mask, size_t elements, T threshold, Stream& stream) {
        // Copy to the CPU and compute the map there.
        Noa::Memory::PtrHost<T> h_mask(elements);
        copy(mask, h_mask.get(), elements, stream);
        Stream::synchronize(stream);
        auto[h_free_map, elements_mapped] = Noa::Memory::getMap(h_mask.get(), elements, threshold);
        Noa::Memory::PtrHost<size_t> h_map(h_free_map, elements_mapped); // capture

        // Copy map to GPU
        PtrDevice<size_t> d_map(elements_mapped);
        copy(h_map.get(), d_map.get(), d_map.elements(), stream);
        Stream::synchronize(stream); // don't destruct h_map until the copy is done.
        return {d_map.release(), elements_mapped};
    }

    template<typename T>
    std::pair<size_t*, size_t> getMap(const T* mask, size_t mask_pitch, size3_t mask_shape,
                                      T threshold, Stream& stream) {
        // Back and forth to the CPU.
        Noa::Memory::PtrHost<T> h_mask(getElements(mask_shape));
        copy(mask, mask_pitch, h_mask.get(), mask_shape.x, mask_shape, stream);
        Stream::synchronize(stream);
        auto[h_free_map, elements_mapped] = Noa::Memory::getMap(h_mask.get(), h_mask.elements(), threshold);
        Noa::Memory::PtrHost<size_t> h_map(h_free_map, elements_mapped); // capture

        // Copy map to GPU
        PtrDevice<size_t> d_map(elements_mapped);
        copy(h_map.get(), d_map.get(), d_map.elements(), stream);
        Stream::synchronize(stream); // don't destruct h_map until the copy is done.
        return {d_map.release(), elements_mapped};
    }

    template<typename T>
    void extract(const T* i_sparse, size_t i_sparse_elements, T* o_dense, size_t o_dense_elements,
                 const size_t* i_map, uint batches, Stream& stream) {
        uint threads = 192U;
        uint blocks = Math::min((static_cast<uint>(o_dense_elements) + threads - 1) / threads, 32768U);
        NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                        extractMap_,
                        i_sparse, i_sparse_elements, o_dense, o_dense_elements, i_map, batches);
    }

    template<typename T>
    void insert(const T* i_dense, size_t i_dense_elements, T* o_sparse, size_t o_sparse_elements,
                const size_t* i_map, uint batches, Stream& stream) {
        uint threads = 192U;
        uint blocks = Math::min((static_cast<uint>(i_dense_elements) + threads - 1) / threads, 32768U);
        NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                        insertMap_,
                        i_dense, i_dense_elements, o_sparse, o_sparse_elements, i_map, batches);
    }

    #define INSTANTIATE_EXTRACT_INSERT(T)                                                                                   \
    template void extract<T>(const T*, size_t, size3_t, T*, size_t, size3_t, const size3_t*, uint, BorderMode, T, Stream&); \
    template void extract<T>(const T*, size_t, size3_t, T*, size_t, size3_t, size3_t, BorderMode, T, Stream&);              \
    template void insert<T>(const T*, size_t, size3_t, size3_t, T*, size_t, size3_t, Stream&);                              \
    template std::pair<size_t*, size_t> getMap<T>(const T*, size_t, T, Stream&);                                            \
    template std::pair<size_t*, size_t> getMap<T>(const T*, size_t, size3_t, T, Stream&);                                   \
    template void extract<T>(const T*, size_t, T*, size_t, const size_t*, uint, Stream&);                                   \
    template void insert<T>(const T*, size_t, T*, size_t, const size_t*, uint, Stream&)

    INSTANTIATE_EXTRACT_INSERT(short);
    INSTANTIATE_EXTRACT_INSERT(int);
    INSTANTIATE_EXTRACT_INSERT(long);
    INSTANTIATE_EXTRACT_INSERT(long long);
    INSTANTIATE_EXTRACT_INSERT(unsigned short);
    INSTANTIATE_EXTRACT_INSERT(unsigned int);
    INSTANTIATE_EXTRACT_INSERT(unsigned long);
    INSTANTIATE_EXTRACT_INSERT(unsigned long long);
    INSTANTIATE_EXTRACT_INSERT(float);
    INSTANTIATE_EXTRACT_INSERT(double);
}
