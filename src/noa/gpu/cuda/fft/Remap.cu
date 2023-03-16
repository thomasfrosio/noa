#include "noa/algorithms/fft/Remap.hpp"
#include "noa/gpu/cuda/memory/Copy.hpp"
#include "noa/gpu/cuda/fft/Exception.hpp"
#include "noa/gpu/cuda/fft/Remap.hpp"
#include "noa/gpu/cuda/utils/Block.cuh"
#include "noa/gpu/cuda/utils/Iwise.cuh"
#include "noa/gpu/cuda/utils/Pointers.hpp"

namespace {
    using namespace noa;
    using Remap = ::noa::fft::Remap;
    constexpr uint32_t MAX_THREADS = 256;

    // In-place, Y and Z dimensions have both an even number of elements.
    template<typename T>
    __global__ __launch_bounds__(MAX_THREADS)
    void h2hc_inplace_(Accessor<T, 4, u32> output, Shape3<u32> shape_fft) {
        const u32 batch = blockIdx.z;
        const Vec2<u32> gid{blockIdx.y, blockIdx.x};
        const u32 iz = noa::math::ifft_shift(gid[0], shape_fft[0]);
        const u32 iy = noa::math::ifft_shift(gid[1], shape_fft[1]);
        const auto input_row = output[batch][iz][iy];
        const auto output_row = output[batch][gid[0]][gid[1]];

        T* shared = cuda::utils::block_dynamic_shared_resource<T>();
        u32 count = 0;
        for (u32 x = threadIdx.x; x < shape_fft[2]; x += blockDim.x, ++count) {
            shared[x - count * blockDim.x] = output_row[x];
            output_row[x] = input_row[x];
            input_row[x] = shared[x - count * blockDim.x];
        }
    }
}

namespace noa::cuda::fft {
    template<typename T, typename>
    void remap(Remap remap,
               const T* input, Strides4<i64> input_strides,
               T* output, Strides4<i64> output_strides,
               Shape4<i64> shape, Stream& stream) {
        NOA_ASSERT(noa::all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());

        // Reordering is only possible for some remaps... and this
        // entirely depends on the algorithm we use. Regardless, the
        // batch dimension cannot be reordered.
        if (remap == noa::fft::FC2F || remap == noa::fft::F2FC ||
            remap == noa::fft::FC2H || remap == noa::fft::F2H) {
            const auto order_3d = noa::indexing::order(output_strides.pop_front(), shape.pop_front());
            if (noa::any(order_3d != Vec3<i64>{0, 1, 2})) {
                const auto order = (order_3d + 1).push_front(0);
                input_strides = input_strides.reorder(order);
                output_strides = output_strides.reorder(order);
                shape = shape.reorder(order);
            }
        }

        switch (remap) {
            case Remap::H2H:
            case Remap::HC2HC:
                if (input != output)
                    noa::cuda::memory::copy(input, input_strides, output, output_strides, shape.fft(), stream);
                break;
            case Remap::F2F:
            case Remap::FC2FC:
                if (input != output)
                    noa::cuda::memory::copy(input, input_strides, output, output_strides, shape, stream);
                break;
            case Remap::H2HC: {
                if (input == output) {
                    NOA_ASSERT((shape[2] == 1 || !(shape[2] % 2)) && (shape[1] == 1 || !(shape[1] % 2)));
                    NOA_ASSERT(noa::all(input_strides == output_strides));

                    const auto shape_fft = shape.pop_front().as_safe<u32>().fft();
                    const u32 threads = std::min(MAX_THREADS, noa::math::next_multiple_of(shape_fft[2], Constant::WARP_SIZE));
                    const dim3 blocks(std::max(shape_fft[1] / 2, 1u), shape_fft[0], shape[0]);
                    const auto output_accessor = Accessor<T, 4, uint32_t>(output, output_strides.as_safe<u32>());
                    const auto config = LaunchConfig{blocks, threads, threads * sizeof(T)};
                    return stream.enqueue("h2hc_inplace_", h2hc_inplace_<T>, config, output_accessor, shape_fft);
                } else {
                    const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::H2HC>(
                            input, input_strides, output, output_strides, shape);
                    return noa::cuda::utils::iwise_4d("remap", iwise_shape, kernel, stream);
                }
            }
            case Remap::HC2H: {
                const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::HC2H>(
                        input, input_strides, output, output_strides, shape);
                return noa::cuda::utils::iwise_4d("remap", iwise_shape, kernel, stream);
            }
            case Remap::H2F: {
                const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::H2F>(
                        input, input_strides, output, output_strides, shape);
                return noa::cuda::utils::iwise_4d("remap", iwise_shape, kernel, stream);
            }
            case Remap::F2H: {
                const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::F2H>(
                        input, input_strides, output, output_strides, shape);
                return noa::cuda::utils::iwise_4d("remap", iwise_shape, kernel, stream);
            }
            case Remap::F2FC: {
                const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::F2FC>(
                        input, input_strides, output, output_strides, shape);
                return noa::cuda::utils::iwise_4d("remap", iwise_shape, kernel, stream);
            }
            case Remap::FC2F: {
                const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::FC2F>(
                        input, input_strides, output, output_strides, shape);
                return noa::cuda::utils::iwise_4d("remap", iwise_shape, kernel, stream);
            }
            case Remap::HC2F: {
                const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::HC2F>(
                        input, input_strides, output, output_strides, shape);
                return noa::cuda::utils::iwise_4d("remap", iwise_shape, kernel, stream);
            }
            case Remap::F2HC: {
                const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::F2HC>(
                        input, input_strides, output, output_strides, shape);
                return noa::cuda::utils::iwise_4d("remap", iwise_shape, kernel, stream);
            }
            case Remap::FC2H: {
                const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::FC2H>(
                        input, input_strides, output, output_strides, shape);
                return noa::cuda::utils::iwise_4d("remap", iwise_shape, kernel, stream);
            }
            case Remap::FC2HC: {
                const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::FC2HC>(
                        input, input_strides, output, output_strides, shape);
                return noa::cuda::utils::iwise_4d("remap", iwise_shape, kernel, stream);
            }
            case noa::fft::Remap::HC2FC: {
                const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::HC2FC>(
                        input, input_strides, output, output_strides, shape);
                return noa::cuda::utils::iwise_4d("remap", iwise_shape, kernel, stream);
            }
            case noa::fft::Remap::H2FC: {
                const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::H2FC>(
                        input, input_strides, output, output_strides, shape);
                return noa::cuda::utils::iwise_4d("remap", iwise_shape, kernel, stream);
            }
        }
    }

    #define NOA_INSTANTIATE_RESIZE_(T) \
    template void remap<T, void>(Remap, const T*, Strides4<i64>, T*, Strides4<i64>, Shape4<i64>, Stream&)

    NOA_INSTANTIATE_RESIZE_(f16);
    NOA_INSTANTIATE_RESIZE_(f32);
    NOA_INSTANTIATE_RESIZE_(f64);
    NOA_INSTANTIATE_RESIZE_(c16);
    NOA_INSTANTIATE_RESIZE_(c32);
    NOA_INSTANTIATE_RESIZE_(c64);
}
