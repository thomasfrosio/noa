#include "noa/algorithms/fft/Remap.hpp"
#include "noa/cpu/fft/Remap.hpp"
#include "noa/cpu/memory/Copy.hpp"
#include "noa/cpu/utils/Iwise.hpp"

namespace {
    using namespace noa;
    using Remap = noa::fft::Remap;

    template<typename T>
    void h2hc_inplace_(T* output, Strides4<i64> output_strides, Shape4<i64> shape, i64 threads) {
        // E.g. from h = [0,1,2,3,-4,-3,-2,-1] to hc = [-4,-3,-2,-1,0,1,2,3]
        // Simple swap is OK.
        const auto output_accessor = AccessorRestrict<T, 4, i64>(output, output_strides);
        const auto iwise_shape = Shape3<i64>{shape[0], shape[1], std::max(shape[2] / 2, i64{1})};
        const auto shape_3d = shape.pop_front();

        const auto kernel = [=](i64 i, i64 j, i64 k) {
            const i64 base_j = noa::fft::fftshift(j, shape_3d[0]);
            const i64 base_k = noa::fft::fftshift(k, shape_3d[1]);
            const auto i_in = output_accessor[i][j][k];
            const auto i_out = output_accessor[i][base_j][base_k];
            for (i64 l = 0; l < shape_3d[2] / 2 + 1; ++l)
                std::swap(i_in[l], i_out[l]);
        };
        noa::cpu::utils::iwise_3d(iwise_shape, kernel, threads);
    }
}

namespace noa::cpu::fft {
    template<typename T, typename>
    void remap(Remap remap,
               const T* input, Strides4<i64> input_strides,
               T* output, Strides4<i64> output_strides,
               Shape4<i64> shape, i64 threads) {
        NOA_ASSERT(input && output && noa::all(shape > 0));

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
                    noa::cpu::memory::copy(input, input_strides, output, output_strides, shape.rfft(), threads);
                break;
            case Remap::F2F:
            case Remap::FC2FC:
                if (input != output)
                    noa::cpu::memory::copy(input, input_strides, output, output_strides, shape, threads);
                break;
            case Remap::H2HC: {
                if (input == output) {
                    NOA_ASSERT((shape[2] == 1 || !(shape[2] % 2)) && (shape[1] == 1 || !(shape[1] % 2)));
                    NOA_ASSERT(noa::all(input_strides == output_strides));
                    return h2hc_inplace_(output, output_strides, shape, threads);
                } else {
                    const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::H2HC>(
                            input, input_strides, output, output_strides, shape);
                    return noa::cpu::utils::iwise_4d(iwise_shape, kernel, threads);
                }
            }
            case Remap::HC2H: {
                const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::HC2H>(
                        input, input_strides, output, output_strides, shape);
                return noa::cpu::utils::iwise_4d(iwise_shape, kernel, threads);
            }
            case Remap::H2F: {
                const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::H2F>(
                        input, input_strides, output, output_strides, shape);
                return noa::cpu::utils::iwise_4d(iwise_shape, kernel, threads);
            }
            case Remap::F2H: {
                const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::F2H>(
                        input, input_strides, output, output_strides, shape);
                return noa::cpu::utils::iwise_4d(iwise_shape, kernel, threads);
            }
            case Remap::F2FC: {
                const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::F2FC>(
                        input, input_strides, output, output_strides, shape);
                return noa::cpu::utils::iwise_4d(iwise_shape, kernel, threads);
            }
            case Remap::FC2F: {
                const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::FC2F>(
                        input, input_strides, output, output_strides, shape);
                return noa::cpu::utils::iwise_4d(iwise_shape, kernel, threads);
            }
            case Remap::HC2F: {
                const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::HC2F>(
                        input, input_strides, output, output_strides, shape);
                return noa::cpu::utils::iwise_4d(iwise_shape, kernel, threads);
            }
            case Remap::F2HC: {
                const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::F2HC>(
                        input, input_strides, output, output_strides, shape);
                return noa::cpu::utils::iwise_4d(iwise_shape, kernel, threads);
            }
            case Remap::FC2H: {
                const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::FC2H>(
                        input, input_strides, output, output_strides, shape);
                return noa::cpu::utils::iwise_4d(iwise_shape, kernel, threads);
            }
            case Remap::FC2HC: {
                const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::FC2HC>(
                        input, input_strides, output, output_strides, shape);
                return noa::cpu::utils::iwise_4d(iwise_shape, kernel, threads);
            }
            case Remap::HC2FC: {
                const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::HC2FC>(
                        input, input_strides, output, output_strides, shape);
                return noa::cpu::utils::iwise_4d(iwise_shape, kernel, threads);
            }
            case Remap::H2FC: {
                const auto [kernel, iwise_shape] = noa::algorithm::fft::remap<Remap::H2FC>(
                        input, input_strides, output, output_strides, shape);
                return noa::cpu::utils::iwise_4d(iwise_shape, kernel, threads);
            }
        }
    }

    #define NOA_INSTANTIATE_RESIZE_(T) \
    template void remap<T, void>(Remap, const T*, Strides4<i64>, T*, Strides4<i64>, Shape4<i64>, i64)

//    NOA_INSTANTIATE_RESIZE_(f16);
    NOA_INSTANTIATE_RESIZE_(f32);
//    NOA_INSTANTIATE_RESIZE_(f64);
//    NOA_INSTANTIATE_RESIZE_(c16);
    NOA_INSTANTIATE_RESIZE_(c32);
//    NOA_INSTANTIATE_RESIZE_(c64);
}
