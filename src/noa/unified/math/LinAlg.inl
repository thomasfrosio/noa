#ifndef NOA_UNIFIED_LINALG_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

#include "noa/cpu/math/LinAlg.h"
#include "noa/unified/Array.h"
#include "noa/unified/memory/Copy.h"

namespace noa::math {
    template<typename T, typename U, typename>
    void lstsq(const Array<T>& a, const Array<T>& b, const Array<T>& x,
               float cond, const Array<U>& svd) {
        NOA_CHECK(!a.empty() && !b.empty() && !x.empty(), "Empty array detected");

        // Check shape
        NOA_CHECK(a.shape()[0] == b.shape()[0] && a.shape()[0] == x.shape()[0],
                  "The number of batches does not match, got a:{}, b:{} and x:{}",
                  a.shape()[0], b.shape()[0], x.shape()[0]);
        NOA_CHECK(a.shape()[1] == 1 && b.shape()[0] == 1 && x.shape()[0] == 1,
                  "3D matrices are not supported, but got shape a:{}, b:{} and x:{}",
                  a.shape(), b.shape(), x.shape());

        const dim_t m = a.shape()[2];
        const dim_t n = a.shape()[3];
        const dim_t mn_max = std::max({dim_t{1}, m, n});
        const dim_t nrhs = b.shape()[3];

        // Check layout of a:
        const bool is_row_major = indexing::isRowMajor(a.strides());
        const dim_t lda = a.strides()[2 + !is_row_major];
        NOA_CHECK(is_row_major ?
                  (lda >= n && a.strides()[3] == 1) :
                  (lda >= m && a.strides()[2] == 1),
                  "The matrix a should be contiguous in its innermost dimension, and contiguous or padded in "
                  "its second-most dimension, but got stride {}", a.strides());

        // Check b and x:
        Array<T> tmp_b;
        if (x.get() == b.get()) {
            // No need to preserve b since it is equal to the solution matrix. Just make sure it can fit the
            // solution matrix and has same layout as a. Allow x to have the same shape as b, so that the user
            // doesn't have to extract the NxK b just for this call.
            NOA_CHECK(indexing::isRowMajor(b.strides()) == is_row_major,
                      "The matrix b should have the same layout as the matrix a");
            NOA_CHECK(b.shape()[2] == mn_max,
                      "Given the {}-by-{} matrix a, the number of rows in the matrix b should be {} or {} but got {}",
                      m, n, m, n, b.shape()[2]);
            const dim_t ldb = b.strides()[2 + !is_row_major];
            NOA_CHECK(is_row_major ?
                      (ldb >= nrhs && b.strides()[3] == 1) :
                      (ldb >= mn_max && b.strides()[2] == 1),
                      "The matrix b should be contiguous in its innermost dimension, and contiguous or padded in "
                      "its second-most dimension, but got stride {}", b.strides());

            tmp_b = b;
        } else {
            NOA_CHECK(b.shape()[2] == m,
                      "Given the {}-by-{} matrix a, the number of rows in the matrix b should be {} but got {}",
                      m, n, m, b.shape()[2]);

            // The implementation overwrites b with the solution matrix, so we need create a new array large
            // enough to fit b and x, and copy b inside this matrix.
            tmp_b = Array<T>(dim4_t{a.shape()[0], 1, mn_max, nrhs}, a.options());
            if (!is_row_major) { // FIXME Add factory function able to create column-major array
                dim4_t column_major_stride = tmp_b.strides();
                std::swap(column_major_stride[2], column_major_stride[3]);
                tmp_b = Array<T>(tmp_b.share(), tmp_b.shape(), column_major_stride, tmp_b.options());
            }
            using namespace indexing;
            memory::copy(b, tmp_b.subregion(ellipsis_t{}, slice_t{0, m}, full_extent_t{}));
        }

        // Check x:
        NOA_CHECK(x.shape()[2] == n && x.shape()[3] == nrhs,
                  "Given the {}-by-{} matrix a and {}-by-{} matrix b, the solution matrix x "
                  "should be a {}-by-{} matrix but got a {}-by-{} matrix",
                  m, n, m, nrhs, n, nrhs, x.shape()[2], x.shape()[3]);
        NOA_CHECK(indexing::isRowMajor(x.strides()) == is_row_major,
                  "The matrix x should have the same layout as the matrix a");

        if (!svd.empty()) {
            const dim_t mn_min = std::min(m, n);
            NOA_CHECK(a.shape()[0] == svd.shape()[0],
                      "The number of batches does not match, got a:{} and svd:{}",
                      a.shape()[0], svd.shape()[0]);
            NOA_CHECK(indexing::isVector(svd.shape(), true) && svd.contiguous(),
                      "The output singular values should be a contiguous (batched) vector, "
                      "but got shape:{} and stride:{}", svd.shape(), svd.strides());
            NOA_CHECK(dim3_t{svd.shape().get(1)}.elements() == mn_min,
                      "Given the {}-by-{} matrix a, the output singular values should have a size of {}, but got {}",
                      m, n, mn_min, dim3_t{svd.shape().get(1)}.elements());
        }

        const Device device = a.device();
        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(device == tmp_b.device() && (svd.empty() || svd.device() == device),
                      "The input and output arrays should be on the same device");
            cpu::math::lstsq(a.share(), a.strides(), a.shape(),
                             tmp_b.share(), tmp_b.strides(), tmp_b.shape(),
                             cond, svd.share(), stream.cpu());
        } else {
            NOA_THROW("math::lstsq() is currently not supported on the GPU");
        }

        if (x.get() != b.get()) {
            // Copy the solution matrix in x:
            using namespace indexing;
            memory::copy(tmp_b.subregion(ellipsis_t{}, slice_t{0, n}, full_extent_t{}), x);
        }
    }

    template<typename T, typename>
    void surface(const Array<T>& input, int order,
                 const Array<T>& output, bool subtract,
                 const Array<T>& parameters) {
        NOA_CHECK(!input.empty(), "Empty array detected");
        const Device device = input.device();

        if (!output.empty()) {
            NOA_CHECK(!subtract || all(input.shape() == output.shape()),
                      "The surface is about to be subtracted to the input. In this case, the input and output arrays "
                      "should have the same shape. Got input:{} and output:{}",
                      input.shape(), output.shape());
            NOA_CHECK(device == output.device(),
                      "The input and output arrays should be on the same device. Got input:{} and output:{}",
                      device, output.device());
        }

        if (!parameters.empty()) {
            const dim_t batches = input.shape()[0];
            const dim_t expected_parameters_size = (order == 3 ? 10 : static_cast<dim_t>(order * 3)) * batches;
            NOA_CHECK(parameters.device() == device,
                      "The input and parameter arrays should be on the same device. Got input:{} and parameter:{}",
                      device, parameters.device());
            NOA_CHECK(indexing::isVector(parameters.shape(), true) &&
                      parameters.contiguous() &&
                      parameters.elements() == expected_parameters_size,
                      "The output parameter array, specified as a contiguous vector, doesn't match the expected"
                      "number of elements. Got shape:{}, expected number of elements: {}",
                      parameters.shape(), expected_parameters_size);
        }

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::math::surface(input.share(), input.strides(), input.shape(),
                               output.share(), output.strides(), output.shape(),
                               subtract, order, parameters.share(), stream.cpu());
        } else {
            NOA_THROW("math::surface() is currently not supported on the GPU");
        }
    }
}
