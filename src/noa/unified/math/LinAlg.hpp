#pragma once

#include "noa/cpu/math/LinAlg.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/memory/Copy.hpp"

namespace noa::math::details {
    template<typename T, typename U>
    constexpr bool is_valid_lstsq_t =
            noa::traits::is_any_v<noa::traits::value_type_t<T>, f32, f64, c32, c64> &&
            std::is_same_v<noa::traits::value_type_twice_t<T>, noa::traits::value_type_t<U>>;
}

namespace noa::math {
    /// Computes least-squares solution to equation Ax = b.
    /// \details Computes a vector x such that the 2-norm |b - A x| is minimized. Several right hand side vectors b
    ///          and solution vectors x can be handled in a single call, in which case they are stored as the columns
    ///          of the matrix b. Furthermore, the entire problem can be batched.
    ///
    /// \tparam A, B, X         Array or View of f32, f64, c32 or c64.
    /// \tparam U               Array or View of f32, f64 (should match the precision of A, B and X).
    /// \param[in,out] a        Dense M-by-N matrix, where M >= N. It is overwritten.
    /// \param[in,out] b        Dense M-by-K matrix, where K can be 1. It is overwritten.
    /// \param[out] x           Dense N-by-K solution matrix. Can point to the same memory as \p b.
    /// \param cond             Used to determine effective rank of \p a. Cutoff for "small" singular values.
    ///                         Singular values smaller than \p cond * largest_singular_value are considered zero.
    ///                         If <= 0, machine precision is used instead.
    /// \param[out] svd         The min(M,N) singular values of \p a in decreasing order.
    ///                         If empty, a complete orthogonal factorization of \p a is used to solve the problem,
    ///                         which can be slightly faster on many problems but the SVD method is more stable.
    ///
    /// \note In most cases, it is more efficient if \p a, \p b and \p x (if K > 1) are column major and contiguous.
    ///       In any case, the innermost dimension should be contiguous and the second-most dimension can either be
    ///       contiguous or padded.
    /// \note This function is currently not supported on the GPU.
    template<typename MatrixA, typename MatrixB, typename MatrixX, typename Real = f32,
             typename MatrixU = View<noa::traits::value_type_twice_t<MatrixA>>,
             typename = std::enable_if_t<
                     noa::traits::is_real_v<Real> &&
                     noa::traits::are_array_or_view_v<MatrixA, MatrixB, MatrixX, MatrixU> &&
                     noa::traits::are_same_value_type_v<MatrixA, MatrixB, MatrixX> &&
                     !std::is_const_v<noa::traits::value_type_t<MatrixA>> &&
                     details::is_valid_lstsq_t<MatrixA, MatrixU>>>
    void lstsq(const MatrixA& a, const MatrixB& b, const MatrixX& x,
               Real cond = 0, const MatrixU& svd = {}) {
        NOA_CHECK(!a.is_empty() && !b.is_empty() && !x.is_empty(), "Empty array detected");

        // Check batches and 2D matrices:
        NOA_CHECK(a.shape()[0] == b.shape()[0] && a.shape()[0] == x.shape()[0],
                  "The number of batches does not match, got a:{}, b:{} and x:{}",
                  a.shape()[0], b.shape()[0], x.shape()[0]);
        NOA_CHECK(a.shape()[1] == 1 && b.shape()[1] == 1 && x.shape()[1] == 1,
                  "3D matrices are invalid, but got shape a:{}, b:{} and x:{}",
                  a.shape(), b.shape(), x.shape());

        const i64 m_samples = a.shape()[2];
        const i64 n_variables = a.shape()[3];
        const i64 k_solutions = b.shape()[3];

        // Check a:
        const bool is_row_major = noa::indexing::is_row_major(a.strides());
        const i64 lda = a.strides()[2 + !is_row_major];
        NOA_CHECK(is_row_major ?
                  (lda >= n_variables && a.strides()[3] == 1) :
                  (lda >= m_samples && a.strides()[2] == 1),
                  "The MxN matrix A should be contiguous in its innermost dimension, "
                  "and contiguous or padded in its second-most dimension, but got stride {}",
                  a.strides());
        NOA_CHECK(m_samples >= n_variables,
                  "The API is currently limited to MxN matrix A, where M >= N. Got {}x{} instead",
                  a.shape()[2], a.shape()[3]);

        // Check b:
        NOA_CHECK(noa::indexing::is_row_major(b.strides()) == is_row_major,
                  "The matrix B should have the same layout as the matrix A");
        const i64 ldb = b.strides()[2 + !is_row_major];
        NOA_CHECK(is_row_major ?
                  (ldb >= k_solutions && b.strides()[3] == 1) :
                  (ldb >= m_samples && b.strides()[2] == 1),
                  "The matrix B should be contiguous in its innermost dimension, "
                  "and contiguous or padded in its second-most dimension, but got stride {}",
                  b.strides());
        NOA_CHECK(b.shape()[2] == m_samples,
                  "Given the {}x{} matrix A, the number of rows in the matrix B should be {} but got {}",
                  m_samples, n_variables, m_samples, b.shape()[2]);

        // Check x:
        NOA_CHECK(x.shape()[2] == n_variables && x.shape()[3] == k_solutions,
                  "Given the {}x{} matrix A and {}x{} matrix B, the solution matrix X "
                  "should be a {}x{} matrix but got a {}x{} matrix",
                  m_samples, n_variables, m_samples, k_solutions, n_variables, k_solutions,
                  x.shape()[2], x.shape()[3]);
        NOA_CHECK(noa::indexing::is_row_major(x.strides()) == is_row_major,
                  "The matrix x should have the same layout as the matrix a");

        // Check svd:
        if (!svd.is_empty()) {
            const i64 mn_min = std::min(m_samples, n_variables);
            NOA_CHECK(a.shape()[0] == svd.shape()[0],
                      "The number of batches does not match, got {} batches in A and {} batches in SVD",
                      a.shape()[0], svd.shape()[0]);
            NOA_CHECK(noa::indexing::is_contiguous_vector_batched(svd),
                      "The output singular values should be a contiguous (batched) vector, "
                      "but got shape:{} and stride:{}", svd.shape(), svd.strides());
            NOA_CHECK(svd.shape().pop_front().elements() == mn_min,
                      "Given the {}x{} matrix A, the output SVD array should have a size of {}, but got {}",
                      m_samples, n_variables, mn_min, svd.shape().pop_front().elements());
        }

        const Device device = a.device();
        NOA_CHECK(device == b.device() && (svd.is_empty() || svd.device() == device),
                  "The input and output arrays should be on the same device");

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            stream.cpu().enqueue([=](){
                cpu::math::lstsq(a.get(), a.strides(), a.shape(),
                                 b.get(), b.strides(), b.shape(),
                                 static_cast<f32>(cond), svd.get());
            });
        } else {
            NOA_THROW("noa::math::lstsq() is currently not supported on the GPU");
        }

        if (x.get() != b.get()) {
            // Copy the NxK solution matrix in x:
            auto b_output = b.subregion(
                    noa::indexing::Ellipsis{},
                    noa::indexing::Slice{0, n_variables},
                    noa::indexing::FullExtent{});
            b_output.to(x);
        }
    }
}
