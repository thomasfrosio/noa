#pragma once

#include "noa/base/Complex.hpp"
#include "noa/runtime/core/Shape.hpp"

#include "noa/fft/core/Transform.hpp"

namespace noa::fft::cpu {
    /// Returns the optimum even size, greater or equal than "size".
    /// An optimum size is an even integer satisfying (2^a)*(3^b)*(5^c)*(7^d)*(11^e)*(13^f), with e + f = 0 or 1.
    /// If "size" is >16896, this function will simply return the next even number and will not necessarily
    /// satisfy the aforementioned requirements.
    auto fast_size(isize size) -> isize;

    /// Returns the optimum BDHW logical shape.
    template<typename T, usize N>
    auto fast_shape(Shape<T, N> shape) noexcept -> Shape<T, N> {
        for (usize i = 1; i < N; ++i) // ignore batch dimension
            if (shape[i] > 1)
                shape[i] = static_cast<T>(fast_size(static_cast<isize>(shape[i])));
        return shape;
    }

    /// Wrapper for FFTW flags.
    enum Flag : u32 {
        /// -- Planning-rigor field -- ///

        /// Instead of actual measurements of different algorithms, a simple heuristic is used to pick a
        /// (probably suboptimal) plan quickly. With this flag, the input/output arrays are not overwritten
        /// during planning. This is often enough.
        ESTIMATE = 1u << 6,

        /// Find and optimized plan by actually computing several FFTs and measuring their execution time.
        /// Depending on your machine, this can take some time (often a few seconds).
        MEASURE = 0u,

        /// Same as \a MEASURE, but considers a wider range of algorithms and often produces a "more optimal"
        /// plan (especially for large transforms), but at the expense of several times longer planning time
        /// (especially for large transforms).
        PATIENT = 1u << 5,

        /// Same as \a PATIENT, but considers an even wider range of algorithms, including many that we think
        /// are unlikely to be fast, to produce the most optimal plan but with a substantially increased planning time.
        EXHAUSTIVE = 1u << 3,

        /// -- Algorithm-restriction field -- ///

        /// Specifies that an out-of-place transform is allowed to overwrite its input array with arbitrary data;
        /// this can sometimes allow more efficient algorithms to be employed.
        DESTROY_INPUT = 1u << 0,

        /// Specifies that an out-of-place transform must not change its input array. This is ordinarily the default,
        /// except for C2R transforms for which DESTROY_INPUT is the default. In the latter cases, passing this
        /// flag will attempt to use algorithms that do not destroy the input, at the expense of worse performance;
        /// for multidimensional C2R transforms, however, no input-preserving algorithms are implemented and the
        /// fft::Plan will throw an exception.
        PRESERVE_INPUT = 1u << 4,
    };

    /// Wrapper managing FFTW plans.
    /// NOTE: This object does not keep track of the associated data.
    ///       It is the user's responsibility to create, delete and keep track of the input/output arrays.
    /// NOTE: For R2C/C2R transforms, the 2D/3D arrays should be in the rightmost order for best performance since the
    ///       library currently always assumes the non-redundant dimension is the rows, i.e. the rightmost dimension.
    ///       The current exception is for column vectors, which are explicitly detected and supported.
    /// NOTE: In-place R2C/C2R transforms are allowed (input == output). In this case, the input requires extra
    ///       padding: the rows should have an extra float if the dimension is odd, or two extra floats if it is even.
    ///       If strides are provided, this padding should be reflected in the strides.
    /// NOTE: The FFTW planner is intended to be called from a single thread. Thus, even if the constructors of this
    ///       class are thread safe, understand that you may be holding for a long time and other threads might have
    ///       to wait, which might be undesirable.
    /// NOTE: For C2C, column-major is also supported.
    ///       If strides are not provided, arrays should be C-contiguous.
    ///       Any of the FFT flags is accepted.
    template<typename T>
    class Plan {
    public:
        static_assert(nt::any_of<T, f32, f64>);
        using real_type = T;
        using complex_type = Complex<T>;

    public:
        Plan(real_type* input, const Strides4& input_strides,
             complex_type* output, const Strides4& output_strides,
             const Shape4& shape, u32 flags, isize max_n_threads);

        Plan(complex_type* input, const Strides4& input_strides,
             real_type* output, const Strides4& output_strides,
             const Shape4& shape, u32 flags, isize max_n_threads);

        Plan(complex_type* input, const Strides4& input_strides,
             complex_type* output, const Strides4& output_strides,
             const Shape4& shape, nf::Sign sign, u32 flags, isize max_n_threads);

    public:
        Plan(const Plan&) = delete;
        Plan& operator=(const Plan&) = delete;
        Plan(Plan&&) noexcept = default;
        Plan& operator=(Plan&&) noexcept = default;
        ~Plan() noexcept;

        void execute() noexcept;

        static void clear_cache() noexcept;
        static auto cache_size() noexcept -> i32;

    private:
        void* m_plan{};
    };

    inline auto clear_cache() noexcept -> i32 {
        Plan<f32>::clear_cache();
        Plan<f64>::clear_cache();
        return 0;
    }

    inline auto cache_size() noexcept -> i32 {
        i32 size = Plan<f32>::cache_size();
        size += Plan<f64>::cache_size();
        return size;
    }
}
