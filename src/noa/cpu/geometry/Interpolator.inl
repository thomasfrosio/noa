#ifndef NOA_CPU_INTERPOLATOR_
#error "This is an internal header. Include the corresponding .h file instead"
#endif

// Implementation:
namespace noa::cpu::geometry {
    template<typename T, AccessorTraits TRAITS>
    template<typename U>
    Interpolator1D<T, TRAITS>::Interpolator1D(T* input, dim_t strides, dim_t size, U value) noexcept
            : m_data(input, &strides), m_size(static_cast<index_t>(size)), m_value(static_cast<mutable_value_t>(value)) {}

    template<typename T, AccessorTraits TRAITS>
    template<typename I0, typename T1>
    Interpolator1D<T, TRAITS>::Interpolator1D(const Accessor<T, 1, I0, TRAITS>& input, dim_t size, T1 value) noexcept
            : m_data(input.get(), input.strides()),
              m_size(static_cast<index_t>(size)),
              m_value(static_cast<mutable_value_t>(value)) {}

    template<typename T, AccessorTraits TRAITS>
    template<typename I0, typename T1>
    Interpolator1D<T, TRAITS>::Interpolator1D(const AccessorReference<T, 1, I0, TRAITS>& input, dim_t size, T1 value) noexcept
            : m_data(input.get(), input.strides()),
              m_size(static_cast<index_t>(size)),
              m_value(static_cast<mutable_value_t>(value)) {}

    template<typename T, AccessorTraits TRAITS>
    template<BorderMode BORDER>
    auto Interpolator1D<T, TRAITS>::nearest_(accessor_reference_t data, float x) const {
        mutable_value_t out;
        const auto idx = static_cast<index_t>(noa::math::round(x));
        if constexpr (BORDER == BORDER_ZERO) {
            out = idx >= 0 && idx < m_size ? data[idx] : mutable_value_t{0};
        } else if constexpr (BORDER == BORDER_VALUE) {
            out = idx >= 0 && idx < m_size ? data[idx] : static_cast<mutable_value_t>(m_value);
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            out = data[indexing::at<BORDER>(idx, m_size)];
        } else {
            static_assert(traits::always_false_v<T>);
        }
        return out;
    }

    template<typename T, AccessorTraits TRAITS>
    template<BorderMode BORDER, bool COSINE>
    auto Interpolator1D<T, TRAITS>::linear_(accessor_reference_t data, float x) const {
        const auto idx0 = static_cast<index_t>(noa::math::floor(x));
        const index_t idx1 = idx0 + 1;
        mutable_value_t values[2];
        if constexpr (BORDER == BORDER_ZERO || BORDER == BORDER_VALUE) {
            const bool cond[2] = {idx0 >= 0 && idx0 < m_size, idx1 >= 0 && idx1 < m_size};
            if constexpr (BORDER == BORDER_ZERO) {
                values[0] = cond[0] ? data[idx0] : mutable_value_t{0};
                values[1] = cond[1] ? data[idx1] : mutable_value_t{0};
            } else {
                values[0] = cond[0] ? data[idx0] : static_cast<mutable_value_t>(m_value);
                values[1] = cond[1] ? data[idx1] : static_cast<mutable_value_t>(m_value);
            }
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            values[0] = data[indexing::at<BORDER>(idx0, m_size)];
            values[1] = data[indexing::at<BORDER>(idx1, m_size)];
        } else {
            static_assert(traits::always_false_v<T>);
        }
        const float fraction = x - static_cast<float>(idx0);
        if constexpr (COSINE)
            return cosine1D(values[0], values[1], fraction);
        else
            return linear1D(values[0], values[1], fraction);
    }

    template<typename T, AccessorTraits TRAITS>
    template<BorderMode BORDER, bool BSPLINE>
    auto Interpolator1D<T, TRAITS>::cubic_(accessor_reference_t data, float x) const {
        const auto idx1 = static_cast<index_t>(noa::math::floor(x));
        const index_t idx0 = idx1 - 1;
        const index_t idx2 = idx1 + 1;
        const index_t idx3 = idx1 + 2;
        mutable_value_t values[4];
        if constexpr (BORDER == BORDER_ZERO || BORDER == BORDER_VALUE) {
            const bool cond[4] = {idx0 >= 0 && idx0 < m_size,
                                  idx1 >= 0 && idx1 < m_size,
                                  idx2 >= 0 && idx2 < m_size,
                                  idx3 >= 0 && idx3 < m_size};
            if constexpr (BORDER == BORDER_ZERO) {
                values[0] = cond[0] ? data[idx0] : mutable_value_t{0};
                values[1] = cond[1] ? data[idx1] : mutable_value_t{0};
                values[2] = cond[2] ? data[idx2] : mutable_value_t{0};
                values[3] = cond[3] ? data[idx3] : mutable_value_t{0};
            } else {
                values[0] = cond[0] ? data[idx0] : static_cast<mutable_value_t>(m_value);
                values[1] = cond[1] ? data[idx1] : static_cast<mutable_value_t>(m_value);
                values[2] = cond[2] ? data[idx2] : static_cast<mutable_value_t>(m_value);
                values[3] = cond[3] ? data[idx3] : static_cast<mutable_value_t>(m_value);
            }
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            values[0] = data[indexing::at<BORDER>(idx0, m_size)];
            values[1] = data[indexing::at<BORDER>(idx1, m_size)];
            values[2] = data[indexing::at<BORDER>(idx2, m_size)];
            values[3] = data[indexing::at<BORDER>(idx3, m_size)];
        } else {
            static_assert(traits::always_false_v<T>);
        }
        const float fraction = x - static_cast<float>(idx1);
        if constexpr (BSPLINE)
            return cubicBSpline1D(values[0], values[1], values[2], values[3], fraction);
        else
            return cubic1D(values[0], values[1], values[2], values[3], fraction);
    }

    template<typename T, AccessorTraits TRAITS>
    template<InterpMode INTERP, BorderMode BORDER>
    auto Interpolator1D<T, TRAITS>::get(float x) const {
        return get<INTERP, BORDER>(x, 0);
    }

    template<typename T, AccessorTraits TRAITS>
    template<InterpMode INTERP, BorderMode BORDER>
    auto Interpolator1D<T, TRAITS>::get(float x, dim_t offset) const {
        const accessor_reference_t data(m_data.get() + offset, m_data.strides());
        if constexpr (INTERP == INTERP_NEAREST) {
            return nearest_<BORDER>(data, x);
        } else if constexpr (INTERP == INTERP_LINEAR || INTERP == INTERP_LINEAR_FAST) {
            return linear_<BORDER, false>(data, x);
        } else if constexpr (INTERP == INTERP_COSINE || INTERP == INTERP_COSINE_FAST) {
            return linear_<BORDER, true>(data, x);
        } else if constexpr (INTERP == INTERP_CUBIC) {
            return cubic_<BORDER, false>(data, x);
        } else if constexpr (INTERP == INTERP_CUBIC_BSPLINE || INTERP == INTERP_CUBIC_BSPLINE_FAST) {
            return cubic_<BORDER, true>(data, x);
        } else {
            static_assert(traits::always_false_v<T>);
        }
    }

    // -- 2D -- //
    template<typename T, AccessorTraits TRAITS>
    template<typename U>
    Interpolator2D<T, TRAITS>::Interpolator2D(T* input, dim2_t strides, dim2_t shape, U value) noexcept
            : m_data(input, strides.get()),
              m_shape(shape),
              m_value(static_cast<mutable_value_t>(value)) {}

    template<typename T, AccessorTraits TRAITS>
    template<typename I0, typename T1>
    Interpolator2D<T, TRAITS>::Interpolator2D(const Accessor<T, 2, I0, TRAITS>& input, dim2_t shape, T1 value) noexcept
            : m_data(input.get(), input.strides()),
              m_shape(shape),
              m_value(static_cast<mutable_value_t>(value)) {}

    template<typename T, AccessorTraits TRAITS>
    template<typename I0, typename T1>
    Interpolator2D<T, TRAITS>::Interpolator2D(const AccessorReference<T, 2, I0, TRAITS>& input, dim2_t shape, T1 value) noexcept
            : m_data(input.get(), input.strides()),
              m_shape(shape),
              m_value(static_cast<mutable_value_t>(value)) {}

    template<typename T, AccessorTraits TRAITS>
    template<BorderMode BORDER>
    auto Interpolator2D<T, TRAITS>::nearest_(accessor_reference_t data, float y, float x) const {
        mutable_value_t out;
        index2_t idx{noa::math::round(y), noa::math::round(x)};
        if constexpr (BORDER == BORDER_ZERO) {
            if (idx[1] < 0 || idx[1] >= m_shape[1] || idx[0] < 0 || idx[0] >= m_shape[0])
                out = mutable_value_t{0};
            else
                out = data(idx[0], idx[1]);
        } else if constexpr (BORDER == BORDER_VALUE) {
            if (idx[1] < 0 || idx[1] >= m_shape[1] || idx[0] < 0 || idx[0] >= m_shape[0])
                out = static_cast<mutable_value_t>(m_value);
            else
                out = data(idx[0], idx[1]);
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            idx[0] = indexing::at<BORDER>(idx[0], m_shape[0]);
            idx[1] = indexing::at<BORDER>(idx[1], m_shape[1]);
            out = data(idx[0], idx[1]);;
        } else {
            static_assert(traits::always_false_v<T>);
        }
        return out;
    }

    template<typename T, AccessorTraits TRAITS>
    template<BorderMode BORDER, bool COSINE>
    auto Interpolator2D<T, TRAITS>::linear_(accessor_reference_t data, float y, float x) const {
        const index2_t idx0{noa::math::floor(y), noa::math::floor(x)};
        const index2_t idx1(idx0 + 1);
        mutable_value_t values[4]; // v00, v10, v01, v11
        if constexpr (BORDER == BORDER_ZERO || BORDER == BORDER_VALUE) {
            const bool cond_y[2] = {idx0[0] >= 0 && idx0[0] < m_shape[0], idx1[0] >= 0 && idx1[0] < m_shape[0]};
            const bool cond_x[2] = {idx0[1] >= 0 && idx0[1] < m_shape[1], idx1[1] >= 0 && idx1[1] < m_shape[1]};
            if constexpr (BORDER == BORDER_ZERO) {
                constexpr mutable_value_t ZERO{0};
                values[0] = cond_y[0] && cond_x[0] ? data(idx0[0], idx0[1]) : ZERO; // v00
                values[1] = cond_y[0] && cond_x[1] ? data(idx0[0], idx1[1]) : ZERO; // v01
                values[2] = cond_y[1] && cond_x[0] ? data(idx1[0], idx0[1]) : ZERO; // v10
                values[3] = cond_y[1] && cond_x[1] ? data(idx1[0], idx1[1]) : ZERO; // v11
            } else {
                values[0] = cond_y[0] && cond_x[0] ? data(idx0[0], idx0[1]) : m_value;
                values[1] = cond_y[0] && cond_x[1] ? data(idx0[0], idx1[1]) : m_value;
                values[2] = cond_y[1] && cond_x[0] ? data(idx1[0], idx0[1]) : m_value;
                values[3] = cond_y[1] && cond_x[1] ? data(idx1[0], idx1[1]) : m_value;
            }
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            const index_t tmp[4] = {indexing::at<BORDER>(idx0[1], m_shape[1]),
                                    indexing::at<BORDER>(idx1[1], m_shape[1]),
                                    indexing::at<BORDER>(idx0[0], m_shape[0]),
                                    indexing::at<BORDER>(idx1[0], m_shape[0])};
            values[0] = data(tmp[2], tmp[0]); // v00
            values[1] = data(tmp[2], tmp[1]); // v01
            values[2] = data(tmp[3], tmp[0]); // v10
            values[3] = data(tmp[3], tmp[1]); // v11
        } else {
            static_assert(traits::always_false_v<T>);
        }
        float2_t fraction{x - static_cast<float>(idx0[1]), y - static_cast<float>(idx0[0])};
        if constexpr (COSINE)
            return cosine2D(values[0], values[1], values[2], values[3], fraction[0], fraction[1]);
        else
            return linear2D(values[0], values[1], values[2], values[3], fraction[0], fraction[1]);
    }

    template<typename T, AccessorTraits TRAITS>
    template<BorderMode BORDER, bool BSPLINE>
    auto Interpolator2D<T, TRAITS>::cubic_(accessor_reference_t data, float y, float x) const {
        const index2_t idx{noa::math::floor(y), noa::math::floor(x)};
        mutable_value_t square[4][4]; // [y][x]
        if constexpr (BORDER == BORDER_ZERO || BORDER == BORDER_VALUE) {
            const bool cond_y[4] = {idx[0] - 1 >= 0 && idx[0] - 1 < m_shape[0],
                                    idx[0] + 0 >= 0 && idx[0] + 0 < m_shape[0],
                                    idx[0] + 1 >= 0 && idx[0] + 1 < m_shape[0],
                                    idx[0] + 2 >= 0 && idx[0] + 2 < m_shape[0]};
            const bool cond_x[4] = {idx[1] - 1 >= 0 && idx[1] - 1 < m_shape[1],
                                    idx[1] + 0 >= 0 && idx[1] + 0 < m_shape[1],
                                    idx[1] + 1 >= 0 && idx[1] + 1 < m_shape[1],
                                    idx[1] + 2 >= 0 && idx[1] + 2 < m_shape[1]};
            constexpr index_t offset[4] = {-1, 0, 1, 2};
            for (index_t j = 0; j < 4; ++j) {
                const index_t idx_y = idx[0] + offset[j];
                for (index_t i = 0; i < 4; ++i) {
                    const index_t idx_x = idx[1] + offset[i];
                    if constexpr (BORDER == BORDER_ZERO)
                        square[j][i] = cond_x[i] && cond_y[j] ? data(idx_y, idx_x) : mutable_value_t{0};
                    else
                        square[j][i] = cond_x[i] && cond_y[j] ? data(idx_y, idx_x) : m_value;
                }
            }
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            const index_t tmp_y[4] = {indexing::at<BORDER>(idx[0] - 1, m_shape[0]),
                                      indexing::at<BORDER>(idx[0] + 0, m_shape[0]),
                                      indexing::at<BORDER>(idx[0] + 1, m_shape[0]),
                                      indexing::at<BORDER>(idx[0] + 2, m_shape[0])};
            const index_t tmp_x[4] = {indexing::at<BORDER>(idx[1] - 1, m_shape[1]),
                                      indexing::at<BORDER>(idx[1] + 0, m_shape[1]),
                                      indexing::at<BORDER>(idx[1] + 1, m_shape[1]),
                                      indexing::at<BORDER>(idx[1] + 2, m_shape[1])};
            for (index_t j = 0; j < 4; ++j)
                for (index_t i = 0; i < 4; ++i)
                    square[j][i] = data(tmp_y[j], tmp_x[i]);

        } else {
            static_assert(traits::always_false_v<T>);
        }
        const float2_t fraction{x - static_cast<float>(idx[1]), y - static_cast<float>(idx[0])};
        if constexpr (BSPLINE)
            return cubicBSpline2D(square, fraction[0], fraction[1]);
        else
            return cubic2D(square, fraction[0], fraction[1]);
    }

    template<typename T, AccessorTraits TRAITS>
    template<InterpMode INTERP, BorderMode BORDER>
    auto Interpolator2D<T, TRAITS>::get(float2_t coords) const {
        return get<INTERP, BORDER>(coords, 0);
    }

    template<typename T, AccessorTraits TRAITS>
    template<InterpMode INTERP, BorderMode BORDER>
    auto Interpolator2D<T, TRAITS>::get(float2_t coords, dim_t offset) const {
        const accessor_reference_t data(m_data.get() + offset, m_data.strides());
        if constexpr (INTERP == INTERP_NEAREST) {
            return nearest_<BORDER>(data, coords[0], coords[1]);
        } else if constexpr (INTERP == INTERP_LINEAR || INTERP == INTERP_LINEAR_FAST) {
            return linear_<BORDER, false>(data, coords[0], coords[1]);
        } else if constexpr (INTERP == INTERP_COSINE || INTERP == INTERP_COSINE_FAST) {
            return linear_<BORDER, true>(data, coords[0], coords[1]);
        } else if constexpr (INTERP == INTERP_CUBIC) {
            return cubic_<BORDER, false>(data, coords[0], coords[1]);
        } else if constexpr (INTERP == INTERP_CUBIC_BSPLINE || INTERP == INTERP_CUBIC_BSPLINE_FAST) {
            return cubic_<BORDER, true>(data, coords[0], coords[1]);
        } else {
            static_assert(traits::always_false_v<T>);
        }
    }

    // -- 3D -- //
    template<typename T, AccessorTraits TRAITS>
    template<typename U>
    Interpolator3D<T, TRAITS>::Interpolator3D(T* input, dim3_t strides, dim3_t shape, U value) noexcept
            : m_data(input, strides.get()),
              m_shape(shape),
              m_value(static_cast<mutable_value_t>(value)) {}

    template<typename T, AccessorTraits TRAITS>
    template<typename I0, typename T1>
    Interpolator3D<T, TRAITS>::Interpolator3D(const Accessor<T, 3, I0, TRAITS>& input, dim3_t shape, T1 value) noexcept
            : m_data(input.get(), input.strides()),
              m_shape(shape),
              m_value(static_cast<mutable_value_t>(value)) {}

    template<typename T, AccessorTraits TRAITS>
    template<typename I0, typename T1>
    Interpolator3D<T, TRAITS>::Interpolator3D(const AccessorReference<T, 3, I0, TRAITS>& input, dim3_t shape, T1 value) noexcept
            : m_data(input.get(), input.strides()),
              m_shape(shape),
              m_value(static_cast<mutable_value_t>(value)) {}

    template<typename T, AccessorTraits TRAITS>
    template<BorderMode BORDER>
    auto Interpolator3D<T, TRAITS>::nearest_(accessor_reference_t data, float z, float y, float x) const {
        mutable_value_t out;
        index3_t idx{noa::math::round(z), noa::math::round(y), noa::math::round(x)};
        if constexpr (BORDER == BORDER_ZERO) {
            if (idx[2] < 0 || idx[2] >= m_shape[2] ||
                idx[1] < 0 || idx[1] >= m_shape[1] ||
                idx[0] < 0 || idx[0] >= m_shape[0])
                out = mutable_value_t{0};
            else
                out = data(idx[0], idx[1], idx[2]);
        } else if constexpr (BORDER == BORDER_VALUE) {
            if (idx[2] < 0 || idx[2] >= m_shape[2] ||
                idx[1] < 0 || idx[1] >= m_shape[1] ||
                idx[0] < 0 || idx[0] >= m_shape[0])
                out = static_cast<mutable_value_t>(m_value);
            else
                out = data(idx[0], idx[1], idx[2]);
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            idx[2] = indexing::at<BORDER>(idx[2], m_shape[2]);
            idx[1] = indexing::at<BORDER>(idx[1], m_shape[1]);
            idx[0] = indexing::at<BORDER>(idx[0], m_shape[0]);
            out = data(idx[0], idx[1], idx[2]);
        } else {
            static_assert(traits::always_false_v<T>);
        }
        return out;
    }

    template<typename T, AccessorTraits TRAITS>
    template<BorderMode BORDER, bool COSINE>
    auto Interpolator3D<T, TRAITS>::linear_(accessor_reference_t data, float z, float y, float x) const {
        index3_t idx[2];
        idx[0] = index3_t{noa::math::floor(z), noa::math::floor(y), noa::math::floor(x)};
        idx[1] = idx[0] + 1;

        mutable_value_t values[8];
        if constexpr (BORDER == BORDER_ZERO || BORDER == BORDER_VALUE) {
            const bool cond_z[2] = {idx[0][0] >= 0 && idx[0][0] < m_shape[0], idx[1][0] >= 0 && idx[1][0] < m_shape[0]};
            const bool cond_y[2] = {idx[0][1] >= 0 && idx[0][1] < m_shape[1], idx[1][1] >= 0 && idx[1][1] < m_shape[1]};
            const bool cond_x[2] = {idx[0][2] >= 0 && idx[0][2] < m_shape[2], idx[1][2] >= 0 && idx[1][2] < m_shape[2]};

            mutable_value_t cval;
            if constexpr (BORDER == BORDER_ZERO)
                cval = mutable_value_t{0};
            else
                cval = m_value;
            const index_t off_z[2] = {idx[0][0] * data.stride(0), idx[1][0] * data.stride(0)};
            const index_t off_y[2] = {idx[0][1] * data.stride(1), idx[1][1] * data.stride(1)};
            const index_t off_x[2] = {idx[0][2] * data.stride(2), idx[1][2] * data.stride(2)};
            values[0] = cond_z[0] && cond_y[0] && cond_x[0] ? data.get()[off_z[0] + off_y[0] + off_x[0]] : cval; // v000
            values[1] = cond_z[0] && cond_y[0] && cond_x[1] ? data.get()[off_z[0] + off_y[0] + off_x[1]] : cval; // v001
            values[2] = cond_z[0] && cond_y[1] && cond_x[0] ? data.get()[off_z[0] + off_y[1] + off_x[0]] : cval; // v010
            values[3] = cond_z[0] && cond_y[1] && cond_x[1] ? data.get()[off_z[0] + off_y[1] + off_x[1]] : cval; // v011
            values[4] = cond_z[1] && cond_y[0] && cond_x[0] ? data.get()[off_z[1] + off_y[0] + off_x[0]] : cval; // v100
            values[5] = cond_z[1] && cond_y[0] && cond_x[1] ? data.get()[off_z[1] + off_y[0] + off_x[1]] : cval; // v101
            values[6] = cond_z[1] && cond_y[1] && cond_x[0] ? data.get()[off_z[1] + off_y[1] + off_x[0]] : cval; // v110
            values[7] = cond_z[1] && cond_y[1] && cond_x[1] ? data.get()[off_z[1] + off_y[1] + off_x[1]] : cval; // v111

        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            const index_t tmp[6] = {indexing::at<BORDER>(idx[0][2], m_shape[2]),
                                    indexing::at<BORDER>(idx[1][2], m_shape[2]),
                                    indexing::at<BORDER>(idx[0][1], m_shape[1]),
                                    indexing::at<BORDER>(idx[1][1], m_shape[1]),
                                    indexing::at<BORDER>(idx[0][0], m_shape[0]),
                                    indexing::at<BORDER>(idx[1][0], m_shape[0])};
            values[0] = data(tmp[4], tmp[2], tmp[0]); // v000
            values[1] = data(tmp[4], tmp[2], tmp[1]); // v001
            values[2] = data(tmp[4], tmp[3], tmp[0]); // v010
            values[3] = data(tmp[4], tmp[3], tmp[1]); // v011
            values[4] = data(tmp[5], tmp[2], tmp[0]); // v100
            values[5] = data(tmp[5], tmp[2], tmp[1]); // v101
            values[6] = data(tmp[5], tmp[3], tmp[0]); // v110
            values[7] = data(tmp[5], tmp[3], tmp[1]); // v111
        } else {
            static_assert(traits::always_false_v<T>);
        }
        const float3_t fraction{x - static_cast<float>(idx[0][2]),
                                y - static_cast<float>(idx[0][1]),
                                z - static_cast<float>(idx[0][0])};
        if constexpr (COSINE)
            return cosine3D(values[0], values[1], values[2], values[3],
                            values[4], values[5], values[6], values[7],
                            fraction[0], fraction[1], fraction[2]);
        else
            return linear3D(values[0], values[1], values[2], values[3],
                            values[4], values[5], values[6], values[7],
                            fraction[0], fraction[1], fraction[2]);
    }

    template<typename T, AccessorTraits TRAITS>
    template<BorderMode BORDER, bool BSPLINE>
    auto Interpolator3D<T, TRAITS>::cubic_(accessor_reference_t data, float z, float y, float x) const {
        index3_t idx{noa::math::floor(z), noa::math::floor(y), noa::math::floor(x)};
        mutable_value_t values[4][4][4]; // [z][y][x]
        if constexpr (BORDER == BORDER_ZERO || BORDER == BORDER_VALUE) {
            const bool cond_z[4] = {idx[0] - 1 >= 0 && idx[0] - 1 < m_shape[0],
                                    idx[0] + 0 >= 0 && idx[0] + 0 < m_shape[0],
                                    idx[0] + 1 >= 0 && idx[0] + 1 < m_shape[0],
                                    idx[0] + 2 >= 0 && idx[0] + 2 < m_shape[0]};
            const bool cond_y[4] = {idx[1] - 1 >= 0 && idx[1] - 1 < m_shape[1],
                                    idx[1] + 0 >= 0 && idx[1] + 0 < m_shape[1],
                                    idx[1] + 1 >= 0 && idx[1] + 1 < m_shape[1],
                                    idx[1] + 2 >= 0 && idx[1] + 2 < m_shape[1]};
            const bool cond_x[4] = {idx[2] - 1 >= 0 && idx[2] - 1 < m_shape[2],
                                    idx[2] + 0 >= 0 && idx[2] + 0 < m_shape[2],
                                    idx[2] + 1 >= 0 && idx[2] + 1 < m_shape[2],
                                    idx[2] + 2 >= 0 && idx[2] + 2 < m_shape[2]};
            mutable_value_t cval;
            if constexpr (BORDER == BORDER_ZERO)
                cval = mutable_value_t{0};
            else
                cval = m_value;
            constexpr index_t offset[4] = {-1, 0, 1, 2};
            for (index_t i = 0; i < 4; ++i) {
                const index_t idx_z = idx[0] + offset[i];
                for (index_t j = 0; j < 4; ++j) {
                    const index_t idx_y = idx[1] + offset[j];
                    for (index_t k = 0; k < 4; ++k) {
                        values[i][j][k] = cond_z[i] && cond_y[j] && cond_x[k] ?
                                          data(idx_z, idx_y, idx[2] + offset[k]) : cval;
                    }
                }
            }
        } else if constexpr (BORDER == BORDER_CLAMP || BORDER == BORDER_PERIODIC ||
                             BORDER == BORDER_MIRROR || BORDER == BORDER_REFLECT) {
            const index_t tmp_z[4] = {indexing::at<BORDER>(idx[0] - 1, m_shape[0]),
                                      indexing::at<BORDER>(idx[0] + 0, m_shape[0]),
                                      indexing::at<BORDER>(idx[0] + 1, m_shape[0]),
                                      indexing::at<BORDER>(idx[0] + 2, m_shape[0])};
            const index_t tmp_y[4] = {indexing::at<BORDER>(idx[1] - 1, m_shape[1]),
                                      indexing::at<BORDER>(idx[1] + 0, m_shape[1]),
                                      indexing::at<BORDER>(idx[1] + 1, m_shape[1]),
                                      indexing::at<BORDER>(idx[1] + 2, m_shape[1])};
            const index_t tmp_x[4] = {indexing::at<BORDER>(idx[2] - 1, m_shape[2]),
                                      indexing::at<BORDER>(idx[2] + 0, m_shape[2]),
                                      indexing::at<BORDER>(idx[2] + 1, m_shape[2]),
                                      indexing::at<BORDER>(idx[2] + 2, m_shape[2])};
            for (index_t i = 0; i < 4; ++i)
                for (index_t j = 0; j < 4; ++j)
                    for (index_t k = 0; k < 4; ++k)
                        values[i][j][k] = data(tmp_z[i], tmp_y[j], tmp_x[k]);

        } else {
            static_assert(traits::always_false_v<T>);
        }
        const float3_t fraction{x - static_cast<float>(idx[2]),
                                y - static_cast<float>(idx[1]),
                                z - static_cast<float>(idx[0])};
        if constexpr (BSPLINE)
            return cubicBSpline3D(values, fraction[0], fraction[1], fraction[2]);
        else
            return cubic3D(values, fraction[0], fraction[1], fraction[2]);
    }

    template<typename T, AccessorTraits TRAITS>
    template<InterpMode INTERP, BorderMode BORDER>
    auto Interpolator3D<T, TRAITS>::get(float3_t coords) const {
        return get<INTERP, BORDER>(coords, 0);
    }

    template<typename T, AccessorTraits TRAITS>
    template<InterpMode INTERP, BorderMode BORDER>
    auto Interpolator3D<T, TRAITS>::get(float3_t coords, dim_t offset) const {
        const accessor_reference_t data(m_data.get() + offset, m_data.strides());
        if constexpr (INTERP == INTERP_NEAREST) {
            return nearest_<BORDER>(data, coords[0], coords[1], coords[2]);
        } else if constexpr (INTERP == INTERP_LINEAR || INTERP == INTERP_LINEAR_FAST) {
            return linear_<BORDER, false>(data, coords[0], coords[1], coords[2]);
        } else if constexpr (INTERP == INTERP_COSINE || INTERP == INTERP_COSINE_FAST) {
            return linear_<BORDER, true>(data, coords[0], coords[1], coords[2]);
        } else if constexpr (INTERP == INTERP_CUBIC) {
            return cubic_<BORDER, false>(data, coords[0], coords[1], coords[2]);
        } else if constexpr (INTERP == INTERP_CUBIC_BSPLINE || INTERP == INTERP_CUBIC_BSPLINE_FAST) {
            return cubic_<BORDER, true>(data, coords[0], coords[1], coords[2]);
        } else {
            static_assert(traits::always_false_v<T>);
        }
    }
}
