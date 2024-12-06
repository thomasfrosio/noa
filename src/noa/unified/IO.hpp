#pragma once

#include <utility>
#include "noa/core/io/IO.hpp"
#include "noa/core/io/Encoding.hpp"
#include "noa/core/io/ImageFile.hpp"
#include "noa/unified/Array.hpp"

namespace noa::io {
    struct ReadOption {
        /// Whether to enforce the output array to be a stack of 2d images, instead of a single 3d volume.
        bool enforce_2d_stack{};

        /// Whether the decoded values should be clamped to the output type range.
        bool clamp{true};

        /// Number of threads to read and decode the data.
        i32 n_threads{1};
    };

    struct WriteOption {
        /// Data type used for the encoding.
        /// If Encoding::UNKNOWN, let the file encoder decide the best format given the input value type.
        Encoding::Type dtype{Encoding::UNKNOWN};

        /// Whether the input values should be clamped to the serialized values range.
        bool clamp{true};

        /// Number of threads to encode and the data.
        i32 n_threads{1};
    };

    /// Loads the file into a new array with a given type T.
    /// \return BDHW C-contiguous output array containing the whole data array of the file, and its spacing.
    template<nt::numeric T>
    [[nodiscard]] auto read(
        const Path& path,
        ReadOption read_option = {},
        ArrayOption array_option = {}
    ) -> Pair<Array<T>, Vec<f64, 3>> {
        auto file = ImageFile(path, Open{.read = true});
        auto data = Array<T>(file.shape(), array_option);

        if (array_option.is_dereferenceable()) {
            file.read_all(data.span(), {.clamp = read_option.clamp, .n_threads = read_option.n_threads});
        } else {
            auto tmp = Array<T>(file.shape());
            file.read_all(tmp.span(), {.clamp = read_option.clamp, .n_threads = read_option.n_threads});
            std::move(tmp).to(data);
        }

        Vec<f64, 3> pixel_size = file.spacing();
        const auto& shape = data.shape();
        if (read_option.enforce_2d_stack and (shape[0] == 1 and shape[1] > 1))
            data = std::move(data).reshape(shape.filter(1, 0, 2, 3));
        auto out = Pair{std::move(data), pixel_size};
        return out;
    }

    /// Loads the file data into a new array.
    /// Same as the overload above, but without loading the spacing.
    template<nt::numeric T>
    [[nodiscard]] auto read_data(
        const Path& path,
        ReadOption read_option = {},
        ArrayOption array_option = {}
    ) -> Array<T> {
        auto tmp = read<T>(path, read_option, array_option);
        return tmp.first;
    }

    /// Saves the input array into a new file.
    /// \param[in] input    Array to save to disk.
    /// \param spacing      (D)HW spacing.
    /// \param[in] filename Path of the new file.
    /// \param write_option Options.
    template<nt::readable_varray_decay_of_numeric Input, nt::vec_real_size<2, 3> PixelSize>
    void write(
        Input&& input,
        const PixelSize& spacing,
        const Path& filename,
        WriteOption write_option = {}
    ) {
        using value_t = nt::mutable_value_type_t<Input>;

        Vec<f64, 3> file_spacing;
        if constexpr (nt::vec_of_size<PixelSize, 2>)
            file_spacing = spacing.template as<f64>().push_front(1);
        else
            file_spacing = spacing.template as<f64>();

        auto dtype = write_option.dtype;
        if (dtype == Encoding::UNKNOWN)
            dtype = ImageFile::closest_supported_dtype<value_t>(filename.extension().string());

        Array<value_t> tmp;
        Span<const value_t, 4> span;
        if (input.is_dereferenceable()) {
            span = input.eval().reinterpret_as_cpu({.prefetch = true}).span();
        } else {
            tmp = std::forward<Input>(input).to_cpu().eval();
            span = tmp.span();
        }

        ImageFile(filename, Open{.write = true}, {
            .shape = input.shape(),
            .spacing = file_spacing,
            .dtype = dtype,
        }).write_all(span, {
            .clamp = write_option.clamp,
            .n_threads = write_option.n_threads,
        });
    }

    /// Saves the input array into a new file.
    /// Same as the above overload, but without setting a pixel size.
    template<nt::readable_varray_decay_of_numeric Input>
    void write(Input&& input, const Path& filename, WriteOption write_option = {}) {
        write(std::forward<Input>(input), Vec{0., 0., 0.}, filename, write_option);
    }
}

// Expose read/write to noa.
namespace noa {
    using noa::io::read;
    using noa::io::read_data;
    using noa::io::write;
}
