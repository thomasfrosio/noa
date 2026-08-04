#pragma once

#include <utility>

#include "noa/runtime/Array.hpp"

#include "noa/io/IO.hpp"
#include "noa/io/Encoding.hpp"
#include "noa/io/ImageFile.hpp"

namespace noa::io {
    struct ReadOption {
        /// Rank of the returned array.
        /// If 0, the returned ND array is returned as encoded in the file.
        /// If 1, 2 or 3, the ND array is reshaped to the ranked shape before returning. The ranked shape collapses
        /// batch axes to the leftmost axis. Setting the rank is usually used to correct wrongly encoded MRC files.
        /// This only has an effect for N > (rank + 1).
        usize rank{0};

        /// Whether the decoded values should be clamped to the output type range.
        bool clamp{true};

        /// Output dtype.
        /// This is only used for the read_image<std::byte> overload.
        /// If UNKNOWN, defaults to file.dtype().closest_static_type().
        DataType dtype{DataType::UNKNOWN};

        /// Number of threads to read and decode the data.
        i32 n_threads{1};
    };

    template<typename T, usize N>
    struct ReadOutput {
        Array<T, N> data;
        ImageFile::Header header;
    };

    /// Returns a new C-contiguous array of type T containing the whole file data, and the file header.
    /// Files map to BDHW dimensions (see ImageFile) but this function can reshape the 4D shape to ND, possibly ranked, shape.
    template<nt::numeric T, usize N = 4>
    [[nodiscard]] auto read_image(
        const Path& path,
        ReadOption read_option = {},
        ArrayOption array_option = {}
    ) -> ReadOutput<T, N> {
        auto file = ImageFile(path, Open{.read = true});
        auto data = Array<T, 4>(file.shape(), array_option);

        if (array_option.is_dereferenceable()) {
            file.read_all(data.span(), {
                .clamp = read_option.clamp,
                .n_threads = read_option.n_threads,
            });
        } else {
            auto tmp = Array<T, 4>(data.shape());
                file.read_all(tmp.span(), {
                .clamp = read_option.clamp,
                .n_threads = read_option.n_threads,
            });
            std::move(tmp).to(data);
        }

        auto data_nd = std::move(data).template as_nd<N>();
        if (read_option.rank != 0)
            data_nd = std::move(data_nd).reshape(data_nd.shape().ranked(read_option.rank));
        return {std::move(data_nd), file.header()};
    }

    /// Returns a new C-contiguous array of bytes containing the whole file data, and the file header.
    /// Files map to BDHW dimensions (see ImageFile) but this function can reshape the 4D shape to ND, possibly ranked, shape.
    template<nt::byte T, usize N>
    [[nodiscard]] auto read_image(
        const Path& path,
        ReadOption read_option = {},
        ArrayOption array_option = {}
    ) -> ReadOutput<T, N> {
        auto file = ImageFile(path, Open{.read = true});

        auto dtype = read_option.dtype;
        if (dtype == DataType::UNKNOWN)
            dtype = file.dtype().closest_static_type();

        const auto n_bytes_per_element = dtype.n_bytes(1);
        auto shape_byte = file.shape();
        shape_byte[3] *= n_bytes_per_element;
        auto data = Array<std::byte, 4>(shape_byte, array_option);

        if (array_option.is_dereferenceable()) {
            file.read_all(data.span_1d(), dtype, {
                .clamp = read_option.clamp,
                .n_threads = read_option.n_threads,
            });
        } else {
            auto tmp = Array<std::byte, 4>(shape_byte);
            file.read_all(tmp.span_1d(), dtype, {
                .clamp = read_option.clamp,
                .n_threads = read_option.n_threads,
            });
            std::move(tmp).to(data);
        }

        auto data_nd = std::move(data).as_nd<N>();
        if (read_option.rank != 0)
            data_nd = std::move(data_nd).reshape(data_nd.shape().ranked(read_option.rank));
        return {std::move(data_nd), file.header()};
    }

    struct WriteOption {
        /// Rank of the written array.
        /// If 0, the ND array is written as is in the file.
        /// If 1, 2 or 3, the ND array is reshaped to the ranked shape before writing. The ranked shape collapses
        /// batch axes to the leftmost axis. Setting the rank is usually used to distinguish between DHW and BHW 3D arrays.
        /// This only has an effect for N > rank.
        usize rank{0};

        /// DHW spacing (in Angstrom/pix) of the new file.
        Vec<f64, 3> spacing{};

        /// Desired data type of the new file.
        /// Note that encoders are allowed to select a different data-type (usually the closest related)
        /// if this one is not supported. If DataType::UNKNOWN, let the file encoder decide the best
        /// format given the input value type.
        DataType dtype{};

        /// Compression scheme of the new file.
        /// Note that encoders are allowed to select a different compression scheme
        /// (a similar scheme or Compression::NONE) if this one is not supported.
        ///  For MRC, compression is not supported and this is ignored.
        Compression compression{};

        /// Statistics of the data in the new file.
        /// This is for the entire file; there's currently no way to specify it per image/volume within the file.
        /// Note that encoders can ignore certain fields, so check with the stats.has_* functions before use.
        /// For MRC, every field should be specified, otherwise none are saved.
        /// For TIFF, only min and max are used, and both of them should be specified in order to be saved.
        ImageFile::Stats stats{};

        /// Whether the input values should be clamped to the serialized values range.
        bool clamp{true};

        /// Number of threads to encode and the data.
        i32 n_threads{1};
    };

    /// Saves the input array into a new file.
    /// \param[in] input    Array to save to disk. Should be reshapeable to 4D.
    /// \param[in] filename Path of the new file.
    /// \param write_option Options.
    template<nt::readable_array_decay_of_numeric Input>
    void write_image(
        Input&& input,
        const Path& filename,
        WriteOption write_option = {}
    ) {
        using value_t = nt::mutable_value_type_t<Input>;

        auto dtype = write_option.dtype;
        if (dtype == DataType::UNKNOWN)
            dtype = ImageFile::closest_supported_dtype<value_t>(filename.extension().string());

        constexpr auto N = nt::array_size_v<Input>;
        Array<value_t, N> tmp;
        Span<const value_t, N> span;
        if (input.device().is_cpu()) {
            // Unfortunately, the IO is currently part of the core, thus is not stream-aware.
            // To account for asynchronous CPU streams, it is important to synchronize here!
            span = input.eval().span();
        } else if (input.is_dereferenceable()) {
            // The input is on the GPU, reinterpret_as will prefetch and sync, making sure
            // the GPU is done with the input and that the memory can be accessed by the CPU.
            span = input.reinterpret_as_cpu({.prefetch = true}).span();
        } else {
            // The input is on the GPU but cannot be reinterpreted, so copy to cpu.
            // to_cpu() synchronizes both the CPU and GPU stream in this case.
            tmp = std::forward<Input>(input).to_cpu();
            span = tmp.span();
        }

        auto span_4d = span.template as_nd<4>();
        if (write_option.rank != 0)
            span_4d = span_4d.reshape(span_4d.shape().ranked(write_option.rank));

        ImageFile(filename, Open{.write = true}, {
            .shape = span_4d.shape(),
            .spacing = write_option.spacing,
            .dtype = dtype,
            .compression = write_option.compression,
            .stats = write_option.stats,
        }).write_all(span_4d, {
            .clamp = write_option.clamp,
            .n_threads = write_option.n_threads,
        });
    }

    /// Saves the type-erased input array into a new file.
    /// \param[in] input    Type-erased C-contiguous array to save to disk.
    /// \param input_dtype  Input dtype.
    /// \param[in] filename Path of the new file.
    /// \param write_option Options.
    template<nt::readable_array_decay_of_byte Input>
    void write_image(
        Input&& input,
        DataType input_dtype,
        const Path& filename,
        WriteOption write_option = {}
    ) {
        using byte_t = nt::mutable_value_type_t<Input>;
        check(input.is_contiguous(),
              "Input array should be C-contiguous, but got input:shape={}, input:strides={}",
              input.shape(), input.strides());

        constexpr auto N = nt::array_size_v<Input>;
        const auto type_erased_shape = input.shape();
        const auto n_bytes_per_element = input_dtype.n_bytes(1);
        check(is_multiple_of(type_erased_shape[N - 1], n_bytes_per_element),
              "Type-erased input array is not valid, got input:shape={}, input:dtype={}",
              type_erased_shape, input_dtype);
        const auto width = type_erased_shape[N - 1] / n_bytes_per_element;

        auto dtype = write_option.dtype;
        if (dtype == DataType::UNKNOWN)
            dtype = ImageFile::closest_supported_dtype(filename.extension().string(), input_dtype);

        Array<byte_t, N> tmp;
        Span<const byte_t, N> span;
        if (input.device().is_cpu()) {
            // Unfortunately, the IO is currently part of the core, thus is not stream-aware.
            // To account for asynchronous CPU streams, it is important to synchronize here!
            span = input.eval().span();
        } else if (input.is_dereferenceable()) {
            // The input is on the GPU, reinterpret_as will prefetch and sync, making sure
            // the GPU is done with the input and that the memory can be accessed by the CPU.
            span = input.reinterpret_as_cpu({.prefetch = true}).span();
        } else {
            // The input is on the GPU but cannot be reinterpreted, so copy to cpu.
            // to_cpu() synchronizes both the CPU and GPU stream in this case.
            tmp = std::forward<Input>(input).to_cpu();
            span = tmp.span();
        }

        auto span_4d = span.template as_nd<4>();
        if (write_option.rank != 0)
            span_4d = span_4d.reshape(span_4d.shape().ranked(write_option.rank));

        ImageFile(filename, Open{.write = true}, {
            .shape = span_4d.shape().template set<3>(width),
            .spacing = write_option.spacing,
            .dtype = dtype,
            .compression = write_option.compression,
            .stats = write_option.stats,
        }).write_all(span_4d.as_1d(), input_dtype, {
            .clamp = write_option.clamp,
            .n_threads = write_option.n_threads,
        });
    }
}

// Expose read/write to noa.
namespace noa {
    using noa::io::read_image;
    using noa::io::write_image;
}
