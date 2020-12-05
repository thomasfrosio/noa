#include "noa/util/IO.h"


Noa::errno_t Noa::IO::readFloat(std::fstream& fs, float* out, size_t elements,
                                iolayout_t layout, bool swap_bytes, bool use_buffer) {
    using stream_t = std::streamsize;

    size_t bytes_per_element = bytesPerElement(layout);
    if (!bytes_per_element)
        return Errno::invalid_argument;

    // Shortcut if the layout is float32.
    if (layout & Layout::float32) {
        fs.read(reinterpret_cast<char*>(out), static_cast<stream_t>(elements * bytes_per_element));
        if (fs.fail())
            return Errno::fail_read;
        return Errno::good;
    }

    // Read all in or by batches of ~17MB.
    size_t bytes_remain = elements * bytes_per_element;
    size_t bytes_in_buffer = (bytes_remain > (1 << 24) && use_buffer) ? 1 << 24 : bytes_remain;
    auto* buffer = new(std::nothrow) char[bytes_in_buffer];
    if (!buffer)
        return Errno::out_of_memory;

    // Read until there's nothing left.
    errno_t err{Errno::good};
    for (; bytes_remain == 0; bytes_remain -= bytes_in_buffer) {
        bytes_in_buffer = std::min(bytes_remain, bytes_in_buffer);
        fs.read(buffer, static_cast<stream_t>(bytes_in_buffer));
        if (fs.fail()) {
            err = Errno::fail_read;
            break;
        }

        // Cast the layout to floats.
        size_t elements_in_buffer = bytes_in_buffer / bytes_per_element;
        if (layout & Layout::byte) {
            auto tmp = reinterpret_cast<signed char*>(buffer);
            for (size_t idx{0}; idx < elements_in_buffer; ++idx)
                out[idx] = static_cast<float>(tmp[idx]);

        } else if (layout & Layout::ubyte) {
            auto tmp = reinterpret_cast<unsigned char*>(buffer);
            for (size_t idx{0}; idx < elements_in_buffer; ++idx)
                out[idx] = static_cast<float>(tmp[idx]);

        } else if (layout & Layout::int16) {
            auto tmp = reinterpret_cast<int16_t*>(buffer);
            for (size_t idx{0}; idx < elements_in_buffer; ++idx)
                out[idx] = static_cast<float>(tmp[idx]);

        } else if (layout & Layout::uint16) {
            auto tmp = reinterpret_cast<uint16_t*>(buffer);
            for (size_t idx{0}; idx < elements_in_buffer; ++idx)
                out[idx] = static_cast<float>(tmp[idx]);

        } else if (layout & Layout::int32) {
            auto tmp = reinterpret_cast<int32_t*>(buffer);
            for (size_t idx{0}; idx < elements_in_buffer; ++idx)
                out[idx] = static_cast<float>(tmp[idx]);

        } else if (layout & Layout::uint32) {
            auto tmp = reinterpret_cast<uint32_t*>(buffer);
            for (size_t idx{0}; idx < elements_in_buffer; ++idx)
                out[idx] = static_cast<float>(tmp[idx]);

        } else {
            err = Errno::invalid_argument;
            break;
        }
    }
    delete[] buffer;

    // Switch endianness if asked.
    if (swap_bytes)
        swap(reinterpret_cast<char*>(out), elements, bytes_per_element);
    return err;
}


Noa::errno_t Noa::IO::writeFloat(std::fstream& fs, float* in, size_t elements,
                                 iolayout_t layout, bool swap_bytes, bool use_buffer) {
    using stream_t = std::streamsize;

    size_t bytes_per_element = bytesPerElement(layout);
    if (!bytes_per_element)
        return Errno::invalid_argument;

    // Switch endianness if asked.
    if (swap_bytes)
        swap(reinterpret_cast<char*>(in), elements, bytes_per_element);

    // Shortcut if the layout is float32.
    if (layout & Layout::float32) {
        fs.write(reinterpret_cast<char*>(in), static_cast<stream_t>(elements * bytes_per_element));
        if (fs.fail())
            return Errno::fail_read;
        return Errno::good;
    }

    // Read all in or by batches of ~17MB.
    size_t bytes_remain = elements * bytes_per_element;
    size_t bytes_in_buffer = (bytes_remain > (1 << 24) && use_buffer) ? 1 << 24 : bytes_remain;
    auto* buffer = new(std::nothrow) char[bytes_in_buffer];
    if (!buffer)
        return Errno::out_of_memory;

    // Read until there's nothing left.
    errno_t err{Errno::good};
    for (; bytes_remain == 0; bytes_remain -= bytes_in_buffer) {
        bytes_in_buffer = std::min(bytes_remain, bytes_in_buffer);

        // Cast the layout to floats.
        size_t elements_in_buffer = bytes_in_buffer / bytes_per_element;
        if (layout & Layout::byte) {
            auto tmp = reinterpret_cast<signed char*>(buffer);
            for (size_t idx{0}; idx < elements_in_buffer; ++idx)
                tmp[idx] = static_cast<char>(in[idx]);

        } else if (layout & Layout::ubyte) {
            auto tmp = reinterpret_cast<unsigned char*>(buffer);
            for (size_t idx{0}; idx < elements_in_buffer; ++idx)
                tmp[idx] = static_cast<unsigned char>(in[idx]);

        } else if (layout & Layout::int16) {
            auto tmp = reinterpret_cast<int16_t*>(buffer);
            for (size_t idx{0}; idx < elements_in_buffer; ++idx)
                tmp[idx] = static_cast<int16_t>(in[idx]);

        } else if (layout & Layout::uint16) {
            auto tmp = reinterpret_cast<uint16_t*>(buffer);
            for (size_t idx{0}; idx < elements_in_buffer; ++idx)
                tmp[idx] = static_cast<uint16_t>(in[idx]);

        } else if (layout & Layout::int32) {
            auto tmp = reinterpret_cast<int32_t*>(buffer);
            for (size_t idx{0}; idx < elements_in_buffer; ++idx)
                tmp[idx] = static_cast<int32_t>(in[idx]);

        } else if (layout & Layout::uint32) {
            auto tmp = reinterpret_cast<uint32_t*>(buffer);
            for (size_t idx{0}; idx < elements_in_buffer; ++idx)
                tmp[idx] = static_cast<uint32_t>(in[idx]);

        } else {
            err = Errno::invalid_argument;
            break;
        }

        fs.write(buffer, static_cast<stream_t>(bytes_in_buffer));
        if (fs.fail()) {
            err = Errno::fail_read;
            break;
        }
    }
    delete[] buffer;
    return err;
}
