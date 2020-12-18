/**
 * @file AbstractHeader.h
 * @brief Header abstract class. Headers handle the metadata of image files.
 * @author Thomas - ffyr2w
 * @date 31/10/2020
 */
#pragma once

#include "noa/Base.h"
#include <noa/util/OS.h>
#include "noa/util/IO.h"


namespace Noa {
    class AbstractHeader {
    protected:
        IO::Layout m_layout{IO::Layout::unset};
        bool m_is_big_endian{false};

    public:
        AbstractHeader() = default;
        virtual ~AbstractHeader() = default;

        [[nodiscard]] inline IO::Layout getLayout() const { return m_layout; }
        [[nodiscard]] inline bool isBigEndian() const { return m_is_big_endian; }
        [[nodiscard]] inline bool isSwapped() const { return isBigEndian() != OS::isBigEndian(); }

        // Below are all the functions that derived classes
        // should override to be accepted by the Header class.
        //  ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓

        virtual errno_t setLayout(IO::Layout layout) = 0;

        virtual void setEndianness(bool is_big_endian) = 0;

        [[nodiscard]] virtual size_t getOffset() const = 0;

        virtual errno_t read(std::fstream& fstream) = 0;
        virtual errno_t write(std::fstream& fstream) = 0;

        virtual void reset() = 0;

        [[nodiscard]] virtual Int3<size_t> getShape() const = 0;
        virtual errno_t setShape(Int3<size_t>) = 0;

        [[nodiscard]] virtual Float3<float> getPixelSize() const = 0;
        virtual errno_t setPixelSize(Float3<float>) = 0;

        [[nodiscard]] virtual std::string toString(bool brief) const = 0;
    };
}
