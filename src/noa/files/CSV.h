/**
 * @file TextFile.h
 * @brief Text file class.
 * @author Thomas - ffyr2w
 * @date 9 Oct 2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/utils/String.h"
#include "noa/files/Text.h"
#include "noa/managers/Table.h"


namespace Noa::File {

    /**
     * Read and write CSV files + get and set data blocks.
     */
    class CSV : public Text {
        // read line, retrieve line
    private:
        std::string m_header{};
        std::unordered_map<char, std::string> m_variables{};

        std::map<std::string, Table<std::string>> m_data{};

        /**
         * This is set of bitmasks.
         *  - TextFile::Option::long_wait   Wait for the file to exist for up to 15*2s.
         *                                  By default, wait for 5*10ms
         */
        union Status {
            static constexpr uint8_t is_holding_data{0x01};
            static constexpr uint8_t is_columns_set{0x02};
        };
        uint8_t m_status{0};


    public:
        /**
         * Parse the CSV file and save its content to @c m_data.
         * @param[in] prefix    Prefix of the variables. Can be empty.
         * @param[in] reserve   Initial size of the table (i.e. c@ std::vector) holding the data block.
         *                      The table grows exponentially, so it can keep up with large blocks.
         *
         * @note To retrieve values of a given line, use @c get().
         * @note Layout. The file is divided into two parts:
         *       - 1: the header, located at the top the top of the file (more precisely, before
         *       the data block, more on that later). The header has two purposes, a) setting user
         *       variables and b) setting the @c columns variable. Variables have the following
         *       format: @c {prefix}{number}={values}, where @c {number} is a number from 0 to 9
         *       and @c {values} is the value or values of the variable. Spaces _between_
         *       @c {prefix}{number}, the equal sign and @c {values} are ignored. (inline) Comments
         *       are allowed and starts with a "#". Quoting is _not_ possible; quotes will be
         *       treated as any other character. See @c String::parse() for more details. Entering
         *       the same variable multiple time is possible and will result into appending the
         *       values. The variables are used to simplify the data block: the @c {X} pattern,
         *       where X is a known @c {number}, in the data block will be replaced by the
         *       corresponding @c {values}. This is especially useful for paths. Setting the
         *       @c columns variable is _not_ optional and the @c values of this variable should
         *       contain a comma separated list of the, case sensitive, columns name.
         *       - 2) the data blocks, located after the header, starts with the delimiter
         *       @c :data:begin: and ends with @c :data:end: (case sensitive). Inside the data block,
         *       the parser will start to store the data. The number of columns must match the
         *       number of names that were set in the @c columns variable.
         */
        void parse(const std::string& prefix, size_t reserve = 500);

        // parse: read m_path, parse its content and save it into m_data

        // save: write m_data to m_path
    };
}
