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


namespace Noa::File {

    /**
     * Read and write Project files + get and set data blocks.
     */
    class Project : public Text {
        // read line, retrieve line
    private:
        std::string m_header{};

        /**
         * Store the variables found in the @c beg|end:{name}:head sections.
         * Map format: @c {name}=[{var}={values}].
         * @see getHead()
         */
        std::map<std::string, std::map<std::string, std::string>> m_head{};

        /**
         * Store the {meta} tables found in the @c beg|end:{name}:meta sections.
         * The tables are simply stored and left un-parsed.
         * Map format: @c {name}={table}.
         * @see getMeta()
         */
        std::unordered_map<std::string, std::vector<std::string>> m_meta{};

        /**
         * Store the {zone} tables found in the @c beg|end:{name}:zone sections.
         * The tables are simply stored and left un-parsed.
         * Map format: @c {name}=[{nb}={table}].
         * @see getMeta()
         */
        std::unordered_map<std::string, std::map<size_t, std::vector<std::string>>> m_zone{};

        /**
         * This is set of bitmasks.
         *
         */
        union Status {
            static constexpr uint8_t is_holding_data{0x01};
            static constexpr uint8_t is_columns_set{0x02};
        };
        uint8_t m_status{0};

    public:
        /**
         * Set and open the underlying @c Text class.
         * @see                 ::Noa::File::Text
         * @tparam T            A valid path, by lvalue or rvalue.
         * @param[in] path      Filename to store in the current instance.
         * @param[in] mode      Any of the open mode (in|out|trunc|app|ate|binary).
         * @param[in] long_wait Wait for the file to exist for 10*30s, otherwise wait for 5*10ms.
         */
        template<typename T,
                typename = std::enable_if_t<std::is_constructible_v<std::filesystem::path, T>>>
        explicit Project(T&& path,
                         std::ios_base::openmode mode = std::ios::in | std::ios::out,
                         bool long_wait = false)
                : Text(std::forward<T>(path), mode, long_wait) {}


        /**
         * Parse the Project file and save its content to @c m_data.
         * @param[in] prefix    Prefix of the variables. Can be empty.
         * @param[in] reserve   Initial size of the table (i.e. c@ std::vector) holding the data block.
         *                      The table grows exponentially, so it can keep up with large blocks.
         *
         * @details Files are composed of one header and blocks. The header is effectively any text
         *          located before the beginning of the first block. This is saved by the parser and
         *          will be rewritten by @c save(), but its content will not be read. Blocks have
         *          the following format:
         *          @code
         *          :beg:{name}:{section}:
         *          <...>
         *          :end:{name}:{section}:
         *          @endcode
         *          where @c {name} is the name of the block, e.g. tilt1, and @c {section} is the
         *          section type of the block. There are currently 3 types of section, all of which
         *          must be specified for each @c beg|end:{name} block: 1) @c head, which contains
         *          variables, like @c path and @c zone (more on that later), note that variables
         *          can be entered multiple times, which causes the values to be appended, 2) @c meta,
         *          which contains a csv table with metadata information, usually per image, e.g.
         *          exposure, rotation, shift, etc., and 3) @c zone, which contains a csv table with
         *          information on a per particle basis (like STAR files). The @c zone sections need
         *          2 additional information: the number of zones and the coordinates of each zone.
         *          These information should be specified on the @c head section. For example, if
         *          one wants to divide a stack in 2 zones, here's the expected layout.
         *          @code
         *          :beg:{name}:head:
         *          path=example/path
         *          zone={coordinates}
         *          :end:{name}:head:
         *          :beg:{name}:meta:
         *          <...>
         *          :end:{name}:meta:
         *          :beg:{name}:zone:0:
         *          <...>
         *          :end:{name}:zone:0:
         *          :beg:{name}:zone:1:
         *          <...>
         *          :end:{name}:zone:1:
         *          @endcode
         *          In SPA-cryoEM, zones are usually not necessary and one doesn't need to enter
         *          the @c zone={coordinates} line. However, if one want to create zones, it is
         *          possible and @c {coordinates} should be a comma separated list such as:
         *          @c xmin,xmax,ymin,ymax. These coordinates are in (unbinned) pixels and
         *          specify the sub-region (i.e. the rectangle) in the image (i.e. the sum of the
         *          frames) that can contain particles. If one doesn't enter the @c zone={coordinates}
         *          line, the @c {coordinates} will be set to the image dimension and the parser
         *          will expect one zone (i.e. the @c :beg|end:{name}:zone:0: zone).
         *          In SPA-cryoET, this is where zones becomes really useful. The format is the same
         *          as described for SPA-cryoEM except that 1) @c {coordinates} have 2 additional
         *          values @c zmin,zmax and 2) the @c zone variable is _not_ optional. To declare
         *          multiple zones, just enter the coordinates sequentially (this can be done over
         *          multiple lines by re-entering the @c zone). The first 6 values will describe the
         *          coordinates for the @c zone:0, the next 6 values for @c zone:1, etc.
         */
        void parse(const std::string& prefix, size_t reserve = 500);

        // bool save() noexcept: write m_data to m_path


        /**
         *
         * @return
         */
        [[nodiscard]] inline size_t stackCount() const noexcept {
            return m_head.size();
        }


        inline std::map<std::string, std::string>& getHead(const std::string& stack) {
            return m_head[stack];
        }


        inline std::string& getHead(const std::string& stack, const std::string& variable) {
            return m_head[stack][variable];
        }


        inline std::vector<std::string>& getMeta(const std::string& stack) {
            return m_meta[stack];
        }


        inline std::map<size_t, std::vector<std::string>>& getZone(const std::string& stack) {
            return m_zone[stack];
        }


        inline std::vector<std::string>& getZone(const std::string& stack, size_t zone) {
            return m_zone[stack][zone];
        }


    private:
        void parseHead_(const std::string& name, const std::string& prefix);

        void parseMeta_(const std::string& name);

        void parseZone_(const std::string&, size_t zone);
    };
}
