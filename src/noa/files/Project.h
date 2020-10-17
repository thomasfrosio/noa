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
     * Read and write project files.
     * @details This class can red a valid project file and parse its content into the instance
     *          containers (i.e. @c m_header, @c m_head, @c m_meta and @c m_zone). These containers
     *          can be access with the @c get*() member functions to get _and_ set data. Finally,
     *          the @c save() member function takes whatever is in the instance containers and
     *          saves it into a valid project file.
     */
    class NOA_API Project : public Text {
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
         * @see getZone()
         */
        std::unordered_map<std::string, std::map<size_t, std::vector<std::string>>> m_zone{};

    public:
        /**
         * Set the underlying @c Text class. The file isn't open.
         * @see         ::Noa::File::Text
         * @tparam T    A valid path, by lvalue or rvalue.
         * @param path  Filename to store in the current instance.
         */
        template<typename T,
                typename = std::enable_if_t<std::is_constructible_v<std::filesystem::path, T>>>
        explicit Project(T&& path): Text(std::forward<T>(path)) {}


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
         * Load the project file and save its content into the containers.
         * @param[in] prefix    Prefix of the variables. Can be empty.
         *
         * @details Files are composed of one header and blocks. The header is effectively any text
         *          located before the beginning of the first block. This is saved by the parser and
         *          will be rewritten by @c save(), but its content will not be read. Blocks have
         *          the following format:
         *          @code
         *          :beg:{name}:{type}:
         *          <...>
         *          :end:{name}:{type}:
         *          @endcode
         *          where @c {name} is the name of the stack, e.g. tilt1, and @c {type} is the
         *          type of the block. There are currently 3 types of section: 1) @c head, which
         *          contains variables, like paths variable and the @c zone variable (more on that
         *          later). Note that variables can be entered multiple times, which causes the
         *          values to be appended, 2) @c meta, which contains a csv table with metadata
         *          information, usually per image, e.g. exposure, rotation, shift, etc. @c meta
         *          blocks must have a corresponding @c head block. 3) @c zone, which contains a csv
         *          table with information on a per particle basis (like STAR files). @c zone blocks
         *          must have a corresponding @c meta block (and therefore a corresponding @c head
         *          block).The @c zone blocks need 2 additional information: the number of zones and
         *          the coordinates of each zone. These information should be specified on the
         *          @c zone variable, in the corresponding @c head block. For example, if one wants
         *          to divide a stack in 2 zones, here's the minimum layout:
         *          @code
         *          :beg:{name}:head:
         *          zone={coordinates}
         *          :end:{name}:head:
         *          :beg:{name}:meta:
         *          {table}
         *          :end:{name}:meta:
         *          :beg:{name}:zone:0:
         *          {table}
         *          :end:{name}:zone:0:
         *          :beg:{name}:zone:1:
         *          {table}
         *          :end:{name}:zone:1:
         *          @endcode
         *          In SPA-cryoEM, zones are usually not necessary but the parser still requires
         *          the @c zone={coordinates} line. To create zones, the @c {coordinates} should be
         *          a comma separated list such as: @c xmin,xmax,ymin,ymax. These coordinates are in
         *          (unbinned) pixels, from 0 to the number of pixels in the corresponding axis, and
         *          they specify the zones (i.e. sub-regions or rectangles) in the image that can
         *          contain particles. If one want to include the entire image, the @c {coordinates}
         *          should be set to the image dimension. Zones are numbered from 0. To declare
         *          multiple zones, just enter the coordinates sequentially (this can be done over
         *          multiple lines by re-entering the @c zone). The first 4 values should describe
         *          the coordinates of the first zone (i.e. @c zone:0), the next 4 values describe
         *          the second zone (i.e. @c zone:1), etc.
         *          In SPA-cryoET, this is where zones becomes really useful. The format is the same
         *          as described for SPA-cryoEM except that the @c {coordinates} have the following
         *          format: @c xmin,xmax,ymin,ymax,zmin,zmax.
         */
        void load(const std::string& prefix);


        /**
         * Save the stored data into a project file.
         * @param[in] name  Name of the project file to create or overwrite.
         *
         * @note    This function opens (or reopen if path == m_path) the file stream @c m_file in
         *          in std::ios::out | std::ios::trunc mode. This means that 1) if the file exists,
         *          its original content will be lost and 2)
         */
        void save(const std::string& path);


        /**
         *
         */
        inline void save() {
            save(m_path);
        }


        inline std::map<std::string, std::string>& getHead(const std::string& stack) {
            return m_head[stack];
        }


        inline std::vector<std::string>& getMeta(const std::string& stack) {
            return m_meta[stack];
        }


        inline std::map<size_t, std::vector<std::string>>& getZone(const std::string& stack) {
            return m_zone[stack];
        }

    private:
        void parseHead_(const std::string& name, const std::string& prefix);

        void parseMeta_(const std::string& name);

        void parseZone_(const std::string&, size_t zone);
    };
}
