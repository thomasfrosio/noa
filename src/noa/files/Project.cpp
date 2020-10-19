#include "Project.h"


void Noa::File::Project::load(const std::string& prefix) {
    if (!m_file->is_open()) {
        NOA_CORE_ERROR("the file is not open. Open it with open() or reopen()");
    }

    bool within_header{true};
    std::string line;
    m_file->seekg(0);
    while (std::getline(*m_file, line)) {
        size_t idx_inc = line.find_first_not_of(" \t");
        if (idx_inc == std::string::npos)
            continue;

        if (line.rfind(":beg:", idx_inc) != idx_inc) /* not a block */ {
            if (within_header) {
                m_header += line;
                m_header += '\n';
            }
            continue;
        } else {
            within_header = false;
        }

        uint8_t status{0};

        // Blocks are formatted as follow: :beg:{name}:{type}:, where {type} is either
        // head, meta or zone, all of which are 4 characters.
        size_t idx_type = line.find(':', idx_inc + 5);  // :beg:>
        if (idx_type == std::string::npos || !(line.size() >= idx_type + 6 &&
                                               line[idx_type + 5] == ':')) {
            NOA_CORE_ERROR("\"{}\": block format is not recognized: {}", m_path.c_str(), line);
        }

        // Parse and store the block. The parse*_() functions are reading lines until the block
        // stops, so the next iteration starts outside the block.
        std::string name{line.data() + idx_inc + 5, idx_type - idx_inc - 5};
        std::string_view type{line.data() + idx_type + 1, 4};
        if (type == "zone") {
            size_t idx_end = line.find(':', 5); // :beg:name:zone:>
            if (idx_end == std::string::npos) {
                NOA_CORE_ERROR("\"{}\": zone block format is not recognized: {}",
                               m_path.c_str(), line);
            }
            size_t zone = String::toInt<size_t>({line.data() + idx_type + 6,
                                                 idx_end - idx_type + 5},
                                                status);
            if (status) {
                NOA_CORE_ERROR("\"{}\": zone block number is not valid. It should be a positive "
                               "number, got {}", m_path.c_str(), line);
            }
            parseZone_(name, zone);
        } else if (type == "head") {
            parseHead_(name, prefix);
        } else if (type == "meta") {
            parseMeta_(name);
        } else {
            NOA_CORE_ERROR("\"{}\": type block \"{}\" is not recognized. It should be "
                           "\"head\", \"meta\" or \"zone\"", m_path.c_str(), type);
        }
    }
    if (m_file->bad()) {
        NOA_CORE_ERROR("\"{}\": error while loading the project file: {}",
                       m_path.c_str(), std::strerror(errno));
    }

    // Some checks.
    for (const auto& pair: m_zone) {
        if (m_meta.count(pair.first) != 1) {
            NOA_CORE_ERROR("\"{}\": zone block(s) \"{}\" are without meta",
                           m_path.c_str(), pair.first);
        } else if (m_head.count(pair.first) != 1) {
            NOA_CORE_ERROR("\"{}\": zone block(s) \"{}\" are without head",
                           m_path.c_str(), pair.first);
        } else if (m_head[pair.first].count("zone") != 1) {
            NOA_CORE_ERROR("\"{}\": head block \"{}\" is missing its zone variable",
                           m_path.c_str(), pair.first);
        }
    }
}


void Noa::File::Project::parseHead_(const std::string& name, const std::string& prefix) {
    std::string line;
    std::map<std::string, std::string>& map = m_head[name];
    std::string end_delim = fmt::format(":end:{}:head", name);
    bool is_closed{false};
    while (std::getline(*m_file, line)) {
        size_t idx_inc = line.find_first_not_of(" \t");
        if (idx_inc == std::string::npos)
            continue;

        if (line.rfind(end_delim, 0) == 0) {
            is_closed = true;
            break;
        } else if (line.rfind(prefix, idx_inc) != idx_inc)
            continue;

        // Get idx range of the right side of the equal sign.
        size_t idx_start = idx_inc + prefix.size();
        size_t idx_end = line.find('#', idx_start);
        size_t idx_equal = line.find('=', idx_start);
        if (idx_equal == std::string::npos || idx_equal + 1 >= idx_end ||
            idx_start == idx_equal || std::isspace(line[idx_start]))
            continue;

        // Make sure the value to be parsed isn't only whitespaces.
        std::string_view value{line.data() + idx_equal + 1,
                               (idx_end == std::string::npos) ?
                               line.size() - idx_equal - 1 : idx_end - idx_equal - 1};
        if (value.find_first_not_of(" \t") == std::string::npos)
            continue;

        std::string& saved = map[String::rightTrim(line.substr(idx_start, idx_equal - idx_start))];
        if (saved.empty()) {
            saved = value;
        } else if (saved[saved.size() - 1] == ',' || value[0] == ',') {
            saved += value;
        } else {
            saved += ',';
            saved += value;
        }
    }
    if (!is_closed) {
        NOA_CORE_ERROR_LAMBDA("load", "\"{}\": the head block for \"{}\" is not closed",
                              m_path.c_str(), name);
    }
}


void Noa::File::Project::parseMeta_(const std::string& name) {
    std::string line;
    std::vector<std::string>& table = m_meta[name];
    table.reserve(60);
    std::string end_delim = fmt::format(":end:{}:meta", name);
    bool is_closed{false};
    while (std::getline(*m_file, line)) {
        if (line.find_first_not_of(" \t") == std::string::npos)
            continue;
        if (line.rfind(end_delim, 0) == 0) {
            is_closed = true;
            break;
        }
        table.emplace_back(line);
    }
    if (!is_closed) {
        NOA_CORE_ERROR_LAMBDA("load", "\"{}\": the meta block for \"{}\" is not closed",
                              m_path.c_str(), name);
    }
}


void Noa::File::Project::parseZone_(const std::string& name, size_t zone) {
    std::string line;
    std::vector<std::string>& table = m_zone[name][zone];
    table.reserve(500);  // image * particle
    std::string end_delim = fmt::format(":end:{}:zone:{}", name, zone);
    bool is_closed{false};
    while (std::getline(*m_file, line)) {
        if (line.find_first_not_of(" \t") == std::string::npos)
            continue;
        if (line.rfind(end_delim, 0) == 0) {
            is_closed = true;
            break;
        }
        table.emplace_back(line);
    }
    if (!is_closed) {
        NOA_CORE_ERROR_LAMBDA("load", "\"{}\": the zone block for \"{}:{}\" is not closed",
                              m_path.c_str(), name, zone);
    }
}


void Noa::File::Project::save(const std::string& path) {
    std::ofstream ofstream(path, std::ios::out | std::ios::trunc);
    open(path, ofstream, std::ios::out | std::ios::trunc, false);

    std::string buffer(m_header);
    buffer.reserve(500000);  // 0.5KB and then let the string handle it if one stack need more.

    // Go through heads and for each name, write meta and zone.
    // Anything without a head is ignored.
    for (auto&[name, variables]: m_head) {
        // head block
        buffer += fmt::format(":beg:{}:head:\n", name);
        for (auto&[v_name, v_value]: variables) {
            buffer += fmt::format("{}={}\n", v_name, v_value);
        }
        buffer += fmt::format(":end:{}:head:\n", name);

        // meta block
        buffer += fmt::format("\n:beg:{}:meta:\n", name);
        for (auto& line: m_meta[name]) {
            buffer += line;
            buffer += '\n';
        }
        buffer += fmt::format(":beg:{}:meta:\n", name);

        // zone block(s)
        for (auto&[zone, table]: m_zone[name]) {
            buffer += fmt::format("\n:beg:{}:zone:{}:\n", name, zone);
            for (auto& line: table) {
                buffer += line;
                buffer += '\n';
            }
            buffer += fmt::format(":end:{}:zone:{}:\n", name, zone);
        }

        // flush
        ofstream.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        buffer = "";  // capacity should be preserved
    }
}


