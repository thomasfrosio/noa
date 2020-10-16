#include "CSV.h"


void Noa::File::Project::parse(const std::string& prefix, const size_t reserve) {
    if (!m_file->is_open()) {
        NOA_CORE_ERROR("the file isn't open. Open it with open() or reopen()");
    } else if (m_status & Status::is_holding_data)
        NOA_CORE_ERROR("instance is holding data. Better to stop");

    bool within_header{true};
    std::string line;
    m_file->seekg(0);
    while (std::getline(*m_file, line)) {
        size_t idx_inc = line.find_first_not_of(" \t");
        if (idx_inc == std::string::npos)
            continue;

        if (line.rfind(":beg:", idx_inc) != idx_inc) /* not a block */{
            if (within_header)
                m_header += line;
            continue;
        } else {
            within_header = false;
        }

        uint8_t status;

        // Blocks are formatted as follow: :beg:{name}:{type}:, where {type} is either
        // head, meta or zone, all of which are 4 characters.
        size_t idx_type = line.find(':', idx_inc + 5);  // :beg:>
        if (idx_type == std::string::npos || !(line.size() >= idx_type + 5 && line[idx_type + 9])) {
            NOA_CORE_ERROR("block format isn't recognized: {}", line);
        }

        // Parse and store the block. The parse*_() functions are reading lines until the block
        // stops, so the next iteration starts outside the block.
        std::string name{line.data() + idx_inc + 5, idx_type};
        std::string_view type{line.data() + idx_type + 1, 4};
        if (type == "zone") {
            size_t idx_end = line.find(':', 5); // :beg:name:zone:>
            if (idx_end == std::string::npos) {
                NOA_CORE_ERROR("block format isn't recognized: {}", line);
            }
            size_t zone = String::toInt<size_t>({line.data() + idx_type + 5, idx_end}, status);
            parseZone_(name, zone);
        } else if (type == "head") {
            parseHead_(name, prefix);
        } else if (type == "meta") {
            parseMeta_(name);
        } else {
            NOA_CORE_ERROR("project file");
        }
    }
    m_status ^= Status::is_holding_data;

    // check that things checks up: meta and zone block must have a corresponding head block
    // with the zone variable entered.
}


void Noa::File::Project::parseHead_(const std::string& name, const std::string& prefix) {
    std::string line;
    std::map<std::string, std::string>& table = m_head[name];
    std::string end_delim = fmt::format(":end:{}:head", name);
    while (std::getline(*m_file, line)) {
        size_t idx_inc = line.find_first_not_of(" \t");
        if (idx_inc == std::string::npos)
            continue;

        if (line.rfind(end_delim, 0) == 0)
            break;
        else if (line.rfind(prefix, idx_inc) != idx_inc)
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

        // Add or append the variable.
        table["columns"] += String::rightTrim(line.substr(idx_start, idx_equal - idx_start));
    }
}


void Noa::File::Project::parseMeta_(const std::string& name) {
    std::string line;
    std::vector<std::string>& table = m_meta[name];
    table.reserve(60);
    std::string end_delim = fmt::format(":end:{}:meta", name);
    while (std::getline(*m_file, line)) {
        if (line.find_first_not_of(" \t") == std::string::npos)
            continue;
        if (line.rfind(end_delim, 0) == 0)
            break;
        table.emplace_back(line);
    }
}


void Noa::File::Project::parseZone_(const std::string& name, size_t zone) {
    std::string line;
    std::vector<std::string>& table = m_zone[name][zone];
    table.reserve(150);
    std::string end_delim = fmt::format(":end:{}:zone:{}", name, zone);
    while (std::getline(*m_file, line)) {
        if (line.find_first_not_of(" \t") == std::string::npos)
            continue;
        if (line.rfind(end_delim, 0) == 0)
            break;
        table.emplace_back(line);
    }
}
