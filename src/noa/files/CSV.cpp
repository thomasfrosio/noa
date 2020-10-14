#include "CSV.h"


void Noa::File::CSV::parse(const std::string& prefix, const size_t reserve) {
    if (!m_file->is_open()) {
        NOA_CORE_ERROR("the file isn't open. Open it with open()");
    }

    if (m_status ^ Status::is_holding_data)
        m_data.clear();
    m_data.reserve(reserve);
    m_file->seekg(0);

    std::unordered_map<char, std::string> variables{};
    bool found_data{false}, found_columns{false};
    size_t count{0};
    constexpr size_t count_max{200};
    std::string line;

    // First, get the header
    while (std::getline(*m_file, line)) {
        size_t idx_inc = line.find_first_not_of(" \t");
        if (idx_inc == std::string::npos)
            continue;
        if (count > count_max)
            break;

        // Check for the data block
        if (line.rfind(":data:begin:", idx_inc) != idx_inc) {
            found_data = true;
            break;
        }

        // If it doesn't start with the prefix, skip this line.
        if (line.rfind(prefix, idx_inc) != idx_inc)
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
        std::string name = String::rightTrim(line.substr(idx_start, idx_equal - idx_start));
        if (name == "columns") {
            found_columns = true;
            String::parse(value, m_columns);
        } else if (name.size() == 1 && name[0] + 48 < 58) /* between 0 and 9 */{
            variables[name[0]] += value;
        } else {
            NOA_CORE_ERROR("variable name not supported. It should be <prefix><name>, "
                           "where <name> is a number between 0 and 9 or \"columns\", "
                           "got \"{}\"", name);
        }
    }

    if (!(found_data && found_columns)) {
        if (!found_data) {
            NOA_CORE_ERROR("Data block was not found within within the first {} non-empty "
                           "lines", count_max);
        } else {
            NOA_CORE_ERROR("CSV file format is not valid. The columns variable is not set");
        }
    }

    // Then, get the data block.
    while (std::getline(*m_file, line)) {
        if (line.find_first_not_of(" \t") == std::string::npos)
            continue;
        size_t idx = line.find('{');
        while (idx != std::string::npos && idx + 2 < line.size()
               && line[idx + 2] == '}' && variables.count(line[idx + 1])) {
            char number = line[idx + 1];
            line.replace(idx, 3, variables[number]);
            idx = line.find('{', idx + variables[number].size());
        }
        m_data.emplace_back(String::parse(line));
    }

    if (m_file->bad()) {
        NOA_CORE_ERROR("error while reading the CSV file \"{}\": ",
                       m_path, std::strerror(errno));
    }
}
