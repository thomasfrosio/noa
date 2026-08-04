#include <noa/base/Strings.hpp>

#include <noa/io/TextFile.hpp>

#include "Catch.hpp"

using namespace ::noa::types;

TEST_CASE("io::TextFile") {
    namespace fs = std::filesystem;

    const fs::path test_dir = "test_TextFile";
    const fs::path test_file1 = test_dir / "file1.txt";
    const fs::path test_file2 = test_dir / "subdir/file2.txt";
    noa::io::remove_all(test_dir);

    AND_WHEN("file should exists") {
        noa::io::InputTextFile file;

        REQUIRE(not file.fail());
        REQUIRE_THROWS_AS(file.open(test_file2, {.read=true}), noa::Exception);
        REQUIRE(not file.fail());
        REQUIRE_FALSE(file.is_open());

        REQUIRE_THROWS_AS(file.open(test_file2, {.read=true, .write=true}), noa::Exception);
        REQUIRE_FALSE(file.is_open());
        REQUIRE(not file.fail());
    }

    AND_WHEN("creating a file and its parent path") {
        noa::io::OutputTextFile file;
        REQUIRE_FALSE(noa::io::is_file(test_file2));
        file.open(test_file2, {.write=true, .append=true});
        REQUIRE(not file.fail());
        REQUIRE(file.is_open());
        REQUIRE(noa::io::is_file(test_file2));
        file.close();
        REQUIRE_FALSE(file.is_open());
    }

    AND_WHEN("write and read") {
        noa::io::TextFile file;
        file.open(test_file1, {.write=true, .append=true});
        file.write(fmt::format("Here are some arguments: {}, {} ", 123, 124));
        file.write("I'm about to close the file...");
        file.close();
        REQUIRE(not file.fail());

        // read_all() needs the file stream to be opened.
        REQUIRE_THROWS_AS(file.read_all(), noa::Exception);
        file.clear_flags();

        const std::string expected = "Here are some arguments: 123, 124 "
                                     "I'm about to close the file...";
        REQUIRE(file.size() == expected.size());

        file.open(test_file1, {.read=true});
        REQUIRE(file.read_all() == expected);
        REQUIRE(not file.fail());

        REQUIRE_THROWS_AS(noa::io::InputTextFile(test_dir / "not_existing", {.read=true}), noa::Exception);
    }

    AND_WHEN("backup copy and backup move") {
        noa::io::TextFile file(test_file2, {.write=true});
        file.write("number: 2");
        file.close();
        REQUIRE(file.ssize() == 9);

        // Backup copy: open an existing file in writing mode.
        const fs::path test_file2_backup = test_file2.string() + '~';
        file.open(test_file2, {.read=true, .write=true});
        REQUIRE(file.is_open());
        REQUIRE(file.read_all() == "number: 2");
        REQUIRE(noa::io::file_size(test_file2_backup) == file.ssize());

        noa::io::remove(test_file2_backup);

        // Backup move: open an existing file in overwriting mode.
        file.open(test_file2, {.write=true});
        REQUIRE(file.is_open());
        REQUIRE(noa::io::is_file(test_file2_backup));
        REQUIRE(file.read_all().empty());
        REQUIRE(noa::io::file_size(test_file2_backup) == 9);
    }

    AND_THEN("next_line and fstream") {
        noa::io::TextFile file(test_file2, {.write=true, .truncate=true});
        REQUIRE(file.is_open());
        const std::string str = "line1\nline2\nline3\nline4\n";
        file.write(str);
        file.close();
        REQUIRE(not file.fail());

        file.open(test_file2, {.read=true});
        std::string line;
        i32 count{};
        while (file.next_line(line)) {
            ++count;
            REQUIRE(line == fmt::format("line{}", count));
        }
        REQUIRE((count == 4 and not file.bad() and file.eof()));

        file.open(test_file2, {.read=true});
        count = 0;
        while (file.next_line_or_throw(line)) {
            ++count;
            REQUIRE(line == fmt::format("line{}", count));
        }
        REQUIRE((count == 4 and not file.bad() and file.eof()));
    }

    AND_WHEN("append") {
        noa::io::TextFile file;
        file.open(test_file1, {.write=true, .truncate=true});
        file.write("0");
        file.close();
        file.open(test_file1, {.write=true, .append=true});
        file.write("1");
        file.close();

        file.open(test_file1, {.read=true});
        REQUIRE(file.read_all() == "01");
    }

    AND_WHEN("line iterator") {
        constexpr auto LINES = std::array{"line1", "line2", "line3", "line4"};
        auto f = noa::io::TextFile(test_file1, {.write=true, .truncate=true});
        for (auto line : LINES) {
            f.write(line);
            f.write("\n");
        }
        f.close();
        f.open(test_file1, {.read=true});

        size_t i{};
        for (auto line : f)
            REQUIRE(line == LINES[i++]);
        REQUIRE(i == 4);
    }

    noa::io::remove_all(test_dir);
}
