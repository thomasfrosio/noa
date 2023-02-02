#include <noa/core/io/TextFile.hpp>
#include <noa/core/string/Format.hpp>

#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("core::io::TextFile", "[noa][core]") {
    const fs::path test_dir = "test_TextFile";
    const fs::path test_file1 = test_dir / "file1.txt";
    const fs::path test_file2 = test_dir / "subdir/file2.txt";
    os::remove_all(test_dir);

    AND_WHEN("file should exists") {
        io::TextFile<std::ifstream> file;

        REQUIRE(file);
        REQUIRE_THROWS_AS(file.open(test_file2, io::READ), noa::Exception);
        REQUIRE(file);
        REQUIRE(!file.is_open());

        REQUIRE_THROWS_AS(file.open(test_file2, io::READ | io::WRITE), noa::Exception);
        REQUIRE(!file.is_open());
        REQUIRE(file);
    }

    AND_WHEN("creating a file and its parent path") {
        io::TextFile<std::ofstream> file;
        REQUIRE(!os::is_file(test_file2));
        file.open(test_file2, io::WRITE | io::APP);
        REQUIRE(file);
        REQUIRE(file.is_open());
        REQUIRE(os::is_file(test_file2));
        file.close();
        REQUIRE(!file.is_open());
    }

    AND_WHEN("write and read") {
        io::TextFile file;
        file.open(test_file1, io::APP);
        file.write(string::format("Here are some arguments: {}, {} ", 123, 124));
        file.write("I'm about to close the file...");
        file.close();
        REQUIRE(file);

        // read_all() needs the file stream to be opened.
        REQUIRE_THROWS_AS(file.read_all(), noa::Exception);
        file.clear_flags();

        const std::string expected = "Here are some arguments: 123, 124 "
                                     "I'm about to close the file...";
        REQUIRE(file.size() == static_cast<i64>(expected.size()));

        file.open(test_file1, io::READ);
        REQUIRE(file.read_all() == expected);
        REQUIRE(file);

        REQUIRE_THROWS_AS(io::TextFile<std::ifstream>(test_dir / "not_existing", io::READ), noa::Exception);
    }

    AND_WHEN("backup copy and backup move") {
        io::TextFile file(test_file2, io::WRITE);
        file.write("number: 2");
        file.close();
        REQUIRE(file.size() == 9);

        // Backup copy: open an existing file in writing mode.
        const fs::path test_file2_backup = test_file2.string() + '~';
        file.open(test_file2, io::WRITE | io::READ);
        REQUIRE(file.is_open());
        REQUIRE(file.read_all() == "number: 2");
        REQUIRE(os::file_size(test_file2_backup) == file.size());

        os::remove(test_file2_backup);

        // Backup move:  open an existing file in overwriting mode.
        file.open(test_file2, io::WRITE);
        REQUIRE(file.is_open());
        REQUIRE(os::is_file(test_file2_backup));
        REQUIRE(file.read_all().empty());
        REQUIRE(os::file_size(test_file2_backup) == 9);
    }

    AND_THEN("get_line and fstream") {
        io::TextFile file(test_file2, io::WRITE | io::TRUNC);
        REQUIRE(file.is_open());
        const std::string str = "line1\nline2\nline3\nline4\n";
        file.write(str);
        file.close();
        REQUIRE(file);

        file.open(test_file2, io::READ);
        std::string line;
        int count{0};
        while (file.get_line(line)) {
            ++count;
            REQUIRE(line == fmt::format("line{}", count));
        }
        REQUIRE(count == 4);
        REQUIRE((!file.bad() && file.eof()));

        file.open(test_file2, io::READ);
        count = 0;
        while (file.get_line_or_throw(line)) {
            ++count;
            REQUIRE(line == fmt::format("line{}", count));
        }
        REQUIRE(count == 4);
        REQUIRE((!file.bad() && file.eof()));
    }

    AND_WHEN("append") {
        io::TextFile file;
        file.open(test_file1, io::WRITE | io::TRUNC);
        file.write("0");
        file.close();
        file.open(test_file1, io::APP | io::ATE);
        file.write("1");
        file.close();

        file.open(test_file1, io::READ);
        REQUIRE(file.read_all() == "01");
    }

    os::remove_all(test_dir);
}
