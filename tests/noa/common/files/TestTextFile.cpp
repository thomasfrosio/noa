#include <noa/common/io/TextFile.h>
#include <noa/common/string/Format.h>

#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("TextFile:", "[noa][file]") {
    fs::path test_dir = "test_TextFile";
    fs::path test_file1 = test_dir / "file1.txt";
    fs::path test_file2 = test_dir / "subdir/file2.txt";
    os::removeAll(test_dir);

    AND_WHEN("file should exists") {
        io::TextFile<std::ifstream> file;

        REQUIRE_THROWS_AS(file.open(test_file2, io::READ), noa::Exception);
        REQUIRE(!file.isOpen());
        REQUIRE(!file);

        file.clear();
        REQUIRE_THROWS_AS(file.open(test_file2, io::READ | io::WRITE), noa::Exception);
        REQUIRE(!file.isOpen());
        REQUIRE(!file);
    }

    AND_WHEN("creating a file and its parent path") {
        io::TextFile<std::ofstream> file;
        REQUIRE(!os::existsFile(test_file2));
        file.open(test_file2, io::WRITE | io::APP);
        REQUIRE(file);
        REQUIRE(file.isOpen());
        REQUIRE(os::existsFile(test_file2));
        file.close();
        REQUIRE(!file.isOpen());
    }

    AND_WHEN("write and read") {
        io::TextFile file;
        file.open(test_file1, io::APP);
        file.write(string::format("Here are some arguments: {}, {} ", 123, 124));
        file.write("I'm about to close the file...");
        file.close();
        REQUIRE(file);

        // readAll() needs the file stream to be opened.
        REQUIRE_THROWS_AS(file.readAll(), noa::Exception);
        file.clear();

        std::string expected = "Here are some arguments: 123, 124 "
                               "I'm about to close the file...";
        REQUIRE(file.size() == expected.size());

        file.open(test_file1, io::READ);
        REQUIRE(file.readAll() == expected);
        REQUIRE(file);

        REQUIRE_THROWS_AS(io::TextFile<std::ifstream>(test_dir / "not_existing", io::READ), noa::Exception);
    }

    AND_WHEN("backup copy and backup move") {
        io::TextFile file(test_file2, io::WRITE);
        file.write("number: 2");
        file.close();
        REQUIRE(file.size() == 9);

        // Backup copy: open an existing file in writing mode.
        fs::path test_file2_backup = test_file2.string() + '~';
        file.open(test_file2, io::WRITE | io::READ);
        REQUIRE(file.isOpen());
        REQUIRE(file.readAll() == "number: 2");
        REQUIRE(os::size(test_file2_backup) == file.size());

        os::remove(test_file2_backup);

        // Backup move:  open an existing file in overwriting mode.
        file.open(test_file2, io::WRITE);
        REQUIRE(file.isOpen());
        REQUIRE(os::existsFile(test_file2_backup));
        REQUIRE(file.readAll().empty());
        REQUIRE(os::size(test_file2_backup) == 9);
    }

    AND_THEN("getLine and fstream") {
        io::TextFile file(test_file2, io::WRITE | io::TRUNC);
        REQUIRE(file.isOpen());
        std::string str = "line1\nline2\nline3\nline4\n";
        file.write(str);
        file.close();
        REQUIRE(file);

        file.open(test_file2, io::READ);
        std::string line, expected;
        int count{0};
        while (file.getLine(line)) {
            ++count;
            REQUIRE(line == fmt::format("line{}", count));
        }
        REQUIRE(count == 4);
        REQUIRE(!file.bad());
    }

    os::removeAll(test_dir);
}
