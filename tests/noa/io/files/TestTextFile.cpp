#include <noa/io/files/TextFile.h>
#include <noa/util/string/Format.h>

#include <catch2/catch.hpp>

using namespace ::Noa;

TEST_CASE("TextFile:", "[noa][file]") {
    fs::path test_dir = "test_TextFile";
    fs::path test_file1 = test_dir / "file1.txt";
    fs::path test_file2 = test_dir / "subdir/file2.txt";
    OS::removeAll(test_dir);

    AND_WHEN("file should exists") {
        TextFile<std::ifstream> file;

        REQUIRE_THROWS_AS(file.open(test_file2, IO::READ), Noa::Exception);
        REQUIRE(!file.isOpen());
        REQUIRE(!file);

        file.clear();
        REQUIRE_THROWS_AS(file.open(test_file2, IO::READ | IO::WRITE), Noa::Exception);
        REQUIRE(!file.isOpen());
        REQUIRE(!file);
    }

    AND_WHEN("creating a file and its parent path") {
        TextFile<std::ofstream> file;
        REQUIRE(!OS::existsFile(test_file2));
        file.open(test_file2, IO::WRITE | IO::APP);
        REQUIRE(file);
        REQUIRE(file.isOpen());
        REQUIRE(OS::existsFile(test_file2));
        file.close();
        REQUIRE(!file.isOpen());
    }

    AND_WHEN("write and read") {
        TextFile file;
        file.open(test_file1, IO::APP);
        file.write(String::format("Here are some arguments: {}, {} ", 123, 124));
        file.write("I'm about to close the file...");
        file.close();
        REQUIRE(file);

        // read() needs the file stream to be opened.
        REQUIRE_THROWS_AS(file.read(), Noa::Exception);
        file.clear();

        std::string expected = "Here are some arguments: 123, 124 "
                               "I'm about to close the file...";
        REQUIRE(file.size() == expected.size());

        file.open(IO::READ);
        REQUIRE(file.read() == expected);
        REQUIRE(file);

        REQUIRE_THROWS_AS(TextFile<std::ifstream>(test_dir / "not_existing", IO::READ), Noa::Exception);
    }

    AND_WHEN("backup copy and backup move") {
        TextFile file(test_file2, IO::WRITE);
        file.write("number: 2");
        file.close();
        REQUIRE(file.size() == 9);

        // Backup copy: open an existing file in writing mode.
        fs::path test_file2_backup = test_file2.string() + '~';
        file.open(IO::WRITE | IO::READ);
        REQUIRE(file.isOpen());
        REQUIRE(file.read() == "number: 2");
        REQUIRE(OS::size(test_file2_backup) == file.size());

        OS::remove(test_file2_backup);

        // Backup move:  open an existing file in overwriting mode.
        file.open(test_file2, IO::WRITE);
        REQUIRE(file.isOpen());
        REQUIRE(OS::existsFile(test_file2_backup));
        REQUIRE(file.read().empty());
        REQUIRE(OS::size(test_file2_backup) == 9);
    }

    AND_THEN("getLine and fstream") {
        TextFile file(test_file2, IO::WRITE | IO::TRUNC);
        REQUIRE(file.isOpen());
        std::string str = "line1\nline2\nline3\nline4\n";
        file.write(str);
        file.close();
        REQUIRE(file);

        file.open(IO::READ);
        std::string line, expected;
        int count{0};
        while (file.getLine(line)) {
            ++count;
            REQUIRE(line == fmt::format("line{}", count));
        }
        REQUIRE(count == 4);
        REQUIRE(!file.bad());
    }

    OS::removeAll(test_dir);
}
