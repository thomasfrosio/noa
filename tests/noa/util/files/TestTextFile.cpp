#include <catch2/catch.hpp>
#include "../../../Helpers.h"

#include "noa/util/files/TextFile.h"

using namespace ::Noa;


TEST_CASE("TextFile:", "[noa][file]") {
    fs::path test_dir = "test_TextFile";
    fs::path test_file1 = test_dir / "file1.txt";
    fs::path test_file2 = test_dir / "subdir/file2.txt";
    OS::removeAll(test_dir);

    Errno err{Errno::good};

    AND_WHEN("file should exists") {
        TextFile<std::ifstream> file;

        REQUIRE_THROWS_AS(file.open(test_file2, std::ios::in), Noa::Exception);
        REQUIRE(!file.isOpen());
        REQUIRE(!file);

        file.clear();
        REQUIRE_THROWS_AS(file.open(test_file2, std::ios::in | std::ios::out), Noa::Exception);
        REQUIRE(!file.isOpen());
        REQUIRE(!file);
    }

    AND_WHEN("creating a file and its parent path") {
        TextFile<std::ofstream> file;
        REQUIRE(!OS::existsFile(test_file2));
        file.open(test_file2, std::ios::out | std::ios::app);
        REQUIRE(file);
        REQUIRE(file.isOpen());
        REQUIRE(OS::existsFile(test_file2));
        file.close();
        REQUIRE(!file.isOpen());
    }

    AND_WHEN("write and toString") {
        TextFile file;
        file.open(test_file1, std::ios::app);
        file.write("Here are some arguments: {}, {} ", 123, 124);
        file.write("I'm about to close the file...");
        file.close();
        REQUIRE(file);

        // toString() needs the file stream to be opened.
        REQUIRE_THROWS_AS(file.toString(), Noa::Exception);
        file.clear();

        std::string expected = "Here are some arguments: 123, 124 "
                               "I'm about to close the file...";
        REQUIRE(file.size() == expected.size());

        file.open(std::ios::in, true);
        REQUIRE(file.toString() == expected);
        REQUIRE(file);

        REQUIRE_THROWS_AS(TextFile<std::ifstream>(test_dir / "not_existing", std::ios::in), Noa::Exception);
    }

    AND_WHEN("backup copy and backup move") {
        TextFile file(test_file2, std::ios::out);
        file.write("number: {}", 2);
        file.close();
        REQUIRE(file.size() == 9);

        // Backup copy: open an existing file in writing mode.
        fs::path test_file2_backup = test_file2.string() + '~';
        file.open(std::ios::out | std::ios::in, true);
        REQUIRE(file.isOpen());
        REQUIRE(file.toString() == "number: 2");
        REQUIRE(OS::size(test_file2_backup) == file.size());

        OS::remove(test_file2_backup);

        // Backup move:  open an existing file in overwriting mode.
        file.open(test_file2, std::ios::out);
        REQUIRE(file.isOpen());
        REQUIRE(OS::existsFile(test_file2_backup));
        REQUIRE(file.toString().empty());
        REQUIRE(OS::size(test_file2_backup) == 9);
        REQUIRE_ERRNO_GOOD(err);
    }

    AND_THEN("getLine and fstream") {
        TextFile file(test_file2, std::ios::out | std::ios::trunc);
        REQUIRE(file.isOpen());
        std::string str = "line1\nline2\nline3\nline4\n";
        file.write(str);
        file.close();
        REQUIRE(file);

        file.open(std::ios::in);
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
