#include <catch2/catch.hpp>
#include "../../Helpers.h"

#include "noa/files/TextFile.h"

using namespace ::Noa;


TEST_CASE("TextFile:", "[noa][file]") {
    fs::path test_dir = "test_TextFile";
    fs::path test_file1 = test_dir / "file1.txt";
    fs::path test_file2 = test_dir / "subdir/file2.txt";
    OS::removeAll(test_dir);

    Noa::Flag<Errno> err{Errno::good};

    AND_WHEN("file should exists") {
        TextFile<std::ifstream> file;

        err = file.open(test_file2, std::ios::in);
        REQUIRE(!file.isOpen());
        REQUIRE(!file);
        REQUIRE(err == Errno::fail_open);
        REQUIRE(file.state() == Errno::fail_open);

        file.clear();
        err = file.open(test_file2, std::ios::in | std::ios::out);
        REQUIRE(!file.isOpen());
        REQUIRE(!file);
        REQUIRE(err == Errno::fail_open);
        REQUIRE(file.state() == Errno::fail_open);
    }

    AND_WHEN("creating a file and its parent path") {
        TextFile<std::ofstream> file;
        REQUIRE(!OS::existsFile(test_file2, err));
        err = file.open(test_file2, std::ios::out | std::ios::app);
        REQUIRE(file);
        REQUIRE(file.isOpen());
        REQUIRE(OS::existsFile(test_file2, err));
        REQUIRE_ERRNO_GOOD(err);

        err = file.close();
        REQUIRE(!file.isOpen());
        REQUIRE_ERRNO_GOOD(err);
    }

    AND_WHEN("write and toString") {
        TextFile file;
        file.open(test_file1, std::ios::app);
        file.write("Here are some arguments: {}, {} ", 123, 124);
        file.write("I'm about to close the file...");
        err = file.close();
        REQUIRE(file);
        REQUIRE_ERRNO_GOOD(file.state());

        // toString() needs the file stream to be opened.
        file.toString();
        REQUIRE(file.state() == Errno::fail_read);
        file.clear();

        std::string expected = "Here are some arguments: 123, 124 "
                               "I'm about to close the file...";
        REQUIRE(file.size() == expected.size());

        file.open(std::ios::in, true);
        REQUIRE(file.toString() == expected);
        REQUIRE(file);

        TextFile<std::ifstream> file1(test_dir / "not_existing", std::ios::in);
        REQUIRE(file1.state() == Errno::fail_open);
    }

    AND_WHEN("backup copy and backup move") {
        TextFile file(test_file2, std::ios::out);
        file.write("number: {}", 2);
        file.close();
        REQUIRE(file.size() == 9);

        // Backup copy: open an existing file in writing mode.
        fs::path test_file2_backup = test_file2.string() + '~';
        err = file.open(std::ios::out | std::ios::in, true);
        REQUIRE(file.isOpen());
        REQUIRE(file.toString() == "number: 2");
        REQUIRE(OS::size(test_file2_backup, err) == file.size());
        REQUIRE_ERRNO_GOOD(err);

        OS::remove(test_file2_backup);

        // Backup move:  open an existing file in overwriting mode.
        err = file.open(test_file2, std::ios::out);
        REQUIRE(file.isOpen());
        REQUIRE(OS::existsFile(test_file2_backup, err));
        REQUIRE(file.toString().empty());
        REQUIRE(OS::size(test_file2_backup, err) == 9);
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
