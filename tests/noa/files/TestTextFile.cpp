#include <catch2/catch.hpp>
#include "../../Helpers.h"

#include "noa/files/TextFile.h"

using namespace ::Noa;


SCENARIO("Files::Text:", "[noa][file]") {
    errno_t err{Errno::good};
    fs::path test_dir = "testTextFile";
    fs::path test_file = test_dir / "file1.txt";

    OS::removeAll(test_dir);

    AND_THEN("write and toString") {
        TextFile test;
        test.open(test_file, std::ios::out | std::ios::ate);
        test.write("Here are some arguments: {}, {}\n", 123, 124);
        test.write("I'm about to close the file...\n");
        err = test.close();
        REQUIRE_ERRNO_GOOD(err);

        // toString() needs the file stream to be opened.
        test.toString(err);
        REQUIRE(err == Errno::fail_read);

        err = Errno::good;
        test.open(std::ios::in, true);
        std::string str = test.toString(err);
        REQUIRE_ERRNO_GOOD(err);
        REQUIRE(str == std::string{"Here are some arguments: 123, 124\n"
                                   "I'm about to close the file...\n"});

        TextFile file("something");
        file.open(std::ios::out);
    }

    AND_THEN("getLine and fstream") {
        TextFile file(test_file, std::ios::out | std::ios::trunc);
        REQUIRE(file.isOpen());
        std::string str = "line1\nline2\nline3\nline4\n";
        file.fstream().write(str.data(), static_cast<std::streamsize>(str.size()));
        file.close();

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

    AND_WHEN("remove and rename") {
        TextFile file(test_file);
        err = file.remove(); // file doesn't exist, does nothing.
        REQUIRE_ERRNO_GOOD(err);

        file.open(std::ios::out);
        file.write("something...\n");

        fs::path new_file = test_dir / "file2.txt";
        file.rename(new_file, std::ios::in | std::ios::out);
        REQUIRE(file.isOpen());
        REQUIRE(OS::existsFile(new_file, err));
        REQUIRE_ERRNO_GOOD(err);
    }

    OS::removeAll(test_dir);
}
