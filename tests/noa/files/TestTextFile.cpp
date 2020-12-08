/*
 * Test noa/files/Text.h
 */

#include <catch2/catch.hpp>
#include "noa/files/TextFile.h"


SCENARIO("Noa::Files::Text: check the basic functionalities work", "[noa][file]") {
    using namespace Noa;

    errno_t err{Errno::good};
    TextFile test("./test_text/TestText_1.txt", std::ios::out, true);
    test.write("Here are some arguments: {}, {}\n", 123, 124);
    test.write("I'm about to close the file...\n");
    test.close();
    test.toString(err);
    REQUIRE(err == Errno::fail_read);

    err = Errno::good;
    test.reopen(std::ios::in);
    std::string str = test.toString(err);
    REQUIRE(str == std::string{"Here are some arguments: 123, 124\n"
                               "I'm about to close the file...\n"});

    std::filesystem::remove_all("./test_text");
}
