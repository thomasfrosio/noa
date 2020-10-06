/*
 * Test noa/utils/Traits.h
 */

#include <catch2/catch.hpp>
#include "noa/managers/Inputs.h"


SCENARIO("Inputs: create a basic session and retrieve user inputs") {
    using namespace Noa;

    GIVEN("a valid scenario") {
        WHEN("asking help") {
            std::array<std::vector<std::string>, 7> cmdline;
            cmdline[0] = {"./noa"};
            cmdline[1] = {"./noa", "--help"};
            cmdline[2] = {"./noa", "-h"};
            cmdline[3] = {"./noa", "help"};
            cmdline[4] = {"./noa", "h"};
            cmdline[5] = {"./noa", "-help"};
            cmdline[6] = {"./noa", "--h"};

            for (auto& e: cmdline) {
                InputManager im(e);
                auto& cmd = im.setCommand({"cmd1", "doc1"});
                REQUIRE(cmd == std::string("help"));
            }
        }

        WHEN("asking version") {
            std::array<std::vector<std::string>, 6> cmdline;
            cmdline[0] = {"./noa", "--version"};
            cmdline[1] = {"./noa", "-v"};
            cmdline[2] = {"./noa", "version"};
            cmdline[3] = {"./noa", "v"};
            cmdline[4] = {"./noa", "-version"};
            cmdline[5] = {"./noa", "--v"};

            for (auto& e: cmdline) {
                InputManager im(e);
                auto& cmd = im.setCommand({"cmd1", "doc1"});
                REQUIRE(cmd == std::string("version"));
            }
        }

        WHEN("asking for the command") {
            InputManager im(std::vector<std::string>{"./noa", "cmd1", "..."});
            auto& cmd = im.setCommand({"cmd1", "doc1"});
            REQUIRE(cmd == std::string("cmd1"));
        }

        WHEN("registering commands") {
            InputManager im(std::vector<std::string>{"./noa", "cmd1", "..."});
            auto& cmd1 = im.setCommand({"cmd1", "doc1"});
            REQUIRE(cmd1 == std::string("cmd1"));

            std::vector<std::string> a{"cmd1", "doc1", "cmd2", "doc2"};
            auto& cmd3 = im.setCommand(a);
            REQUIRE(cmd3 == std::string("cmd1"));

            InputManager im1(std::vector<std::string>{"./noa", "cmd2", "..."});
            std::vector<std::string> b{"cmd0", "doc0", "cmd1", "doc1", "cmd2", "doc2"};
            auto& cmd4 = im1.setCommand(b);
            REQUIRE(cmd4 == std::string("cmd2"));
        }

        WHEN("registering options") {
            // this is just to make sure no error are raised.
            InputManager im(std::vector<std::string>{"./noa", "cmd1", "..."});
            auto& cmd1 = im.setCommand({"cmd1", "doc1"});
            im.setOption({"opt1_longname", "opt1_shortname", "SI", "", "opt1_doc"});
            im.setOption({"opt1_longname", "opt1_shortname", "SI", "", "opt1_doc",
                          "opt2_longname", "opt2_shortname", "PF", "1.5,0.5", "opt2_doc"});
        }

        WHEN("parsing the command line and getting formatted values") {
            WHEN("no default values, long-names") {
                std::vector<std::string> cmdline{
                        "./exe", "command1",
                        "--option10", "value1",
                        "--option11", "value1,", "value2",
                        "--option12", "value1,value2,value3",
                        "--option13", "v1,", "v2", ",v3,", "v4,v5",

                        "--option20", "-1",
                        "--option21", "21,21",
                        "--option22", "-10,10,2",
                        "--option23", "-1",

                        "--option30", "0.3423",
                        "--option31", "2323", "231",
                        "--option32", "5.,.5", ".9e8",
                        "--option33", "-1,-0.5,0.555,1.5e-9", ",23,", "-232.12",

                        "--option40", "0",
                        "--option41", "1,", "true",
                        "--option42", "0,0,TRUE",
                        "--option43", "False",
                };
                InputManager im(cmdline);
                auto& cmd1 = im.setCommand({"command1", "doc1"});
                im.setOption({"option10", "opt10", "SS", "", "doc...",
                              "option11", "opt11", "PS", "", "doc...",
                              "option12", "opt12", "TS", "", "doc...",
                              "option13", "opt13", "AS", "", "doc...",

                              "option20", "opt20", "SI", "", "doc...",
                              "option21", "opt21", "PI", "", "doc...",
                              "option22", "opt22", "TI", "", "doc...",
                              "option23", "opt23", "AI", "", "doc...",

                              "option30", "opt30", "SF", "", "doc...",
                              "option31", "opt31", "PF", "", "doc...",
                              "option32", "opt32", "TF", "", "doc...",
                              "option33", "opt33", "AF", "", "doc...",

                              "option40", "opt40", "SB", "", "doc...",
                              "option41", "opt41", "PB", "", "doc...",
                              "option42", "opt42", "TB", "", "doc...",
                              "option43", "opt43", "AB", "", "doc...",
                             });
                REQUIRE(im.parse() == true);

                using str = std::string;
                REQUIRE(im.get<str>("option10") == "value1");
                REQUIRE(im.get<std::array<str, 2>, 2>("option11") ==
                        std::array<str, 2>{"value1", "value2"});
                REQUIRE(im.get<std::vector<str>, 3>("option12") ==
                        std::vector<str>{"value1", "value2", "value3"});
                REQUIRE(im.get<std::vector<str>, -1>("option13") ==
                        std::vector<str>{"v1", "v2", "v3", "v4", "v5"});

                using ll = long long;
                REQUIRE(im.get<long>("option20") == -1L);
                REQUIRE(im.get<std::vector<int>, 2>("option21") == std::vector<int>{21, 21});
                REQUIRE(im.get<std::vector<int>, 3>("option22") == std::vector<int>{-10, 10, 2});
                REQUIRE(im.get<std::array<ll, 1>, -1>("option23") == std::array<ll, 1>{-1LL});

                REQUIRE(im.get<float>("option30") == 0.3423f);
                REQUIRE(im.get<std::vector<double>, 2>("option31") ==
                        std::vector<double>{2323, 231});
                REQUIRE(im.get<std::vector<float>, 3>("option32") ==
                        std::vector<float>{5.f, .5f, .9e8f});
                REQUIRE(im.get<std::vector<float>, -1>("option33") ==
                        std::vector<float>{-1.f, -0.5f, 0.555f, 1.5e-9f, 23.f, -232.12f});

                REQUIRE(im.get<bool>("option40") == false);
                REQUIRE(im.get<std::array<bool, 2>, 2>("option41") ==
                        std::array<bool, 2>{true, true});
                REQUIRE(im.get<std::array<bool, 3>, 3>("option42") ==
                        std::array<bool, 3>{false, false, true});
                REQUIRE(im.get<std::vector<bool>, -1>("option43") ==
                        std::vector<bool>{false});
            }

            WHEN("with default values, short-names") {
                std::vector<std::string> cmdline{
                        "./exe", "my_command_test",
                        "-opt10", "v1",
                        "-opt11", "v1,",
                        "-opt12", "my file.txt", ",", ",", "something else.txt",
                        "-opt13", "v1,v2,v3,v4,v5",

                        "-opt21", ",21",
                        "-opt22", "-10,10,2",
                        "-opt23", "-1", ",1,2,-2,-10,1",

                        "-opt30", "0.3423",
                        "-opt31", "2323,",
                        "-opt32", "5.,", ".9e8", ",",

                        "-opt40", "0",
                        "-opt42", "0,,",
                };
                InputManager im(cmdline);
                auto& cmd1 = im.setCommand({"my_command_test", "doc_test..."});
                im.setOption({"option10", "opt10", "SS", "d1", "doc...",
                              "option11", "opt11", "PS", "d1,d2", "doc...",
                              "option12", "opt12", "TS", "file1.txt, file2.txt,file3.txt", "doc...",
                              "option13", "opt13", "AS", "d1,d2,d3,d4,d5,d6,d7", "doc...",

                              "option20", "opt20", "SI", "-100", "doc...",
                              "option21", "opt21", "PI", "-101,101", "doc...",
                              "option22", "opt22", "TI", "1,1,1", "doc...",
                              "option23", "opt23", "AI", "123,123,123", "doc...",

                              "option30", "opt30", "SF", ".55", "doc...",
                              "option31", "opt31", "PF", "1000,1001", "doc...",
                              "option32", "opt32", "TF", "1.2,1.3,1.4", "doc...",
                              "option33", "opt33", "AF", ".1,.2,.3", "doc...",

                              "option40", "opt40", "SB", "true", "doc...",
                              "option41", "opt41", "PB", "true,true", "doc...",
                              "option42", "opt42", "TB", "0,0,1", "doc...",
                              "option43", "opt43", "AB", "true", "doc...",
                             });
                REQUIRE(im.parse() == true);

                using str = std::string;
                REQUIRE(im.get<str>("option10") == "v1");
                REQUIRE(im.get<std::array<str, 2>, 2>("option11") ==
                        std::array<str, 2>{"v1", "d2"});
                REQUIRE(im.get<std::vector<str>, 3>("option12") ==
                        std::vector<str>{"my file.txt", "file2.txt", "something else.txt"});
                REQUIRE(im.get<std::vector<str>, -1>("option13") ==
                        std::vector<str>{"v1", "v2", "v3", "v4", "v5"});

                using ll = long long;
                REQUIRE(im.get<long>("option20") == -100L);
                REQUIRE(im.get<std::vector<int>, 2>("option21") == std::vector<int>{-101, 21});
                REQUIRE(im.get<std::vector<int>, 3>("option22") == std::vector<int>{-10, 10, 2});
                REQUIRE(im.get<std::vector<ll>, -1>("option23") ==
                        std::vector<ll>{-1LL, 1LL, 2LL, -2LL, -10LL, 1LL});

                REQUIRE(im.get<float>("option30") == 0.3423f);
                REQUIRE(im.get<std::vector<double>, 2>("option31") ==
                        std::vector<double>{2323, 1001});
                REQUIRE(im.get<std::array<float, 3>, 3>("option32") ==
                        std::array<float, 3>{5.f, .9e8f, 1.4f});
                REQUIRE(im.get<std::vector<float>, -1>("option33") ==
                        std::vector<float>{.1f, .2f, .3f});

                REQUIRE(im.get<bool>("option40") == false);
                REQUIRE(im.get<std::array<bool, 2>, 2>("option41") ==
                        std::array<bool, 2>{true, true});
                REQUIRE(im.get<std::vector<bool>, 3>("option42") ==
                        std::vector<bool>{false, false, true});
                REQUIRE(im.get<std::vector<bool>, -1>("option43") ==
                        std::vector<bool>{true});
            }
        }
    }

    GIVEN("an invalid scenario") {
        WHEN("registering not correctly formatted commands should raise an error") {
            InputManager im(std::vector<std::string>{"./noa", "cmd1", "..."});
            REQUIRE_THROWS_AS(im.setCommand({}), Noa::ErrorCore);
            REQUIRE_THROWS_AS(im.setCommand({"", "doc1"}), Noa::ErrorCore);
            REQUIRE_THROWS_AS(im.setCommand({"cmd1"}), Noa::ErrorCore);
            REQUIRE_THROWS_AS(im.setCommand({"cmd1", "doc1", "cmd2"}), Noa::ErrorCore);
        }
    }
}


