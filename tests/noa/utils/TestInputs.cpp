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
            im.setOption({"opt1_longname", "opt1_shortname", "1I", "", "opt1_doc"});
            im.setOption({"opt1_longname", "opt1_shortname", "1I", "", "opt1_doc",
                          "opt2_longname", "opt2_shortname", "2F", "1.5,0.5", "opt2_doc"});
        }

        WHEN("parsing the command line and getting formatted values") {
            WHEN("no default values, long-names") {
                std::vector<std::string> cmdline{
                        "./exe", "command1",
                        "--option10", "value1",
                        "--option11", "value1,", "value2,value3,value4", ",value5",
                        "--option12", "value1,value2,value3",
                        "--option13", "v1,", "v2", ",v3,", "v4,v5",

                        "--option20", "-1",
                        "--option21", "21,21,-1,2",
                        "--option22", "-10,10,2",
                        "--option23", "-1",

                        "--option30", "0.3423", ",", "-0.23", ",", ".13", ",", "0.2e10", ",-.5,0.2",
                        "--option31", "2323", "231",
                        "--option32", "5.,.5", ".9e8",
                        "--option33", "-1,-0.5,0.555,1.5e-9", ",23,", "-232.12",

                        "--option40", "0,n,0,y,1,n,1,1,y",
                        "--option41", "1,", "true", "1,", "false,0",
                        "--option42", "0,0,TRUE",
                        "--option43", "False",
                };
                InputManager im(cmdline);
                auto& cmd1 = im.setCommand({"command1", "doc1"});
                im.setOption({"option10", "opt10", "1S", "", "doc...",
                              "option11", "opt11", "5S", "", "doc...",
                              "option12", "opt12", "3S", "", "doc...",
                              "option13", "opt13", "0S", "", "doc...",

                              "option20", "opt20", "1I", "", "doc...",
                              "option21", "opt21", "4I", "", "doc...",
                              "option22", "opt22", "3I", "", "doc...",
                              "option23", "opt23", "0I", "", "doc...",

                              "option30", "opt30", "6F", "", "doc...",
                              "option31", "opt31", "2F", "", "doc...",
                              "option32", "opt32", "3F", "", "doc...",
                              "option33", "opt33", "0F", "", "doc...",

                              "option40", "opt40", "9B", "", "doc...",
                              "option41", "opt41", "5B", "", "doc...",
                              "option42", "opt42", "3B", "", "doc...",
                              "option43", "opt43", "0B", "", "doc...",
                             });
                REQUIRE(im.parse() == true);

                using str = std::string;
                REQUIRE(im.get<str>("option10") == "value1");
                REQUIRE(im.get<std::array<str, 5>, 5>("option11") ==
                        std::array<str, 5>{"value1", "value2", "value3", "value4", "value5"});
                REQUIRE(im.get<std::vector<str>, 3>("option12") ==
                        std::vector<str>{"value1", "value2", "value3"});
                REQUIRE(im.get<std::vector<str>, 0>("option13") ==
                        std::vector<str>{"v1", "v2", "v3", "v4", "v5"});

                using ll = long long;
                REQUIRE(im.get<long>("option20") == -1L);
                REQUIRE(im.get<std::vector<int>, 4>("option21") == std::vector<int>{21, 21, -1, 2});
                REQUIRE(im.get<std::vector<int>, 3>("option22") == std::vector<int>{-10, 10, 2});
                REQUIRE(im.get<std::vector<ll>, 0>("option23") == std::vector<ll>{-1LL});

                REQUIRE(im.get<std::array<float, 6>, 6>("option30") ==
                        std::array<float, 6>{0.3423f, -0.23f, .13f, 0.2e10f, -.5f, 0.2f});
                REQUIRE(im.get<std::vector<double>, 2>("option31") ==
                        std::vector<double>{2323, 231});
                REQUIRE(im.get<std::vector<float>, 3>("option32") ==
                        std::vector<float>{5.f, .5f, .9e8f});
                REQUIRE(im.get<std::vector<float>, 0>("option33") ==
                        std::vector<float>{-1.f, -0.5f, 0.555f, 1.5e-9f, 23.f, -232.12f});

                REQUIRE(im.get<std::vector<bool>, 9>("option40") ==
                        std::vector<bool>{0, 0, 0, 1, 1, 0, 1, 1, 1});
                REQUIRE(im.get<std::array<bool, 5>, 5>("option41") ==
                        std::array<bool, 5>{1, 1, 1, 0, 0});
                REQUIRE(im.get<std::array<bool, 3>, 3>("option42") ==
                        std::array<bool, 3>{false, false, true});
                REQUIRE(im.get<std::vector<bool>, 0>("option43") ==
                        std::vector<bool>{false});
            }

            WHEN("with default values, short-names") {
                std::vector<std::string> cmdline{
                        "./exe", "my_command_test",
                        "-opt10", "v1",
                        "--opt11", "v1,",
                        "-opt12", "my file.txt", ",", ",", "something else.txt",
                        "-opt13", "v1,v2,v3,v4,v5",

                        "-opt21", ",21",
                        "--opt22", "-10,10,2",
                        "-opt23", "-1", ",1,2,-2,-10,1",

                        "-opt30", "0.3423",
                        "-opt31", "2323,",
                        "--opt32", "5.,", ".9e8", ",",

                        "-opt40", "0",
                        "-opt42", "0,,",
                };
                InputManager im(cmdline);
                auto& cmd1 = im.setCommand({"my_command_test", "doc_test..."});
                im.setOption({"option10", "opt10", "1S", "d1", "doc...",
                              "option11", "opt11", "2S", "d1,d2", "doc...",
                              "option12", "opt12", "3S", "file1.txt, file2.txt,file3.txt", "doc...",
                              "option13", "opt13", "0S", "d1,d2,d3,d4,d5,d6,d7", "doc...",

                              "option20", "opt20", "1I", "-100", "doc...",
                              "option21", "opt21", "2I", "-101,", "doc...",
                              "option22", "opt22", "3I", "1,1,1", "doc...",
                              "option23", "opt23", "0I", "123,123,123", "doc...",

                              "option30", "opt30", "1F", ".55", "doc...",
                              "option31", "opt31", "2F", ",1001", "doc...",
                              "option32", "opt32", "3F", ",1.3,1.4", "doc...",
                              "option33", "opt33", "0F", ".1,.2,.3", "doc...",

                              "option40", "opt40", "1B", "true", "doc...",
                              "option41", "opt41", "2B", "true,true", "doc...",
                              "option42", "opt42", "3B", "1,0,1", "doc...",
                              "option43", "opt43", "0B", "true", "doc...",
                             });
                REQUIRE(im.parse() == true);

                using str = std::string;
                REQUIRE(im.get<str>("option10") == "v1");
                REQUIRE(im.get<std::array<str, 2>, 2>("option11") ==
                        std::array<str, 2>{"v1", "d2"});
                REQUIRE(im.get<std::vector<str>, 3>("option12") ==
                        std::vector<str>{"my file.txt", "file2.txt", "something else.txt"});
                REQUIRE(im.get<std::vector<str>, 0>("option13") ==
                        std::vector<str>{"v1", "v2", "v3", "v4", "v5"});

                using ll = long long;
                REQUIRE(im.get<long>("option20") == -100L);
                REQUIRE(im.get<std::vector<int>, 2>("option21") == std::vector<int>{-101, 21});
                REQUIRE(im.get<std::vector<int>, 3>("option22") == std::vector<int>{-10, 10, 2});
                REQUIRE(im.get<std::vector<ll>, 0>("option23") ==
                        std::vector<ll>{-1LL, 1LL, 2LL, -2LL, -10LL, 1LL});

                REQUIRE(im.get<float>("option30") == 0.3423f);
                REQUIRE(im.get<std::vector<double>, 2>("option31") ==
                        std::vector<double>{2323, 1001});
                REQUIRE(im.get<std::array<float, 3>, 3>("option32") ==
                        std::array<float, 3>{5.f, .9e8f, 1.4f});
                REQUIRE(im.get<std::vector<float>, 0>("option33") ==
                        std::vector<float>{.1f, .2f, .3f});

                REQUIRE(im.get<bool>("option40") == false);
                REQUIRE(im.get<std::array<bool, 2>, 2>("option41") ==
                        std::array<bool, 2>{true, true});
                REQUIRE(im.get<std::vector<bool>, 3>("option42") ==
                        std::vector<bool>{false, false, true});
                REQUIRE(im.get<std::vector<bool>, 0>("option43") ==
                        std::vector<bool>{true});
            }
        }
    }

    GIVEN("an invalid scenario") {
        WHEN("command line format isn't valid, raise an error") {
            std::vector<std::vector<std::string>> cmdlines{
                    {"./noa", "cmd1", "-opt1"},
                    {"./noa", "cmd1", "--opt1"},
                    {"./noa", "cmd1", "-opt1", "--opt2"},
                    {"./noa", "cmd1", "-opt1", "1", "-opt1"},
                    {"./noa", "cmd1", "-opt1", "1", "--opt1", "1"},
            };
            for (auto& e: cmdlines) {
                InputManager im(e);
                im.setCommand({"cmd1", "doc..."});
                REQUIRE_THROWS_AS(im.parse(), Noa::ErrorCore);
            }
        }

        WHEN("registering not correctly formatted commands should raise an error") {
            InputManager im(std::vector<std::string>{"./noa", "cmd1", "..."});
            REQUIRE_THROWS_AS(im.setCommand({}), Noa::ErrorCore);
            REQUIRE_THROWS_AS(im.setCommand({"", "doc1"}), Noa::ErrorCore);
            REQUIRE_THROWS_AS(im.setCommand({"cmd1"}), Noa::ErrorCore);
            REQUIRE_THROWS_AS(im.setCommand({"cmd1", "doc1", "cmd2"}), Noa::ErrorCore);
            REQUIRE_THROWS_AS(im.setCommand({"cmd10", "doc10", "cmd2", "doc2"}), Noa::ErrorCore);
        }

        WHEN("registering not correctly formatted options should raise an error") {
            InputManager im(std::vector<std::string>{"./noa", "cmd1",
                                                     "-opt11", "1",
                                                     "-opt12", "1",
                                                     "-opt13", "1",
                                                     "-opt14", "1"});
            auto& cmd1 = im.setCommand({"cmd1", "doc_test..."});

            REQUIRE_THROWS_AS(im.setOption({"option1", "opt1", "1S", ""}), Noa::ErrorCore);
            REQUIRE_THROWS_AS(im.setOption({"option1"}), Noa::ErrorCore);
            REQUIRE_THROWS_AS(im.setOption({"", "", "", "", "", ""}), Noa::ErrorCore);
            REQUIRE_THROWS_AS(im.setOption({"option1", "opt1"}), Noa::ErrorCore);

            WHEN("retrieving options that were set with invalid type") {
                InputManager im1(std::vector<std::string>{"./noa", "cmd1",
                                                          "-opt11", "1",
                                                          "-opt12", "1",
                                                          "-opt13", "1",
                                                          "-opt14", "1"});
                im1.setCommand({"cmd1", "doc_test..."});
                im1.setOption({"option11", "opt11", "10S", "", "doc...",
                               "option12", "opt12", "", "", "doc...",
                               "option13", "opt13", "ss", "", "doc...",
                               "option14", "opt14", "pb", "", "doc..."});
                REQUIRE(im1.parse() == true);
                REQUIRE_THROWS_AS(im1.get<bool>("option11"), Noa::ErrorCore);
                REQUIRE_THROWS_AS(im1.get<int>("option12"), Noa::ErrorCore);
                REQUIRE_THROWS_AS(im1.get<float>("option13"), Noa::ErrorCore);
                REQUIRE_THROWS_AS(im1.get<std::string>("option14"), Noa::ErrorCore);
            }

            WHEN("retrieving options but asking for the wrong type") {
                InputManager im2(std::vector<std::string>{"./noa", "cmd1",
                                                          "-opt21", "1",
                                                          "-opt22", "1",
                                                          "-opt23", "1",
                                                          "-opt24", "1"});
                im2.setCommand({"cmd1", "doc_test..."});
                im2.setOption({"option21", "opt21", "1S", "", "doc...",
                               "option22", "opt22", "2B", "0,0", "doc...",
                               "option23", "opt23", "3I", "10,11,12", "doc...",
                               "option24", "opt24", "0F", "", "doc..."});
                REQUIRE(im2.parse() == true);
                REQUIRE_THROWS_AS((im2.get<int>("option21")), Noa::ErrorCore);
                REQUIRE_THROWS_AS(im2.get<bool>("option22"), Noa::ErrorCore);
                REQUIRE_THROWS_AS((im2.get<std::vector<float>, 2>("option23")), Noa::ErrorCore);
                REQUIRE_THROWS_AS((im2.get<std::vector<int>, 0>("option24")), Noa::ErrorCore);
            }
        }

        WHEN("retrieving options that are not registered") {
            InputManager im(std::vector<std::string>{"./noa", "cmd1", "--opt1", "1"});
            im.setCommand({"cmd0", "doc0", "cmd1", "doc1"});
            im.setOption({"option1", "opt1", "1S", "", "doc...",
                          "option2", "opt2", "2S", "", "doc..."});
            REQUIRE(im.parse() == true);
            REQUIRE_THROWS_AS((im.get<int>("option11")), Noa::ErrorCore);
            REQUIRE_THROWS_AS((im.get<int>("Option1")), Noa::ErrorCore);
            REQUIRE_THROWS_AS((im.get<bool>("")), Noa::ErrorCore);
            REQUIRE_THROWS_AS((im.get<int>("opt11")), Noa::ErrorCore);
        }

        WHEN("retrieve options that were not specified and don't have a default value") {
            InputManager im(std::vector<std::string>{"./noa", "cmd1",
                                                     "--opt3", "1,",
                                                     "-opt2", "1,,3",
                                                     "-opt7", "1.,2.,3.,",
                                                     "--option9", "1,1,1,,,"});
            im.setCommand({"cmd0", "doc0", "cmd1", "doc1"});
            im.setOption({"option1", "opt1", "1I", "", "doc...",
                          "option2", "opt2", "3I", "", "doc...",
                          "option3", "opt3", "0F", "", "doc...",
                          "option4", "opt4", "1S", "", "doc...",
                          "option5", "opt5", "2F", "5,", "doc...",
                          "option6", "opt6", "3B", "", "doc...",
                          "option7", "opt7", "0F", ".1,.2f,-.03,12.4", "doc...",
                          "option8", "opt8", "5S", "v1,v2,,v4,v5", "doc...",
                          "option9", "opt9", "6B", "1,1,false,false,,", "doc...",
                          "option10", "opt10", "7I", "-1,-2,-3,-4,,,", "doc..."});
            REQUIRE(im.parse() == true);
            REQUIRE_THROWS_AS((im.get<int>("option1")), Noa::ErrorCore);
            REQUIRE_THROWS_AS((im.get<std::array<int, 3>, 3>("option2")), Noa::ErrorCore);
            REQUIRE_THROWS_AS((im.get<std::vector<int>, 0>("option3")), Noa::ErrorCore);
            REQUIRE_THROWS_AS((im.get<std::string>("option4")), Noa::ErrorCore);
            REQUIRE_THROWS_AS((im.get<std::array<double, 2>, 2>("option5")), Noa::ErrorCore);
            REQUIRE_THROWS_AS((im.get<std::array<bool, 3>, 3>("option6")), Noa::ErrorCore);
            REQUIRE_THROWS_AS((im.get<std::vector<bool>, 0>("option7")), Noa::ErrorCore);
            REQUIRE_THROWS_AS((im.get<std::vector<std::string>, 5>("option8")), Noa::ErrorCore);
            REQUIRE_THROWS_AS((im.get<std::array<bool, 6>, 6>("option9")), Noa::ErrorCore);
            REQUIRE_THROWS_AS((im.get<std::vector<long>, 7>("option10")), Noa::ErrorCore);
        }
    }
}


