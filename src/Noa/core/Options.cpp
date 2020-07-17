//
// Created by thomas on 12/07/2020.
//

#include <iostream>
#include "Options.h"

namespace Noa {

    /**
     * @short       This constructor only saves the parser. The goal of the subclasses constructors
     *              is to get and format all of their attributes. This parent class defines the
     *              getValues() methods, which greatly simplifies this task. Usually, attributes
     *              can and should be defined with a member initializer list, which is more efficient.
     *
     * @a_parser    Parser defined at the EntryPoint.
     *              It contains the parsed parameter (command line and parameter file).
     *
     * @example     class Subclass : Noa::Options {
     *              public:
     *                  std::string opt1;
     *                  std::vector<float> opt2;
     *                  bool opt3;
     *                  Example(Parser* a_parser) :
     *                      opt1(getValues("Input", "i", 1, 'S')),
     *                      opt2(getValues("Dimensions", "dim", 3, 'F', {2.4, 2.4, 2.4})),
     *                      opt3(getValues("Fast", "f", 1, 'B', true)) {};
     *              }
     */
    Options::Options(Parser* a_parser_ptr)
            : m_parser(a_parser_ptr),
              opt1(getValues<std::string>("Input", "i", 1, 'S', "File")),
              opt2(getValues<std::vector<int>>("Dimensions", "dim", 3, 'F', {2, 2, 2})),
              opt3(getValues<bool>("Fast", "f", 1, 'B', true)),
              opt4(getValues<std::vector<std::string>>("Names", "n", 2, 'S', {"thomas", "frosio"})) {}


}
