/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "CommandLineParser.hh"

void Analytical::CommandLineParser::parse(int argc, char** argv) noexcept(
    false) {
  values.clear();
  help_requested = false;
  for (int i = 1; i < argc; ++i) {
    std::string token(argv[i]);
    if (token.rfind("--", 0) != 0) {
      throw ParsingError("unexpected positional argument: " + token);
    }
    token.erase(0, 2);
    std::string name = token;
    std::vector<std::string> option_values;
    auto equal = token.find('=');
    if (equal != std::string::npos) {
      name = token.substr(0, equal);
      option_values.push_back(token.substr(equal + 1));
    }
    if (options.find(name) == options.end()) {
      throw ParsingError("unrecognized option: --" + name);
    }
    if (option_values.empty()) {
      while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        option_values.emplace_back(argv[++i]);
      }
      if (option_values.empty()) option_values.push_back("true");
    }
    values[name] = std::move(option_values);
    if (name == "help") help_requested = true;
  }
}

void Analytical::CommandLineParser::print_help_message_if_required()
    const noexcept {
  if (help_requested) {
    std::cout << "Command Line Options:\n";
    for (const auto& [name, option] : options) {
      std::cout << "  --" << name << "\t" << option.explanation << "\n";
    }
    exit(0);
  }
}
