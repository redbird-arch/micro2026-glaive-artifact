/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __COMMANDLINEPARSER_HH__
#define __COMMANDLINEPARSER_HH__

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace Analytical {
class CommandLineParser {
 public:
  class ParsingError : public std::runtime_error {
   public:
    explicit ParsingError(const std::string& message)
        : std::runtime_error(message) {}
  };

  CommandLineParser() noexcept { add_option("help", "Prints help message"); }

  void parse(int argc, char** argv) noexcept(false);

  void print_help_message_if_required() const noexcept;

  template <typename T>
  void add_command_line_option(
      const char* name,
      const char* explanation) noexcept {
    add_option(name, explanation);
  }

  template <typename T>
  void add_command_line_multitoken_option(
      const char* name,
      const char* explanation) noexcept {
    add_option(name, explanation);
  }

  template <typename T>
  void set_if_defined(const char* arg_name, T* target_var) const {
    auto it = values.find(arg_name);
    if (it != values.end()) {
      *target_var = parse_value<T>(it->second);
    }
  }

 private:
  struct Option {
    std::string explanation;
  };

  std::map<std::string, Option> options;
  std::map<std::string, std::vector<std::string>> values;
  bool help_requested = false;

  void add_option(const char* name, const char* explanation) noexcept {
    options.emplace(name, Option{explanation});
  }

  template <typename T>
  static T parse_scalar(const std::string& value) {
    if constexpr (std::is_same_v<T, std::string>) {
      return value;
    } else if constexpr (std::is_same_v<T, bool>) {
      return value == "1" || value == "true" || value == "on" ||
          value == "yes";
    } else if constexpr (std::is_same_v<T, int>) {
      return std::stoi(value);
    } else if constexpr (std::is_same_v<T, float>) {
      return std::stof(value);
    } else if constexpr (std::is_same_v<T, double>) {
      return std::stod(value);
    } else {
      static_assert(!sizeof(T), "Unsupported command-line scalar type");
    }
  }

  template <typename T>
  static T parse_value(const std::vector<std::string>& raw) {
    if constexpr (std::is_same_v<T, std::vector<int>>) {
      T result;
      for (const auto& value : raw) result.push_back(parse_scalar<int>(value));
      return result;
    } else if constexpr (std::is_same_v<T, std::vector<double>>) {
      T result;
      for (const auto& value : raw) result.push_back(parse_scalar<double>(value));
      return result;
    } else {
      if (raw.empty()) throw ParsingError("missing value");
      return parse_scalar<T>(raw.front());
    }
  }
};
} // namespace Analytical

#endif
