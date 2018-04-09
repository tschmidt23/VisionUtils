#pragma once

/***
 * © Tanner Schmidt 2018
 */

#include <string>
#include <vector>

namespace vu {

std::string CompileDirectory();

std::vector<std::string> GetDirectoryContents(const std::string & dirName);

} // namespace vu
