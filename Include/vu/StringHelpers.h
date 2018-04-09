#pragma once

/***
 * © Tanner Schmidt 2018
 */

#include <string>
#include <vector>

namespace vu {

std::vector<std::string> Split(const std::string s, const char delimeter);

const std::string StringFormat(const std::string format, ...);

} // namespace vu
