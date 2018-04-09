/***
 * © Tanner Schmidt 2018
 */

#include <vu/StringHelpers.h>

#include <cstdarg>
#include <sstream>

namespace vu {

std::vector<std::string> Split(const std::string s, const char delimeter) {

    std::vector<std::string> pieces;

    if (!s.empty()) {

        std::stringstream ss(s);

        while (ss.good()) {

            std::string piece;
            getline( ss, piece, delimeter );
            pieces.push_back(piece);

        }

    }

    return pieces;

}

const std::string StringFormat(const std::string format, ...) {
    int size = 100;
    std::string str;
    va_list ap;
    while (1) {
        str.resize(size);
        va_start(ap, format);
        int n = vsnprintf((char *)str.c_str(), size, format.c_str(), ap);
        va_end(ap);
        if (n > -1 && n < size) {
            str.resize(n);
            return str;
        }
        if (n > -1)
            size = n + 1;
        else
            size *= 2;
    }
    return str;
}


} // namespace vu
