/***
 * © Tanner Schmidt 2018
 */

#include <vu/FileHelpers.h>

#include <dirent.h>

#define TOKEN_TO_STRING(token) # token
#define STRINGITIZE_TOKEN(token) TOKEN_TO_STRING(token)

namespace vu {


std::string CompileDirectory() {
    static std::string dir = STRINGITIZE_TOKEN(COMPILE_DIR);
    return dir;
}

std::vector<std::string> GetDirectoryContents(const std::string & dirName) {

    std::vector<std::string> contents;

    DIR * dir = opendir(dirName.c_str());
    if (dir != NULL) {

        struct dirent * ent;

        while ((ent = readdir(dir)) != NULL) {
            std::string name(ent->d_name);
            contents.push_back(name);
        }

    }

    closedir(dir);

    return contents;

}

} // namespace vu
