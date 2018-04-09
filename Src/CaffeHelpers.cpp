/***
 * © Tanner Schmidt 2018
 */

#ifdef HAS_CAFFE

#include <vu/CaffeHelpers.h>
#include <vu/FileHelpers.h>

#include <regex>

namespace vu {

void InitCaffe(char * executableName) {

    char * arg = executableName; //buffer.data();
    char * * argv = &arg;
    int argc = 1;
    caffe::GlobalInit(&argc,&argv);

}

void CaffeSwizzle(const NDT::ConstVolume<unsigned char> & source,
                  NDT::Volume<float> & destination) {

    for (int c = 0; c < destination.DimensionSize(2); ++c) {
        for (int y = 0; y < destination.DimensionSize(1); ++y) {
            for (int x = 0; x < destination.DimensionSize(0); ++x) {
                destination(x,y,c) = source(c,x,y) / 255.f;
            }
        }
    }

}

int GetLatestSnapshot(const std::string & experimentDir) {

    int latestSnap = 0;
    std::vector<std::string> experimentDirContents = GetDirectoryContents(experimentDir);
    std::regex snapRegex("snap_iter_(\\d+).caffemodel");
    for (std::string & content : experimentDirContents) {
        std::smatch match;
        if (std::regex_search(content,match,snapRegex)) {
            latestSnap = std::max(latestSnap,atoi(match.str(1).c_str()));
        }
    }
    return latestSnap;

}


} // namespace vu

#endif // HAS_CAFFE