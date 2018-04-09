/***
 * © Tanner Schmidt 2018
 */

#include <vu/Camera/Camera.h>

namespace vu {

template <typename T>
CameraBase<T>::CameraBase(const picojson::value & cameraSpec) {

    if (!cameraSpec.contains("width")) {
        throw std::runtime_error("camera spec does not give the width of the image");
    }

    if (!cameraSpec.contains("height")) {
        throw std::runtime_error("camera spec does not give the height of the camera");
    }

    const picojson::value & widthSpec = cameraSpec["width"];
    const picojson::value & heightSpec = cameraSpec["height"];

    width_ = atoi(widthSpec.to_str().c_str());
    height_ = atoi(heightSpec.to_str().c_str());

}

template class CameraBase<float>;
template class CameraBase<double>;

} // namespace vu
