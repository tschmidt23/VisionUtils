#pragma once

/***
 * © Tanner Schmidt 2018
 */

#include <memory>
#include <vector>
#include <pangolin/utils/picojson.h>
#include <vu/Camera/Camera.h>
#include <sophus/se3.hpp>

namespace vu {

template <typename T>
class Rig {
public:

    Rig(const picojson::value & rigSpec);

    const std::size_t NumCameras() const {
        return cameras_.size();
    }

    inline const CameraBase<T> & Camera(const std::size_t index) const {
        return *cameras_[index];
    }

    inline const Sophus::SE3<T> & TransformCameraToRig(const std::size_t index) const {
        return transformsCameraToRig_[index];
    }

private:

    std::vector<std::shared_ptr<CameraBase<T> > > cameras_;
    std::vector<Sophus::SE3<T> > transformsCameraToRig_;

};


} // namespace vu
