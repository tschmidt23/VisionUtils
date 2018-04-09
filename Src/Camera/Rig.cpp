/***
 * © Tanner Schmidt 2018
 */

#include <vu/Camera/Rig.h>

#include <vu/Camera/CameraFactory.h>

namespace vu {

template <typename T>
inline Sophus::SE3<T> PoseFromJson(const picojson::value & poseSpec) {

    if (poseSpec.size() != 3) {
        throw std::runtime_error("pose spec must have 3 rows");
    }

    Eigen::Matrix<T,4,4> M = Eigen::Matrix<T,4,4>::Identity();
    for (int r = 0; r < 3; ++r) {
        const picojson::value & rowSpec = poseSpec[r];
        if (rowSpec.size() != 4) {
            throw std::runtime_error("each row in pose spec must have 4 values");
        }
        for (int c = 0; c < 4; ++c) {
            M(r,c) = atof(rowSpec[c].to_str().c_str());
        }
    }

    // orthonormalization
    Eigen::Matrix<T,3,3> R = M.template block<3,3>(0,0);
    Eigen::AngleAxis<T> aa(R);
    M.template block<3,3>(0,0) = aa.toRotationMatrix();

    return Sophus::SE3<T>(M);

}

template <typename T>
Rig<T>::Rig(const picojson::value & rigSpec) {

    const picojson::value & camsSpec = rigSpec["camera"];

    const std::size_t nCameras = camsSpec.size();

    cameras_.resize(nCameras);
    transformsCameraToRig_.resize(nCameras);

    CameraFactory<T> & cameraFactory = CameraFactory<T>::Instance();

    for (std::size_t i = 0; i < nCameras; ++i) {

        const picojson::value & camSpec = camsSpec[i];
        if (!camSpec.contains("camera_model")) {
            throw std::runtime_error("camera spec does not contain a camera model");
        }

        cameras_[i].reset(cameraFactory.CreateCamera(camSpec["camera_model"]));

        if (!camSpec.contains("pose")) {
            throw std::runtime_error("camera spec does not contain a pose");
        }

        transformsCameraToRig_[i] = PoseFromJson<T>(camSpec["pose"]);

    }
}

template class Rig<float>;
template class Rig<double>;

} // namespace vu
