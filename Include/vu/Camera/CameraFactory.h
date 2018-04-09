#pragma once

/***
 * © Tanner Schmidt 2018
 */


#include <functional>
#include <map>

#include <pangolin/utils/picojson.h>

#include <vu/Camera/Camera.h>

namespace vu {

template <typename T>
class CameraFactory {
public:

    static CameraFactory & Instance();

    typedef std::function<CameraBase<T> *(const picojson::value &)> CameraCreator;

    CameraBase<T> * CreateCamera(const picojson::value & cameraSpec);

    void RegisterCameraCreator(const std::string name, CameraCreator creator);

private:

    CameraFactory() { }
    CameraFactory(const CameraFactory &);
    CameraFactory & operator=(const CameraFactory &);
    ~CameraFactory() { }

    std::map<std::string,CameraCreator> cameraCreators_;

};

namespace internal {

template <typename T>
struct CameraModelRegistration {
CameraModelRegistration(const std::string name,
                        typename CameraFactory<T>::CameraCreator creator) {
    CameraFactory<T>::Instance().RegisterCameraCreator(name,creator);
}
};

#define REGISTER_CAMERA_MODEL(name)                                                                                        \
    template <typename T>                                                                                                  \
    CameraBase<T> * Create##name##CameraModel(const picojson::value & cameraSpec) {                                  \
        return new Camera<name##CameraModel,T>(cameraSpec);                                                                \
    }                                                                                                                      \
    static internal::CameraModelRegistration<float> name##CameraRegistration_f(#name, Create##name##CameraModel<float>);   \
    static internal::CameraModelRegistration<double> name##CameraRegistration_d(#name, Create##name##CameraModel<double>)

} // namespace internal

} // namespace vu
