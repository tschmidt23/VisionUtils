#pragma once

/***
 * © Tanner Schmidt 2018
 */

#include <pangolin/utils/picojson.h>
#include <vu/Camera/CameraModel.h>
#include <Eigen/Core>
#include <string>

namespace vu {

template <typename T>
class CameraBase {
public:

    CameraBase(const picojson::value & cameraSpec);

    CameraBase(const int width, const int height)
        : width_(width), height_(height) { }

    inline unsigned int Width() const { return width_; }

    inline unsigned int Height() const { return height_; }

    virtual const T * Params() const = 0;

    virtual std::size_t NumParams() const = 0;

    virtual std::string ModelName() const = 0;

    virtual Eigen::Matrix<T,2,1> Project(const Eigen::Matrix<T,3,1> point3d) const = 0;

    virtual Eigen::Matrix<T,3,1> Unproject(const Eigen::Matrix<T,2,1> point2d, const T depth) const = 0;

private:

    int width_;
    int height_;

};

template <template <typename> class ModelT, typename T>
class Camera : public CameraBase<T> {
public:

    Camera(const picojson::value & cameraSpec)
        : CameraBase<T>(cameraSpec), model_(cameraSpec) { }

    Camera(const int width, const int height,
           const ModelT<T> & model)
        : CameraBase<T>(width,height), model_(model) { }

    template <typename U>
    Camera<ModelT,U> Cast() const {
        return Camera<ModelT,U>(this->width_, this->height_, model_.Cast<U>());
    }

    inline const T * Params() const override {
        return model_.Params();
    }

    inline std::size_t NumParams() const override {
        return internal::CameraModelTraits<ModelT>::NumParams;
    }

    inline std::string ModelName() const override {
        return model_.ModelName();
    }

    inline Eigen::Matrix<T,2,1> Project(const Eigen::Matrix<T,3,1> point3d) const override {
        return model_.Project(point3d);
    }

    inline Eigen::Matrix<T,3,1> Unproject(const Eigen::Matrix<T,2,1> point2d, const T depth) const override {
        return model_.Unproject(point2d,depth);
    }

    inline ModelT<T> & Model() { return model_; }

    inline const ModelT<T> & Model() const { return model_; }

private:
    ModelT<T> model_;
};


} // namespace vu
