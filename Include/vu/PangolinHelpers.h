#pragma once

#include <vu/Camera/Camera.h>

#include <pangolin/pangolin.h>

namespace vu {

template <typename Scalar>
pangolin::OpenGlMatrixSpec ProjectionMatrixRDF_TopLeft(const CameraBase<Scalar> & camera,
                                                       const Scalar zNear, const Scalar zFar,
                                                       const bool flipY = true);

template <typename Scalar>
pangolin::OpenGlMatrixSpec ProjectionMatrixRDF_BottomLeft(const CameraBase<Scalar> & camera,
                                                          const Scalar zNear, const Scalar zFar);

} // namespace vu