#ifdef HAS_PANGOLIN

#include <vu/PangolinHelpers.h>

#include <pangolin/pangolin.h>

namespace vu {

template <typename Scalar>
pangolin::OpenGlMatrixSpec ProjectionMatrixRDF_TopLeft(const CameraBase<Scalar> & camera,
                                                       const Scalar zNear, const Scalar zFar,
                                                       const bool flipY) {

    return pangolin::ProjectionMatrixRDF_TopLeft(camera.Width(),camera.Height(),
                                                 camera.Params()[0],
                                                 flipY ? -camera.Params()[1] : camera.Params()[1],
                                                 camera.Params()[2]+0.5,
                                                 flipY ? (camera.Height()-(camera.Params()[3]+0.5)) : camera.Params()[3] + 0.5,
                                                 zNear, zFar);

}

#define PROJECTION_MATRIX_RDF_TOP_LEFT_EXPLICIT_INSTANTIATION(type)                           \
    template pangolin::OpenGlMatrixSpec ProjectionMatrixRDF_TopLeft(const CameraBase<type> &, \
                                                                    const type, const type, const bool)

PROJECTION_MATRIX_RDF_TOP_LEFT_EXPLICIT_INSTANTIATION(float);
PROJECTION_MATRIX_RDF_TOP_LEFT_EXPLICIT_INSTANTIATION(double);


template <typename Scalar>
pangolin::OpenGlMatrixSpec ProjectionMatrixRDF_BottomLeft(const CameraBase<Scalar> & camera,
                                                          const Scalar zNear, const Scalar zFar) {

    return pangolin::ProjectionMatrixRDF_BottomLeft(camera.Width(),camera.Height(),
                                                    camera.Params()[0],camera.Params()[1],
                                                    camera.Params()[2],camera.Params()[3],
                                                    zNear, zFar);

}

#define PROJECTION_MATRIX_RDF_BOTTOM_LEFT_EXPLICIT_INSTANTIATION(type)                           \
    template pangolin::OpenGlMatrixSpec ProjectionMatrixRDF_BottomLeft(const CameraBase<type> &, \
                                                                    const type, const type)

PROJECTION_MATRIX_RDF_BOTTOM_LEFT_EXPLICIT_INSTANTIATION(float);
PROJECTION_MATRIX_RDF_BOTTOM_LEFT_EXPLICIT_INSTANTIATION(double);


} // namespace vu

#endif // HAS_PANGOLIN