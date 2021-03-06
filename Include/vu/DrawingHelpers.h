#pragma once

/***
 * © Tanner Schmidt 2018
 */

#include <NDT/Tensor.h>

#include <pangolin/display/view.h>

#include <GL/glew.h>

#include <Eigen/Geometry>

#include <sophus/se3.hpp>

#include <vu/EigenHelpers.h>


namespace vu {

template <int VecD, int TensorD, typename T>
void DrawPoints(const NDT::ConstTensor<TensorD, Vec<VecD, T> > & points, const GLuint mode = GL_POINTS);

void DrawPoints(const NDT::Image<Vec3<float> > & points,
                const NDT::Image<Vec3<float> > & normals,
                const GLuint mode = GL_POINTS);

//template <int D>
//void DrawPoints(const NDT::Vector<Vec<D, float> > & points, const GLuint mode = GL_POINTS);

void DrawPoints(const NDT::Vector<Vec3<float> > & points,
                const NDT::Vector<int> & indices,
                const GLuint mode = GL_POINTS);

void DrawPoints(const NDT::Vector<Vec3<float> > & points,
                const NDT::Vector<Vec3<float> > & normals,
                const NDT::Vector<int> & indices,
                const GLuint mode = GL_POINTS);

void DrawPoints(const NDT::ConstVector<Vec3<float> > & points,
                const NDT::ConstVector<Vec3<float> > & normals,
                const NDT::ConstVector<Vec3<unsigned char> > & colors,
                const NDT::ConstVector<int> & indices,
                const GLuint mode = GL_POINTS);

template <int D>
void DrawPoints(const NDT::Image<Vec<D, float> > & points,
                const NDT::Image<Vec3<unsigned char> > & colors,
                const GLuint mode = GL_POINTS);

template <int D>
void DrawPoints(const NDT::Vector<Vec<D, float> > & points,
                const NDT::Vector<Vec3<unsigned char> > & colors,
                const GLuint mode = GL_POINTS);

template <int D>
void DrawPoints(const NDT::Vector<Vec<D, float> > & points,
                const NDT::Vector<Vec3<unsigned char> > & colors,
                const NDT::Vector<int> & indices,
                const GLuint mode = GL_POINTS);

// TODO: I think this can be superceded by the above
void DrawPoints(const NDT::Vector<Vec3<float> > & points,
                const NDT::Vector<Vec3<unsigned char> > & colors,
                const NDT::Vector<Vec3<int> > & faces);

void DrawPoints(const NDT::Vector<Vec3<float> > & points,
                const NDT::Vector<Vec3<float> > & normals,
                const GLenum mode = GL_POINTS);

void DrawPoints(const NDT::Vector<Vec3<float> > & points,
                const NDT::Vector<Vec3<float> > & normals,
                const NDT::Vector<Vec3<unsigned char> > & colors);

void DrawPoints(const NDT::Vector<Vec3<float> > & points,
                const NDT::Vector<Vec3<float> > & normals,
                const NDT::Vector<Vec3<int> > & faces);

void ActivateImageCoordinates(pangolin::View & view, const Vec2<int> & imageDims);

template <typename T, NDT::Residency R>
void ActivateImageCoordinates(pangolin::View & view, const NDT::Tensor<2,T,R> & image) {

    ActivateImageCoordinates(view, image.Dimensions().template cast<int>());

}

void ActivateViewportCoordinates(pangolin::View & view, const Eigen::AlignedBox2f & viewport);

float OscillatingValue(const float d0, const float amplitude, const float frequency = 1.f);

void GlOscillatingPointSize(const float d0, const float amplitude, const float frequency = 1.f);

template <typename Scalar>
inline void GlMultMatrix(const Sophus::SE3<Scalar> & T) {
    glMultMatrixf(T.matrix().transpose().data());
}

} // namespace vu
