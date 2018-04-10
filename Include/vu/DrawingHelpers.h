#pragma once

/***
 * © Tanner Schmidt 2018
 */

#include <NDT/Tensor.h>

#include <pangolin/display/view.h>

#include <GL/glew.h>

namespace vu {

template <typename Scalar>
using Vec2 = Eigen::Matrix<Scalar,2,1,Eigen::DontAlign>;

template <typename Scalar>
using Vec3 = Eigen::Matrix<Scalar,3,1,Eigen::DontAlign>;

template <typename Scalar>
using Vec4 = Eigen::Matrix<Scalar,4,1,Eigen::DontAlign>;


void DrawPoints(const NDT::Image<Vec3<float> > & points, const GLuint mode = GL_POINTS);

void DrawPoints(const NDT::Image<Vec3<float> > & points,
                const NDT::Image<Vec3<float> > & normals);

void DrawPoints(const NDT::Vector<Vec3<float> > & points, const GLuint mode = GL_POINTS);

template <int D>
void DrawPoints(const NDT::Image<Eigen::Matrix<float,D,1,Eigen::DontAlign> > & points,
                const NDT::Image<Vec3<unsigned char> > & colors);

void DrawPoints(const NDT::Vector<Vec3<float> > & points,
                const NDT::Vector<Vec3<unsigned char> > & colors);

void DrawPoints(const NDT::Vector<Vec3<float> > & points,
                const NDT::Vector<Vec3<float> > & normals,
                const GLenum mode = GL_POINTS);

void DrawPoints(const NDT::Vector<Vec3<float> > & points,
                const NDT::Vector<Vec3<float> > & normals,
                const NDT::Vector<Vec3<unsigned char> > & colors);

void DrawPoints(const NDT::Vector<Vec3<float> > & points,
                const NDT::Vector<Vec3<float> > & normals,
                const NDT::Vector<Vec3<int> > & faces);


template <typename T, NDT::Residency R>
void ActivateImageCoordinates(pangolin::View & view, const NDT::Tensor<2,T,R> & image) {

    view.ActivatePixelOrthographic();

    glScalef( view.GetBounds().w / static_cast<double>(image.Width()),
              -view.GetBounds().h / static_cast<double>(image.Height()),
              1);
    glTranslatef(0, -1.f * static_cast<int>(image.Height()), 0);

}

float OscillatingValue(const float d0, const float amplitude, const float frequency = 1.f);

void GlOscillatingPointSize(const float d0, const float amplitude, const float frequency = 1.f);

} // namespace vu