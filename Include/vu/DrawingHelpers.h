#pragma once

/***
 * © Tanner Schmidt 2018
 */

#include <NDT/Tensor.h>

#include <pangolin/display/view.h>

#include <GL/glew.h>

#include <vu/EigenHelpers.h>

namespace vu {

void DrawPoints(const NDT::Image<Vec3<float> > & points, const GLuint mode = GL_POINTS);

void DrawPoints(const NDT::Image<Vec3<float> > & points,
                const NDT::Image<Vec3<float> > & normals);

void DrawPoints(const NDT::Vector<Vec3<float> > & points, const GLuint mode = GL_POINTS);

template <int D>
void DrawPoints(const NDT::Image<Vec<D, float> > & points,
                const NDT::Image<Vec3<unsigned char> > & colors);

template <int D>
void DrawPoints(const NDT::Vector<Vec<D, float> > & points,
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

void ActivateImageCoordinates(pangolin::View & view, const Vec2<int> & imageDims);

template <typename T, NDT::Residency R>
void ActivateImageCoordinates(pangolin::View & view, const NDT::Tensor<2,T,R> & image) {

    ActivateImageCoordinates(view, image.Dimensions().template cast<int>());

}

float OscillatingValue(const float d0, const float amplitude, const float frequency = 1.f);

void GlOscillatingPointSize(const float d0, const float amplitude, const float frequency = 1.f);

} // namespace vu