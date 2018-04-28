/***
 * © Tanner Schmidt 2018
 */

#include <vu/DrawingHelpers.h>

#include <GL/gl.h>
#include <pangolin/utils/timer.h>

namespace vu {

template <typename Derived>
struct DrawingTraits;

void ActivateImageCoordinates(pangolin::View & view, const Vec2<int> & imageDims) {

    view.ActivatePixelOrthographic();

    glScalef( view.GetBounds().w / static_cast<double>(imageDims(0)),
              -view.GetBounds().h / static_cast<double>(imageDims(1)),
              1);
    glTranslatef(0, -1.f * static_cast<int>(imageDims(1)), 0);

}

void DrawPoints(const NDT::Image<Vec3<float> > & points, const GLuint mode) {

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, points.Data());
    glDrawArrays(mode, 0, points.Count());
    glDisableClientState(GL_VERTEX_ARRAY);

}

void DrawPoints(const NDT::Image<Vec3<float> > & points,
                const NDT::Image<Vec3<float> > & normals) {

    assert(points.Count() == normals.Count());

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, points.Data());
    glEnableClientState(GL_NORMAL_ARRAY);
    glNormalPointer(GL_FLOAT, 0, normals.Data());
    glDrawArrays(GL_POINTS, 0, points.Count());
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

}

void DrawPoints(const NDT::Image<Vec3<float> > & points,
                const NDT::Image<Vec3<unsigned char> > & colors) {

    assert(points.Count() == colors.Count());

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(4, GL_FLOAT, 0, points.Data());
    glEnableClientState(GL_COLOR_ARRAY);
    glColorPointer(3, GL_UNSIGNED_BYTE, 0, colors.Data());
    glDrawArrays(GL_POINTS, 0, points.Count());
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

}

template <int D>
void DrawPoints(const NDT::Image<Vec<D, float> > & points,
                const NDT::Image<Vec3<unsigned char> > & colors) {

    assert(points.Count() == colors.Count());

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(D, GL_FLOAT, 0, points.Data());
    glEnableClientState(GL_COLOR_ARRAY);
    glColorPointer(3, GL_UNSIGNED_BYTE, 0, colors.Data());
    glDrawArrays(GL_POINTS, 0, points.Count());
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

}

template void DrawPoints(const NDT::Image<Vec3<float> > &, const NDT::Image<Vec3<unsigned char> > &);

template void DrawPoints(const NDT::Image<Vec4<float> > &, const NDT::Image<Vec3<unsigned char> > &);

void DrawPoints(const NDT::Vector<Vec3<float> > & points, const GLuint mode) {

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, points.Data());
    glDrawArrays(mode, 0, points.Count());
    glDisableClientState(GL_VERTEX_ARRAY);

}

template <int D>
void DrawPoints(const NDT::Vector<Vec<D, float> > & points,
                const NDT::Vector<Vec3<unsigned char> > & colors) {

    assert(points.Length() == colors.Length());

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(D, GL_FLOAT, 0, points.Data());
    glEnableClientState(GL_COLOR_ARRAY);
    glColorPointer(3, GL_UNSIGNED_BYTE, 0, colors.Data());
    glDrawArrays(GL_POINTS, 0, points.Count());
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

}

template void DrawPoints(const NDT::Vector<Vec2<float> > &, const NDT::Vector<Vec3<unsigned char> > &);

template void DrawPoints(const NDT::Vector<Vec3<float> > &, const NDT::Vector<Vec3<unsigned char> > &);

void DrawPoints(const NDT::Vector<Vec3<float> > & points,
                const NDT::Vector<Vec3<float> > & normals,
                const GLenum mode) {

    assert(points.Length() == normals.Length());

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, points.Data());
    glEnableClientState(GL_NORMAL_ARRAY);
    glNormalPointer(GL_FLOAT, 0, normals.Data());
    glDrawArrays(mode, 0, points.Count());
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

}

void DrawPoints(const NDT::Vector<Vec3<float> > & points,
                const NDT::Vector<Vec3<float> > & normals,
                const NDT::Vector<Vec3<unsigned char> > & colors) {

    assert(points.Length() == colors.Length());

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, points.Data());
    glEnableClientState(GL_NORMAL_ARRAY);
    glNormalPointer(GL_FLOAT, 0, normals.Data());
    glEnableClientState(GL_COLOR_ARRAY);
    glColorPointer(3, GL_UNSIGNED_BYTE, 0, colors.Data());
    glDrawArrays(GL_POINTS, 0, points.Count());
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

}

void DrawPoints(const NDT::Vector<Vec3<float> > & points,
                const NDT::Vector<Vec3<float> > & normals,
                const NDT::Vector<Vec3<int> > & faces) {
    assert(points.Length() == normals.Length());

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, points.Data());
    glEnableClientState(GL_NORMAL_ARRAY);
    glNormalPointer(GL_FLOAT, 0, normals.Data());
    glDrawElements(GL_TRIANGLES, faces.Count() * 3, GL_UNSIGNED_INT, faces.Data());
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

}

float OscillatingValue(const float d0, const float amplitude, const float frequency) {

    return d0 + amplitude * std::sin(2 * M_PI * 1e-6 * frequency * pangolin::TimeNow_us());

}

void GlOscillatingPointSize(const float d0, const float amplitude, const float frequency) {

    glPointSize(d0 + amplitude * std::sin(2 * M_PI * 1e-6 * frequency * pangolin::TimeNow_us()));

}

} // namespace vu