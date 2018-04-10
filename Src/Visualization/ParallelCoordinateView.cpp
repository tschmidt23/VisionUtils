/***
 * © Tanner Schmidt 2018
 */

#ifdef HAS_PANGOLIN

#include <vu/Visualization/ParallelCoordinateView.h>

namespace vu {

void ParallelCoordinateView::PreRender() {

    glClearColor(1,1,1,0);
    this->ActivateScissorAndClear();
    this->ActivatePixelOrthographic();

    glPushMatrix();

    glScalef(handler_.Zoom()*handler_.Aspect(),handler_.Zoom(),1.f);
    glTranslatef(-handler_.TranslationX(),-handler_.TranslationY(),0.f);

    // draw axis
    const int w = this->GetBounds().w;
    const int h = this->GetBounds().h;

    glColor3ub(192,192,192);
    glBegin(GL_LINES);
    for (int d=0; d<D_; ++d) {
        glVertex2f((d + 0.5)*(1.0/D_)*w,0.1*h);
        glVertex2f((d + 0.5)*(1.0/D_)*w,0.9*h);
    }
    glEnd();

    glClearColor(0,0,0,0);

}

void ParallelCoordinateView::PostRender() {

    glPopMatrix();

}

void ParallelCoordinateView::Render(const float * points, const int nPoints) {

    const int w = this->GetBounds().w;
    const int h = this->GetBounds().h;

    std::vector<float> points2D(nPoints*(D_-1)*2*2);

    int index = 0;
    for (int i=0; i<nPoints; ++i) {
        float lastVal = points[D_*i];
        for (int d=1; d<D_; ++d) {
            const float val = points[d + D_*i];
            points2D[index++] = (d - 0.5)*(1.0/D_)*w;
            points2D[index++] = ((lastVal + 1)/2*0.8 + 0.1)*h;
            points2D[index++] = (d + 0.5)*(1.0/D_)*w;
            points2D[index++] = ((    val + 1)/2*0.8 + 0.1)*h;
            lastVal = val;
        }
    }

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2,GL_FLOAT,0,points2D.data());
    glDrawArrays(GL_LINES,0,points2D.size()/2);
    glDisableClientState(GL_VERTEX_ARRAY);

}

void ParallelCoordinateView::Render(const float * points, const uchar3 * colors, const int nPoints) {

    const int w = this->GetBounds().w;
    const int h = this->GetBounds().h;

    std::vector<float> points2D(nPoints*(D_-1)*2*2);
    std::vector<uchar3> pointColors(nPoints*(D_-1)*2);

    int pointIndex = 0;
    int colorIndex = 0;
    for (int i=0; i<nPoints; ++i) {
        float lastVal = points[D_*i];
        for (int d=1; d<D_; ++d) {
            const float val = points[d + D_*i];
            points2D[pointIndex++] = (d - 0.5)*(1.0/D_)*w;
            points2D[pointIndex++] = ((lastVal + 1)/2*0.8 + 0.1)*h;
            points2D[pointIndex++] = (d + 0.5)*(1.0/D_)*w;
            points2D[pointIndex++] = ((    val + 1)/2*0.8 + 0.1)*h;
            pointColors[colorIndex++] = colors[i];
            pointColors[colorIndex++] = colors[i];
            lastVal = val;
        }
    }

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2,GL_FLOAT,0,points2D.data());
    glEnableClientState(GL_COLOR_ARRAY);
    glColorPointer(3,GL_UNSIGNED_BYTE,0,pointColors.data());
    glDrawArrays(GL_LINES,0,points2D.size()/2);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

}

} // namespace vu

#endif // HAS_PANGOLIN