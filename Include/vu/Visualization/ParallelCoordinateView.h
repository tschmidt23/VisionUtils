#pragma once

/***
 * © Tanner Schmidt 2018
 */

#include <vector_types.h>
#include <pangolin/pangolin.h>

//#include "embeddingView.h"

namespace vu {

class ParallelCoordinateHandler : public pangolin::Handler {
public:

    ParallelCoordinateHandler()
        : zoom_("panel.zoom",1.f,0.5f,2.f),
          aspect_("panel.aspect",1.f,0.5f,2.f),
          transX_("panel.trans x",0.f,-1.f,1.f),
          transY_("panel.trans y",0.f,-1.f,1.f),
          leftPressed_(false),
          rightPressed_(false) { }

    inline void ClampZoom() {

        const float minZoom = 1.f/aspect_;

        zoom_ = std::max((float)zoom_,minZoom);

    }

    inline void DoZoom(pangolin::View & v, const int x, const int y, float multiplier) {

        const float zoomX = (x-v.GetBounds().l) / (zoom_ * aspect_) + transX_;

        const float zoomY = (y-v.GetBounds().b) / zoom_ + transY_;

        zoom_ = zoom_ * multiplier;

        transX_ = -(x-v.GetBounds().l) / (zoom_ * aspect_) - zoomX;

        transY_ = -(y-v.GetBounds().b) / zoom_ - zoomY;

        ClampZoom();

    }

    void Mouse(pangolin::View & v, pangolin::MouseButton button, int x, int y, bool pressed, int button_state) {

        if (button == pangolin::MouseButtonLeft) {

            leftPressed_ = pressed;

            lastX_ = x; lastY_ = y;

        } else if (button == pangolin::MouseButtonRight) {

            rightPressed_ = pressed;

            lastX_ = x; lastY_ = y;

        } else if (button == pangolin::MouseWheelUp) {

            DoZoom(v,x,y,zoomRate_);

        } else if (button == pangolin::MouseWheelDown) {

            DoZoom(v,x,y,1./zoomRate_);

        }

    }

    void MouseMotion(pangolin::View & v, int x, int y, int button_state) {

        if (leftPressed_) {

            if (rightPressed_) {

            } else {
                transX_ = transX_ - (x - lastX_) / (zoom_ * aspect_); // / static_cast<float>(v.GetBounds().w);
                transY_ = transY_ - (y - lastY_) / zoom_; // / static_cast<float>(v.GetBounds().h);
            }

        }

        lastX_ = x; lastY_ = y;

    }

    inline float Zoom() {
        return zoom_;
    }

    inline float Aspect() {
        return aspect_;
    }

    inline float TranslationX() {
        return transX_;
    }

    inline float TranslationY() {
        return transY_;
    }

private:

    static constexpr float zoomRate_ = 1.07;

    bool leftPressed_, rightPressed_;
    int lastX_, lastY_;

    pangolin::Var<float> zoom_;
    pangolin::Var<float> aspect_;
    pangolin::Var<float> transX_;
    pangolin::Var<float> transY_;

};

class ParallelCoordinateView : public pangolin::View /*: public EmbeddingView*/ {
public:

    ParallelCoordinateView(/*const std::string name,*/ const int dimensionality)
        : /*EmbeddingView(name),*/ D_(dimensionality) {

        SetHandler(&handler_);

    }

    void PreRender();

    void PostRender();

    void Render(const float * points, const int nPoints);

    void Render(const float * points, const uchar3 * colors, const int nPoints);

    inline int Dimensionality() const { return D_; }


private:

    ParallelCoordinateHandler handler_;

    ParallelCoordinateView & operator=(const ParallelCoordinateView & other) = delete;
    ParallelCoordinateView(const ParallelCoordinateView & other) = delete;

    const int D_;

};

} // namespace vu