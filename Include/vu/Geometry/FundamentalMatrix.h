#pragma once

#include <Eigen/SVD>

#include <vu/EigenHelpers.h>
#include <NDT/Tensor.h>

namespace vu {

//cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
//{
//    const int N = vP1.size();
//
//    cv::Mat A(N,9,CV_32F);
//
//    for(int i=0; i<N; i++)
//    {
//        const float u1 = vP1[i].x;
//        const float v1 = vP1[i].y;
//        const float u2 = vP2[i].x;
//        const float v2 = vP2[i].y;
//
//        A.at<float>(i,0) = u2*u1;
//        A.at<float>(i,1) = u2*v1;
//        A.at<float>(i,2) = u2;
//        A.at<float>(i,3) = v2*u1;
//        A.at<float>(i,4) = v2*v1;
//        A.at<float>(i,5) = v2;
//        A.at<float>(i,6) = u1;
//        A.at<float>(i,7) = v1;
//        A.at<float>(i,8) = 1;
//    }
//
//    cv::Mat u,w,vt;
//
//    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
//
//    cv::Mat Fpre = vt.row(8).reshape(0, 3);
//
//    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
//
//    w.at<float>(2)=0;
//
//    return  u*cv::Mat::diag(w)*vt;
//}

template <typename Scalar>
Eigen::Matrix<Scalar, 3, 3> ComputeFundamentalMatrix(const NDT::Vector <Vec2<Scalar>> & sourcePoints,
                                                     const NDT::Vector <Vec2<Scalar>> & targetPoints);

} // namespace vu