#pragma once

/***
 * © Tanner Schmidt 2018
 */

#include <nanoflann.hpp>

#include <NDT/Tensor.h>

#include <vu/EigenHelpers.h>

namespace vu {

template <int D>
struct KdVertexList {
public:

    KdVertexList(const NDT::ConstVector<Vec<D, float> > & points)
            : points_(points) { }

    KdVertexList(const NDT::ConstImage<Vec<D, float> > & points)
            : points_(points.Count(), points.Data()) { }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return points_.Count();
    }

    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
    inline float kdtree_distance(const float * p1, const size_t idx_p2,size_t /*size*/) const {
        Eigen::Map<const Vec<D, float> > p(p1);
//        if (p(2) > 0) {
        return (p - points_(idx_p2)).squaredNorm();
//        } else {
//            return std::numeric_limits<float>::infinity();
//        }
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline float kdtree_get_pt(const size_t idx, int dim) const {
        return points_(idx)(dim);
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /*bb*/) const { return false; }

private:

    NDT::ConstVector<Vec<D, float> > points_;

};

template <int D>
using KdVertexListTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, KdVertexList<D> >, KdVertexList<D>, D, int>;


using KdPointCloud = KdVertexList<3>;

using KdPointCloudTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, KdPointCloud>, KdPointCloud, 3, int>;



struct KdIndexedPointCloud {
public:

    KdIndexedPointCloud(const NDT::ConstVector<Vec3<float> > & points,
                        const NDT::ConstVector<int> & indices)
            : points_(points), indices_(indices) { }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return indices_.Count();
    }

    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
    inline float kdtree_distance(const float * p1, const size_t idx_p2, size_t /*size*/) const {
        Eigen::Map<const Vec3<float> > p(p1);
//        if (p(2) > 0) {
        return (p - points_(indices_(idx_p2))).squaredNorm();
//        } else {
//            return std::numeric_limits<float>::infinity();
//        }
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline float kdtree_get_pt(const size_t idx, int dim) const {
        return points_(indices_(idx))(dim);
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /*bb*/) const { return false; }

private:

    NDT::ConstVector<Vec3<float> > points_;
    NDT::ConstVector<int> indices_;

};

using KdIndexedPointCloudTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, KdIndexedPointCloud>, KdIndexedPointCloud, 3, int>;




template <int D>
struct KdListOfVertexLists {
public:

    KdListOfVertexLists(const std::vector<NDT::Vector<Vec<D, float> > > & points) {

        for (auto el : points) {
            points_.push_back(el);
        }

        count_ = std::accumulate(points_.cbegin(), points_.cend(), 0,
                                 [](const int soFar, const NDT::ConstVector<Vec<D, float> > & el) { return soFar + el.Count(); });

    }

    KdListOfVertexLists(const std::vector<NDT::ConstVector<Vec<D, float> > > & points)
            : points_(points) {

        count_ = std::accumulate(points_.cbegin(), points_.cend(), 0,
                                 [](const int soFar, const NDT::ConstVector<Vec<D, float> > & el) { return soFar + el.Count(); });

    }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return count_;
    }

    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
    inline float kdtree_distance(const float * p1, const size_t idx_p2,size_t /*size*/) const {

        int n = 0;
        size_t localIndex = idx_p2;

        while (localIndex >= points_[n].Count()) {
            localIndex -= points_[n].Count();
            ++n;
        }

        Eigen::Map<const Vec<D, float> > p(p1);

        return (p - points_[n](localIndex)).squaredNorm();

    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline float kdtree_get_pt(const size_t idx, int dim) const {

        int n = 0;
        size_t localIndex = idx;

        while (localIndex >= points_[n].Count()) {
            localIndex -= points_[n].Count();
            ++n;
        }

        return points_[n](localIndex)(dim);

    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /*bb*/) const { return false; }

private:

    int count_;
    std::vector<NDT::ConstVector<Vec<D, float> > > points_;

};

template <int D>
using KdListOfVertexListsTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, KdListOfVertexLists<D> >, KdListOfVertexLists<D>, D, int>;


//struct KdVertMap {
//public:
//
//    KdVertMap(const NDT::Image<Vec3<float> > & points)
//            : points_(points.Count(), points.Data()) { }
//
//    // Must return the number of data points
//    inline size_t kdtree_get_point_count() const {
//        return points_.Count();
//    }
//
//    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
//    inline float kdtree_distance(const float * p1, const size_t idx_p2,size_t /*size*/) const {
//        Eigen::Map<const Vec3<float> > p(p1);
//        if (points_(idx_p2)(2) > 0) {
//            return (p - points_(idx_p2)).squaredNorm();
//        } else {
//            return std::numeric_limits<float>::infinity();
//        }
//    }
//
//    // Returns the dim'th component of the idx'th point in the class:
//    // Since this is inlined and the "dim" argument is typically an immediate value, the
//    //  "if/else's" are actually solved at compile time.
//    inline float kdtree_get_pt(const size_t idx, int dim) const {
//        return points_(idx)(dim);
//    }
//
//    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
//    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
//    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
//    template <class BBOX>
//    bool kdtree_get_bbox(BBOX & /*bb*/) const { return false; }
//
//private:
//
//    NDT::ConstVector<Vec3<float> > points_;
//
//};

struct KdVertMap {
public:

    KdVertMap(const NDT::Image<Vec3<float> > & points)
            : points_(points.Count(), points.Data()) {

        for (int i = 0; i < points_.Count(); ++i) {
            if (points_(i)(2) > 0) {
                validIndices_.push_back(i);
            }
        }

    }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return validIndices_.size();
    }

    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
    inline float kdtree_distance(const float * p1, const size_t idx_p2,size_t /*size*/) const {
        Eigen::Map<const Vec3<float> > p(p1);
        return (p - points_(validIndices_[idx_p2])).squaredNorm();

    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline float kdtree_get_pt(const size_t idx, int dim) const {
        return points_(validIndices_[idx])(dim);
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /*bb*/) const { return false; }

    inline int GetOriginalIndex(const int validIndex) const {
        return validIndices_[validIndex];
    }

private:

    std::vector<int> validIndices_;
    NDT::ConstVector<Vec3<float> > points_;

};

using KdVertMapTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, KdVertMap>, KdVertMap, 3, int>;

template <typename Scalar, int C, int X, int Y>
struct KdFeatureMap;

template <typename Scalar>
struct KdFeatureMap<Scalar, 2, 0, 1> {
public:

    KdFeatureMap(const NDT::ConstVolume<Scalar> & featureMap)
            : linearizedFeatureMap_(featureMap.DimensionSize(0) * featureMap.DimensionSize(1),
                                    featureMap.DimensionSize(2), featureMap.Data()) { }

    inline size_t kdtree_get_point_count() const {
        return linearizedFeatureMap_.DimensionSize(0);
    }

    inline Scalar kdtree_distance(const Scalar * p1, const size_t idx_p2, size_t /*size*/) const {

        Scalar distanceSquared(0);

        for (int c = 0; c < linearizedFeatureMap_.DimensionSize(1); ++c) {

            const Scalar diff = linearizedFeatureMap_(idx_p2, c) - p1[c];

            distanceSquared += diff * diff;

        }

        return distanceSquared;

    }

    inline Scalar kdtree_get_pt(const size_t idx, int dim) const {

        return linearizedFeatureMap_(idx, dim);

    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /*bb*/) const {
        return false;
    }

private:

    // The linearized feature map flattens the first two dimensions so that they can be indexed
    // linearly w/o specifying both x and y. This should speed up search a bit. The first dimension
    // should be of length W * H and the second is the number of feature channels.
    NDT::ConstHostTensor2<Scalar> linearizedFeatureMap_;

};

template <typename Scalar>
struct KdFeatureMap<Scalar, 0, 1, 2> {
public:

    KdFeatureMap(const NDT::ConstVolume<Scalar> & featureMap)
            : linearizedFeatureMap_(featureMap.DimensionSize(0),
                                    featureMap.DimensionSize(1) * featureMap.DimensionSize(2),
                                    featureMap.Data()) { }

    inline size_t kdtree_get_point_count() const {
        return linearizedFeatureMap_.DimensionSize(1);
    }

    inline Scalar kdtree_distance(const Scalar * p1, const size_t idx_p2, size_t /*size*/) const {

        Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> > p2(&linearizedFeatureMap_(0, idx_p2), linearizedFeatureMap_.DimensionSize(1));

        return (Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1> >(p1, linearizedFeatureMap_.DimensionSize(1)) - p2).squaredNorm();


//        Scalar distanceSquared(0);
//
//        for (int c = 0; c < linearizedFeatureMap_.DimensionSize(0); ++c) {
//
//            const Scalar diff = linearizedFeatureMap_(c, idx_p2) - p1[c];
//
//            distanceSquared += diff * diff;
//
//        }
//
//        return distanceSquared;

    }

    inline Scalar kdtree_get_pt(const size_t idx, int dim) const {

        return linearizedFeatureMap_(dim, idx);

    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /*bb*/) const {
        return false;
    }

private:

    // The linearized feature map flattens the first two dimensions so that they can be indexed
    // linearly w/o specifying both x and y. This should speed up search a bit. The first dimension
    // should be of length W * H and the second is the number of feature channels.
    NDT::ConstHostTensor2<Scalar> linearizedFeatureMap_;

};

template <typename Scalar, int C = 0, int X = 1, int Y = 2>
using KdFeatureMapTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<Scalar, KdFeatureMap<Scalar, C, X, Y> >, KdFeatureMap<Scalar, C, X, Y>, -1, int>;


struct KdMatrix {
public:

    KdMatrix(const NDT::ConstTensor2f & matrix)
            : matrix_(matrix) { }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return matrix_.DimensionSize(1);
    }

    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
    inline float kdtree_distance(const float * p1, const size_t idx_p2,size_t /*size*/) const {
        return (Eigen::Map<const Eigen::VectorXf>(p1, matrix_.DimensionSize(0)) -
                Eigen::Map<const Eigen::VectorXf>(&matrix_(0, idx_p2), matrix_.DimensionSize(0))).squaredNorm();
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline float kdtree_get_pt(const size_t idx, int dim) const {
        return matrix_(dim, idx);
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /*bb*/) const { return false; }

private:

    NDT::ConstTensor2f matrix_;

};

using KdMatrixTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, KdMatrix>, KdMatrix, -1, int>;

} // namespace vu