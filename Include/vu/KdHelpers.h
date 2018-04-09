#pragma once

#include <nanoflann.hpp>
#include <NDT/Tensor.h>

namespace vu {

struct KdPointCloud {
public:

    KdPointCloud(const NDT::Vector<Vec3<float> > & points)
            : points_(points) { }

    KdPointCloud(const NDT::Image<Vec3<float> > & points)
            : points_(points.Count(), points.Data()) { }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return points_.Count();
    }

    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
    inline float kdtree_distance(const float * p1, const size_t idx_p2,size_t /*size*/) const {
        Eigen::Map<const Vec3<float> > p(p1);
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

    NDT::ConstVector<Vec3<float> > points_;

};

using KdPointCloudTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, KdPointCloud>, KdPointCloud, 3, int>;

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

template <typename Scalar>
struct KdFeatureMap {
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
using KdFeatureMapTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<Scalar, KdFeatureMap<Scalar> >, KdFeatureMap<Scalar>, -1, int>;

} // namespace vu