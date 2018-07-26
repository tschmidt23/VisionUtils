#pragma once

#include <vu/EigenHelpers.h>

#include <NDT/Tensor.h>

namespace vu {

class MeshSampler {
public:

    MeshSampler(const NDT::ConstVector<Vec3<float> > & vertices,
                const NDT::ConstVector<Vec3<int> > & faces);

    Vec3<float> SamplePoint() const;

    inline double TotalSurfaceArea() const {
        return totalSurfaceArea_;
    }

private:

    const NDT::ConstVector<Vec3<float> > vertices_;
    const NDT::ConstVector<Vec3<int> > faces_;

    std::vector<double> cumulativeSurfaceArea_;
    double totalSurfaceArea_;

};

} // namespace vu