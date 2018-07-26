#include <vu/Mesh/Sampling.h>

#include <vu/RandomHelpers.h>

namespace vu {

MeshSampler::MeshSampler(const NDT::ConstVector<Vec3<float> > & vertices,
                         const NDT::ConstVector<Vec3<int> > & faces)
        : vertices_(vertices), faces_(faces), cumulativeSurfaceArea_(faces.Count()) {

    for (int n = 0; n < faces.Count(); ++n) {

        const Vec3<float> & vA = vertices(faces(n)(0));
        const Vec3<float> & vB = vertices(faces(n)(1));
        const Vec3<float> & vC = vertices(faces(n)(2));

        const double lAB = (vA - vB).norm();
        const double lBC = (vB - vC).norm();
        const double lCA = (vC - vA).norm();

        const double s = (lAB + lBC + lCA) / 2.0;

        const double surfaceArea = std::sqrt( s * (s - lAB) * (s - lBC)*(s - lCA));

        if (n == 0) {
            cumulativeSurfaceArea_[n] = surfaceArea;
        } else {
            // TODO: numerical stability issues here potentially, maybe do a tree reduction in the future
            cumulativeSurfaceArea_[n] = cumulativeSurfaceArea_[n - 1] + surfaceArea;
        }

    }

    totalSurfaceArea_ = cumulativeSurfaceArea_.back();

    const double oneOverTotalSurfaceArea = 1.0 / totalSurfaceArea_;

    for (double & val : cumulativeSurfaceArea_) {
        val *= oneOverTotalSurfaceArea;
    }

}

Vec3<float> MeshSampler::SamplePoint() const {

    // pick a face
    const Vec3<int> & face = faces_(SelectIndexFromCumulativeDistribution(cumulativeSurfaceArea_));

    // pick a vertex
    const double r1 = UniformSample<double>();

    const double r2 = UniformSample<double>();

    const double sqrtR1 = std::sqrt(r1);

    return (1 - sqrtR1) * vertices_(face(0)) +
           sqrtR1 * (1 - r2) * vertices_(face(1)) +
           sqrtR1 * r2 * vertices_(face(2));

}

} // namespace vu

//void sampleUniformlyBySurfaceArea(const TriangleMesh & mesh, std::vector<float3> & sampledPoints, const float sampleDensity) {
//
//    std::vector<double> cumulativeSurfaceArea(mesh.numFaces());
//    for (int n=0; n<mesh.numFaces(); ++n) {
//
//        const uint3 & f = mesh.face(n);
//        const float3 & A = mesh.vertex(f.x);
//        const float3 & B = mesh.vertex(f.y);
//        const float3 & C = mesh.vertex(f.z);
//
//        double lAB = length(A - B);
//        double lBC = length(B - C);
//        double lCA = length(C - A);
//
//        double s = (lAB + lBC + lCA)/2;
//
//        double surfaceArea = std::sqrt(s*(s-lAB)*(s-lBC)*(s-lCA));
//
//        if (std::isnan(surfaceArea)) {
//            surfaceArea = 0;
//        }
//
//        if (n == 0) { cumulativeSurfaceArea[n] = surfaceArea; }
//        else { cumulativeSurfaceArea[n] = cumulativeSurfaceArea[n-1] + surfaceArea; } // TODO: numerical stability issues here, potentially
//
//    }
//
//    double totalSurfaceArea = cumulativeSurfaceArea.back();
//    double oneOverTotalSurfaceArea = 1./totalSurfaceArea;
//    for (double & p : cumulativeSurfaceArea) {
//        p *= oneOverTotalSurfaceArea;
//    }
//
//    const int nSamplePoints = std::round(totalSurfaceArea*sampleDensity);
//
//    sampledPoints.resize(nSamplePoints);
//
//    for (float3 & sample : sampledPoints) {
//
//        // pick a face
//        const uint3 & f = mesh.face(selectIndexFromCumulativeDistribution(cumulativeSurfaceArea));
//
//        // pick a point
//        const float r1 = uniformUnitSample();
//
//        const float r2 = uniformUnitSample();
//        sample = (1 - std::sqrt(r1))*mesh.vertex(f.x) +
//                 std::sqrt(r1)*(1-r2)*mesh.vertex(f.y) +
//                 std::sqrt(r1)*r2*mesh.vertex(f.z);
//
//    }
//
//}