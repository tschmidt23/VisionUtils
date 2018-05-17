#pragma  once

/***
 * © Tanner Schmidt 2018
 */

#include <pangolin/pangolin.h>

#include <NDT/Tensor.h>

#include <vu/EigenHelpers.h>

namespace vu {

class BufferedMesh {
public:

    BufferedMesh(const std::string meshFilename);

    inline NDT::Vector<Vec3<float> > & Vertices() { return meshVertices_; }

    inline NDT::Vector<Vec3<float> > & Normals() { return meshNormals_; }

    inline NDT::Vector<Vec3<unsigned char> > & Colors() { return meshColors_; }

    inline NDT::Vector<Vec3<int> > & Faces() { return meshFaces_; }

    inline pangolin::GlBuffer & VertexBuffer() { return vertexBuffer_; }

    inline pangolin::GlBuffer & NormalBuffer() { return normalBuffer_; }

    inline pangolin::GlBuffer & ColorBuffer() { return colorBuffer_; }

    inline pangolin::GlBuffer & IndexBuffer() { return indexBuffer_; }

    void RenderVCI() const;

    void RenderVNCI(const pangolin::GlBuffer & colorBuffer) const;

private:

    NDT::ManagedVector<Vec3<float> > meshVertices_, meshNormals_;

    NDT::ManagedVector<Vec3<unsigned char> > meshColors_;

    NDT::ManagedVector<Vec3<int> > meshFaces_;

    pangolin::GlBuffer vertexBuffer_, normalBuffer_, colorBuffer_, indexBuffer_;

};

} // namespace vu
