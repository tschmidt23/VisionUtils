#pragma  once

/***
 * © Tanner Schmidt 2018
 */

#include <string>

#include <NDT/Tensor.h>

#include <vu/EigenHelpers.h>

namespace vu {

void WritePly(const NDT::ConstVector<Vec3<float> > & vertices,
              const NDT::ConstVector<Vec3<float> > & normals,
              const std::string filename);

void WritePly(const NDT::ConstVector<Vec3<float> > & vertices,
              const NDT::ConstVector<Vec3<float> > & normals,
              const NDT::ConstVector<Vec3<int> > & faces,
              const std::string filename);

void WritePly(const NDT::ConstVector<Vec3<float> > & vertices,
              const NDT::ConstVector<Vec3<float> > & normals,
              const NDT::ConstVector<Vec3<unsigned char> > & colors,
              const NDT::ConstVector<Vec3<int> > & faces,
              const std::string filename);

void WritePly(const NDT::ConstVector<Vec3<float> > & vertices,
              const NDT::ConstVector<Vec3<int> > & faces,
              const std::string filename);

void WritePly(const NDT::ConstVector<Vec3<float> > & vertices,
              const NDT::ConstVector<Vec3<unsigned char> > & colors,
              const std::string filename);

void WritePly(const NDT::ConstVector<Vec3<float> > & vertices,
              const NDT::ConstVector<Vec3<unsigned char> > & colors,
              const NDT::ConstVector<Vec3<int> > & faces,
              const std::string filename);

void ReadPly(NDT::ManagedVector<Vec3<float> > & vertices,
             const std::string filename);

void ReadPly(NDT::ManagedVector<Vec3<float> > & vertices,
             NDT::ManagedVector<Vec3<float> > & normals,
             const std::string filename);

void ReadPly(NDT::ManagedVector<Vec3<float> > & vertices,
             NDT::ManagedVector<Vec3<float> > & normals,
             const NDT::Vector<Vec3<unsigned char> > & colors,
             const std::string filename);

void ReadPly(NDT::ManagedVector<Vec3<float> > & vertices,
             NDT::ManagedVector<Vec3<int> > & faces,
             const std::string filename);

void ReadPly(NDT::ManagedVector<Vec3<float> > & vertices,
             NDT::ManagedVector<Vec3<float> > & normals,
             NDT::ManagedVector<Vec3<int> > & faces,
             const std::string filename);

void ReadPly(NDT::ManagedVector<Vec3<float> > & vertices,
             NDT::ManagedVector<Vec3<float> > & normals,
             NDT::ManagedVector<Vec3<unsigned char> > & colors,
             NDT::ManagedVector<Vec3<int> > & faces,
             const std::string filename);

int ReadPlyVertexCount(const std::string filename);

} // namespace vu