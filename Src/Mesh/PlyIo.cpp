/***
* © Tanner Schmidt 2018
*/

#include <vu/Mesh/PlyIo.h>

#include <fstream>

namespace vu {

void WritePly(const NDT::Vector<Vec3<float> > & vertices,
              const NDT::Vector<Vec3<float> > & normals,
              const std::string filename) {

    std::ofstream stream(filename);
    stream << "ply" << std::endl;
    stream << "format ascii 1.0" << std::endl;
    stream << "element vertex " << vertices.Count() << std::endl;
    stream << "property float x" << std::endl;
    stream << "property float y" << std::endl;
    stream << "property float z" << std::endl;
    stream << "property float nx" << std::endl;
    stream << "property float ny" << std::endl;
    stream << "property float nz" << std::endl;
    stream << "end_header" << std::endl;

    for (int i = 0; i < vertices.Count(); ++i) {
        const Vec3<float> & v = vertices(i);
        const Vec3<float> & n = normals(i);
        stream << v(0) << " " << v(1) << " " << v(2) << " " << -n(0) << " " << -n(1) << " " << -n(2) << std::endl;
    }

}

void WritePly(const NDT::Vector<Vec3<float> > & vertices,
              const NDT::Vector<Vec3<float> > & normals,
              const NDT::Vector<Vec3<int> > & faces,
              const std::string filename) {

    std::ofstream stream(filename);
    stream << "ply" << std::endl;
    stream << "format ascii 1.0" << std::endl;
    stream << "element vertex " << vertices.Count() << std::endl;
    stream << "property float x" << std::endl;
    stream << "property float y" << std::endl;
    stream << "property float z" << std::endl;
    stream << "property float nx" << std::endl;
    stream << "property float ny" << std::endl;
    stream << "property float nz" << std::endl;
    stream << "element face " << faces.Count() << std::endl;
    stream << "property list uchar int vertex_index" << std::endl;
    stream << "end_header" << std::endl;
//    for (const Vec3<float> & v : vertices) {
    for (int i = 0; i < vertices.Count(); ++i) {
        const Vec3<float> & v = vertices(i);
        const Vec3<float> & n = normals(i);
        stream << v(0) << " " << v(1) << " " << v(2) << " " << -n(0) << " " << -n(1) << " " << -n(2) << std::endl;
    }
    for (int i = 0; i < faces.Count(); ++i) {
        const Vec3<int> & f = faces(i);
        stream << "3 " << f(0) << " " << f(2) << " " << f(1) << std::endl;
    }

}

void WritePly(const NDT::Vector<Vec3<float> > & vertices,
              const NDT::Vector<Vec3<int> > & faces,
              const std::string filename) {

    std::ofstream stream(filename);
    stream << "ply" << std::endl;
    stream << "format ascii 1.0" << std::endl;
    stream << "element vertex " << vertices.Count() << std::endl;
    stream << "property float x" << std::endl;
    stream << "property float y" << std::endl;
    stream << "property float z" << std::endl;
    stream << "element face " << faces.Count() << std::endl;
    stream << "property list uchar int vertex_index" << std::endl;
    stream << "end_header" << std::endl;
    for (int i = 0; i < vertices.Count(); ++i) {
        const Vec3<float> & v = vertices(i);
        stream << v(0) << " " << v(1) << " " << v(2) << std::endl;
    }
    for (int i = 0; i < faces.Count(); ++i) {
        const Vec3<int> & f = faces(i);
        stream << "3 " << f(0) << " " << f(2) << " " << f(1) << std::endl;
    }

}

void WritePly(const NDT::Vector<Vec3<float> > & vertices,
              const NDT::Vector<Vec3<unsigned char> > & colors,
              const std::string filename) {

    std::ofstream stream(filename);
    stream << "ply" << std::endl;
    stream << "format ascii 1.0" << std::endl;
    stream << "element vertex " << vertices.Count() << std::endl;
    stream << "property float x" << std::endl;
    stream << "property float y" << std::endl;
    stream << "property float z" << std::endl;
    stream << "property uchar red" << std::endl;
    stream << "property uchar green" << std::endl;
    stream << "property uchar blue" << std::endl;
    stream << "end_header" << std::endl;
    for (int i = 0; i < vertices.Count(); ++i) {
        const Vec3<float> & v = vertices(i);
        const Vec3<unsigned char> & c = colors(i);
        stream << v(0) << " " << v(1) << " " << v(2) << " " << (int)c(0) << " " << (int)c(1) << " " << (int)c(2) << std::endl;
    }

}

void WritePly(const NDT::Vector<Vec3<float> > & vertices,
              const NDT::Vector<Vec3<unsigned char> > & colors,
              const NDT::Vector<Vec3<int> > & faces,
              const std::string filename) {

    std::ofstream stream(filename);
    stream << "ply" << std::endl;
    stream << "format ascii 1.0" << std::endl;
    stream << "element vertex " << vertices.Count() << std::endl;
    stream << "property float x" << std::endl;
    stream << "property float y" << std::endl;
    stream << "property float z" << std::endl;
    stream << "property uchar red" << std::endl;
    stream << "property uchar green" << std::endl;
    stream << "property uchar blue" << std::endl;
    stream << "element face " << faces.Count() << std::endl;
    stream << "property list uchar int vertex_index" << std::endl;
    stream << "end_header" << std::endl;
    for (int i = 0; i < vertices.Count(); ++i) {
        const Vec3<float> & v = vertices(i);
        const Vec3<unsigned char> & c = colors(i);
        stream << v(0) << " " << v(1) << " " << v(2) << " " << (int)c(0) << " " << (int)c(1) << " " << (int)c(2) << std::endl;
    }
    for (int i = 0; i < faces.Count(); ++i) {
        const Vec3<int> & f = faces(i);
        stream << "3 " << f(0) << " " << f(2) << " " << f(1) << std::endl;
    }

}

void WritePly(const NDT::Vector<Vec3<float> > & vertices,
              const NDT::Vector<Vec3<float> > & normals,
              const NDT::Vector<Vec3<unsigned char> > & colors,
              const NDT::Vector<Vec3<int> > & faces,
              const std::string filename) {

    std::ofstream stream(filename);
    stream << "ply" << std::endl;
    stream << "format ascii 1.0" << std::endl;
    stream << "element vertex " << vertices.Count() << std::endl;
    stream << "property float x" << std::endl;
    stream << "property float y" << std::endl;
    stream << "property float z" << std::endl;
    stream << "property float nx" << std::endl;
    stream << "property float ny" << std::endl;
    stream << "property float nz" << std::endl;
    stream << "property uchar red" << std::endl;
    stream << "property uchar green" << std::endl;
    stream << "property uchar blue" << std::endl;
    stream << "element face " << faces.Count() << std::endl;
    stream << "property list uchar int vertex_index" << std::endl;
    stream << "end_header" << std::endl;
    for (int i = 0; i < vertices.Count(); ++i) {
        const Vec3<float> & v = vertices(i);
        const Vec3<float> & n = normals(i);
        const Vec3<unsigned char> & c = colors(i);
        stream << v(0) << " " << v(1) << " " << v(2) << " " <<
               -n(0) << " " << -n(1) << " " << -n(2) << " " <<
               (int)c(0) << " " << (int)c(1) << " " << (int)c(2) << std::endl;
    }
    for (int i = 0; i < faces.Count(); ++i) {
        const Vec3<int> & f = faces(i);
        stream << "3 " << f(0) << " " << f(2) << " " << f(1) << std::endl;
    }

}

void ReadPly(NDT::ManagedVector<Vec3<float> > & vertices,
             NDT::ManagedVector<Vec3<float> > & normals,
             NDT::ManagedVector<Vec3<unsigned char> > & colors,
             NDT::ManagedVector<Vec3<int> > & faces,
             const std::string filename) {

    std::ifstream stream(filename);
    std::string word;

    auto getExpectedWord = [&stream, &word, &filename](const std::string expected) {

        stream >> word;
        if (word != expected) {
            throw std::runtime_error("while reading " + filename + ":\n did not find expected word " + expected +
                                     "\n instead, we get " + word);
        }

    };

    getExpectedWord("ply");
    getExpectedWord("format");
    getExpectedWord("ascii");

    double version;
    stream >> version;

    if (version != 1.0) {
        throw std::runtime_error("expected version 1");
    }

    getExpectedWord("element");
    getExpectedWord("vertex");

    int vertexCount;
    stream >> vertexCount;

    vertices.Resize(vertexCount);
    normals.Resize(vertexCount);
    colors.Resize(vertexCount);

    getExpectedWord("property");
    getExpectedWord("float");
    getExpectedWord("x");

    getExpectedWord("property");
    getExpectedWord("float");
    getExpectedWord("y");

    getExpectedWord("property");
    getExpectedWord("float");
    getExpectedWord("z");

    getExpectedWord("property");
    getExpectedWord("float");
    getExpectedWord("nx");

    getExpectedWord("property");
    getExpectedWord("float");
    getExpectedWord("ny");

    getExpectedWord("property");
    getExpectedWord("float");
    getExpectedWord("nz");

    getExpectedWord("property");
    getExpectedWord("uchar");
    getExpectedWord("red");

    getExpectedWord("property");
    getExpectedWord("uchar");
    getExpectedWord("green");

    getExpectedWord("property");
    getExpectedWord("uchar");
    getExpectedWord("blue");

    getExpectedWord("element");
    getExpectedWord("face");

    int faceCount;
    stream >> faceCount;

    faces.Resize(faceCount);

    getExpectedWord("property");
    getExpectedWord("list");
    getExpectedWord("uchar");
    getExpectedWord("int");
    getExpectedWord("vertex_index");

    getExpectedWord("end_header");

    for (int i = 0; i < vertexCount; ++i) {
        stream >> vertices(i)(0);
        stream >> vertices(i)(1);
        stream >> vertices(i)(2);
        stream >> normals(i)(0);
        stream >> normals(i)(1);
        stream >> normals(i)(2);
        int v;
        stream >> v; colors(i)(0) = v;
        stream >> v; colors(i)(1) = v;
        stream >> v; colors(i)(2) = v;
        normals(i) *= -1;
    }

    for ( int i = 0; i < faceCount; ++i) {
        int nIndices;
        stream >> nIndices;
        if (nIndices != 3) {
            throw std::runtime_error("expecting triangle mesh");
        }

        stream >> faces(i)(0);
        stream >> faces(i)(2);
        stream >> faces(i)(1);

    }

}

} // namespace vu