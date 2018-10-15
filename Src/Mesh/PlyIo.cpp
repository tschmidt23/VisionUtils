/***
* © Tanner Schmidt 2018
*/

#include <vu/EigenHelpers.h>
#include <vu/Mesh/PlyIo.h>

#include <fstream>
#include <regex>

namespace vu {

using namespace operators;

void WritePly(const NDT::ConstVector<Vec3<float> > & vertices,
              const NDT::ConstVector<Vec3<float> > & normals,
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
        stream << v(0) << " " << v(1) << " " << v(2) << " " << n(0) << " " << n(1) << " " << n(2) << std::endl;
    }

}

void WritePly(const NDT::ConstVector<Vec3<float> > & vertices,
              const NDT::ConstVector<Vec3<float> > & normals,
              const NDT::ConstVector<Vec3<int> > & faces,
              const std::string filename) {

    std::cout << "not flipping normals" << std::endl;

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
        stream << v(0) << " " << v(1) << " " << v(2) << " " << n(0) << " " << n(1) << " " << n(2) << std::endl;
    }
    for (int i = 0; i < faces.Count(); ++i) {
        const Vec3<int> & f = faces(i);
        stream << "3 " << f(0) << " " << f(2) << " " << f(1) << std::endl;
    }

}

void WritePly(const NDT::ConstVector<Vec3<float> > & vertices,
              const NDT::ConstVector<Vec3<int> > & faces,
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

void WritePly(const NDT::ConstVector<Vec3<float> > & vertices,
              const NDT::ConstVector<Vec3<unsigned char> > & colors,
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

void WritePly(const NDT::ConstVector<Vec3<float> > & vertices,
              const NDT::ConstVector<Vec3<unsigned char> > & colors,
              const NDT::ConstVector<Vec3<int> > & faces,
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

void WritePly(const NDT::ConstVector<Vec3<float> > & vertices,
              const NDT::ConstVector<Vec3<float> > & normals,
              const NDT::ConstVector<Vec3<unsigned char> > & colors,
              const NDT::ConstVector<Vec3<int> > & faces,
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
               n(0) << " " << n(1) << " " << n(2) << " " <<
               (int)c(0) << " " << (int)c(1) << " " << (int)c(2) << std::endl;
    }
    for (int i = 0; i < faces.Count(); ++i) {
        const Vec3<int> & f = faces(i);
        stream << "3 " << f(0) << " " << f(2) << " " << f(1) << std::endl;
    }

}

void ParsePlyHeader(std::ifstream & stream,
                    std::vector<std::string> & elementTypeStrings,
                    std::vector<int> & elementCounts,
                    std::vector<std::vector<std::string> > & elementPropertyTypeStrings,
                    std::vector<std::vector<std::string> > & elementPropertyNames) {

    std::regex firstLineRegex("ply\\s*");
    std::regex secondLineRegex("format\\s+ascii\\s+1\\.0\\s*");

    auto CheckStreamDieOnError = [&stream]() {
        if (!stream.good()) {
            throw std::runtime_error("unexpected EOF");
        }
    };

    std::string s;

    std::getline(stream, s);

    CheckStreamDieOnError();

    if (!std::regex_match(s, firstLineRegex)) {
        throw std::runtime_error("expected 'ply' on first line");
    }

    std::getline(stream, s);

    CheckStreamDieOnError();

    if (!std::regex_match(s, secondLineRegex)) {
        throw std::runtime_error("expected ascii format, version 1.0, on second line");
    }

    stream >> s;

    while (s != "end_header") {

        if (s == "element") {

            stream >> s;
            elementTypeStrings.push_back(s);

            int count;
            stream >> count;
            elementCounts.push_back(count);

            elementPropertyTypeStrings.emplace_back();
            elementPropertyNames.emplace_back();

        } else if (s == "property") {

            if (elementPropertyTypeStrings.empty()) {
                throw std::runtime_error("expected 'element' before 'property'");
            }

            std::getline(stream, s);

            std::regex propertyRegex("\\s+(.*)\\s+([^\\s]+)\\s*");

            std::smatch match;

            if (!std::regex_match(s, match, propertyRegex)) {
                throw std::runtime_error("could not parse property");
            }

            elementPropertyTypeStrings.back().push_back(match[1]);

            elementPropertyNames.back().push_back(match[2]);

//            stream >> s;
//            elementPropertyTypeStrings.back().push_back(s);
//
//            stream >> s;
//            elementPropertyNames.back().push_back(s);

        } else if (s == "comment") {

            std::getline(stream, s);

        } else {

            throw std::runtime_error("unexpected token '" + s + "'");

        }

        stream >> s;

        CheckStreamDieOnError();

    }

}

class Reader {
public:

    virtual void Read(std::ifstream & stream, const int /*index*/) = 0;

};

template <typename T>
class ThrowawayReader : public Reader {
public:

    void Read(std::ifstream & stream, const int /*index*/) {
        T element;
        stream >> element;

//        static bool printed = false;
//        if (!printed) {
//            std::cout << element << std::endl << std::endl;
//            printed = true;
//        }

    }

};

template <>
class ThrowawayReader<unsigned char> : public Reader {
public:

    void Read(std::ifstream & stream, const int /*index*/) {
        int element;
        stream >> element;
    }

};

template <typename T>
class KeepReader : public Reader {
public:

    KeepReader(NDT::Vector<T> vec) : vec_(vec) {}

    void Read(std::ifstream & stream, const int index) {
        stream >> vec_(index);

//        static bool printed = false;
//        if (!printed) {
//            std::cout << vec_(index) << std::endl << std::endl;
//            printed = true;
//        }

    }

private:

    NDT::Vector<T> vec_;

};

template <int Length>
class ListReader : public Reader {
public:

    ListReader(NDT::Vector<Vec<Length, int> > & vec) : vec_(vec) { }

    void Read(std::ifstream & stream, const int index) {

        int length;
        stream >> length;

        if (length != Length) {
            throw std::runtime_error("expected length " + std::to_string(Length) + ", got " + std::to_string(length) +
             " @ index " + std::to_string(index));
        }

        stream >> vec_(index);

    }

private:

    NDT::Vector<Vec<Length, int> > vec_;

};

class ThrowawayListReader : public Reader {
public:

    void Read(std::ifstream & stream, const int index) {

        unsigned char length;
        stream >> length;

        int i;
        for (int l = 0; l < length; ++l) {
            stream >> i;
        }

    }

};

int ReadPlyVertexCount(const std::string filename) {

    std::ifstream stream(filename);

    std::vector<std::string> elementTypeStrings;
    std::vector<int> elementCounts;
    std::vector<std::vector<std::string> > elementPropertyTypeStrings;
    std::vector<std::vector<std::string> > elementPropertyNames;

    ParsePlyHeader(stream, elementTypeStrings, elementCounts, elementPropertyTypeStrings, elementPropertyNames);

    for (int i = 0; i < elementTypeStrings.size(); ++i) {

        const std::string & s = elementTypeStrings[i];

        if (s == "vertex") {

            return elementCounts[i];

        }

    }

    return -1;

}

void ReadPly(NDT::ManagedVector<Vec3<float> > * vertices,
             NDT::ManagedVector<Vec3<float> > * normals,
             NDT::ManagedVector<Vec3<unsigned char> > * colors,
             NDT::ManagedVector<Vec3<int> > * faces,
             const std::string filename) {

    std::ifstream stream(filename);

    std::vector<std::string> elementTypeStrings;
    std::vector<int> elementCounts;
    std::vector<std::vector<std::string> > elementPropertyTypeStrings;
    std::vector<std::vector<std::string> > elementPropertyNames;

    ParsePlyHeader(stream, elementTypeStrings, elementCounts, elementPropertyTypeStrings, elementPropertyNames);

    for (int i = 0; i < elementTypeStrings.size(); ++i) {

        const int nElements = elementCounts[i];

        // set up the readers
        std::vector<std::shared_ptr<Reader> > readers;

        const std::vector<std::string> & propertyTypeStrings = elementPropertyTypeStrings[i];

        const std::vector<std::string> & propertyNames = elementPropertyNames[i];

        const std::string & s = elementTypeStrings[i];

//        std::cout << "element type " << i << ": " << s << std::endl;

        if (s == "vertex") {

            for (int j = 0; j < propertyNames.size(); ++j) {

                // TODO: lots of duplicate code
                if (propertyNames[j] == "x") {

                    const std::string coordinateType = propertyTypeStrings[j];

                    ++j;

                    if (propertyNames[j] != "y") {
                        throw std::runtime_error("expected 'y' after 'x'");
                    }

                    if (propertyTypeStrings[j] != coordinateType) {
                        throw std::runtime_error("expected the same type for all coordinates");
                    }

                    ++j;

                    if (propertyNames[j] != "z") {
                        throw std::runtime_error("expected 'z' after 'y'");
                    }

                    if (propertyTypeStrings[j] != coordinateType) {
                        throw std::runtime_error("expected the same type for all coordinates");
                    }

                    if (coordinateType == "float") {
                        if (vertices) {
//                            std::cout << "adding vertex keep reader" << std::endl;
                            vertices->Resize(nElements);
                            readers.push_back(std::make_shared<KeepReader<Vec3<float> > >(*vertices));
                        } else {
//                            std::cout << "adding vertex throwaway reader" << std::endl;
                            readers.push_back(std::make_shared<ThrowawayReader<Vec3<float> > >());
                        }
                    } else {
                        throw std::runtime_error("unhandled vertex coordinate type '" + coordinateType + "'");
                    }

                } else if (propertyNames[j] == "nx") {

                    const std::string coordinateType = propertyTypeStrings[j];

                    ++j;

                    if (propertyNames[j] != "ny") {
                        throw std::runtime_error("expected 'ny' after 'nx'");
                    }

                    if (propertyTypeStrings[j] != coordinateType) {
                        throw std::runtime_error("expected the same type for all normals");
                    }

                    ++j;

                    if (propertyNames[j] != "nz") {
                        throw std::runtime_error("expected 'nz' after 'ny'");
                    }

                    if (propertyTypeStrings[j] != coordinateType) {
                        throw std::runtime_error("expected the same type for all normals");
                    }

                    if (coordinateType == "float") {
                        if (normals) {
//                            std::cout << "adding normal keep reader" << std::endl;
                            normals->Resize(nElements);
                            readers.push_back(std::make_shared<KeepReader<Vec3<float> > >(*normals));
                        } else {
//                            std::cout << "adding normal throwaway reader" << std::endl;
                            readers.push_back(std::make_shared<ThrowawayReader<Vec3<float> > >());
                        }
                    } else {
                        throw std::runtime_error("unhandled normal type '" + coordinateType + "'");
                    }

                } else if (propertyNames[j] == "red") {

                    const std::string coordinateType = propertyTypeStrings[j];

                    ++j;

                    if (propertyNames[j] != "green") {
                        throw std::runtime_error("expected 'green' after 'red'");
                    }

                    if (propertyTypeStrings[j] != coordinateType) {
                        throw std::runtime_error("expected the same type for all colors");
                    }

                    ++j;

                    if (propertyNames[j] != "blue") {
                        throw std::runtime_error("expected 'blue' after 'green'");
                    }

                    if (propertyTypeStrings[j] != coordinateType) {
                        throw std::runtime_error("expected the same type for all colors");
                    }

                    if (coordinateType == "uchar") {
                        if (colors) {
//                            std::cout << "adding color keep reader" << std::endl;
                            colors->Resize(nElements);
                            readers.push_back(std::make_shared<KeepReader<Vec3<unsigned char> > >(*colors));
                        } else {
//                            std::cout << "adding color throwaway reader" << std::endl;
                            readers.push_back(std::make_shared<ThrowawayReader<Vec3<unsigned char> > >());
                        }
                    } else {
                        throw std::runtime_error("unhandled normal type '" + coordinateType + "'");
                    }

                } else if (propertyNames[j] == "alpha") {

                    // TODO: might actually want to keep this sometimes
                    if (propertyTypeStrings[j] == "uchar") {
                        readers.push_back(std::make_shared<ThrowawayReader<unsigned char> >());
                    } else {
                        throw std::runtime_error("unexpected alpha type");
                    }

                } else {

                    throw std::runtime_error("unrecongized property " + propertyNames[j]);

                }

            }



        } else if (s == "face") {

            for (int j = 0; j < propertyNames.size(); ++j) {

                if (propertyNames[j] == "vertex_index" || propertyNames[j] == "vertex_indices") {

                    const std::string indexType = propertyTypeStrings[j];

                    if (indexType == "list uchar int") {
                        if (faces) {
                            faces->Resize(nElements);
                            readers.push_back(std::make_shared<ListReader<3> >(*faces));
                        } else {
                            readers.push_back(std::make_shared<ThrowawayListReader>());
                        }
                    } else {
                        throw std::runtime_error("unhandled face type index '" + indexType + "'");
                    }

                } else {

                    throw std::runtime_error("unrecognized property " + propertyNames[j]);

                }

            }

        } else {

            throw std::runtime_error("unrecognized element type '" + s);

        }

        // read the elements
        for (int n = 0; n < nElements; ++n) {

            for (auto reader : readers) {

                reader->Read(stream, n);

            }

        }

    }


}

void ReadPly(NDT::ManagedVector<Vec3<float> > & vertices,
             const std::string filename) {

    ReadPly(&vertices, nullptr, nullptr, nullptr, filename);

}

void ReadPly(NDT::ManagedVector<Vec3<float> > & vertices,
             NDT::ManagedVector<Vec3<float> > & normals,
             const std::string filename) {

    ReadPly(&vertices, &normals, nullptr, nullptr, filename);

}

void ReadPly(NDT::ManagedVector<Vec3<float> > & vertices,
             NDT::ManagedVector<Vec3<float> > & normals,
             NDT::ManagedVector<Vec3<unsigned char> > & colors,
             const std::string filename) {

    ReadPly(&vertices, &normals, &colors, nullptr, filename);

}

void ReadPly(NDT::ManagedVector<Vec3<float> > & vertices,
             NDT::ManagedVector<Vec3<int> > & faces,
             const std::string filename) {

    ReadPly(&vertices, nullptr, nullptr, &faces, filename);

}

void ReadPly(NDT::ManagedVector<Vec3<float> > & vertices,
             NDT::ManagedVector<Vec3<float> > & normals,
             NDT::ManagedVector<Vec3<int> > & faces,
             const std::string filename) {

    ReadPly(&vertices, &normals, nullptr, &faces, filename);

}

void ReadPly(NDT::ManagedVector<Vec3<float> > & vertices,
             NDT::ManagedVector<Vec3<float> > & normals,
             NDT::ManagedVector<Vec3<unsigned char> > & colors,
             NDT::ManagedVector<Vec3<int> > & faces,
             const std::string filename) {

    ReadPly(&vertices, &normals, &colors, &faces, filename);

//    std::ifstream stream(filename);
//    std::string word;
//
//    auto getExpectedWord = [&stream, &word, &filename](const std::string expected) {
//
//        stream >> word;
//        if (word != expected) {
//            throw std::runtime_error("while reading " + filename + ":\n did not find expected word " + expected +
//                                     "\n instead, we get " + word);
//        }
//
//    };
//
//    getExpectedWord("ply");
//    getExpectedWord("format");
//    getExpectedWord("ascii");
//
//    double version;
//    stream >> version;
//
//    if (version != 1.0) {
//        throw std::runtime_error("expected version 1");
//    }
//
//    getExpectedWord("element");
//    getExpectedWord("vertex");
//
//    int vertexCount;
//    stream >> vertexCount;
//
//    vertices.Resize(vertexCount);
//    normals.Resize(vertexCount);
//    colors.Resize(vertexCount);
//
//    getExpectedWord("property");
//    getExpectedWord("float");
//    getExpectedWord("x");
//
//    getExpectedWord("property");
//    getExpectedWord("float");
//    getExpectedWord("y");
//
//    getExpectedWord("property");
//    getExpectedWord("float");
//    getExpectedWord("z");
//
//    getExpectedWord("property");
//    getExpectedWord("float");
//    getExpectedWord("nx");
//
//    getExpectedWord("property");
//    getExpectedWord("float");
//    getExpectedWord("ny");
//
//    getExpectedWord("property");
//    getExpectedWord("float");
//    getExpectedWord("nz");
//
//    getExpectedWord("property");
//    getExpectedWord("uchar");
//    getExpectedWord("red");
//
//    getExpectedWord("property");
//    getExpectedWord("uchar");
//    getExpectedWord("green");
//
//    getExpectedWord("property");
//    getExpectedWord("uchar");
//    getExpectedWord("blue");
//
//    getExpectedWord("element");
//    getExpectedWord("face");
//
//    int faceCount;
//    stream >> faceCount;
//
//    faces.Resize(faceCount);
//
//    getExpectedWord("property");
//    getExpectedWord("list");
//    getExpectedWord("uchar");
//    getExpectedWord("int");
//    getExpectedWord("vertex_index");
//
//    getExpectedWord("end_header");
//
//    for (int i = 0; i < vertexCount; ++i) {
//        stream >> vertices(i)(0);
//        stream >> vertices(i)(1);
//        stream >> vertices(i)(2);
//        stream >> normals(i)(0);
//        stream >> normals(i)(1);
//        stream >> normals(i)(2);
//        int v;
//        stream >> v; colors(i)(0) = v;
//        stream >> v; colors(i)(1) = v;
//        stream >> v; colors(i)(2) = v;
//        normals(i) *= -1;
//    }
//
//    for ( int i = 0; i < faceCount; ++i) {
//        int nIndices;
//        stream >> nIndices;
//        if (nIndices != 3) {
//            throw std::runtime_error("expecting triangle mesh");
//        }
//
//        stream >> faces(i)(0);
//        stream >> faces(i)(2);
//        stream >> faces(i)(1);
//
//    }

}


} // namespace vu