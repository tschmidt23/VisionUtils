/***
 * © Tanner Schmidt 2018
 */

#ifdef HAS_PANGOLIN

#include <vu/Mesh/BufferedMesh.h>
#include <vu/Mesh/PlyIo.h>

namespace vu {

BufferedMesh::BufferedMesh(const std::string meshFilename) {

    ReadPly(meshVertices_, meshNormals_, meshColors_, meshFaces_, meshFilename);

    vertexBuffer_.Reinitialise(pangolin::GlArrayBuffer, meshVertices_.Count(), GL_FLOAT, 3, GL_STATIC_DRAW);
    vertexBuffer_.Upload(meshVertices_.Data(), meshVertices_.SizeBytes());

    normalBuffer_.Reinitialise(pangolin::GlArrayBuffer, meshNormals_.Count(), GL_FLOAT, 3, GL_STATIC_DRAW);
    normalBuffer_.Upload(meshNormals_.Data(), meshNormals_.SizeBytes());

    colorBuffer_.Reinitialise(pangolin::GlArrayBuffer, meshColors_.Count(), GL_UNSIGNED_BYTE, 3, GL_STATIC_DRAW);
    colorBuffer_.Upload(meshColors_.Data(), meshColors_.SizeBytes());

    indexBuffer_.Reinitialise(pangolin::GlElementArrayBuffer, meshFaces_.Count() * 3, GL_INT, 3, GL_STATIC_DRAW);
    indexBuffer_.Upload(meshFaces_.Data(), meshFaces_.SizeBytes());

}

void BufferedMesh::RenderVCI() const {

    glEnableClientState(GL_VERTEX_ARRAY);
    vertexBuffer_.Bind();
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glEnableClientState(GL_COLOR_ARRAY);
    colorBuffer_.Bind();
    glColorPointer(3, GL_UNSIGNED_BYTE, 0, 0);
    indexBuffer_.Bind();
    glDrawElements(GL_TRIANGLES, indexBuffer_.num_elements, GL_UNSIGNED_INT,0);
    indexBuffer_.Unbind();
    colorBuffer_.Unbind();
    vertexBuffer_.Unbind();
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

}

void BufferedMesh::RenderVNCI(const pangolin::GlBuffer & colorBuffer) const {

    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    glEnableClientState(GL_VERTEX_ARRAY);
    vertexBuffer_.Bind();
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glEnableClientState(GL_NORMAL_ARRAY);
    normalBuffer_.Bind();
    glNormalPointer(GL_FLOAT, 0, 0);
    glEnableClientState(GL_COLOR_ARRAY);
    colorBuffer.Bind();
    glColorPointer(3, GL_UNSIGNED_BYTE, 0, 0);
    indexBuffer_.Bind();
    glDrawElements(GL_TRIANGLES, indexBuffer_.num_elements, GL_UNSIGNED_INT,0);
    indexBuffer_.Unbind();
    colorBuffer.Unbind();
    normalBuffer_.Unbind();
    vertexBuffer_.Unbind();
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

    glDisable(GL_LIGHTING);

}

} // namespace vu

#endif // HAS_PANGOLIN