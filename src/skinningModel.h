#ifndef SKINNING_MODEL_H
#define SKINNING_MODEL_H

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

struct Mesh
{
  std::vector<V3i> triangles;
  std::vector<V3f> vertices;
  std::vector<V3f> normals;
  std::vector<V3f> colors;
};

std::vector<V3f> calcVertexNormals(const std::vector<V3f>& vertices,const std::vector<V3i>& triangles)
{
  std::vector<V3f> faceNormals(triangles.size());
  for(int i=0;i<triangles.size();i++)
  {
    const V3i t = triangles[i];
    faceNormals[i] = normalize(cross(vertices[t[1]]-vertices[t[0]],
                                     vertices[t[2]]-vertices[t[0]]));
  }

  std::vector<V3f> vertexNormals = std::vector<V3f>(vertices.size(),V3f(0,0,0));
  for(int i=0;i<triangles.size();i++)
  {
    for(int j=0;j<3;j++)
    {
      vertexNormals[triangles[i][j]] += faceNormals[i];
    }
  }
  for(int i=0;i<vertexNormals.size();i++)
  {
    vertexNormals[i] = normalize(vertexNormals[i]);
  }

  return vertexNormals;
}

Mesh loadMeshFromOBJ(const std::string& fileName)
{
  printf("fileName=%s\n",fileName.c_str());

  Mesh mesh;

  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;

  std::string err;
  bool ret = tinyobj::LoadObj(&attrib,&shapes,&materials,&err,fileName.c_str(),"",true);

  if (!err.empty()) { printf("%s\n", err.c_str()); return mesh; }

  if (!ret) { return mesh; }

  int numTriangles = 0;
  for (int s=0;s<shapes.size();s++)
  {
    numTriangles += shapes[s].mesh.num_face_vertices.size();
  }
  mesh.triangles = std::vector<V3i>(numTriangles);

  int triangleIndex = 0;
  for (int s=0;s<shapes.size();s++)
  {
    for (int f=0;f<shapes[s].mesh.num_face_vertices.size();f++)
    {
      mesh.triangles[triangleIndex] = V3i(shapes[s].mesh.indices[3*f+0].vertex_index,
                                          shapes[s].mesh.indices[3*f+1].vertex_index,
                                          shapes[s].mesh.indices[3*f+2].vertex_index);
      triangleIndex++;
    }
  }

  const int numVertices = attrib.vertices.size()/3;
  mesh.vertices = std::vector<V3f>(numVertices);
  for (int i=0;i<numVertices;i++)
  {
    mesh.vertices[i] = V3f(attrib.vertices[i*3+0],
                           attrib.vertices[i*3+1],
                           attrib.vertices[i*3+2]);
  }

  mesh.normals = calcVertexNormals(mesh.vertices,mesh.triangles);

  mesh.colors.resize(mesh.vertices.size());

  return mesh;
}

bool saveMeshToObj(const std::string& filename, const std::vector<V3f>& vertices, const std::vector<V3i>& triangles)
{
  FILE* fw = fopen(filename.c_str(), "w");
  if (!fw)
  {
    return false;
  }

  int N = vertices.size();
  int M = triangles.size();

  for (int i=0;i<N;i++)
  {
    const V3f& v = vertices[i];
    fprintf(fw, "v %f %f %f\n", v(0), v(1), v(2));
  }
  for (int i=0;i<M;i++)
  {
    const V3i& f = triangles[i];
    fprintf(fw, "f %d %d %d\n", f(0)+1, f(1)+1, f(2)+1);
  }

  fclose(fw);

  return true;
}

template<typename T>
struct Joint
{
  Vec<3,T> offset;
  int parentId;
};

class SkinningModel
{
  V3f evalJointPosition(const int jointId)
  {
    const int parentId = joints[jointId].parentId;

    if (parentId >= 0)
    {
      return joints[jointId].offset + evalJointPosition(parentId);
    }
    else
    {
      return joints[jointId].offset;
    }
  }

public:
  std::string objFileName;
  std::vector<Joint<float>> joints;
  std::vector<V3f> vertices;
  std::vector<V3i> triangles;
  std::vector<V3f> normals;
  std::vector<V3f> colors;
  A2f weights;
  
  std::vector<V3f> JColors;

  bool load(const char* filename)
  {
    FILE* fr = fopen(filename, "r");

    if (!fr) { return false; }

    char strBuffer[256];
    fscanf(fr, "%s\n", strBuffer);
    objFileName = std::string(strBuffer);
    Mesh mesh = loadMeshFromOBJ(objFileName);
    vertices = mesh.vertices;
    triangles = mesh.triangles;
    int N = vertices.size();
    int numJoints;
    fscanf(fr, "%d\n", &numJoints);
    joints.resize(numJoints);
    for (int i=0;i<numJoints;i++)
    {
      V3f& offset = joints[i].offset;
      fscanf(fr, "%f %f %f %d\n", &(offset[0]), &(offset[1]), &(offset[2]), &(joints[i].parentId));
    }
    weights = A2f(numJoints, N);
    for (int i=0;i<N;i++)
    {
      for (int j=0;j<numJoints-1;j++)
      {
        fscanf(fr, "%f ", &(weights(j, i)));
      }
      fscanf(fr, "%f\n", &(weights(numJoints-1, i)));
    }

    fclose(fr);

    normals = calcVertexNormals(vertices, triangles);
    // compute vertex colors
    {
      JColors.resize(24);
      JColors[0] = V3f(1.0f, 0.0f, 0.0f); // kostrc
      JColors[1] = V3f(0.0f, 1.0f, 0.0f); // leva kycel
      JColors[2] = V3f(0.0f, 0.0f, 1.0f); // prava kycel
      JColors[3] = V3f(1.0f, 1.0f, 0.0f); // pupek
      JColors[4] = V3f(1.0f, 0.0f, 1.0f); // levy koleno
      JColors[5] = V3f(0.0f, 1.0f, 1.0f); // pravy koleno
      JColors[6] = V3f(0.5f, 0.5f, 0.5f); // nadpupek
      JColors[7] = V3f(0.5f, 1.0f, 0.0f); // levy kotnik
      JColors[8] = V3f(0.5f, 0.0f, 0.0f); // pravy kotnik
      JColors[9] = V3f(1.0f, 0.5f, 0.0f); // hrudnik
      JColors[10] = V3f(0.5f, 0.125f, 0.125f); // leva spicka
      JColors[11] = V3f(0.0f, 0.0f, 0.5f); // prava spicka
      JColors[12] = V3f(0.0f, 0.5f, 0.0f); // krk
      JColors[13] = V3f(0.0f, 0.5f, 0.5f); // leva lopatka
      JColors[14] = V3f(0.5f, 0.5f, 0.0f); // prava lopatka
      JColors[15] = V3f(0.0f, 0.0f, 0.0f); // hlava
      JColors[16] = V3f(1.0f, 0.5f, 0.5f); // levy rameno
      JColors[17] = V3f(0.9f, 0.6f, 0.125f); // pravy rameno
      JColors[18] = V3f(0.25f, 0.0f, 0.5f); // levy loket
      JColors[19] = V3f(0.5f, 0.0f, 0.25f); // pravy loket
      JColors[20] = V3f(0.25f,0.75f,0.25f); // levy zapesti
      JColors[21] = V3f(0.25f,0.25f,0.75f); // pravy zapesti
      JColors[22] = V3f(0.75f,0.25f,0.25f); // konec levy ruky
      JColors[23] = V3f(1.0f,1.0f,1.0f); // konec pravy ruky

      colors.resize(N);
      for (int i=0;i<N;i++)
      {
        V3f c(0,0,0);
        for (int j=0;(j<numJoints) && (j<22);j++)
        {
          c += weights(j,i)*JColors[j];
        }
        colors[i] = c;
      }
    }

    return true;
  }

  //void save(const std::string& skinFileName, const std::string& objFileName)
  void save(const char* node, const char* dir, const char* fname)
  {
    char skinFullPath[_MAX_PATH];
    char objFullPath[_MAX_PATH];

    _makepath(skinFullPath, node, dir, fname, ".skin");
    _makepath(objFullPath, node, dir, fname, ".obj");

    //FILE* fw = fopen(skinFileName.c_str(), "w");
    FILE* fw = fopen(skinFullPath, "w");

    if (!fw) { return; }

    //fprintf(fw, "%s\n", objFileName.c_str());
    fprintf(fw, "%s.obj\n", fname);

    //saveMeshToObj(objFileName, vertices, triangles);
    saveMeshToObj(objFullPath, vertices, triangles);

    int N = vertices.size();
    
    int numJoints = weights.width();
    fprintf(fw, "%d\n", numJoints);
    for (int i=0;i<numJoints;i++)
    {
      V3f& offset = joints[i].offset;
      fprintf(fw, "%f %f %f %d\n", offset[0], offset[1], offset[2], joints[i].parentId);
    }
    for (int i=0;i<N;i++)
    {
      for (int j=0;j<numJoints-1;j++)
      {
        fprintf(fw, "%f ", weights(j, i));
      }
      fprintf(fw, "%f\n", weights(numJoints-1, i));
    }

    fclose(fw);
  }

  std::vector<V3f> getRestPoseJointPositions()
  {
    std::vector<V3f> restPoseJointPositions(joints.size());

    for (int i=0;i<restPoseJointPositions.size();i++)
    {
      restPoseJointPositions[i] = evalJointPosition(i);
    }

    return restPoseJointPositions;
  }
};

#endif
