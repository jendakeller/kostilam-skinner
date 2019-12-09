#include "glew.h"
#include "gui_GL.h"

#include "jzq.h"
#include "gui.h"

#include "geometry.h"
#include "jzq_bvh.h"

#include "Dual.h"
#include "lbfgs.h"

#include "skinningModel.h"

#include "timer.h"

#include <stdio.h>
#include <stdlib.h>

#define ID __COUNTER__


void drawLineGL(const V3f& p,
                const V3f& q)
{
  glVertex3f(p(0),p(1),p(2));
  glVertex3f(q(0),q(1),q(2));
}

void drawGrid(float y)
{
  float k = 0.25f;
  int r = 8;
  glColor3f(0.4,0.4,0.4);
  glLineWidth(1);
  glBegin(GL_LINES);
  for(int i=-r;i<=+r;i++)
  {
    drawLineGL(V3f(-float(r)*k,y,float(i)*k),
               V3f(+float(r)*k,y,float(i)*k));

    drawLineGL(V3f(float(i)*k,y,-float(r)*k),
               V3f(float(i)*k,y,+float(r)*k));
  }
  glEnd();
}

template<typename T>
Vec<3,T> e2p(const Vec<2,T>& x) { return Vec<3,T>(x(0),x(1),T(1)); }

template<typename T>
Vec<2,T> p2e(const Vec<3,T>& x) { return Vec<2,T>(x(0)/x(2),x(1)/x(2)); }

template<typename T>
Vec<3,T> p2e(const Vec<4,T>& x) { return Vec<3,T>(x(0)/x(3),x(1)/x(3),x(2)/x(3)); }

template<typename T>
Vec<4,T> e2p(const Vec<3,T>& x) { return Vec<4,T>(x(0),x(1),x(2),T(1)); }



#define BETAS_SIZE 10
#define JOINT_SIZE 24

template<typename T>
class SMPL
{
  std::string objFileName;
  std::vector<V3f> tVertices;  // template mesh vertices
  std::vector<V3i> triangles; // template mesh triangles
  std::vector<std::vector<V3f>> S; // bases vectors
  std::vector<std::vector<float>> JR; // J regressor
  std::vector<T> betas;

  std::vector<Vec<3,T>> J; // Joints
  std::vector<Vec<3,T>> vertices;
  
  void setupMeshAndJoints()
  {
    int N = tVertices.size();

    vertices.resize(N);
    for (int i=0;i<N;i++)
    {
      vertices[i] = Vec<3,T>(tVertices[i][0], tVertices[i][1], tVertices[i][2]);
    }

    for (int i=0;i<BETAS_SIZE;i++)
    {
      std::vector<V3f>& Si = S[i];
      T beta = betas[i];
      for (unsigned int j=0;j<N;j++)
      {
        vertices[j] += Vec<3,T>(beta*Si[j][0], beta*Si[j][1], beta*Si[j][2]);
      }
    }

    for (int i=0;i<J.size();i++)
    {
      J[i] = Vec<3,T>(0.0f,0.0f,0.0f);

      for (int j=0;j<N;j++)
      {
        float jr = JR[i][j];
        J[i] += Vec<3,T>(jr*vertices[j][0], jr*vertices[j][1], jr*vertices[j][2]);
      }
    }
  }

public:
  std::vector<int> JParents;
  std::vector<V3f> JColors;
  A2f weights;
  
  SMPL()
  {
    J.resize(JOINT_SIZE);

    JColors.resize(JOINT_SIZE);
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

    JParents.resize(JOINT_SIZE);
    JParents[0] = -1;
    JParents[1] = 0;
    JParents[2] = 0;
    JParents[3] = 0;
    JParents[4] = 1;
    JParents[5] = 2;
    JParents[6] = 3;
    JParents[7] = 4;
    JParents[8] = 5;
    JParents[9] = 6;
    JParents[10] = 7;
    JParents[11] = 8;
    JParents[12] = 9;
    JParents[13] = 9;
    JParents[14] = 9;
    JParents[15] = 12;
    JParents[16] = 13;
    JParents[17] = 14;
    JParents[18] = 16;
    JParents[19] = 17;
    JParents[20] = 18;
    JParents[21] = 19;
    JParents[22] = 20;
    JParents[23] = 21;

    betas = std::vector<T>(BETAS_SIZE, 0.0f);
  }
  ~SMPL() {}

  std::vector<V3f> getTriangleNormals(const std::vector<V3f>& vertices, const std::vector<V3f>& normals, const std::vector<V3i>& triangles)
  {
    int N = triangles.size();

    std::vector<V3f> tNormals(N);

    for (int i=0;i<N;i++)
    {
      const V3i& tri = triangles[i];

      const V3f& n0 = normals[tri(0)];
      const V3f& n1 = normals[tri(1)];
      const V3f& n2 = normals[tri(2)];

      tNormals[i] = normalize(n0 + n1 + n2);
    }

    return tNormals;
  }

  inline int bvhGetNearestTriangleID(const BVH& bvh,
                            const std::vector<V3f>& vertices,
                            const std::vector<V3i>& triangles,
                            const std::vector<V3f>& triangleNormals,
                            const V3f& p,
                            const V3f& n,
                            const float dotThr=0.0)
  {
    int stack[64];
    int top = 0;
    stack[top++] = 0;

    float minDist = FLT_MAX;
    int minId = -1;

    while(top>0)
    {
      const int nodeIndex = stack[--top];
      const BVHNode& node = bvh.nodes[nodeIndex];

      if (node.axis==255)
      {
        for(int i=0;i<node.numPrims;i++)
        {
          const int id = bvh.ids[node.firstPrimId+i];

          if (dot(n,triangleNormals[id]) > dotThr)
          {
            const V3i triangle = triangles[id];

            const V3f& v0 = vertices[triangle[0]];
            const V3f& v1 = vertices[triangle[1]];
            const V3f& v2 = vertices[triangle[2]];

            float dist = distanceToTriangleSqr(Vec3f(p),v0,v1,v2);
            if (dist < minDist)
            {
              minDist = dist;
              minId = id;
            }
          }
        }
      }
      else
      {
        int childIds[2] = { nodeIndex+1, node.otherChildId };
        float childDists[2];

        for(int i=0;i<2;i++) { childDists[i] = distanceToBBoxSqr(p,bvh.nodes[childIds[i]].bbox); }

        if (childDists[1]>childDists[0])
        {
          std::swap(childIds[0],childIds[1]);
          std::swap(childDists[0],childDists[1]);
        }

        for(int i=0;i<2;i++)
        {
          if (childDists[i]<=minDist) { stack[top++] = childIds[i]; }
        }
      }
    }

    return minId;
  }

  // Compute barycentric coordinates (u, v, w) for
  // point p with respect to triangle (a, b, c)
  void barycentric(const V3f& p, const V3f& a, const V3f& b, const V3f& c, float* u, float* v, float* w)
  {
    V3f v0 = b - a, v1 = c - a, v2 = p - a;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    *v = (d11 * d20 - d01 * d21) / denom;
    *w = (d00 * d21 - d01 * d20) / denom;
    *u = 1.0f - *v - *w;
  }
  
  SkinningModel export2SkinningModel(const std::vector<V3f>& meshVertices, const std::vector<V3i>& meshTriangles, const std::vector<V3f>& angles, const V3f& translate, const float scale)
  {
    SkinningModel skinningModel;

    int N = meshVertices.size();
    int M = angles.size();

    setupMeshAndJoints();

    std::vector<Joint<float>> joints = makeModelFromSMPL(J, JParents);
    std::vector<Joint<float>> joints0 = joints;

    skinningModel.objFileName = objFileName;
    skinningModel.joints = joints;
    skinningModel.vertices = meshVertices;
    skinningModel.triangles = meshTriangles;
    skinningModel.normals = calcVertexNormals(skinningModel.vertices, skinningModel.triangles);
    skinningModel.weights = A2f(M, N);

    std::vector<V3f> angles0(angles.size(), V3f(0,0,0));
  
    std::vector<V3f> tvertices = scaleAndTranslateVertices(deformVertices(vertices, weights, joints0, angles0, joints, angles0),
                                                           scale,
                                                           translate);

    vertices = scaleAndTranslateVertices(deformVertices(vertices, weights, joints0, angles0, joints, angles),
                                         scale,
                                         translate);

    const BVH bvh = bvhBuild(vertices, triangles);
    
    printf("start remapping\n");
    double t0 = timerGet();
    
    std::vector<V3f> nvertices(N);

    std::vector<V3f> normals = calcVertexNormals(vertices, triangles);
    std::vector<V3f> triangleNormals = getTriangleNormals(vertices, normals, triangles);
    for (int i=0;i<N;i++)
    {
      const V3f& p = meshVertices[i];
      const V3f& n = skinningModel.normals[i];

      int triId = bvhGetNearestTriangleID(bvh, vertices, triangles, triangleNormals, p, n, -1.0f);
      if (triId > -1)
      {
        V3i tri = triangles[triId];

        float u,v,w;
        barycentric(p, vertices[tri(0)], vertices[tri(1)], vertices[tri(2)], &u, &v, &w);
        nvertices[i] = u*tvertices[tri(0)] + v*tvertices[tri(1)] + w*tvertices[tri(2)];
        for (int j=0;j<M;j++)
        {
          skinningModel.weights(j,i) = u*weights(j,tri(0)) + v*weights(j,tri(1)) + w*weights(j,tri(2));
        }
      }
      else
      {
        printf("BVH failed to find nearest triangle!\n");
        for (int j=0;j<M;j++)
        {
          skinningModel.weights(j,i) = 0.0f;
        }
      }
    }
    printf("stop remapping ... %fs\n", (timerGet() - t0)*0.001);

    std::vector<V3f> anglesTPose(angles.size(), V3f(0,0,0));
    skinningModel.vertices = deformVertices(invScaleAndTranslateVertices(skinningModel.vertices, scale, translate), skinningModel.weights, joints0, angles, joints, anglesTPose);
    
    return skinningModel;
  }

  bool loadModel(const std::string smplFileName)
  {
    FILE *fr = fopen(smplFileName.c_str(), "r");

    if (!fr)
    {
      return false;
    }

    char buffer[100];
    fscanf(fr, "%s\n", buffer);
    objFileName = buffer;
    Mesh tMesh = loadMeshFromOBJ(objFileName.c_str());
    if (tMesh.vertices.size() == 0)
    {
      fclose(fr);
      return false;
    }
    tVertices = tMesh.vertices;
    triangles = tMesh.triangles;
    int N = tVertices.size();

    vertices.resize(N);
    for (int i=0;i<N;i++)
    {
      vertices[i] = Vec<3,T>(tVertices[i][0], tVertices[i][1], tVertices[i][2]);
    }

    // load bases vectors
    S.resize(BETAS_SIZE);
    for (int i=0;i<BETAS_SIZE;i++)
    {
      std::vector<V3f>& Si = S[i];
      Si.resize(N);
      for (int j=0;j<N;j++)
      {
        V3f vb;
        fscanf(fr, "%f %f %f\n", &vb(0), &vb(1), &vb(2));
        Si[j] = vb;
      }
    }

    // load weights
    weights = A2f(JOINT_SIZE, N);
    for (int i=0;i<N;i++)
    {
      int j;
      for (j=0;j<JOINT_SIZE-1;j++)
      {
        fscanf(fr, "%f ", &(weights(j,i)));
      }
      fscanf(fr, "%f\n", &(weights(j,i)));

      // compute vertex colors
      V3f c(0.0f, 0.0f, 0.0f);
      for (j=0;j<JOINT_SIZE;j++)
      {
        c += weights(j,i)*JColors[j];
      }
      //mesh.colors[i] = c;
      //TODO: vymyslet jak handlovat barvu
    }

    // load regressor
    JR.resize(JOINT_SIZE);
    for (int i=0;i<JOINT_SIZE;i++)
    {
      std::vector<float>& JRi = JR[i];
      JRi.resize(N);
      for (int j=0;j<N;j++)
      {
        fscanf(fr, "%f\n", &(JRi[j]));
      }
    }

    fclose(fr);

    return true;
  }

  void setupBetas(const std::vector<T>& betas)
  {
    for (int i=0;i<BETAS_SIZE;i++)
    {
      this->betas[i] = betas[i];
    }
  }

  Mesh getMesh()
  {
    setupMeshAndJoints();

    Mesh mesh;
    mesh.vertices = vertices;
    mesh.normals = normals;
    mesh.triangles = triangles;

    return mesh;
  }

  void getMeshAndJoints(std::vector<Vec<3,T>>& vertices, std::vector<V3i>& triangles, std::vector<Vec<3,T>>& joints)
  {
    setupMeshAndJoints();

    vertices = this->vertices;
    triangles = this->triangles;
    joints = this->J;
  }
};

template<typename T>
std::vector<Joint<T>> makeModelFromSMPL(const std::vector<Vec<3,T>>& J,
                                        const std::vector<int>& JParents)
{
  int N = J.size();

  std::vector<Joint<T>> joints(N);

  for (int i=0;i<N;i++)
  {
    if (JParents[i]==-1)
    {
      joints[i].offset = J[i];
    }
    else
    {
      joints[i].offset = J[i] - J[JParents[i]];
    }
    joints[i].parentId = JParents[i];
  }

  return joints;
}


template<typename T>
Mat<3,3,T> rotationMatrix(const Vec<3,T>& angles)
{
  T cx = std::cos(angles(0));
  T cy = std::cos(angles(1));
  T cz = std::cos(angles(2));
  T sx = std::sin(angles(0));
  T sy = std::sin(angles(1));
  T sz = std::sin(angles(2));

  // Rz*Rx*Ry
  return Mat<3,3,T>(-sx*sy*sz+cy*cz,  sx*sy*cz+cy*sz, -cx*sy,
                             -cx*sz,           cx*cz,     sx,
                     sx*cy*sz+sy*cz, -sx*cy*cz+sy*sz,  cx*cy);
}

template<typename T>
const Mat<4,4,T> evalJointTransform(const int jointId,
                                    const std::vector<Joint<T>> joints,
                                    const std::vector<Vec<3,T>> angles)
{
  Mat<3,3,T> R = rotationMatrix(angles[jointId]);

  const Vec<3,T> t = joints[jointId].offset;

  Mat<4,4,T> Rt = Mat<4,4,T>(R(0,0),R(0,1),R(0,2),t(0),
                             R(1,0),R(1,1),R(1,2),t(1),
                             R(2,0),R(2,1),R(2,2),t(2),
                             0,     0,     0,     1);
  
  const int parentId = joints[jointId].parentId;
  if (parentId>=0)
  {
    return evalJointTransform(parentId, joints, angles)*Rt;
  }
  else
  {
    return Rt;
  }
}

template<typename T>
std::vector<Vec<3,T>> deformVertices(const std::vector<Vec<3,T>>& orgVertices,
                                     const A2f& vertexJointWeights,
                                     const std::vector<Joint<T>>& orgJoints,
                                     const std::vector<Vec<3,T>>& orgAngles,
                                     const std::vector<Joint<T>>& curJoints,
                                     const std::vector<Vec<3,T>>& curAngles)
{
  std::vector<Vec<3,T>> deformedVertices(orgVertices.size());

  std::vector<Mat<4,4,T>> Ms(orgJoints.size());

  #pragma omp parallel for
  for (int j=0;j<orgJoints.size();j++)
  {
    const Mat<4,4,T> M0 = evalJointTransform(j,orgJoints,orgAngles);
    const Mat<4,4,T> M1 = evalJointTransform(j,curJoints,curAngles);

    Ms[j] = M1*inverse(M0);
  }

  #pragma omp parallel for
  for (int i=0;i<orgVertices.size();i++)
  {
    const Vec<3,T> ov = orgVertices[i];
    Vec<3,T> dv = Vec<3,T>(0,0,0);
    for (int j=0;j<orgJoints.size();j++)
    {
      Vec<3,T> dvPart = p2e(Ms[j]*e2p(ov));
      float w = vertexJointWeights(j,i);
      dv += Vec<3,T>(w*dvPart(0), w*dvPart(1), w*dvPart(2));
    }
    deformedVertices[i] = dv;
  }

  return deformedVertices;
}

template<typename T>
std::vector<Vec<3,T>> scaleAndTranslateVertices(const std::vector<Vec<3,T>>& inVertices,
                                                const T scale,
                                                const Vec<3,T> translation)
{
  std::vector<Vec<3,T>> outVertices(inVertices.size());

  for (int i=0;i<outVertices.size();i++)
  {
    outVertices[i] = scale*inVertices[i] + translation;
  }

  return outVertices;
}

template<typename T>
std::vector<Vec<3,T>> invScaleAndTranslateVertices(const std::vector<Vec<3,T>>& inVertices,
                                                   const T scale,
                                                   const Vec<3,T> translation)
{
  std::vector<Vec<3,T>> outVertices(inVertices.size());

  for (int i=0;i<outVertices.size();i++)
  {
    outVertices[i] = (inVertices[i] - translation)/scale;
  }

  return outVertices;
}

//------------------------------------------------------------------
A3f makesdf(const std::vector<V3f>& vertices, const std::vector<V3i>& triangles, int resolution, float &maxExtent, V3f& bboxMin, V3f& bboxCenter)
{
  bboxMin = V3f(+FLT_MAX,+FLT_MAX,+FLT_MAX);
  V3f bboxMax = V3f(-FLT_MAX,-FLT_MAX,-FLT_MAX);

  for(int i=0;i<vertices.size();i++)
  {
    const V3f& v = vertices[i];
    bboxMin = std::min(bboxMin,v);
    bboxMax = std::max(bboxMax,v);
  }

 // bboxMin = bboxMin - 0.1f*(bboxMax-bboxMin);
 // bboxMax = bboxMax + 0.1f*(bboxMax-bboxMin);

  bboxCenter = 0.5f * (bboxMax + bboxMin);

  const V3f extents = bboxMax-bboxMin;
  maxExtent = max(extents);

  const V3i gridSize = V3i(std::ceil(float(resolution)*(extents[0]/maxExtent)),
                           std::ceil(float(resolution)*(extents[1]/maxExtent)),
                           std::ceil(float(resolution)*(extents[2]/maxExtent)));

  std::vector<V3f> rescaledVertices(vertices.size());
  for(int i=0;i<vertices.size();i++)
  {
    rescaledVertices[i] = ((vertices[i]-bboxMin)/maxExtent)*float(resolution);
  }

  const BVH bvh = bvhBuild(rescaledVertices, triangles);

  A3f grid(gridSize);
  #pragma omp parallel for
  for(int z=0;z<grid.depth();z++)
  for(int y=0;y<grid.height();y++)
  for(int x=0;x<grid.width();x++)
  {
    grid(x,y,z) = bvhGetSignedDistance(bvh, rescaledVertices, triangles, V3f(x,y,z));
  }

  return grid;
}

template <typename F>
lbfgsfloatval_t lbfgsEvalGradAndValueSparse(void* instance,
                                            const lbfgsfloatval_t* x,
                                            lbfgsfloatval_t* g,
                                            const int n,
                                            const lbfgsfloatval_t step)
{
  static int counter = 0;
  if(counter%100==0) printf("eval grad %d\n",counter);
  counter++;

  F& func = *((F*)instance);

  Vector< Dual<float> > arg(n);

  for(int i=0;i<n;i++)
  {
    arg(i).a = x[i];
    arg(i).b = SparseVector<float>(i);

    g[i] = 0.0f;
  }

  DenseDual<float> value;
  value.a = 0;
  value.b = Vector<float>(n);
  for(int i=0;i<n;i++) { value.b(i) = 0; }

  func(arg,value);

  for(int i=0;i<n;i++)
  {
    g[i] = value.b(i);
  }

  return value.a;
}

template <typename F>
std::vector<float> minimizeLBFGSSparse(const F& f,const std::vector<float>& x0,int maxIter)
{
  const int n = x0.size();

  lbfgsfloatval_t fx;
  lbfgsfloatval_t* x = lbfgs_malloc(n);
  lbfgs_parameter_t param;

  for(int i=0;i<n;i++) { x[i] = x0[i]; }

  lbfgs_parameter_init(&param);
  param.max_iterations = maxIter;
  param.m = 100;

  lbfgs(n,x,&fx,lbfgsEvalGradAndValueSparse<F>,0,(void*)(&f),&param);

  std::vector<float> argmin(n);
  for(int i=0;i<n;i++) { argmin[i] = x[i]; }

  lbfgs_free(x);

  return argmin;
}

struct Cost
{
  SMPL<Dual<float>>* smpl;
  A3f SDF;

  float maxExtent;
  V3f bboxMin;
  V3f bboxCenter;
  int resolution;

  std::vector<Vec<3,Dual<float>>> angles, angles0;

  static float lambdaAngle;
  static float lambdaArea;
  static float lambdaControls;

  std::vector<int>* smplControlPoints;
  std::vector<int>* meshControlPoints;

  std::vector<V3f>* meshVertices;

  bool optimizeTranslation;
  bool optimizeScale;

  double totalTime;

  Cost(std::vector<V3f>* meshVertices, const std::vector<V3i>& triangles, SMPL<Dual<float>>* smpl, std::vector<int>* smplControlPoints, std::vector<int>* meshControlPoints, bool optimizeTranslation, bool optimizeScale)
  : smpl(smpl), smplControlPoints(smplControlPoints), meshControlPoints(meshControlPoints), meshVertices(meshVertices), optimizeTranslation(optimizeTranslation), optimizeScale(optimizeScale)
  {
    std::vector<V3f>& vertices = *meshVertices;

    resolution = 256;
    SDF = makesdf(vertices, triangles, resolution, maxExtent, bboxMin, bboxCenter);
    //a3write(SDF,"sdf.A3f");

    angles0.resize(JOINT_SIZE);
    for (int i=0;i<JOINT_SIZE;i++)
    {
      angles0[i] = Vec<3,Dual<float>>(0.0f,0.0f,0.0f);
    }
    angles = angles0;

    totalTime = 0.0;
  }

  template<typename T>
  void operator()(const Vector<T>& x, DenseDual<float>& sum)
  {
    double t0 = timerGet();

    for (int i=0;i<JOINT_SIZE;i++)
    {
      sum += lambdaAngle*std::abs(x[BETAS_SIZE + i*3]);
      sum += lambdaAngle*std::abs(x[BETAS_SIZE + i*3+1]);
      sum += lambdaAngle*std::abs(x[BETAS_SIZE + i*3+2]);
    }

    Vec<3,T> dBBoxMin(+FLT_MAX,+FLT_MAX,+FLT_MAX);
    Vec<3,T> dBBoxMax(-FLT_MAX,-FLT_MAX,-FLT_MAX);

    std::vector<T> betas(BETAS_SIZE);
    for (int i=0;i<BETAS_SIZE;i++)
    {
      betas[i] = x[i];
    }
    smpl->setupBetas(betas);

    std::vector<Vec<3,T>> smplVertices;
    std::vector<V3i> triangles;
    std::vector<Vec<3,T>> smplJoints;

    smpl->getMeshAndJoints(smplVertices, triangles, smplJoints);
    
    //----------------------------------
    std::vector<Joint<T>> joints, joints0;
    joints = makeModelFromSMPL(smplJoints, smpl->JParents);
    joints0 = joints;

    for (int i=0;i<JOINT_SIZE;i++)
    {
      angles[i](0) = x[BETAS_SIZE + i*3];
      angles[i](1) = x[BETAS_SIZE + i*3+1];
      angles[i](2) = x[BETAS_SIZE + i*3+2];
    }
    Vec<3,T> translation(x[BETAS_SIZE + JOINT_SIZE*3],
                         x[BETAS_SIZE + JOINT_SIZE*3+1],
                         x[BETAS_SIZE + JOINT_SIZE*3+2]);
    
    T scale = x[BETAS_SIZE + JOINT_SIZE*3+3];
    printf("scale: %f\n", scale.a);

    std::vector<Vec<3,T>> vertices;

    if (!optimizeTranslation)
    {
      translation(0).b.n = 0;
      translation(1).b.n = 0;
      translation(2).b.n = 0;
    }
    if (!optimizeScale)
    {
      scale.b.n = 0;
    }

    vertices = scaleAndTranslateVertices(deformVertices(smplVertices, smpl->weights, joints0, angles0, joints, angles),
                                         scale,
                                         translation);
    
    //----------------------------

    int K = std::min<int>(smplControlPoints->size(), meshControlPoints->size());
    for (int i=0;i<K;i++)
    {
      const Vec<3,T>& v0 = vertices[(*smplControlPoints)[i]];
      V3f v1f = (*meshVertices)[(*meshControlPoints)[i]];
      Vec<3,T> v1(v1f(0), v1f(1), v1f(2));
      Vec<3,T> d = v0 - v1;
      sum += lambdaControls*dot(d,d);
    }

    int N = vertices.size();
    
    for (int i=0;i<N;i++)
    {
      Vec<3,T>& v = vertices[i];
      
      v(0) = ((v(0)-bboxMin(0))/maxExtent)*float(resolution);
      v(1) = ((v(1)-bboxMin(1))/maxExtent)*float(resolution);
      v(2) = ((v(2)-bboxMin(2))/maxExtent)*float(resolution);

      for (int j=0;j<3;j++)
      {
        dBBoxMin[j] = std::min(dBBoxMin[j], v[j]);
        dBBoxMax[j] = std::max(dBBoxMax[j], v[j]);
      }
    }

    T lambdaBBox(1.0f);
    if (optimizeTranslation)
    {
      lambdaBBox = T(100.0f);
    }
    for (int j=0;j<3;j++)
    {
      sum += lambdaBBox*(sqr(dBBoxMin[j]) + sqr(dBBoxMax[j] - float(SDF.size(j))));
    }
    
    int cnt = 0;
    int M = triangles.size();
    #pragma omp parallel for
    for (int i=0;i<M;i++)
    {
      V3i tri = triangles[i];
      const Vec<3,T> v0 = vertices[tri(0)];
      const Vec<3,T> v1 = vertices[tri(1)];
      const Vec<3,T> v2 = vertices[tri(2)];
      Vec<3,T> c((v0(0) + v1(0) + v2(0)) / 3.0f,
                 (v0(1) + v1(1) + v2(1)) / 3.0f,
                 (v0(2) + v1(2) + v2(2)) / 3.0f);
      
      float c0 = floor(c(0).a);
      float c1 = floor(c(1).a);
      float c2 = floor(c(2).a);
      Vec<3,T> cf(c0, c1, c2);
      V3i ci0(c0, c1, c2);

      if ((ci0(0) >= 0) && (ci0(0) < SDF.size(0)) &&
          (ci0(1) >= 0) && (ci0(1) < SDF.size(1)) &&
          (ci0(2) >= 0) && (ci0(2) < SDF.size(2)))
      {
        Vec<3,T> cr = c - cf;

        V3i ci1 = std::min(ci0+V3i(1,1,1), SDF.size()-V3i(1,1,1));

        float d000 = SDF(ci0(0), ci0(1), ci0(2));
        float d001 = SDF(ci0(0), ci0(1), ci1(2));
        float d010 = SDF(ci0(0), ci1(1), ci0(2));
        float d011 = SDF(ci0(0), ci1(1), ci1(2));
        float d100 = SDF(ci1(0), ci0(1), ci0(2));
        float d101 = SDF(ci1(0), ci0(1), ci1(2));
        float d110 = SDF(ci1(0), ci1(1), ci0(2));
        float d111 = SDF(ci1(0), ci1(1), ci1(2));

        T d00 = (T(1.0f) - cr(0))*d000 + cr(0)*d100;
        T d01 = (T(1.0f) - cr(0))*d001 + cr(0)*d101;
        T d10 = (T(1.0f) - cr(0))*d010 + cr(0)*d110;
        T d11 = (T(1.0f) - cr(0))*d011 + cr(0)*d111;

        T d0 = (T(1.0f) - cr(1))*d00 + cr(1)*d10;
        T d1 = (T(1.0f) - cr(1))*d01 + cr(1)*d11;

        Vec<3,T> e1(v1(0) - v0(0),
                    v1(1) - v0(1),
                    v1(2) - v0(2));
        Vec<3,T> e2(v2(0) - v0(0),
                    v2(1) - v0(1),
                    v2(2) - v0(2));
        Vec<3,T> e3(e1(1)*e2(2) - e1(2)*e2(1),
                    e1(2)*e2(0) - e1(0)*e2(2),
                    e1(0)*e2(1) - e1(1)*e2(0));
        T area = 0.5f*std::sqrt(e3(0)*e3(0) + e3(1)*e3(1) + e3(2)*e3(2));

        T partialSum = lambdaArea*area*sqr((1.0f - cr(2))*d0 + cr(2)*d1) + sqr((1.0f - cr(2))*d0 + cr(2)*d1);
        #pragma omp critical
        {
          sum += partialSum;
        }
      }
      else
      {
        cnt++;
      }
    }
    printf("nof vertices out of sdf:=%d/%d\n", cnt, M);
    printf("cost():sum=%f ... %fs\n", sum.a, (timerGet() - t0)*0.001);
  }
};

//------------------------------------------------------------------

int intersectModel(const Ray& ray, const std::vector<V3f>& vertices, const std::vector<V3i>& triangles)
{
  int tid = -1;
  
  float tMin = +FLT_MAX;

  int N = triangles.size();
  for (int i=0;i<N;i++)
  {
    const V3i& tri = triangles[i];
    const V3f& v0 = vertices[tri[0]];
    const V3f& v1 = vertices[tri[1]];
    const V3f& v2 = vertices[tri[2]];

    float t;
    if (intersect(ray, 0.0f, tMin, v0, v1, v2, &t))
    {
      tMin = t;
      tid = i;
    }
  }
  if (tid == -1)
  {
    return -1;
  }
  else
  {
    V3f inter = ray.o + tMin*ray.d;

    const V3i& tri = triangles[tid];
    int vid = tri[0];
    V3f d = vertices[vid] - inter;
    float minDist = dot(d,d);
    for (int i=1;i<3;i++)
    {
      d = vertices[tri[i]] - inter;
      float dist = dot(d,d);
      if (dist < minDist)
      {
        minDist = dist;
        vid = tri[i];
      }
    }
    return vid;
  }
}


int maxIters = 100;
float Cost::lambdaArea = 0.1f;
float Cost::lambdaAngle = 100.0f;
float Cost::lambdaControls = 1000000.0f;

std::string joint_names[24] = {
  "pelvis",
  "hipL",
  "hipR",
  "chestLower",
  "kneeL",
  "kneeR",
  "chestMiddle",
  "ankleL",
  "ankleR",
  "chestUpper",
  "footL",
  "footR",
  "neck",
  "clavicleL",
  "clavicleR",
  "head",
  "shoulderL",
  "shoulderR",
  "elbowL",
  "elbowR",
  "wristL",
  "wristR",
  "handL",
  "handR"
};



int main(int argc,char** argv)
{
  if (argc < 2)
  {
    printf("usage: %s input_mesh.obj\n", argv[0]);
    return 1;
  }

  const std::string objFilePath = argv[1];
  char objFileName[_MAX_FNAME];
  {
    char node[1024];
    char dir[_MAX_DIR];
    char ext[_MAX_EXT];

    _splitpath_s(objFilePath.c_str(), node, dir, objFileName, ext);
  }
  
  timerReset();

  double t0;
  t0 = timerGet();

  bool smplsControlTurn = true;
  std::vector<int> smplControlPoints;
  std::vector<int> meshControlPoints;

  std::vector<float> betas(BETAS_SIZE, 0.0f);

  std::vector<Joint<float>> joints, joints0;
  std::vector<V3f> angles(JOINT_SIZE,V3f(0,0,0));
  std::vector<V3f> angles0(JOINT_SIZE,V3f(0,0,0));

  V3f smplTranslate(0.0f, 0.0f, 0.0f);
  float smplScale = 1.0f;

  std::vector<V3f> smplJoints;
  std::vector<V3f> smplVertices;
  std::vector<V3i> smplTriangles;

  SkinningModel skinningModel;

  int modelSel = 0;
  std::vector<std::string> smplFileNames(2);
  std::vector<SMPL<float>> smplModels(2);

  smplFileNames[0] = "basicModel_m.smpl";
  if (!smplModels[0].loadModel(smplFileNames[0]))
  {
    printf("Error occurred on loading model file basicModel_m.smpl.\n");
    return -1;
  }
  smplFileNames[1] = "basicModel_f.smpl";
  if (!smplModels[1].loadModel(smplFileNames[1]))
  {
    printf("Error occurred on loading model file basicModel_f.smpl.\n");
    return -1;
  }

  FILE *fr = fopen("params.txt", "r");
  if (fr)
  {
    char fname[_MAX_FNAME];
    fscanf(fr, "%s\n", fname);
    if (strcmp(objFileName, fname) == 0)
    {
      fscanf(fr, "%d\n", &modelSel);
      fscanf(fr, "%f %f %f %f %f %f %f %f %f %f\n", &betas[0], &betas[1], &betas[2], &betas[3], &betas[4], &betas[5], &betas[6], &betas[7], &betas[8], &betas[9]);
      for (int i=0;i<JOINT_SIZE;i++)
      {
        float x,y,z;
        fscanf(fr, "%f %f %f\n", &x,&y,&z);
        angles[i](0) = x;
        angles[i](1) = y;
        angles[i](2) = z;
      }
      float x,y,z;
      fscanf(fr, "%f %f %f\n", &x,&y,&z);
      smplTranslate(0) = x;
      smplTranslate(1) = y;
      smplTranslate(2) = z;
      fscanf(fr, "%f\n", &smplScale);
    }
    fclose(fr);
  }

  SMPL<float> smpl = smplModels[modelSel];
  smpl.setupBetas(betas);

  Mesh Mmesh = loadMeshFromOBJ(objFilePath);
  if (Mmesh.vertices.size() == 0)
  {
    return -1;
  }

  bool showInputMesh = true;
  bool showSMPLModel = true;
  bool showWeights = false;
  bool showSkinningModel = false;
  bool showSkinningTPose = false;
  bool showJoints = true;
  bool useWireframe = false;

  bool optimizeTranslation = true;
  bool optimizeScale = false;
  
  A2f jointJointWeights(JOINT_SIZE, JOINT_SIZE);
  for (int i=0;i<JOINT_SIZE;i++)
  for (int j=0;j<JOINT_SIZE;j++)
  {
    jointJointWeights(j,i) = (i==j) ? 1.0f : 0.0f;
  }
  
  guiInit();

  GLContext* glCtx = new GLContext();
  glCtx->makeCurrent();
  
  int navigationMode;

  V3f cameraPosition=Vec3f(2.0,2.0,2.0);
  V3f cameraDirection=Vec3f(-0.5,-0.5,-0.5);
  V3f cameraUp=Vec3f(-0.354232,0.847255,-0.395826);
  V3f lookAt=Vec3f(-0.1,-0.1,0.1);

  int lastMouseX = 0;
  int lastMouseY = 0;
  int lastMouseDelta = -1;

  V3f lastCamPos = cameraPosition;
  V3f lastCamDir = cameraDirection;
  V3f lastCamUp  = cameraUp;
  V3f lastLookAt = lookAt;

  float dragStartX;
  float dragStartY;

  V3f dragPlaneN;
  V3f dragStartP;
  float dragPlaneD;

  int dragModel = -1;
  int dragBlob = -1;

  int dragJointId=-1;
  int dragBlobId=-1;

  int betasSelector = 0;
  int visJointWeight = 0;

  while(1)
  {
    WindowBegin(ID,spf("%s - Skinner", objFilePath.c_str()).c_str(),Opts().initialGeometry(64,64,-1,-1));

    if (windowCloseRequest()||keyDown(KeyEscape)) { break; }

    HBoxLayoutBegin(ID);
      glCtx->makeCurrent();
      GLWidgetBegin(ID,glCtx,Opts().minimumWidth(1000).minimumHeight(800));
      
      const Ray ray = Ray(cameraPosition,glUnproject(V2f(mouseX(),mouseY())));

      static int dragJointId = -1;
      static int dragBlobId = -1;
      static int dragHandle = -1;
      static bool dragPoleVector;

      static V3f dragStartHandlePos;
      static V3f dragStartPoleVecPos;
      static V3f dragStartFootPos;
      
      static V3f dragStartRootPosition;
      static V3f dragStartRootAngle;

      if (mouseDown(ButtonLeft)||mouseDown(ButtonRight))
      {
        dragBlobId =-1;
        dragStartX = mouseX();
        dragStartY = mouseY();

        lastCamPos = cameraPosition;
        lastCamDir = cameraDirection;
        lastCamUp  = cameraUp;
        lastLookAt = lookAt;

        float tmin = +FLT_MAX;
      }
      
      // bool makeControlPoint = mouseDown(ButtonLeft) && keyPressed(KeyShift);
      bool makeControlPoint = false;
      if (keyDown(KeyDelete))
      {
        if (smplsControlTurn)
        {
          if (meshControlPoints.size() > 0)
          {
            meshControlPoints.pop_back();
            smplsControlTurn = false;
          }
        }
        else
        {
          if (smplControlPoints.size() > 0)
          {
            smplControlPoints.pop_back();
            smplsControlTurn = true;
          }
        }
      }

      const bool ctrl = keyPressed(KeyControl);

      if      (mouseDown(ButtonLeft)  && dragModel==-1 && !ctrl) navigationMode = 2;
      else if (mouseDown(ButtonRight) && dragModel==-1 && !ctrl) navigationMode = 1;
      else if (mouseDown(ButtonLeft)  && dragHandle!=-1 && !ctrl) navigationMode = 3;
      
      if (mouseUp(ButtonLeft) || mouseUp(ButtonRight))
      {
        navigationMode = 0;
        dragJointId=-1;
        dragBlobId=-1;
        dragHandle = -1;
      }

      if (navigationMode==1)
      {
        float alpha = 0.005f*float(mouseY()-dragStartY);
        float beta = -0.005f*float(mouseX()-dragStartX);

        Mat3x3f R = rotationFromAxisAngle(Vec3f(0,1,0),beta) *
                    rotationFromAxisAngle(normalize(cross(lastCamUp,lastCamDir)),alpha);

        cameraPosition = lookAt + R*(lastCamPos-lookAt);
        cameraDirection = R*lastCamDir;
        cameraUp = R*lastCamUp;
      }
      else if (navigationMode==2)
      {
        V3f cameraRight = cross(cameraDirection,cameraUp);

        V3f shift = (float(mouseX()-dragStartX)*cameraRight +
                    -float(mouseY()-dragStartY)*cameraUp)*0.01f;

        cameraPosition = lastCamPos - shift;
        lookAt = lastLookAt - shift;
      }

      if (mouseWheelDelta()!=0 && navigationMode==0)
      {
        cameraPosition += float(mouseWheelDelta())/120.0f*cameraDirection*0.5f;
      }

      if (useWireframe)
      {
        glPolygonMode(GL_FRONT, GL_LINE);
        glPolygonMode(GL_BACK, GL_LINE);
      }
      else
      {
        glPolygonMode(GL_FRONT, GL_FILL);
        glPolygonMode(GL_BACK, GL_FILL);
      }

      glViewport(0,0,widgetWidth(),widgetHeight());
      glClearColor(0.2,0.2,0.2,0);
      glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluPerspective(40.0f,float(widgetWidth())/float(widgetHeight()),0.001,1000);

      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      glEnable(GL_DEPTH_TEST);

      gluLookAt(cameraPosition(0),cameraPosition(1),cameraPosition(2),
                cameraPosition(0)+cameraDirection(0),cameraPosition(1)+cameraDirection(1),cameraPosition(2)+cameraDirection(2),
                cameraUp(0),cameraUp(1),cameraUp(2));

      glLineWidth(1);
      drawGrid(-1.0f);

      glDisable(GL_CULL_FACE);

      smpl.getMeshAndJoints(smplVertices, smplTriangles, smplJoints);

      joints = makeModelFromSMPL(smplJoints, smpl.JParents);
      joints0 = joints;
      
      const std::vector<V3f> deformedVertices = scaleAndTranslateVertices(deformVertices(smplVertices,smpl.weights,joints0,angles0,joints,angles),
                                                                                         smplScale,
                                                                                         smplTranslate);
      
      const std::vector<V3f> deformedNormals = calcVertexNormals(deformedVertices,smplTriangles);

      const std::vector<V3f> deformedJoints = scaleAndTranslateVertices(deformVertices(smplJoints,jointJointWeights,joints0,angles0,joints,angles),
                                                                        smplScale,
                                                                        smplTranslate);
      
      if (makeControlPoint)
      {
        if (smplsControlTurn)
        {
          int id = intersectModel(ray, deformedVertices, smplTriangles);
          if (id >= 0)
          {
            smplControlPoints.push_back(id);
            smplsControlTurn = false;
          }
        }
        else
        {
          int id = intersectModel(ray, Mmesh.vertices, Mmesh.triangles);
          if (id >= 0)
          {
            meshControlPoints.push_back(id);
            smplsControlTurn = true;
          }
        }
      }
      
      if (showSMPLModel)
      {
        for (int i=0;i<smplControlPoints.size();i++)
        {
          int id = smplControlPoints[i];
          drawSphere(deformedVertices[id], 0.005, V3f(1.0f,0.0f,1.0f));
        }
      }
      if (showInputMesh)
      {
        for (int i=0;i<meshControlPoints.size();i++)
        {
          int id = meshControlPoints[i];
          drawSphere(Mmesh.vertices[id], 0.005, V3f(0.0f,1.0f,1.0f));
        }
      }

      if (showJoints)
      {
        for (int i=0;i<JOINT_SIZE;i++)
        {
          drawSphere(deformedJoints[i], 0.01, V3f(0.0f, 1.0f, 0.0f));
        }

        float k = 0.25f;
        glColor3f(0.7,0.7,0.7);
        glLineWidth(1);
        glBegin(GL_LINES);
        for (int i=1;i<JOINT_SIZE;i++)
        {
          drawLineGL(deformedJoints[i], deformedJoints[smpl.JParents[i]]);
        }
        glEnd();
      }
      glBegin(GL_TRIANGLES);
        if (showSMPLModel)
        {
          for (int j=0;j<smplTriangles.size();j++)
          {
            for(int i=0;i<3;i++)
            {
              const V3f& v = deformedVertices[smplTriangles[j][i]];
              const V3f& n = deformedNormals[smplTriangles[j][i]];
              
              if (showWeights)
              {
                float w = 1.0f*smpl.weights(visJointWeight,smplTriangles[j][i]);
                glColor3f(w,w,w);
              }
              else
              {
                const V3f& c = (n+V3f(1,1,1))*0.5f;
                glColor3f(c[0],c[1],c[2]);
              }
                            
              glVertex3f(v[0],v[1],v[2]);
            }
          }
        }

        if (showInputMesh)
        {
          Mesh& mesh = Mmesh;
          for(int j=0;j<mesh.triangles.size();j++)
          {
            for(int i=0;i<3;i++)
            {
              const V3f& v = mesh.vertices[mesh.triangles[j][i]];
              const V3f& n = mesh.normals[mesh.triangles[j][i]];
              if (showWeights)
              {
                const V3f& c = mesh.colors[mesh.triangles[j][i]];
                glColor3f(c[0],c[1],c[2]);
              }
              else
              {
                const V3f& c = (n+V3f(1,1,1))*0.5f;
                glColor3f(c[0],c[1],c[2]);
              }
              glVertex3f(v[0],v[1],v[2]);
            }
          }
        }

        if (showSkinningModel && skinningModel.vertices.size())
        {
          const std::vector<V3i>& triangles = skinningModel.triangles;
          const std::vector<V3f> vertices   = scaleAndTranslateVertices(deformVertices(skinningModel.vertices, skinningModel.weights, skinningModel.joints, angles0, skinningModel.joints, angles), smplScale, smplTranslate);
          const std::vector<V3f>& normals   = calcVertexNormals(vertices, triangles);
          
          for(int j=0;j<triangles.size();j++)
          {
            for(int i=0;i<3;i++)
            {
              const V3f& v = vertices[triangles[j][i]];
              const V3f& n = normals[triangles[j][i]];
              if (showWeights)
              {
                float w = 1.0f*skinningModel.weights(visJointWeight,triangles[j][i]);
                glColor3f(w,w,w);
              }
              else
              {
                const V3f& c = (n+V3f(1,1,1))*0.5f;
                glColor3f(c[0],c[1],c[2]);
              }
              glVertex3f(v[0],v[1],v[2]);
            }
          }
        }
        if (showSkinningTPose && skinningModel.vertices.size())
        {
          const std::vector<V3i>& triangles = skinningModel.triangles;
          const std::vector<V3f> vertices   = skinningModel.vertices;
          const std::vector<V3f>& normals   = calcVertexNormals(vertices, triangles);
          
          for(int j=0;j<triangles.size();j++)
          {
            for(int i=0;i<3;i++)
            {
              const V3f& v = vertices[triangles[j][i]];
              const V3f& n = normals[triangles[j][i]];
              if (showWeights)
              {
                float w = 1.0f*skinningModel.weights(visJointWeight,triangles[j][i]);
                glColor3f(w,w,w);
              }
              else
              {
                const V3f& c = (n+V3f(1,1,1))*0.5f;
                glColor3f(c[0],c[1],c[2]);
              }
              glVertex3f(v[0],v[1],v[2]);
            }
          }
        }

        glEnd();
      
      glEnable(GL_CULL_FACE);

      GLWidgetEnd();
      
      FrameBegin(ID);
      VBoxLayoutBegin(ID);
        HBoxLayoutBegin(ID);
          HSeparator(ID);
          Label(ID, "Model:",Opts().alignText(AlignHCenter));
          HSeparator(ID);
        HBoxLayoutEnd();
        HBoxLayoutBegin(ID);
          if (RadioButton(ID,"Male",0,&modelSel))
          {
            smpl = smplModels[modelSel];
            smpl.setupBetas(betas);
          }
          if (RadioButton(ID,"Female",1,&modelSel))
          {
            smpl = smplModels[modelSel];
            smpl.setupBetas(betas);
          }
        HBoxLayoutEnd();
        HBoxLayoutBegin(ID);
          HSeparator(ID);
          Label(ID, "Betas:",Opts().alignText(AlignHCenter));
          HSeparator(ID);
        HBoxLayoutEnd();
        {
          const int nofGuiItems=2;
          const int idOffset = 100;
          
          HBoxLayoutBegin(ID);
            for (int i=0;i<BETAS_SIZE;i++)
            {
              if ((i>0) && (i%5==0))
              {
                HBoxLayoutEnd();
                HBoxLayoutBegin(ID+idOffset + i*nofGuiItems);
              }
              RadioButton(ID+idOffset + i*nofGuiItems+1, spf("[%d]", i).c_str(), i, &betasSelector);
            }
          HBoxLayoutEnd();
          if (HSlider(ID,-5.0f,5.0f,&(betas[betasSelector])))
          {
            smpl.setupBetas(betas);
          }
          Label(ID, spf("betas[%d] = %.3f", betasSelector, betas[betasSelector]).c_str());
        }
        if (Button(ID, "Clear All Betas"))
        {
          for (int i=0;i<BETAS_SIZE;i++)
          {
            betas[i] = 0.0f;
          }
          smpl.setupBetas(betas);
        }
        
        HBoxLayoutBegin(ID);
          HSeparator(ID);
          Label(ID, "Thetas:",Opts().alignText(AlignHCenter));
          HSeparator(ID);
        HBoxLayoutEnd();

        {
          const float thetaMin = -4.0f;
          const float thetaMax =  4.0f;

          const int nofGuiItems=2;
          const int idOffset = 200;
          HBoxLayoutBegin(ID);
          for (int i=0;i<joints.size();i++)
          {
            if ((i>0) && (i % 4 == 0))
            {
              HBoxLayoutEnd();
              HBoxLayoutBegin(ID+idOffset + i*nofGuiItems);
            }
            RadioButton(ID+idOffset + i*nofGuiItems+1,spf("[%d]%s", i, joint_names[i]).c_str(),i,&visJointWeight);
          }
          HBoxLayoutEnd();
          Label(ID, spf("[%d]%s = %.3f, %.3f, %.3f", visJointWeight, joint_names[visJointWeight], angles[visJointWeight][0], angles[visJointWeight][1], angles[visJointWeight][2]).c_str());
          HBoxLayoutBegin(ID);
            HSlider(ID, thetaMin, thetaMax, &angles[visJointWeight][0]);
            HSlider(ID, thetaMin, thetaMax, &angles[visJointWeight][1]);
            HSlider(ID, thetaMin, thetaMax, &angles[visJointWeight][2]);
          HBoxLayoutEnd();
          HBoxLayoutBegin(ID);
            if (Button(ID, "Clear Current Joint"))
            {
              angles[visJointWeight] = V3f(0,0,0);
            }
            if (Button(ID, "Clear All Joints"))
            {
              for (int i=0;i<JOINT_SIZE;i++)
              {
                angles[i] = V3f(0,0,0);
              }
            }
            if (Button(ID, "Set A Pose"))
            {
              for (int i=0;i<JOINT_SIZE;i++)
              {
                angles[i] = V3f(0,0,0);
              }
              angles[1][2] = -0.5f;
              angles[2][2] = 0.5f;
              angles[16][2] = 1.0f;
              angles[17][2] = -1.0f;
            }
          HBoxLayoutEnd();
        }

        HBoxLayoutBegin(ID);
          HSeparator(ID);
          Label(ID, "Transformation:", Opts().alignText(AlignHCenter));
          HSeparator(ID);
        HBoxLayoutEnd();
        Label(ID, spf("t = %.3f, %.3f, %.3f", smplTranslate(0), smplTranslate(1), smplTranslate(2)).c_str());
        HBoxLayoutBegin(ID);
          HSlider(ID,-2.0f,2.0f,&smplTranslate(0));
          HSlider(ID,-2.0f,2.0f,&smplTranslate(1));
          HSlider(ID,-2.0f,2.0f,&smplTranslate(2));
        HBoxLayoutEnd();
        if (Button(ID, "Clear Translation"))
        {
          smplTranslate = V3f(0,0,0);
        }
        HBoxLayoutBegin(ID);
          Label(ID, "Scale:");
          SpinBox(ID, 0.01f, 100.0f, &smplScale);
        HBoxLayoutEnd();
        
        HBoxLayoutBegin(ID);
          HSeparator(ID);
          Label(ID, "View Options:", Opts().alignText(AlignHCenter));
          HSeparator(ID);
        HBoxLayoutEnd();
        
        HBoxLayoutBegin(ID);
          VBoxLayoutBegin(ID);
        CheckBox(ID, "Show Input Mesh", &showInputMesh);
        CheckBox(ID, "Show SMPL Model", &showSMPLModel);
        CheckBox(ID, "Show Weights", &showWeights);
          VBoxLayoutEnd();
          VBoxLayoutBegin(ID);
            CheckBox(ID, "Show Output Skinned Mesh", &showSkinningModel, Opts().enabled(skinningModel.vertices.size() > 0));
            CheckBox(ID, "Show Skinned Mesh in T-Pose", &showSkinningTPose, Opts().enabled(skinningModel.vertices.size() > 0));
            CheckBox(ID, "Show Joints", &showJoints);
          VBoxLayoutEnd();
        HBoxLayoutEnd();
        CheckBox(ID, "Use Wireframe", &useWireframe);

        HBoxLayoutBegin(ID);
          HSeparator(ID);
          Label(ID, "Fitting Options:", Opts().alignText(AlignHCenter));
          HSeparator(ID);
        HBoxLayoutEnd();

        HBoxLayoutBegin(ID);
          CheckBox(ID, "Optimize Translation", &optimizeTranslation);
          CheckBox(ID, "Optimize Scale", &optimizeScale);
        HBoxLayoutEnd();

        HBoxLayoutBegin(ID);
          Label(ID, "maxIters:");
          SpinBox(ID, 1, 10000, &maxIters);
        HBoxLayoutEnd();
        // HBoxLayoutBegin(ID);
        //   Label(ID, "lambdaArea:");
        //   SpinBox(ID, 0.01f, 1000000.0f, &Cost::lambdaArea);
        // HBoxLayoutEnd();
        // HBoxLayoutBegin(ID);
        //   Label(ID, "lambdaAngle:");
        //   SpinBox(ID, 0.01f, 1000000.0f, &Cost::lambdaAngle);
        // HBoxLayoutEnd();

        // HBoxLayoutBegin(ID);
        //   Label(ID, "lambdaControls:");
        //   SpinBox(ID, 0.01f, 1000000.0f, &Cost::lambdaControls);
        // HBoxLayoutEnd();

        if (Button(ID, "Coarse Fit"))
        {
          V3f Amin(+FLT_MAX,+FLT_MAX,+FLT_MAX);
          V3f Amax(-FLT_MAX,-FLT_MAX,-FLT_MAX);
          V3f Bmin(+FLT_MAX,+FLT_MAX,+FLT_MAX);
          V3f Bmax(-FLT_MAX,-FLT_MAX,-FLT_MAX);

          for (int i=0;i<Mmesh.vertices.size();i++)
          {
            const V3f& v = Mmesh.vertices[i];
            for (int d=0;d<3;d++)
            {
              Amin[d] = std::min(Amin[d], v[d]);
              Amax[d] = std::max(Amax[d], v[d]);
            }
          }

          for (int i=0;i<smplVertices.size();i++)
          {
            const V3f& v = smplVertices[i];
            for (int d=0;d<3;d++)
            {
              Bmin[d] = std::min(Bmin[d], v[d]);
              Bmax[d] = std::max(Bmax[d], v[d]);
            }
          }

          V3f A = Amax - Amin;
          V3f B = Bmax - Bmin;

          smplScale = dot(A,B) / dot(B,B);
          smplTranslate = 0.5f * (Amax + Amin - Bmax - Bmin);
        }
        if (Button(ID, "Fit Model To Mesh"))
        {
          SMPL<Dual<float>> smplCost;
          smplCost.loadModel(smplFileNames[modelSel]);

          Cost cost(&(Mmesh.vertices), Mmesh.triangles, &smplCost, &smplControlPoints, &meshControlPoints, optimizeTranslation, optimizeScale);

          double t0 = timerGet();
          std::vector<float> x(BETAS_SIZE + JOINT_SIZE*3 + 3 + 1, 0.0f);
          
          for (int i=0;i<BETAS_SIZE;i++)
          {
            x[i] = betas[i];
          }
          for (int i=0;i<JOINT_SIZE;i++)
          {
            x[BETAS_SIZE + i*3]   = angles[i](0);
            x[BETAS_SIZE + i*3+1] = angles[i](1);
            x[BETAS_SIZE + i*3+2] = angles[i](2);
          }
          x[BETAS_SIZE + JOINT_SIZE*3]   = smplTranslate(0);
          x[BETAS_SIZE + JOINT_SIZE*3+1] = smplTranslate(1);
          x[BETAS_SIZE + JOINT_SIZE*3+2] = smplTranslate(2);
          x[BETAS_SIZE + JOINT_SIZE*3+3] = smplScale;

          x = minimizeLBFGSSparse(cost,x,maxIters);
          printf("lbfgs time: %fs\n", (timerGet() - t0)*0.001);

          for (int i=0;i<BETAS_SIZE;i++)
          {
            betas[i] = x[i];
          }
          smpl.setupBetas(betas);

          for (int i=0;i<JOINT_SIZE;i++)
          {
            angles[i](0) = x[BETAS_SIZE + i*3];
            angles[i](1) = x[BETAS_SIZE + i*3+1];
            angles[i](2) = x[BETAS_SIZE + i*3+2];
          }

          smplTranslate(0) = x[BETAS_SIZE + JOINT_SIZE*3];
          smplTranslate(1) = x[BETAS_SIZE + JOINT_SIZE*3+1];
          smplTranslate(2) = x[BETAS_SIZE + JOINT_SIZE*3+2];
          smplScale        = x[BETAS_SIZE + JOINT_SIZE*3+3];

          // save params
          FILE *fw = fopen("params.txt", "w");
          fprintf(fw, "%s\n", objFileName);
          fprintf(fw, "%d\n", modelSel);
          fprintf(fw, "%f", x[0]);
          for (int i=1;i<BETAS_SIZE;i++)
          {
            fprintf(fw, " %f", x[i]);
          }
          fprintf(fw, "\n");
          for (int i=0;i<JOINT_SIZE;i++)
          {
            fprintf(fw, "%f %f %f\n", x[BETAS_SIZE + i*3], x[BETAS_SIZE + i*3+1], x[BETAS_SIZE + i*3+2]);
          }
          fprintf(fw, "%f %f %f\n", x[BETAS_SIZE + JOINT_SIZE*3], x[BETAS_SIZE + JOINT_SIZE*3+1], x[BETAS_SIZE + JOINT_SIZE*3+2]);
          fprintf(fw, "%f\n", x[BETAS_SIZE + JOINT_SIZE*3+3]);
          fclose(fw);
        }
        HBoxLayoutBegin(ID);
          if (Button(ID, "Remap Skinning Weights"))
          {
            skinningModel = smpl.export2SkinningModel(Mmesh.vertices, Mmesh.triangles, angles, smplTranslate, smplScale);
          }
          if (Button(ID, "Save Skinned Mesh", Opts().enabled(skinningModel.vertices.size() > 0)))
          {
            char* skinFullPath = FileSaveDialog("Save Skinned Mesh", 0, "*.skin");
            if (skinFullPath)
            {
              char node[1024];
              char dir[_MAX_DIR];
              char fname[_MAX_FNAME];
              char ext[_MAX_EXT];

              _splitpath_s(skinFullPath, node, dir, fname, ext);
              skinningModel.save(node, dir, fname);
            }
          }
        HBoxLayoutEnd();
      VBoxLayoutEnd();
      FrameEnd();
    HBoxLayoutEnd();

    WindowEnd();
    guiUpdate();
  }

  return 0;
}
