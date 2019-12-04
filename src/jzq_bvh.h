#ifndef JZQ_BVH_H_
#define JZQ_BVH_H_

#include <vector>
#include "jzq.h"

struct BVH;

inline BVH bvhBuild(const std::vector<V3f>& vertices,
                    const std::vector<V3i>& triangles);

inline bool bvhFindRayIntersection(const BVH& bvh,
                                   const std::vector<V3f>& vertices,
                                   const std::vector<V3i>& triangles,
                                   const Ray& ray,
                                   float* tHit,
                                   int*   hitTriangleId);

inline float bvhGetDistance(const BVH& bvh,
                            const std::vector<V3f>& vertices,
                            const std::vector<V3i>& triangles,
                            const V3f& p);

inline float bvhGetSignedDistance(const BVH& bvh,
                                  const std::vector<V3f>& vertices,
                                  const std::vector<V3i>& triangles,
                                  const V3f& p);

///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////

struct BBox
{
  V3f min;
  V3f max;

  BBox() {}
  BBox(const V3f& min,const V3f& max) : min(min),max(max) {}
};

struct BVHNode
{
  BBox bbox;
  union
  {
    int otherChildId;
    int firstPrimId;
  };
  unsigned short numPrims;
  unsigned char axis;
};

struct BVH
{
  std::vector<BVHNode> nodes;
  std::vector<int>     ids;
};

BBox bboxUnion(const std::vector<BBox>& bboxes,int start,int end)
{
  assert(start<end);
  assert(start>=0);
  assert(end<=bboxes.size());

  BBox bbox = BBox(V3f(+FLT_MAX,+FLT_MAX,+FLT_MAX),
                   V3f(-FLT_MAX,-FLT_MAX,-FLT_MAX));

  for(int i=start;i<end;i++)
  {
    bbox.min = std::min(bbox.min,bboxes[i].min);
    bbox.max = std::max(bbox.max,bboxes[i].max);
  }

  return bbox;
}

BBox bboxTriangle(const V3f& v0,const V3f& v1,const V3f& v2)
{
  return BBox(V3f(std::min(std::min(v0(0),v1(0)),v2(0)),
                  std::min(std::min(v0(1),v1(1)),v2(1)),
                  std::min(std::min(v0(2),v1(2)),v2(2))),
              V3f(std::max(std::max(v0(0),v1(0)),v2(0)),
                  std::max(std::max(v0(1),v1(1)),v2(1)),
                  std::max(std::max(v0(2),v1(2)),v2(2))));
}

BVHNode bvhMakeInnerNode(const BBox& bbox,const int axis,const int otherChildId)
{
  BVHNode node;

  node.bbox = bbox;
  node.axis = axis;
  node.numPrims = 0;
  node.otherChildId = otherChildId;

  return node;
}

BVHNode bvhMakeLeafNode(const BBox& bbox,const int firstPrimId,const int numPrims)
{
  BVHNode node;

  node.bbox = bbox;
  node.axis = 255;
  node.numPrims = numPrims;
  node.firstPrimId = firstPrimId;

  return node;
}


template<unsigned int N,typename T>
int argmax(const Vec<N,T>& v)
{
  int valueMax = v(0);
  int indexMax = 0;

  for(int i=1;i<N;i++)
  {
    if (v(i)>valueMax)
    {
      indexMax = i;
      valueMax = v(i);
    }
  }

  return indexMax;
}

int bvhBuildNode(int                   thisNodeIndex,
                 int                   start,
                 int                   end,
                 int                   depth,
                 std::vector<BBox>&    bboxes,
                 std::vector<V3f>&     centroids,
                 std::vector<int>&     ids,
                 std::vector<BVHNode>& nodes)
{
  const int count = end-start;
  const BBox bbox = bboxUnion(bboxes,start,end);

  if (count<=2 || depth > 24)
  {
    if (thisNodeIndex>=nodes.size()) { nodes.resize(thisNodeIndex+1); }
    nodes[thisNodeIndex] = bvhMakeLeafNode(bbox,start,count);
    const int nextNodeIndex = thisNodeIndex+1;
    return nextNodeIndex;
  }

  const int axis = argmax(bbox.max-bbox.min);
  const float split = 0.5f*(bbox.min(axis)+bbox.max(axis));

  int middle = start;
  for(int i=start;i<end;i++)
  {
    if (centroids[i](axis)<split)
    {
      std::swap(centroids[i],centroids[middle]);
      std::swap(bboxes[i],bboxes[middle]);
      std::swap(ids[i],ids[middle]);
      middle++;
    }
  }

  if (middle==start || middle==end)
  {
    middle = (start+end)/2;
  }

  const int leftChildIndex  = thisNodeIndex+1;
  const int rightChildIndex = bvhBuildNode(leftChildIndex ,start ,middle,depth+1,bboxes,centroids,ids,nodes);
  const int nextNodeIndex   = bvhBuildNode(rightChildIndex,middle,end   ,depth+1,bboxes,centroids,ids,nodes);

  if (thisNodeIndex>=nodes.size()) { nodes.resize(thisNodeIndex+1); }
  nodes[thisNodeIndex] = bvhMakeInnerNode(bbox,axis,rightChildIndex);
  return nextNodeIndex;
}


BVH bvhBuild(const std::vector<Vec3f>& vertices,
	           const std::vector<V3i>&   triangles)
{
  BVH bvh;

  const int numTris = triangles.size();

  std::vector<BBox> bboxes(numTris);
  std::vector<V3f> centroids(numTris);

  bvh.ids = std::vector<int>(numTris);

  for(int i=0;i<numTris;i++)
  {
    const BBox bbox = bboxTriangle(vertices[triangles[i][0]],
                                   vertices[triangles[i][1]],
                                   vertices[triangles[i][2]]);
    bboxes[i] = bbox;
    centroids[i] = 0.5f*(bbox.min+bbox.max);
    bvh.ids[i] = i;
  }

  bvhBuildNode(0,0,numTris,0,bboxes,centroids,bvh.ids,bvh.nodes);

  return bvh;
}

bool intersect(const Ray& ray,const float t0,const float t1,const V3f& v0,const V3f v1,const V3f& v2,float* out_t)
{
  const Vec3f e01 = v1-v0;
  const Vec3f e02 = v2-v0;

  const Vec3f s1 = cross(ray.d,e02);
  const float denom = dot(s1,e01);

  if (denom==0.0f) return false;
  const float invDenom = 1.f/denom;

  const Vec3f d = ray.o-v0;
  const float b1 = dot(d,s1)*invDenom;
  if (b1<0.0f || b1>1.0f) return false;

  const Vec3f s2 = cross(d,e01);
  const float b2 = dot(ray.d, s2)*invDenom;
  if (b2<0.0f || b1+b2>1.0f) return false;

  const float t = dot(e02,s2)*invDenom;
  if (t<t0 || t>t1) return false;

  *out_t = t;

  return true;
}

bool intersectRayBBox(const Ray& ray,float t0,float t1,const BBox& bbox)
{
  for (int i=0;i<3;i++)
  {
    //float tnear = (bbox.min(i)-ray.o(i))*ray.invd(i);
    float tnear = (bbox.min(i)-ray.o(i))/ray.d(i);
    //float tfar  = (bbox.max(i)-ray.o(i))*ray.invd(i);
    float tfar  = (bbox.max(i)-ray.o(i))/ray.d(i);
    if (tnear > tfar) std::swap(tnear,tfar);
    t0 = tnear > t0 ? tnear : t0;
    t1 = tfar  < t1 ? tfar  : t1;
    if (t0 > t1) return false;
  }
  return true;
}

bool bvhTraverseNodeRayStack(const Ray& ray,
                             const std::vector<V3i>& triangles,
                             const std::vector<V3f>& vertices,
                             const std::vector<int>& ids,
                             const std::vector<BVHNode>& nodes,
                             float* inout_tnear,
                             int*   out_id)
{
  int stack[64];
  int* stackTop = &stack[0];
  *stackTop++ = -1;

  float& tnear = *inout_tnear;
  int& id = *out_id;

  bool hit = false;

  int nodeIndex = 0;

  while(nodeIndex>=0)
  {
    const BVHNode& node = nodes[nodeIndex];

    if (intersectRayBBox(ray,0.000001f,tnear,node.bbox))
    {
      if (node.axis==255)
      {
        for(int i=0;i<node.numPrims;i++)
        {
          const int triId = ids[node.firstPrimId+i];

          const V3i triangle = triangles[triId];

          const V3f& v0 = vertices[triangle[0]];
          const V3f& v1 = vertices[triangle[1]];
          const V3f& v2 = vertices[triangle[2]];

          if (intersect(ray,0.000001f,tnear,v0,v1,v2,&tnear))
          {
            id = triId;
            hit = true;
          }
        }
        nodeIndex = *--stackTop;
      }
      else
      {
        if (ray.d[node.axis]>0.0f)
        {
          *stackTop++ = node.otherChildId;
          nodeIndex = nodeIndex+1;
        }
        else
        {
          *stackTop++ = nodeIndex+1;
          nodeIndex = node.otherChildId;
        }
      }
    }
    else
    {
      nodeIndex = *--stackTop;
    }
  }

  return hit;
}

inline bool bvhFindRayIntersection(const BVH& bvh,
                                   const std::vector<V3f>& vertices,
                                   const std::vector<V3i>& triangles,
                                   const Ray& ray,
                                   float* tHit,
                                   int*   hitTriangleId)
{
  float tnear = FLT_MAX;
  if (bvhTraverseNodeRayStack(ray,triangles,vertices,bvh.ids,bvh.nodes,&tnear,hitTriangleId))
  {
    *tHit = tnear;
    return true;
  }
  return false;
}

float distanceToBBoxSqr(const V3f& p,const BBox& b)
{
  float sqrDist = 0.0f;
  for (int i=0;i<3;i++)
  {
    float v = p[i];
    if (v<b.min[i]) { sqrDist += (b.min[i]-v)*(b.min[i]-v); }
    if (v>b.max[i]) { sqrDist += (v-b.max[i])*(v-b.max[i]); }
  }
  return sqrDist;
}

float sqrnorm(const Vec3f& x)
{
  return dot(x,x);
}

float distanceToTriangleSqr(const Vec3f& x,const Vec3f& v0,const Vec3f& _v1,const Vec3f& _v2)
{
  Vec3f diff = v0 - x;
  Vec3f _edge0 = _v1 - v0;
  Vec3f _edge1 = _v2 - v0;
  float a00 = sqrnorm(_edge0);
  float a01 = dot(_edge0,_edge1);
  float a11 = sqrnorm(_edge1);
  float b0  = dot(diff,_edge0);
  float b1  = dot(diff,_edge1);
  float c   = sqrnorm(diff);
  float det = abs(a00*a11 - a01*a01);
  float s   = a01*b1 - a11*b0;
  float t   = a01*b0 - a00*b1;
  float sqrDistance;

  //if (a00<10e-7 || a11<10e-7 || det<10e-7) { return FLT_MAX; }

  if (s + t <= det)
  {
    if (s < 0.0)
    {
      if (t < 0.0)  // region 4
      {
        if (b0 < 0.0)
        {
          t = 0.0;
          if (-b0 >= a00)
          {
            s = 1.0;
            sqrDistance = a00 + (2.0)*b0 + c;
          }
          else
          {
            s = -b0/a00;
            sqrDistance = b0*s + c;
          }
        }
        else
        {
          s = 0.0;
          if (b1 >= 0.0)
          {
            t = 0.0;
            sqrDistance = c;
          }
          else if (-b1 >= a11)
          {
            t = 1.0;
            sqrDistance = a11 + (2.0)*b1 + c;
          }
          else
          {
            t = -b1/a11;
            sqrDistance = b1*t + c;
          }
        }
      }
      else  // region 3
      {
        s = 0.0;
        if (b1 >= 0.0)
        {
          t = 0.0;
          sqrDistance = c;
        }
        else if (-b1 >= a11)
        {
          t = 1.0;
          sqrDistance = a11 + (2.0)*b1 + c;
        }
        else
        {
          t = -b1/a11;
          sqrDistance = b1*t + c;
        }
      }
    }
    else if (t < 0.0)  // region 5
    {
      t = 0.0;
      if (b0 >= 0.0)
      {
        s = 0.0;
        sqrDistance = c;
      }
      else if (-b0 >= a00)
      {
        s = 1.0;
        sqrDistance = a00 + (2.0)*b0 + c;
      }
      else
      {
        s = -b0/a00;
        sqrDistance = b0*s + c;
      }
    }
    else  // region 0
    {
      // minimum at interior point
      float invDet = (1.0)/det;

      s *= invDet;
      t *= invDet;
      sqrDistance = s*(a00*s + a01*t + (2.0)*b0) +
                    t*(a01*s + a11*t + (2.0)*b1) + c;
    }
  }
  else
  {
    float tmp0, tmp1, numer, denom;

    if (s < 0.0)  // region 2
    {
      tmp0 = a01 + b0;
      tmp1 = a11 + b1;
      if (tmp1 > tmp0)
      {
        numer = tmp1 - tmp0;
        denom = a00 - (2.0)*a01 + a11;
        if (numer >= denom)
        {
          s = 1.0;
          t = 0.0;
          sqrDistance = a00 + (2.0)*b0 + c;
        }
        else
        {
          s = numer/denom;
          t = 1.0 - s;
          sqrDistance = s*(a00*s + a01*t + (2.0)*b0) +
                        t*(a01*s + a11*t + (2.0)*b1) + c;
        }
      }
      else
      {
        s = 0.0;
        if (tmp1 <= 0.0)
        {
          t = 1.0;
          sqrDistance = a11 + (2.0)*b1 + c;
        }
        else if (b1 >= 0.0)
        {
          t = 0.0;
          sqrDistance = c;
        }
        else
        {
          t = -b1/a11;
          sqrDistance = b1*t + c;
        }
      }
    }
    else if (t < 0.0)  // region 6
    {
      tmp0 = a01 + b1;
      tmp1 = a00 + b0;
      if (tmp1 > tmp0)
      {
        numer = tmp1 - tmp0;
        denom = a00 - (2.0)*a01 + a11;
        if (numer >= denom)
        {
          t = 1.0;
          s = 0.0;
          sqrDistance = a11 + (2.0)*b1 + c;
        }
        else
        {
          t = numer/denom;
          s = 1.0 - t;
          sqrDistance = s*(a00*s + a01*t + (2.0)*b0) +
                        t*(a01*s + a11*t + (2.0)*b1) + c;
        }
      }
      else
      {
        t = 0.0;
        if (tmp1 <= 0.0)
        {
          s = 1.0;
          sqrDistance = a00 + (2.0)*b0 + c;
        }
        else if (b0 >= 0.0)
        {
          s = 0.0;
          sqrDistance = c;
        }
        else
        {
          s = -b0/a00;
          sqrDistance = b0*s + c;
        }
      }
    }
    else  // region 1
    {
      numer = a11 + b1 - a01 - b0;
      if (numer <= 0.0)
      {
          s = 0.0;
          t = 1.0;
          sqrDistance = a11 + (2.0)*b1 + c;
      }
      else
      {
        denom = a00 - (2.0)*a01 + a11;
        if (numer >= denom)
        {
          s = 1.0;
          t = 0.0;
          sqrDistance = a00 + (2.0)*b0 + c;
        }
        else
        {
          s = numer/denom;
          t = 1.0 - s;
          sqrDistance = s*(a00*s + a01*t + (2.0)*b0) +
                        t*(a01*s + a11*t + (2.0)*b1) + c;
        }
      }
    }
  }

  // Account for numerical round-off error.
  if (sqrDistance < 0.0)
  {
    return 0.0;
  }

  return sqrDistance;
}

inline float bvhGetDistance(const BVH& bvh,
                            const std::vector<V3f>& vertices,
                            const std::vector<V3i>& triangles,
 					                  const V3f& p)
{
  int stack[64];
  int top = 0;
  stack[top++] = 0;

  float minDist = FLT_MAX;

  while(top>0)
  {
    const int nodeIndex = stack[--top];
    const BVHNode& node = bvh.nodes[nodeIndex];

    if (node.axis==255)
    {
      for(int i=0;i<node.numPrims;i++)
      {
        const int id = bvh.ids[node.firstPrimId+i];

        const V3i triangle = triangles[id];

        const V3f& v0 = vertices[triangle[0]];
        const V3f& v1 = vertices[triangle[1]];
        const V3f& v2 = vertices[triangle[2]];

        minDist = std::min(minDist,distanceToTriangleSqr(Vec3f(p),v0,
                                                                  v1,
                                                                  v2));
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

  return minDist<FLT_MAX ? std::sqrt(minDist) : minDist;
}

inline float bvhGetSignedDistance(const BVH& bvh,
                                  const std::vector<V3f>& vertices,
                                  const std::vector<V3i>& triangles,
                                  const V3f& p)
{
	static const Vec3f shootDirs[26] =
	{
	  Vec3f(-0.696923,-0.696923,-0.169102),
	  Vec3f(-0.250000,-0.957107, 0.146447),
	  Vec3f( 0.288675,-0.866025, 0.408248),
	  Vec3f(-0.957107,-0.250000, 0.146446),
	  Vec3f(-0.500000,-0.500000, 0.707107),
	  Vec3f( 0.250000,-0.457107, 0.853553),
	  Vec3f(-0.866026, 0.288675, 0.408248),
	  Vec3f(-0.457107, 0.250000, 0.853553),
	  Vec3f( 0.119573, 0.119573, 0.985599),
	  Vec3f(-0.500000,-0.500000,-0.707107),
	  Vec3f( 0.146447,-0.853553,-0.500000),
	  Vec3f( 0.707107,-0.707107, 0.000000),
	  Vec3f(-0.853553, 0.146447,-0.500000),
	  Vec3f( 0.853553,-0.146447, 0.500000),
	  Vec3f(-0.707107, 0.707107,-0.000000),
	  Vec3f(-0.146447, 0.853553, 0.500000),
	  Vec3f( 0.500000, 0.500000, 0.707107),
	  Vec3f(-0.119573,-0.119573,-0.985599),
	  Vec3f( 0.457107,-0.250000,-0.853553),
	  Vec3f( 0.866026,-0.288675,-0.408248),
	  Vec3f(-0.250000, 0.457107,-0.853553),
	  Vec3f( 0.500000, 0.500000,-0.707107),
	  Vec3f( 0.957107, 0.250000,-0.146446),
	  Vec3f(-0.288675, 0.866025,-0.408248),
	  Vec3f( 0.250000, 0.957107,-0.146447),
	  Vec3f( 0.696923, 0.696923, 0.169102)
	};

  int vote = 0;

  for(int i=0;i<26;i++)
  {
    const Vec3f d = shootDirs[i];

    const Ray ray = Ray(p,d);

    float tHit;
    int hitId;
    if (bvhFindRayIntersection(bvh,vertices,triangles,ray,&tHit,&hitId))
    {
      const V3i& triangle = triangles[hitId];

      const V3f& v0 = vertices[triangle[0]];
      const V3f& v1 = vertices[triangle[1]];
      const V3f& v2 = vertices[triangle[2]];

      const V3f e01 = v1-v0;
      const V3f e02 = v2-v0;

      const Vec3f nrm = normalize(cross(e01,e02));

      if (dot(nrm,d)>0)
      {
        vote+= 2;
      }
      else
      {
        vote-= 2;
      }

      if ((std::abs(vote)-((25-i)*2))>0) { break; }
    }
    else
    {
      vote -= 1;
    }
  }

  const float dist = bvhGetDistance(bvh,vertices,triangles,p);

  return (vote > 0 ? -dist : dist);
}

/*
BVH bvhRefit(const BVH& bvh,
	   	     const std::vector<V3f>& vertices,
	         const std::vector<V3i>& triangles);

bool bvhFindNearestPoint(const BVH& bvh,
   	   	                 const std::vector<V3f>& vertices,
		                 const std::vector<V3i>& triangles,
   	                     const V3f& queryPoint,
	                           V3f* nearestPoint,
	                           int* nearestTriangleId);

bool bvhFindRayIntersection(const BVH& bvh,
   	   	                    const std::vector<V3f>& vertices,
		                    const std::vector<V3i>& triangles,
   	                        const Ray& ray,
	                        const float t0,
	                        const float t1,
	                        float* tHit,
	                        int*   hitTriangleId);

bool bvhTestIfInside(const BVH& bvh,
   	   	             const std::vector<V3f>& vertices,
		             const std::vector<V3i>& triangles,
   	                 const V3f& queryPoint);

float bvhGetSignedDistance(const BVH& bvh,
   	   	                   const std::vector<V3f>& vertices,
		                   const std::vector<V3i>& triangles,
		                   const V3f& queryPoint);
*/

#endif
