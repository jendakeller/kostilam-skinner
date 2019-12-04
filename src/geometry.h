// Copyright (C) 2014 Czech Technical University in Prague - All Rights Reserved

#ifndef GEOMETRY_H_
#define GEOMETRY_H_

#include <vector>
#include <utility>

#include "jzq.h"

struct Ray
{
  V3f o;
  V3f d;

  Ray() {}
  Ray(const V3f o,const V3f d) : o(o),d(d) {}
};

struct Splat
{
  V3f p;
  V3f c;
  Splat();
  Splat(const V3f& p,const V3f& c) : p(p),c(c) { }
};

struct BBox2
{
  V2f min;
  V2f max;

  BBox2(const V2f& min,const V2f& max) : min(min),max(max) {}
};

struct BBox3
{
  V3f min;
  V3f max;

  BBox3(const V3f& min,const V3f& max) : min(min),max(max) {}
};

typedef std::vector<BBox2> BBox2es;

inline bool solveQuadratic(float A,float B,float C,float *t0,float *t1)
{
  float discrim = B * B - 4.f * A * C;
  if (discrim < 0.) return false;
  float rootDiscrim = sqrtf(discrim);

  float q;
  if (B < 0) q = -.5f * (B - rootDiscrim);
  else       q = -.5f * (B + rootDiscrim);
  *t0 = q / A;
  *t1 = C / q;
  if (*t0 > *t1) { std::swap(*t0, *t1); }
  return true;
};

Mat3x3f rotationFromAxisAngle(const Vec3f& axis,float angle);
bool intersectSphere(const Ray& r,const V3f& center,float radius,float* tHit);
int intersectNodes(const Ray& ray,std::vector<V3f>& nodes,float radius=0.1f);
Vec3f intersectPlane(const Ray& ray,const V3f planeN,float planeD);

BBox2 updateBBox(const BBox2& oldBBox,const BBox2& newBBox);
BBox2 clipBBox(const BBox2& inBBox,const A2V3uc& image);

Mat3x3f getPrincipalRotation(std::vector<V3f>& points);

V3f e2p(const V2f& x);
V4f e2p(const V3f& x);

V2f p2e(const V3f& x);
V3f p2e(const V4f& x);

V3f projectPoint(const V3f pin, const Mat3x4f& P);

enum RotationType {ROT_X, ROT_Y, ROT_Z, ROT_YZ, ROT_XYZ, EYE};

template <typename E>
int intersectEffectors(const Ray& ray, std::vector<E>& effectors, float radius = 0.04f)
{
  float tmin = 100000.0f;
  int hit = -1;
  for (int i = 0; i<effectors.size(); i++)
  {
    float thit = 0.0f;
    if (intersectSphere(ray, effectors[i].pos, 0.04f, &thit) && thit<tmin)
    {
      tmin = thit;
      hit = i;
    }
  }
  return hit;
}

template<typename T>
Mat<3,3,T> getRotation(RotationType type, const Vec<3,T>& angle){
  switch(type) {
    case ROT_X: {
      T ca = cos(angle(0));
      T sa = sin(angle(0));
      return Mat<3,3,T>(1.0f, 0.0f, 0.0f,
                        0.0f,   ca,   sa,
                        0.0f,  -sa,   ca); }
    case ROT_Y: {
      T ca = cos(angle(0));
      T sa = sin(angle(0));
      return Mat<3,3,T>(  ca, 0.0f,  -sa,
                        0.0f, 1.0f, 0.0f,
                          sa, 0.0f,   ca); }
    case ROT_Z: {
      T ca = cos(angle(0));
      T sa = sin(angle(0));
      return Mat<3,3,T>(  ca,   sa, 0.0f,
                         -sa,   ca, 0.0f,
                        0.0f, 0.0f, 1.0f); }
    case ROT_YZ: {
      T cy = cos(angle(0));
      T cz = cos(angle(1));
      T sy = sin(angle(0));
      T sz = sin(angle(1));
      return Mat<3,3,T>(cy*cz, cy*sz,  -sy,
                          -sz,    cz, 0.0f,
                        sy*cz, sy*sz,   cy); }
    case ROT_XYZ: {
      T cx = cos(angle(0));
      T cy = cos(angle(1));
      T cz = cos(angle(2));
      T sx = sin(angle(0));
      T sy = sin(angle(1));
      T sz = sin(angle(2));
      // one of the 6 possibilities...
      return Mat<3,3,T>(-sx*sy*sz+cy*cz, sx*sy*cz+cy*sz, -cx*sy,
                        -cx*sz,                   cx*cz,     sx,
                          sx*cy*sz+sy*cz, -sx*cy*cz+sy*sz, cx*cy); }
  default:
    return Mat<3,3,T>(1.0f, 0.0f, 0.0f,
                      0.0f, 1.0f, 0.0f,
                      0.0f, 0.0f, 1.0f);
  }
}

template<int N,typename T>
float distance(const Vec<N,T>& p,const Vec<N,T>& q)
{
  return norm(q-p);
}

template<typename T> T sqr(const T& x) { return x*x; }

void rgb2hsv(float r,float g,float b,float* h,float* s,float* v);
V3f rgb2hsv(const V3f& c);
V3f hsvCone(const V3f hsv);
float phi31(float x);

#endif