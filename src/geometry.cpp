// Copyright (C) 2014 Czech Technical University in Prague - All Rights Reserved

#include "geometry.h"

Mat3x3f rotationFromAxisAngle(const Vec3f& axis,float angle)
{
  const float ux = axis(0); const float ux2 = ux*ux;
  const float uy = axis(1); const float uy2 = uy*uy;
  const float uz = axis(2); const float uz2 = uz*uz;

  const float c = cosf(angle);
  const float s = sinf(angle);

  return Mat3x3f(   ux2+(1-ux2)*c,ux*uy*(1-c)-uz*s,ux*uz*(1-c)+uy*s,
                 ux*uy*(1-c)+uz*s,   uy2+(1-uy2)*c,uy*uz*(1-c)-ux*s,
                 ux*uz*(1-c)-uy*s,uy*uz*(1-c)+ux*s,   uz2+(1-uz2)*c);
};

bool intersectSphere(const Ray& r,const V3f& center,float radius,float* tHit)
{
  const Ray ray = Ray(r.o-center,r.d);

  float A = ray.d(0)*ray.d(0) + ray.d(1)*ray.d(1) + ray.d(2)*ray.d(2);
  float B = 2 * (ray.d(0)*ray.o(0) + ray.d(1)*ray.o(1) + ray.d(2)*ray.o(2));
  float C = ray.o(0)*ray.o(0) + ray.o(1)*ray.o(1) +
            ray.o(2)*ray.o(2) - radius*radius;

  float t0, t1;
  if (!solveQuadratic(A, B, C, &t0, &t1))
      return false;

  if (t1 < 0) return false;
  float thit = t0;
  if (t0 < 0) {
      thit = t1;
  }

  *tHit = thit;

  return true;
}

int intersectNodes(const Ray& ray,std::vector<V3f>& nodes,float radius)
{
  float tmin = 100000.0f;
  int hit = -1;
  for(int i=0;i<nodes.size();i++)
  {
    float thit = 0.0f;
    if (intersectSphere(ray,nodes[i],radius,&thit) && thit<tmin)
    {
      tmin = thit;
      hit = i;
    }
  }
  return hit;
}

Vec3f intersectPlane(const Ray& ray,const V3f planeN,float planeD)
{
  const float t = -(dot(ray.o,planeN)+planeD)/dot(ray.d,planeN);
  return ray.o + t*ray.d;
}

Mat3x3f getPrincipalRotation(std::vector<V3f>& points)
{
  V3f bN,bP1,bP2,bP3; Mat3x3f RA,RB; float bd;

  float min=1e16; int iter=0; printf("plane:\n");

  while (iter<1024)
  {
   V3f P1=points[rand()%points.size()];
   V3f P2=points[rand()%points.size()];
   V3f P3=points[rand()%points.size()];

   V3f N=cross(P2-P1,P3-P1);

   float d=dot(P1,N),iN=1./norm(N),err=0;

   for(int i=0;i<points.size();i++) err+=fabs(dot(points[i],N)-d)*iN;

   if (err<min) { min=err; bN=N; bd=d; bP1=P1; bP2=P2; bP3=P3; printf("%d:%f\n",iter,err/points.size()); }

   iter++;
  }

  V3f X=bP2-bP1;
  V3f Y=bP3-bP1;

  V3f RX,RZ,RY,RO=V3f(0,0,1);

  RX = normalize(X);
  RY = normalize(Y);
  RZ = normalize(cross(X,RY));
  RY = normalize(cross(RZ,X));

  RA(0,0)=RX(0); RA(0,1)=RX(1); RA(0,2)=RX(2);
  RA(1,0)=RZ(0); RA(1,1)=RZ(1); RA(1,2)=RZ(2);
  RA(2,0)=RY(0); RA(2,1)=RY(1); RA(2,2)=RY(2);

  for(int i=0;i<points.size();i++) points[i]=RA*points[i];

  min=1e16; iter=0; printf("line:\n");

  while (iter<1024)
  {
   V3f P1=points[rand()%points.size()];
   V3f P2=points[rand()%points.size()];
   V3f dP=P2-P1; dP(1)=0;

   float iN=1./norm(dP),dx=dP(0),dy=dP(2),err=0;

   for(int i=0;i<points.size();i++) err+=iN*fabs(dx*(P1(2)-points[i](2))-(P1(0)-points[i](0))*dy);

   if (err<min) { min=err; bP1=P1; bP2=P2; printf("%d:%f\n",iter,err,dx,dy,iN); }

   iter++;
  }

  X=bP2-bP1; X(1)=0;
  Y=V3f(0,0,1);

  RX = normalize(X);
  RY = normalize(Y);
  RZ = normalize(cross(X,RY));
  RY = normalize(cross(RZ,X));

  RB(0,0)=RX(0); RB(0,1)=RX(1); RB(0,2)=RX(2);
  RB(1,0)=RZ(0); RB(1,1)=RZ(1); RB(1,2)=RZ(2);
  RB(2,0)=RY(0); RB(2,1)=RY(1); RB(2,2)=RY(2);

  for(int i=0;i<points.size();i++) points[i]=RB*points[i];

  return RB*RA;
}

V3f e2p(const V2f& x) { return V3f(x(0),x(1),1.0f); }
V4f e2p(const V3f& x) { return V4f(x(0),x(1),x(2),1.0f); }

V2f p2e(const V3f& x) { return V2f(x(0)/x(2),x(1)/x(2)); }
V3f p2e(const V4f& x) { return V3f(x(0)/x(3),x(1)/x(3),x(2)/x(3)); }

BBox2 updateBBox(const BBox2& oldBBox,const BBox2& newBBox)
{
  BBox2 bbox(V2f(+FLT_MAX,+FLT_MAX),
             V2f(-FLT_MAX,-FLT_MAX));

  bbox.min = std::min(oldBBox.min,newBBox.min);
  bbox.max = std::max(oldBBox.max,newBBox.max);

  return bbox;
}

BBox2 clipBBox(const BBox2& inBBox,const A2V3uc& image)
{
 BBox2 bbox(inBBox);

 const int w=image.width(),h=image.height();

 if (bbox.min(0)<0) bbox.min(0)=0;
 if (bbox.min(1)<0) bbox.min(1)=0;
 if (bbox.max(0)>w) bbox.max(0)=w;
 if (bbox.max(1)>h) bbox.max(1)=h;

 if (bbox.max(0)<0 || bbox.min(0)>w) bbox.min(0)=bbox.max(0)=0;
 if (bbox.max(1)<0 || bbox.min(1)>h) bbox.min(1)=bbox.max(1)=0;

  return bbox;
}

V3f projectPoint(const V3f pin, const Mat3x4f& P)
{
  const float pbx = P(0,0)*pin(0) + P(0,1)*pin(1) + P(0,2)*pin(2) + P(0,3);
  const float pby = P(1,0)*pin(0) + P(1,1)*pin(1) + P(1,2)*pin(2) + P(1,3);
  const float pbz = P(2,0)*pin(0) + P(2,1)*pin(1) + P(2,2)*pin(2) + P(2,3);

  return V3f(pbx/pbz,pby/pbz,pbz);
}

void rgb2hsv(float r,float g,float b,float* h,float* s,float* v)
{
  const float rgbMin = std::min(r,std::min(g,b));
  const float rgbMax = std::max(r,std::max(g,b));

  const float delta = rgbMax - rgbMin;

  *v = rgbMax;

  *s = (rgbMax > 0) ? (delta / rgbMax) : 0;

  if (delta > 0)
  {
    if (r>=g && r>=b) { *h = (       (g-b) / delta) * 60.0f / 360.0f; }
    if (g>=r && g>=b) { *h = (2.0f + (b-r) / delta) * 60.0f / 360.0f; }
    if (b>=r && b>=g) { *h = (4.0f + (r-g) / delta) * 60.0f / 360.0f; }

    if (*h<0.0f) *h = 1.0f+*h;
  }
  else
  {
    *h = 0;
  }
}

V3f rgb2hsv(const V3f& c)
{
  const float r = c(0);
  const float g = c(1);
  const float b = c(2);

  float h,s,v;

  rgb2hsv(r,g,b,&h,&s,&v);

  return V3f(h,s,v);
}

V3f hsvCone(const V3f hsv)
{
  const float h = hsv(0);
  const float s = hsv(1);
  const float v = hsv(2);

  return V3f(cos(6.28318530718f*h),v,sin(6.28318530718f*h));
}

float phi31(float x)
{
  return abs(x) < 1.0f ? (std::max(std::pow(1.0f-abs(x),4.0f),0.0f)*(1.0f+4.0f*abs(x))) : 0.0f;
}
