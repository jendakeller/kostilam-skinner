// Copyright (C) 2014 Czech Technical University in Prague - All Rights Reserved

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #define VC_EXTRALEAN
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#endif

#include <GL/GL.h>
#include <GL/glu.h>

#include "gui_GL.h"
#include <algorithm>

using std::vector;

float shade(const V3f& n)
{
  return (0.6f*pow(std::max(dot(normalize(n),normalize(V3f(-1,1,-1))),0.0f),1.5f)+0.4f);
}

float shade(const V3f& n,const V3f& lightDir)
{
  return (0.6f*pow(std::max(dot(normalize(n),normalize(lightDir)),0.0f),1.5f)+0.4f);
}

void glVertex(const V2f& v)
{
  glVertex2f(v(0),v(1));
}

void glVertex(const V3f& v)
{
  glVertex3f(v(0),v(1),v(2));
}

void glColor(const V3f& c)
{
  glColor3f(c(0),c(1),c(2));
}

void glCircle(const V2f& c,float r)
{
  glBegin(GL_LINE_LOOP);
  for(int i=0;i<128;i++)
  {
    glVertex(c+r*V2f(cos((float(i)/128.0f)*2.0f*3.1415926),
                     sin((float(i)/128.0f)*2.0f*3.1415926)));
  }
  glEnd();
}

void drawAxes(float s)
{
  glLineWidth(2);
  glBegin(GL_LINES);
    glColor3f(1,0,0);
    glVertex3f(0,0,0);
    glVertex3f(s,0,0);

    glColor3f(0,1,0);
    glVertex3f(0,0,0);
    glVertex3f(0,s,0);

    glColor3f(0,0,1);
    glVertex3f(0,0,0);
    glVertex3f(0,0,s);
  glEnd();
}

void normalize(GLfloat *a)
{
    GLfloat d=sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
    a[0]/=d; a[1]/=d; a[2]/=d;
}

void drawSphereSubdiv(GLfloat* a,GLfloat* b,GLfloat* c,int div,float r,const V3f& center,const V3f& color,const V3f& lightDir)
{
  if (div<=0)
  {
    glColor(color*shade(V3f(a[0],a[1],a[2]),lightDir)); glVertex(r*V3f(a[0],a[1],a[2])+center);
    glColor(color*shade(V3f(b[0],b[1],b[2]),lightDir)); glVertex(r*V3f(b[0],b[1],b[2])+center);
    glColor(color*shade(V3f(c[0],c[1],c[2]),lightDir)); glVertex(r*V3f(c[0],c[1],c[2])+center);
  }
  else
  {
    GLfloat ab[3],ac[3],bc[3];
    for (int i=0;i<3;i++)
    {
        ab[i]=(a[i]+b[i])/2;
        ac[i]=(a[i]+c[i])/2;
        bc[i]=(b[i]+c[i])/2;
    }
    normalize(ab); normalize(ac); normalize(bc);
    drawSphereSubdiv(a,ab,ac,div-1,r,center,color,lightDir);
    drawSphereSubdiv(b,bc,ab,div-1,r,center,color,lightDir);
    drawSphereSubdiv(c,ac,bc,div-1,r,center,color,lightDir);
    drawSphereSubdiv(ab,bc,ac,div-1,r,center,color,lightDir);
  }
}

//void drawSphere(const V3f& center,float radius,const V3f& color=V3f(0.6,0.6,0.6),const V3f& lightDir=V3f(-1,1,-1))
void drawSphere(const V3f& center,float radius,const V3f& color,const V3f& lightDir)
{
  static float X =.525731112119133606;
  static float Z =.850650808352039932;
  static GLfloat vdata[12][3] = {
    {-X, 0.0, Z}, {X, 0.0, Z}, {-X, 0.0, -Z}, {X, 0.0, -Z},
    {0.0, Z, X}, {0.0, Z, -X}, {0.0, -Z, X}, {0.0, -Z, -X},
    {Z, X, 0.0}, {-Z, X, 0.0}, {Z, -X, 0.0}, {-Z, -X, 0.0}
  };
  static GLuint tindices[20][3] = {
    {0,4,1}, {0,9,4}, {9,5,4}, {4,5,8}, {4,8,1},
    {8,10,1}, {8,3,10}, {5,3,8}, {5,2,3}, {2,7,3},
    {7,10,3}, {7,6,10}, {7,11,6}, {11,0,6}, {0,1,6},
    {6,1,10}, {9,0,11}, {9,11,2}, {9,2,5}, {7,2,11}
  };

  glBegin(GL_TRIANGLES);
  for (int i=0;i<20;i++) { drawSphereSubdiv(vdata[tindices[i][0]],vdata[tindices[i][1]],vdata[tindices[i][2]],2,radius,center,color,lightDir); }
  glEnd();
}

void drawBox(const V3f& min,const V3f& max)
{
  glBegin(GL_LINES);
    glVertex3f(min(0),min(1),min(2));
    glVertex3f(max(0),min(1),min(2));

    glVertex3f(max(0),min(1),min(2));
    glVertex3f(max(0),max(1),min(2));

    glVertex3f(max(0),max(1),min(2));
    glVertex3f(min(0),max(1),min(2));

    glVertex3f(min(0),max(1),min(2));
    glVertex3f(min(0),min(1),min(2));


    glVertex3f(min(0),min(1),max(2));
    glVertex3f(max(0),min(1),max(2));

    glVertex3f(max(0),min(1),max(2));
    glVertex3f(max(0),max(1),max(2));

    glVertex3f(max(0),max(1),max(2));
    glVertex3f(min(0),max(1),max(2));

    glVertex3f(min(0),max(1),max(2));
    glVertex3f(min(0),min(1),max(2));


    glVertex3f(min(0),min(1),min(2));
    glVertex3f(min(0),min(1),max(2));

    glVertex3f(max(0),min(1),min(2));
    glVertex3f(max(0),min(1),max(2));

    glVertex3f(max(0),max(1),min(2));
    glVertex3f(max(0),max(1),max(2));

    glVertex3f(min(0),max(1),min(2));
    glVertex3f(min(0),max(1),max(2));
  glEnd();
}


const int LEFT = 1;
const int RIGHT = 2;
const int BOTTOM = 4;
const int TOP = 8;

int outcode(double x, double y, double w, double h)
{
 int code = 0;

 if (x < 0) code |= LEFT; else if (x >= w) code |= RIGHT;
 if (y < 0) code |= BOTTOM; else if (y >= h) code |= TOP;

 return code;
}

int clipLine(double& x0, double& y0, double& x1, double& y1, const double w, const double h)
{
 int oc0 = outcode(x0,y0,w,h);
 int oc1 = outcode(x1,y1,w,h);

 int ok = 0;

 while (1)
 {
   if (!(oc0 | oc1)) { ok = 1; break; } else if (oc0 & oc1) break; else
   {
     double x,y; int oc = oc0 ? oc0 : oc1;

     if (oc & TOP)    { x = x0 + (x1 - x0) * (h-1 - y0) / (y1 - y0); y = h-1; } else
     if (oc & BOTTOM) { x = x0 + (x1 - x0) * y0 / (y1 - y0); y = 0; } else
     if (oc & RIGHT)  { y = y0 + (y1 - y0) * (w-1 - x0) / (x1 - x0); x = w-1; } else
     if (oc & LEFT)   { y = y0 + (y1 - y0) * x0 / (x1 - x0); x = 0; }

     if (oc == oc0) { x0 = x; y0 = y; oc0 = outcode(x0,y0,w,h); }
               else { x1 = x; y1 = y; oc1 = outcode(x1,y1,w,h); }
   }
 }

 return ok;
}

void drawLine(A2V4uc &O, const Mat3x4f& P, const Vec3f& a, const Vec3f& b, const V4uc& col,float scale)
{
 const float pax = P(0,0)*a(0) + P(0,1)*a(1) + P(0,2)*a(2) + P(0,3);
 const float pay = P(1,0)*a(0) + P(1,1)*a(1) + P(1,2)*a(2) + P(1,3);
 const float paz = P(2,0)*a(0) + P(2,1)*a(1) + P(2,2)*a(2) + P(2,3);

 const float pbx = P(0,0)*b(0) + P(0,1)*b(1) + P(0,2)*b(2) + P(0,3);
 const float pby = P(1,0)*b(0) + P(1,1)*b(1) + P(1,2)*b(2) + P(1,3);
 const float pbz = P(2,0)*b(0) + P(2,1)*b(1) + P(2,2)*b(2) + P(2,3);

 const float ax = pax/paz * scale;
 const float ay = pay/paz * scale;

 const float bx = pbx/pbz * scale;
 const float by = pby/pbz * scale;

 drawLine(O,(int)ax,(int)ay,(int)bx,(int)by,col);
}

Vec3f glUnproject(const V2f& x)
{
  GLint viewport[4];
  GLdouble modelview[16];
  GLdouble projection[16];
  GLfloat winX, winY, winZ;
  GLdouble posX, posY, posZ;

  glGetDoublev( GL_MODELVIEW_MATRIX, modelview );
  glGetDoublev( GL_PROJECTION_MATRIX, projection );
  glGetIntegerv( GL_VIEWPORT, viewport );

  winX = (float)x(0);
  winY = (float)viewport[3] - (float)x(1);
  winZ = 1;
  gluUnProject(winX,winY,winZ,modelview,projection,viewport,&posX,&posY,&posZ);

  return V3f(posX, posY, posZ);
}
