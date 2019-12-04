// Copyright (C) 2014 Czech Technical University in Prague - All Rights Reserved

#ifndef GUI_GL_H_
#define GUI_GL_H_

#include <vector>
#include <algorithm>

//#include "../kostilam.h"
#include "jzq.h"

void glVertex(const V2f& v);
void glVertex(const V3f& v);
void glColor(const V3f& c);
void glCircle(const V2f& c,float r);
void drawAxes(float s);
//void drawGrid(float y=0);
void drawSphere(const V3f& center,float radius,const V3f& color=V3f(0.6,0.6,0.6),const V3f& lightDir=V3f(-1,1,-1));
void drawBox(const V3f& min,const V3f& max);
Vec3f glUnproject(const V2f& x);

int clipLine(double& x0, double& y0, double& x1, double& y1, const double w, const double h);

template<int N,typename T>
void drawCircle(Array2<Vec<N,T> >& dst,int cx,int cy,int r,const Vec<N,T>& color)
{
  const int x0 = std::max(cx-r-1,0);
  const int x1 = std::min(cx+r+1,dst.width()-1);
  const int y0 = std::max(cy-r-1,0);
  const int y1 = std::min(cy+r+1,dst.height()-1);

  for(int y=y0;y<=y1;y++)
  for(int x=x0;x<=x1;x++)
  {
    if (x>=0 && y>=0 && x<dst.width() && y<dst.height())
    {
      float d = norm(V2f(x,y)-V2f(cx,cy));
      float a = 0.8f*(std::max(std::min(1.0f-abs(float(d)-float(r)),1.0f),0.0f));
      if (a>0.0f)
      {
        dst(x,y) = Vec<N,T>( ((1.0f-a)*Vec<N,float>(dst(x,y))) + ((a)*Vec<N,float>(color)) );
      }
    }
  }
}

template<int N,typename T>
void drawRect(Array2<Vec<N,T> >& dst,int x0,int y0,int x1,int y1,const Vec<N,T>& color)
{
  for(int x=x0;x<=x1;x++)
  {
    if (x>=0&&x<dst.width())
    {
      if (y0>=0 && y0<dst.height()) { dst(x,y0) = color; }
      if (y1>=0 && y1<dst.height()) { dst(x,y1) = color; }
    }
  }

  for(int y=y0;y<=y1;y++)
  {
    if (y>=0&&y<dst.height())
    {
      if (x0>=0 && x0<dst.width()) { dst(x0,y) = color; }
      if (x1>=0 && x1<dst.width()) { dst(x1,y) = color; }
    }
  }
}

template<int N,typename T>
void drawLine(Array2<Vec<N,T>>& dst, int x1,int y1, int x2,int y2, const Vec<N,T>& color)
{
  double x1d=x1,y1d=y1,x2d=x2,y2d=y2;

  if (!clipLine(x1d,y1d,x2d,y2d,dst.width(),dst.height())) return;

  x1=(int)x1d; y1=(int)y1d; x2=(int)x2d; y2=(int)y2d;

  //if(x1 < 0 || x1 >= dst.width() || y1 < 0 || y1 > dst.height()) return;
  //if(x2 < 0 || x2 >= dst.width() || y2 < 0 || y2 > dst.height()) return;

  Vec<N,T>* frameBuffer = dst.data();
  int width = dst.width();

    int dx = x2 - x1;
    int dy = y2 - y1;
    int c0, c1, p, idx;
    int stay, jump;

    if(abs(dy) < abs(dx)){
        // I, IV, V, VIII
        // stepping along the x axis
        if(x1 < x2){
            idx = x1 + y1*width;
        } else {
            idx = x2 + y2*width;
            dx = -dx;
            dy = -dy;
            std::swap(x1,x2);
        }
        stay = 1;
        if(dy < 0){
            dy = -dy;
            jump = 1 - width;
        } else {
            jump = width + 1;
        }
    } else {
        // II, III, VI, VII
        // stepping along the y axis
        std::swap(dx,dy);
        if(y1 < y2){
            idx = x1 + y1*width;
            x2 = y2;
            x1 = y1;
            dy = -dy;
        } else {
            idx = x2 + y2*width;
            dx = -dx;
            x2 = y1;
            x1 = y2;
        }
        if(dy < 0){
            dy = -dy;
            stay = width;
            jump = width + 1;
        } else {
            stay = width;
            jump = width - 1;
        }
    }

    c0 = dy<<1;
    c1 = c0 - (dx<<1);
    p = c0 - dx;
    frameBuffer[idx] = color;
    for (int i = x1 + 1; i <= x2; i++) {
        if (p < 0) {
            p += c0;
            idx += stay;
        } else {
            p += c1;
            idx += jump;
        }
        frameBuffer[idx] = color;
    }
}

#endif
