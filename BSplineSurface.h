#ifndef __LayerDepthMap__BSplineSurface__
#define __LayerDepthMap__BSplineSurface__

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "DataStructures.h"
//#include "BSpline.h"


class BSplineSurface{
  
 public:
  BSplineSurface() {};
  BSplineSurface(const int IMAGE_WIDTH, const int IMAGE_HEIGTH, const double STRIDE_X, const double STRIDE_Y, const int BSPLINE_ORDER);
  void fitBSplineSurface(const std::vector<double> &point_cloud, const std::vector<int> &pixels);
  std::vector<double> getPointCloud(const std::vector<double> &CAMERA_PARAMETERS, const bool USE_PANORAMA) const;
  
  friend std::ostream & operator <<(std::ostream &out_str, const BSplineSurface &surface);
  friend std::istream & operator >>(std::istream &in_str, BSplineSurface &surface);
  
 private:
  int IMAGE_WIDTH_;
  int IMAGE_HEIGHT_;
  int NUM_PIXELS_;
  double STRIDE_X_;
  double STRIDE_Y_;
  int BSPLINE_ORDER_;
  
  std::vector<double> control_point_xs_;
  std::vector<double> control_point_ys_;
  
  std::vector<double> control_point_depths_;
  
  
  void initControlPoints();
  double calcBasisFunctionValue2D(const double x, const double y, const double control_point_x, const double control_point_y, const double stride_x, const double stride_y, const int order) const;
  double calcBasisFunctionValue1D(const double &x, const double &control_point_x, const double &stride_x, const int &order) const;
};

#endif /* defined(__LayerDepthMap__BSplineSurface__) */
