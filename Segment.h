#ifndef Segment_H
#define Segment_H

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Dense>

#include "DataStructures.h"
#include "cv_utils.h"
#include "BSplineSurface.h"
//#include "BSpline.h"


class Segment{
  
 public:
  Segment(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<double> &camera_parameters, const cv_utils::ImageMask &fitting_mask, const DataStatistics &STATISTICS, const std::vector<double> &pixel_weights = std::vector<double>(), const int segment_type = 0, const bool use_panorama = false);
  Segment(const int image_width, const int image_height, const std::vector<double> &camera_parameters, const DataStatistics &STATISTICS, const bool use_panorama = false);
  Segment() {};
  
  friend std::ostream & operator <<(std::ostream &out_str, const Segment &segment);
  friend std::istream & operator >>(std::istream &in_str, Segment &segment);
  
  
  void setGMM(const cv::Ptr<cv::ml::EM> GMM);
  void setGMM(const cv::FileNode GMM_file_node);
  
  cv::Ptr<cv::ml::EM> getGMM() const;
  
  //std::vector<double> getDepthMap() const;
  double getDepth(const int pixel) const;
  double getDepth(const double x_ratio, const double y_ratio) const;
  std::vector<double> getPlane() const;
  cv_utils::ImageMask getMask() const;
  std::vector<double> getSegmentPoint(const int pixel) const;
  int getSegmentType() const { return segment_type_; };
  
  bool checkPixelFitting(const cv::Mat &hsv_image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const int pixel) const;
  double calcPixelFittingCost(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const int pixel, const RepresenterPenalties &PENALTIES, const double weight_3D, const bool behind_room_structure_tolerance) const;
  
  void checkPixelFittingCosts(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const RepresenterPenalties &PENALTIES, const int index) const;
  
  std::vector<int> projectToOtherViewpoints(const int pixel, const double viewpoint_movement);
  
  Eigen::Matrix3d getUnwarpTransform(const std::vector<double> &point_cloud, const std::vector<double> &CAMERA_PARAMETERS) const;
  
  bool getValidity() const { return validity_; };
  std::vector<int> getSegmentPixels() const;
  //int getNumSegmentPixels() const;
  
  bool checkPairwiseConvexity(const int pixel_1, const int pixel_2);
  double getMaxColorLikelihood() const
  {
    return max_color_likelihood_;
  }
  
  
  void refit(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::vector<double> &camera_parameters, const cv_utils::ImageMask &fitting_mask, const cv_utils::ImageMask &occluded_mask, const std::vector<double> &pixel_weights, const int segment_type);
  
  cv_utils::ImageMask getFittingMask() const { return fitting_mask_; };

  double calcColorFittingCost(const cv::Vec3b color) const;

  //void setBehindRoomStructure(const bool behind_room_structure);
  //bool getBehindRoomStructure() const;

  double getConfidence() const;
  
  
 private:
  int IMAGE_WIDTH_;
  int IMAGE_HEIGHT_;
  
  int NUM_PIXELS_;
  std::vector<double> CAMERA_PARAMETERS_;
  
  DataStatistics STATISTICS_;
  bool USE_PANORAMA_;
  
  bool validity_;
  
  std::vector<double> plane_;
  std::vector<double> segment_point_cloud_;
  std::vector<double> segment_normals_;
  
  cv::Ptr<cv::ml::EM> GMM_;
  double max_color_likelihood_;

  std::vector<double> color_model_;
  
  
  cv_utils::ImageMask segment_mask_;
  cv_utils::ImageMask fitting_mask_;
  std::vector<int> segment_growth_map_;
  std::vector<double> segment_center_;
  
  std::shared_ptr<cv_utils::Histogram<double> > point_plane_distance_histogram_;
  std::shared_ptr<cv_utils::Histogram<double> > point_plane_angle_histogram_;
  std::shared_ptr<cv_utils::Histogram<double> > pixel_center_distance_histogram_;
  std::shared_ptr<cv_utils::Histogram<double> > color_likelihood_histogram_;
  
  int segment_type_;
  bool behind_room_structure_;
  BSplineSurface b_spline_surface_;

  double segment_confidence_;
  
  
  void fitSegmentPlane(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const cv_utils::ImageMask &fitting_mask, const std::vector<double> &pixel_weights);
  void fitSegmentBSplineSurface(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const cv_utils::ImageMask &fitting_mask, const std::vector<double> &pixel_weights);
  void calcPointCloud(const std::vector<double> &point_cloud = std::vector<double>());
  void trainGMM(const cv::Mat &image, const std::vector<int> &pixels);
  void calcHistograms(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals);
  void writeSegmentImage(const std::string filename);
  void calcSegmentGrowthMap();

  void trainColorModel(const cv::Mat &image, const std::vector<int> &pixels);

  double predictColorLikelihood(const cv::Vec3b color) const;

  void calcConfidence(const std::vector<double> &point_cloud);
};

#endif
