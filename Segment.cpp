#include "Segment.h"

#include "cv_utils.h"

#include <iostream>

#include "TRW_S/MRFEnergy.h"
#include "utils.h"


using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace Eigen;
using namespace cv_utils;

Segment::Segment(const cv::Mat &image, const std::vector<double> &point_cloud, const vector<double> &normals, const vector<double> &camera_parameters, const ImageMask &fitting_mask, const DataStatistics &STATISTICS, const std::vector<double> &pixel_weights, const int segment_type, const bool use_panorama) : IMAGE_WIDTH_(image.cols), IMAGE_HEIGHT_(image.rows), NUM_PIXELS_(image.cols * image.rows), CAMERA_PARAMETERS_(camera_parameters), STATISTICS_(STATISTICS), USE_PANORAMA_(use_panorama), behind_room_structure_(false)
{
  //ImageMask invalid_point_mask = getInvalidPointMask(point_cloud, IMAGE_WIDTH_, IMAGE_HEIGHT_);
  //ImageMask valid_fitting_mask = fitting_mask - invalid_point_mask;
  segment_type_ = segment_type;
  if (segment_type_ == 0) {
    fitSegmentPlane(image, point_cloud, normals, fitting_mask - getInvalidPointMask(point_cloud, IMAGE_WIDTH_, IMAGE_HEIGHT_), pixel_weights.size() > 0 ? pixel_weights : vector<double>(NUM_PIXELS_, 1));
  } else {
    fitSegmentBSplineSurface(image, point_cloud, normals, fitting_mask - getInvalidPointMask(point_cloud, IMAGE_WIDTH_, IMAGE_HEIGHT_), pixel_weights.size() > 0 ? pixel_weights : vector<double>(NUM_PIXELS_, 1));
  }
  if (validity_ == false)
    return;  
  
  //  trainGMM(image, segment_mask_.getPixels());
  fitting_mask_ = segment_mask_ - (segment_mask_ - fitting_mask);
  trainColorModel(image, fitting_mask_.getPixels());
  calcPointCloud(point_cloud);
  calcConfidence(point_cloud);
  calcSegmentGrowthMap();

  //  calcHistograms(image, point_cloud, normals);
}

Segment::Segment(const int image_width, const int image_height, const vector<double> &camera_parameters, const DataStatistics &STATISTICS, const bool use_panorama) : IMAGE_WIDTH_(image_width), IMAGE_HEIGHT_(image_height), NUM_PIXELS_(image_width * image_height), CAMERA_PARAMETERS_(camera_parameters), STATISTICS_(STATISTICS), validity_(true), USE_PANORAMA_(use_panorama), behind_room_structure_(false)
{
}

void Segment::fitSegmentPlane(const cv::Mat &image, const std::vector<double> &point_cloud, const vector<double> &normals, const ImageMask &fitting_mask, const std::vector<double> &pixel_weights)
{
  vector<int> fitting_pixels = fitting_mask.getPixels();
  // vector<int> valid_fitting_pixels;
  // for (vector<int>::const_iterator pixel_it = fitting_pixels.begin(); pixel_it != fitting_pixels.end(); pixel_it++) {
  //   if (checkPointValidity(point_cloud, *pixel_it) == false)
  //     continue;
  //   valid_fitting_pixels.push_back(*pixel_it);
  // }
  // fitting_pixels = valid_fitting_pixels;
  // ImageMask valid_fitting_mask(fitting_pixels, IMAGE_WIDTH_, IMAGE_HEIGHT_);
  
  if (fitting_pixels.size() < STATISTICS_.small_segment_num_pixels_threshold) {
    validity_ = false;
    segment_mask_ = ImageMask(fitting_pixels, IMAGE_WIDTH_, IMAGE_HEIGHT_);
    return;
  }
  
  const int NUM_ITERATIONS = min(static_cast<int>(fitting_pixels.size() / 3), 100);
  
  double max_sum_inlier_weights = 0;
  vector<double> best_plane;
  for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
    vector<int> initial_pixels;
    while (initial_pixels.size() < 3) {
      int initial_pixel = fitting_pixels[rand() % fitting_pixels.size()];
      vector<int> neighbor_pixels = fitting_mask.findMaskWindowPixels(initial_pixel, 3, USE_PANORAMA_);
      if (neighbor_pixels.size() < 2)
	continue;
      initial_pixels.push_back(initial_pixel);
      neighbor_pixels = randomSampleValues(neighbor_pixels, 2);
      initial_pixels.insert(initial_pixels.end(), neighbor_pixels.begin(), neighbor_pixels.end());
    }
    
    vector<double> initial_points;
    vector<double> initial_normals;
    for (vector<int>::const_iterator pixel_it = initial_pixels.begin(); pixel_it != initial_pixels.end(); pixel_it++) {
      initial_points.insert(initial_points.end(), point_cloud.begin() + *pixel_it * 3, point_cloud.begin() + (*pixel_it + 1) * 3);
      initial_normals.insert(initial_normals.end(), normals.begin() + *pixel_it * 3, normals.begin() + (*pixel_it + 1) * 3);
    }
    
    vector<double> plane = fitPlane(initial_points);
    if (plane.size() == 0)
      continue;
    // bool inversed = false;
    // vector<double> plane_normal(plane.begin(), plane.begin() + 3);
    // for (int point_index = 0; point_index < 3; point_index++) {
    //   if (calcDotProduct(plane_normal, getPoint(initial_normals, point_index)) < 0) {
    // 	inversed = true;
    // 	break;
    //   }
    // }
    if (plane[3] > 0)
      for (int c = 0; c < 4; c++)
	plane[c] = -plane[c];
    
    double sum_inlier_weights = 0;
    int num_inliers = 0;
    for (vector<int>::const_iterator pixel_it = fitting_pixels.begin(); pixel_it != fitting_pixels.end(); pixel_it++) {
      vector<double> point = getPoint(point_cloud, *pixel_it);
      double point_plane_distance = abs(calcPointPlaneDistance(point, plane));
      if (point_plane_distance > STATISTICS_.pixel_fitting_distance_threshold)
	continue;
      
      // vector<double> normal(normals.begin() + *pixel_it * 3, normals.begin() + (*pixel_it + 1) * 3);
      // double angle = calcAngle(vector<double>(plane.begin(), plane.begin() + 3), normal);
      // if (angle > STATISTICS_.pixel_fitting_angle_threshold)
      // 	continue;
      
      sum_inlier_weights += pixel_weights[*pixel_it];
      num_inliers++;
    }
    //cout << sum_inlier_weights << '\t' << num_inliers << endl;
    if (sum_inlier_weights > max_sum_inlier_weights) {
      best_plane = plane;
      max_sum_inlier_weights = sum_inlier_weights;
    }
  }
  //exit(1);
  
  if (best_plane.size() == 0) {
    validity_ = false;
    segment_mask_ = ImageMask(fitting_pixels, IMAGE_WIDTH_, IMAGE_HEIGHT_);
    return;
  }
  
  plane_ = best_plane;
  //  cout << plane_[0] << '\t' << plane_[1] << '\t' << plane_[2] << '\t' << plane_[3] << endl;
  vector<int> fitted_pixels;
  for (vector<int>::const_iterator pixel_it = fitting_pixels.begin(); pixel_it != fitting_pixels.end(); pixel_it++) {
    vector<double> point(point_cloud.begin() + *pixel_it * 3, point_cloud.begin() + (*pixel_it + 1) * 3);
    double point_plane_distance = abs(calcPointPlaneDistance(point, plane_));
    if (point_plane_distance > STATISTICS_.pixel_fitting_distance_threshold)
      continue;
    
    // vector<double> normal(normals.begin() + *pixel_it * 3, normals.begin() + (*pixel_it + 1) * 3);
    // double angle = calcAngle(vector<double>(plane_.begin(), plane_.begin() + 3), normal);
    // if (angle > STATISTICS_.pixel_fitting_angle_threshold)
    //   continue;
    
    if (calcPlaneDepthAtPixel(plane_, *pixel_it, IMAGE_WIDTH_, IMAGE_HEIGHT_, CAMERA_PARAMETERS_, USE_PANORAMA_) <= 0) {
    //   int x = *pixel_it % IMAGE_WIDTH_;
    //   int y = *pixel_it / IMAGE_WIDTH_;
    //   double angle_1 = -M_PI * (y - CAMERA_PARAMETERS_[2]) / CAMERA_PARAMETERS_[0];
    //   double angle_2 = (2 * M_PI) * (x - CAMERA_PARAMETERS_[1]) / IMAGE_WIDTH_;
    //   double Z_ratio = sin(angle_1);
    //   double X_ratio = cos(angle_1) * sin(angle_2);
    //   double Y_ratio = cos(angle_1) * cos(angle_2);
    //   cout << *pixel_it << '\t' << X_ratio << '\t' << Y_ratio << '\t' << Z_ratio << endl;
    //   cout << point[0] << '\t' << point[1] << '\t' << point[2] << endl;
    //   cout << normal[0] << '\t' << normal[1] << '\t' << normal[2] << endl;
    //   exit(1);
      continue;
    }
    
    fitted_pixels.push_back(*pixel_it);
  }
  
  if (fitted_pixels.size() < STATISTICS_.small_segment_num_pixels_threshold) {
    validity_ = false;
    segment_mask_ = ImageMask(fitting_pixels, IMAGE_WIDTH_, IMAGE_HEIGHT_);
    return;
  }
  
  ImageMask fitted_pixel_mask(fitted_pixels, IMAGE_WIDTH_, IMAGE_HEIGHT_);
  ImageMask possible_segment_pixel_mask = fitted_pixel_mask;
  possible_segment_pixel_mask.dilate();
  possible_segment_pixel_mask.dilate();
  
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //   vector<double> point(point_cloud.begin() + pixel * 3, point_cloud.begin() + (pixel + 1) * 3);
  //   double point_plane_distance = calcPointPlaneDistance(point, plane_);
  //   if (point_plane_distance > STATISTICS_.depth_conflict_threshold)
  //     continue;
  //   possible_segment_pixel_mask.set(pixel, true);
  // }
  vector<vector<int> > connected_components = possible_segment_pixel_mask.findConnectedComponents(USE_PANORAMA_);
  
  double selected_component_sum_fitted_weights = 0;
  vector<int> selected_component_fitted_pixels;
  //int selected_component_index = -1;
  for (vector<vector<int> >::const_iterator component_it = connected_components.begin(); component_it != connected_components.end(); component_it++) {
    double component_sum_fitted_weights = 0;
    vector<int> component_fitted_pixels;
    for (vector<int>::const_iterator pixel_it = component_it->begin(); pixel_it != component_it->end(); pixel_it++) {
      if (fitted_pixel_mask.at(*pixel_it)) {
	component_sum_fitted_weights += pixel_weights[*pixel_it];
	component_fitted_pixels.push_back(*pixel_it);
      }
    }
    if (component_fitted_pixels.size() < STATISTICS_.small_segment_num_pixels_threshold)
      continue;
    if (component_sum_fitted_weights > selected_component_sum_fitted_weights) {
      selected_component_sum_fitted_weights = component_sum_fitted_weights;
      selected_component_fitted_pixels = component_fitted_pixels;
      //selected_component_index = component_it - connected_components.begin();
    }
  }
  if (selected_component_fitted_pixels.size() == 0) {
    validity_ = false;
    segment_mask_ = ImageMask(fitted_pixels, IMAGE_WIDTH_, IMAGE_HEIGHT_);
    return;
  }
  
  segment_mask_ = ImageMask(selected_component_fitted_pixels, IMAGE_WIDTH_, IMAGE_HEIGHT_);
  ImageMask eroded_segment_mask = segment_mask_;
  eroded_segment_mask.erode();
  if (eroded_segment_mask.getNumPixels() == 0) {
    validity_ = false;
    segment_mask_ = ImageMask(fitted_pixels, IMAGE_WIDTH_, IMAGE_HEIGHT_);
    return;
  }
  //segment_mask_ = ImageMask(fitted_pixels, IMAGE_WIDTH_, IMAGE_HEIGHT_);
  
  plane_ = fitPlane(getPoints(point_cloud, selected_component_fitted_pixels));
  if (plane_[3] > 0)
    for (int c = 0; c < 4; c++)
      plane_[c] = -plane_[c];
  //  cout << plane_[0] << '\t' << plane_[1] << '\t' << plane_[2] << '\t' << plane_[3] << endl;
  validity_ = true;
}

void Segment::fitSegmentBSplineSurface(const cv::Mat &image, const std::vector<double> &point_cloud, const vector<double> &normals, const ImageMask &fitting_mask, const std::vector<double> &pixel_weights)
{
  vector<int> fitting_pixels = fitting_mask.getPixels();
  
  if (fitting_pixels.size() < STATISTICS_.small_segment_num_pixels_threshold) {
    validity_ = false;
    segment_mask_ = ImageMask(fitting_pixels, IMAGE_WIDTH_, IMAGE_HEIGHT_);
    return;
  }
  
  ImageMask possible_segment_pixel_mask = fitting_mask;
  possible_segment_pixel_mask.dilate();
  
  vector<vector<int> > connected_components = possible_segment_pixel_mask.findConnectedComponents();
  
  double selected_component_sum_fitted_weights = 0;
  vector<int> selected_component_fitted_pixels;
  int selected_component_index = -1;
  for (vector<vector<int> >::const_iterator component_it = connected_components.begin(); component_it != connected_components.end(); component_it++) {
    double component_sum_fitted_weights = 0;
    vector<int> component_fitted_pixels;
    for (vector<int>::const_iterator pixel_it = component_it->begin(); pixel_it != component_it->end(); pixel_it++) {
      component_sum_fitted_weights += pixel_weights[*pixel_it];
      component_fitted_pixels.push_back(*pixel_it);
    }
    if (component_fitted_pixels.size() < STATISTICS_.small_segment_num_pixels_threshold || component_fitted_pixels.size() > STATISTICS_.bspline_surface_num_pixels_threshold)
      continue;
    if (component_sum_fitted_weights > selected_component_sum_fitted_weights) {
      selected_component_sum_fitted_weights = component_sum_fitted_weights;
      selected_component_fitted_pixels = component_fitted_pixels;
      selected_component_index = component_it - connected_components.begin();
    }
  }
  if (selected_component_index == -1) {
    validity_ = false;
    segment_mask_ = ImageMask(fitting_pixels, IMAGE_WIDTH_, IMAGE_HEIGHT_);
    return;
  }
  
  segment_mask_ = ImageMask(connected_components[selected_component_index], IMAGE_WIDTH_, IMAGE_HEIGHT_);
  segment_mask_.erode();
  if (segment_mask_.getNumPixels() < STATISTICS_.small_segment_num_pixels_threshold) {
    validity_ = false;
    segment_mask_ = ImageMask(fitting_pixels, IMAGE_WIDTH_, IMAGE_HEIGHT_);
    return;
  }
  
  b_spline_surface_ = BSplineSurface(IMAGE_WIDTH_, IMAGE_HEIGHT_, 10, 10, segment_type_);
  b_spline_surface_.fitBSplineSurface(point_cloud, connected_components[selected_component_index]);
  
  validity_ = true;
}

void Segment::calcHistograms(const Mat &image, const vector<double> &point_cloud, const vector<double> &normals)
{
  segment_center_ = segment_mask_.getCenter();
  
  vector<int> segment_pixels = segment_mask_.getPixels();
  vector<double> point_plane_distance_values(segment_pixels.size());
  vector<double> point_plane_angle_values(segment_pixels.size());
  vector<double> pixel_center_distance_values(segment_pixels.size());
  vector<double> color_likelihood_values(segment_pixels.size());
  for (vector<int>::const_iterator pixel_it = segment_pixels.begin(); pixel_it != segment_pixels.end(); pixel_it++) {
    vector<double> point = getPoint(point_cloud, *pixel_it);
    double point_plane_distance = abs(calcPointPlaneDistance(point, plane_));
    point_plane_distance_values[pixel_it - segment_pixels.begin()] = point_plane_distance;
    vector<double> normal = getPoint(normals, *pixel_it);
    double point_plane_angle = calcAngle(vector<double>(plane_.begin(), plane_.begin() + 3), normal);
    //point_plane_angle = min(point_plane_angle, M_PI - point_plane_angle);
    point_plane_angle_values[pixel_it - segment_pixels.begin()] = point_plane_angle;
    double pixel_center_distance = calcDistance(segment_center_, getVec(1.0 * (*pixel_it % IMAGE_WIDTH_), 1.0 * (*pixel_it / IMAGE_WIDTH_)));
    pixel_center_distance_values[pixel_it - segment_pixels.begin()] = pixel_center_distance;
    double color_likelihood = predictColorLikelihood(image.at<Vec3b>(*pixel_it / IMAGE_WIDTH_, *pixel_it % IMAGE_WIDTH_));
    //cout << "color likelihood: " << color_likelihood << endl;
    color_likelihood_values[pixel_it - segment_pixels.begin()] = color_likelihood;
  }
  
  point_plane_distance_histogram_.reset(new Histogram<double>(STATISTICS_.num_grams, 1.0 * -STATISTICS_.pixel_fitting_distance_threshold, 1.0 * STATISTICS_.pixel_fitting_distance_threshold, point_plane_distance_values));
  point_plane_angle_histogram_.reset(new Histogram<double>(STATISTICS_.num_grams, -STATISTICS_.pixel_fitting_angle_threshold, STATISTICS_.pixel_fitting_angle_threshold, point_plane_angle_values));
  pixel_center_distance_histogram_.reset(new Histogram<double>(STATISTICS_.num_grams, 0, getMax(pixel_center_distance_values), pixel_center_distance_values));
  color_likelihood_histogram_.reset(new Histogram<double>(STATISTICS_.num_grams, getMin(color_likelihood_values), max_color_likelihood_, color_likelihood_values));
}

void Segment::trainColorModel(const Mat &image, const vector<int> &pixels)
{
  vector<double> color_sums(3, 0);
  for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
    int pixel = *pixel_it;
    int x = pixel % IMAGE_WIDTH_;
    int y = pixel / IMAGE_WIDTH_;
    Vec3b color = image.at<Vec3b>(y, x);
    for (int c = 0; c < 3; c++)
      color_sums[c] += color[c];
  }
  color_model_.assign(3, 0);
  for (int c = 0; c < 3; c++)
    color_model_[c] = color_sums[c] / pixels.size();
}

double Segment::calcColorFittingCost(const Vec3b color) const
{
  double distance = 0;
  for (int c = 0; c < 3; c++)
    distance += pow(color[c] - color_model_[c], 2);
  distance = sqrt(distance);
  return 1 - exp(-pow(distance, 2) / (2 * STATISTICS_.color_diff_var));
}

void Segment::trainGMM(const Mat &image, const vector<int> &pixels)
{
  Mat segment_samples(pixels.size(), 3, CV_32FC1);
  // Mat blurred_image;
  // GaussianBlur(image, blurred_image, cv::Size(3, 3), 0, 0);
  // Mat blurred_hsv_image;
  // blurred_image.convertTo(blurred_hsv_image, CV_32FC3, 1.0 / 255);
  // cvtColor(blurred_hsv_image, blurred_hsv_image, CV_BGR2HSV);
  
  // for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
  //   Vec3f color = blurred_hsv_image.at<Vec3f>(*pixel_it / IMAGE_WIDTH_, *pixel_it % IMAGE_WIDTH_);
  //   segment_samples.at<float>(pixel_it - pixels.begin(), 0) = color[1] * cos(color[0] * M_PI / 180);
  //   segment_samples.at<float>(pixel_it - pixels.begin(), 1) = color[1] * sin(color[0] * M_PI / 180);
  //   //segment_samples.at<float>(pixel_it - pixels.begin(), 2) = color[2] * 0.1;
  
  
  //   // Vec3b color = image.at<Vec3b>(*pixel_it / IMAGE_WIDTH_, *pixel_it % IMAGE_WIDTH_);
  //   // for (int c = 0; c < 3; c++)
  //   //   segment_samples.at<float>(pixel_it - pixels.begin(), c) = 1.0 * color[c] / 256;
  
  //   // segment_samples.at<float>(pixel_it - pixels.begin(), 3) = 1.0 * (*pixel_it % IMAGE_WIDTH_) / IMAGE_WIDTH_;
  //   // segment_samples.at<float>(pixel_it - pixels.begin(), 4) = 1.0 * (*pixel_it / IMAGE_WIDTH_) / IMAGE_HEIGHT_;
  // }
  
  for (vector<int>::const_iterator pixel_it = pixels.begin(); pixel_it != pixels.end(); pixel_it++) {
    Vec3b color = image.at<Vec3b>(*pixel_it / IMAGE_WIDTH_, *pixel_it % IMAGE_WIDTH_);
    segment_samples.at<float>(pixel_it - pixels.begin(), 0) = color[0];
    segment_samples.at<float>(pixel_it - pixels.begin(), 1) = color[1];
    segment_samples.at<float>(pixel_it - pixels.begin(), 2) = color[2];
  }
  
  const int NUM_CLUSTERS = 3;
  GMM_ = EM::create();
  GMM_->setClustersNumber(NUM_CLUSTERS);
  Mat log_likelihoods(pixels.size(), 1, CV_64FC1);
  GMM_->trainEM(segment_samples, log_likelihoods, noArray(), noArray());
  
  max_color_likelihood_ = numeric_limits<double>::lowest();
  for (int cluster_index = 0; cluster_index < NUM_CLUSTERS; cluster_index++) {
    Vec2d prediction = GMM_->predict2(GMM_->getMeans().row(cluster_index), noArray());
    Mat weights = GMM_->getWeights();
    double likelihood = prediction[0] + log(weights.at<double>(0, prediction[1]));
    if (likelihood > max_color_likelihood_)
      max_color_likelihood_ = likelihood;
  }
  //cout << "max color likelihood " << max_color_likelihood_ << endl;
}

void Segment::calcPointCloud(const vector<double> &point_cloud)
{
  segment_point_cloud_ = vector<double>(IMAGE_WIDTH_ * IMAGE_HEIGHT_ * 3, 0);
  if (segment_type_ == 0) {
    for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++) {
      double depth = calcPlaneDepthAtPixel(plane_, pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_, CAMERA_PARAMETERS_, USE_PANORAMA_);
      if (depth <= 0)
	continue;
      vector<double> point = unprojectPixel(pixel, depth, IMAGE_WIDTH_, IMAGE_HEIGHT_, CAMERA_PARAMETERS_, USE_PANORAMA_);
      for (int c = 0; c < 3; c++)
	segment_point_cloud_[pixel * 3 + c] = point[c];
    }
  } else {
    segment_point_cloud_ = b_spline_surface_.getPointCloud(CAMERA_PARAMETERS_, USE_PANORAMA_);
    segment_normals_ = calcNormals(segment_point_cloud_, IMAGE_WIDTH_, IMAGE_HEIGHT_);
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      if (calcDotProduct(getPoint(segment_point_cloud_, pixel), getPoint(segment_normals_, pixel)) > 0)
        for (int c = 0; c < 3; c++)
          segment_normals_[pixel * 3 + c] = -segment_normals_[pixel * 3 + c];
    }
  }
}

void Segment::calcConfidence(const vector<double> &point_cloud)
{
  vector<int> segment_pixels = fitting_mask_.getPixels();
  double fitting_error_sum2 = 0;
  for (vector<int>::const_iterator pixel_it = segment_pixels.begin(); pixel_it != segment_pixels.end(); pixel_it++)
    fitting_error_sum2 += pow(calcDistance(getPoint(point_cloud, *pixel_it), getPoint(segment_point_cloud_, *pixel_it)), 2);
  double fitting_error = sqrt(fitting_error_sum2 / segment_pixels.size());
  //cout << fitting_error << endl;
  segment_confidence_ = exp(-pow(fitting_error, 2) / (2 * pow(STATISTICS_.pixel_fitting_distance_threshold, 2)));
}

// vector<double> Segment::getDepthMap() const
// {
//   return depth_map_;
// }

double Segment::getDepth(const int pixel) const
{
  double depth = calcNorm(vector<double>(segment_point_cloud_.begin() + pixel * 3, segment_point_cloud_.begin() + (pixel + 1) * 3));
  return abs(depth) < 0.000001 ? -1 : depth;
}

double Segment::getDepth(const double x_ratio, const double y_ratio) const
{
  double x = IMAGE_WIDTH_ * x_ratio;
  double y = IMAGE_HEIGHT_ * y_ratio;
  int lower_x = max(static_cast<int>(floor(x)), 0);
  int upper_x = min(static_cast<int>(ceil(x)), IMAGE_WIDTH_ - 1);
  int lower_y = max(static_cast<int>(floor(y)), 0);
  int upper_y = min(static_cast<int>(ceil(y)), IMAGE_HEIGHT_ - 1);
  if (lower_x == upper_x && lower_y == upper_y)
    return getDepth(lower_y * IMAGE_WIDTH_ + lower_x);
  else if (lower_x == upper_x)
    return getDepth(lower_y * IMAGE_WIDTH_ + lower_x) * (upper_y - y) + getDepth(upper_y * IMAGE_WIDTH_ + lower_x) * (y - lower_y);
  else if (lower_y == upper_y)
    return getDepth(lower_y * IMAGE_WIDTH_ + lower_x) * (upper_x - x) + getDepth(lower_y * IMAGE_WIDTH_ + upper_x) * (x - lower_x);
  else {
    double area_1 = (x - lower_x) * (y - lower_y);
    double area_2 = (x - lower_x) * (upper_y - y);
    double area_3 = (upper_x - x) * (y - lower_y);
    double area_4 = (upper_x - x) * (upper_y - y);
    double depth_1 = getDepth(lower_y * IMAGE_WIDTH_ + lower_x);
    double depth_2 = getDepth(upper_y * IMAGE_WIDTH_ + lower_x);
    double depth_3 = getDepth(lower_y * IMAGE_WIDTH_ + upper_x);
    double depth_4 = getDepth(upper_y * IMAGE_WIDTH_ + upper_x);
    
    return depth_1 * area_4 + depth_2 * area_3 + depth_3 * area_2 + depth_4 * area_1;
  }
}

vector<double> Segment::getPlane() const
{
  return plane_;
}

ImageMask Segment::getMask() const
{
  return fitting_mask_;
}

double Segment::getConfidence() const
{
  return segment_confidence_;
}

void Segment::writeSegmentImage(const string filename)
{
  Mat segment_image = segment_mask_.drawMaskImage();
  imwrite(filename, segment_image);
}

// Segment &Segment::operator =(const Segment &segment)
// {
//   IMAGE_WIDTH_ = segment.IMAGE_WIDTH_;
//   IMAGE_HEIGHT_ = segment.IMAGE_HEIGHT_;
//   NUM_PIXELS_ = segment.NUM_PIXELS_;
//   CAMERA_PARAMETERS_ = segment.CAMERA_PARAMETERS_;
//   segment_pixels_ = segment.segment_pixels_;
//   segment_mask_ = segment.segment_mask_;
//   segment_radius_ = segment.segment_radius_;
//   segment_center_ = segment.segment_center_;
//   distance_map_ = segment.distance_map_;
//   segment_type_ = segment.segment_type_;
//   disp_plane_ = segment.disp_plane_;
//   depth_plane_ = segment.depth_plane_;
//   input_statistics_ = segment.input_statistics_;
//   //segment_statistics_ = segment.segment_statistics_;
//   depth_map_ = segment.depth_map_;
//   normals_ = segment.normals_;
//   GMM_ = segment.GMM_;
//   segment_confidence_ = segment.segment_confidence_;


//   // Mat sample(1, 5, CV_32FC1);
//   // for (int c = 0; c < 5; c++)
//   //   sample.at<float>(0, c) = rand() % 256;
//   // Vec2d result_1 = segment.GMM_->predict2(sample, noArray());
//   // Vec2d result_2 = GMM_->predict2(sample, noArray());
//   // cout << result_1[0] << '\t' << result_2[0] << endl;

//   return *this;
// }

ostream & operator <<(ostream &out_str, const Segment &segment)
{
  // out_str << segment.segment_pixels_.size() << endl;
  // for (vector<int>::const_iterator pixel_it = segment.segment_pixels_.begin(); pixel_it != segment.segment_pixels_.end(); pixel_it++)
  //   out_str << *pixel_it << '\t';
  // out_str << endl;
  //out_str << segment.visible_pixels_.size() << endl;
  // for (vector<int>::const_iterator pixel_it = segment.visible_pixels_.begin(); pixel_it != segment.visible_pixels_.end(); pixel_it++)
  //   out_str << *pixel_it << '\t';
  // out_str << endl;
  // for (int c = 0; c < 3; c++)
  //   out_str << segment.disp_plane_[c] << '\t';
  // out_str << endl;
  
  out_str << segment.segment_type_ << endl;
  out_str << static_cast<int>(segment.behind_room_structure_) << endl;
  out_str << segment.segment_confidence_ << endl;
  if (segment.segment_type_ == 0) {
    for (int c = 0; c < 4; c++)
      out_str << segment.plane_[c] << '\t';
    out_str << endl;
  } else
    out_str << segment.b_spline_surface_ << endl;
  for (int c = 0; c < 3; c++)
    out_str << segment.color_model_[c] << '\t';
  out_str << endl;
  out_str << segment.segment_mask_ << endl;
  out_str << segment.fitting_mask_ << endl;
  out_str << segment.max_color_likelihood_ << endl;
  return out_str;
}

istream & operator >>(istream &in_str, Segment &segment)
{
  // int num_segment_pixels;
  // in_str >> num_segment_pixels;
  // segment.segment_pixels_ = vector<int>(num_segment_pixels);
  // for (int pixel_index = 0; pixel_index < num_segment_pixels; pixel_index++)
  //   in_str >> segment.segment_pixels_[pixel_index];
  // segment.calcSegmentMaskInfo();
  // int num_visible_pixels;
  // in_str >> num_visible_pixels; 
  // segment.visible_pixels_ = vector<int>(num_visible_pixels);
  // for (int pixel_index = 0; pixel_index < num_visible_pixels; pixel_index++)
  //   in_str >> segment.visible_pixels_[pixel_index];
  
  // segment.disp_plane_.assign(3, 0);
  // for (int c = 0; c < 3; c++)
  //   in_str >> segment.disp_plane_[c];
  
  in_str >> segment.segment_type_;
  //cout << segment.segment_type_ << endl;
  int behind_room_structure_value;
  in_str >> behind_room_structure_value;
  segment.behind_room_structure_ = behind_room_structure_value > 0;
  in_str >> segment.segment_confidence_;
  if (segment.segment_type_ == 0) {
    segment.plane_.assign(4, 0);
    for (int c = 0; c < 4; c++)
      in_str >> segment.plane_[c];
  } else
    in_str >> segment.b_spline_surface_;
  segment.color_model_.assign(3, 0);
  for (int c = 0; c < 3; c++)
    in_str >> segment.color_model_[c];
  in_str >> segment.segment_mask_;
  in_str >> segment.fitting_mask_;
  in_str >> segment.max_color_likelihood_;
  segment.calcPointCloud();
  segment.calcSegmentGrowthMap();
  //cout << "done" << endl;
  return in_str;
}


double Segment::predictColorLikelihood(const Vec3b color) const
{
  Mat sample(1, 3, CV_64FC1);
  sample.at<double>(0, 0) = color[0];
  sample.at<double>(0, 1) = color[1];
  sample.at<double>(0, 2) = color[2];
  
  Vec2d prediction = GMM_->predict2(sample, noArray());
  Mat weights = GMM_->getWeights();
  //cout << prediction[0] + log(weights.at<double>(0, prediction[1])) << endl;
  return prediction[0] + log(weights.at<double>(0, prediction[1]));
}

void Segment::setGMM(const Ptr<EM> GMM)
{
  GMM_ = GMM;
}

void Segment::setGMM(const cv::FileNode GMM_file_node)
{
  GMM_ = EM::create();
  GMM_->read(GMM_file_node);
}

Ptr<EM> Segment::getGMM() const
{
  return GMM_;
}

vector<int> Segment::getSegmentPixels() const
{
  return segment_mask_.getPixels();
}

bool Segment::checkPixelFitting(const Mat &image, const vector<double> &point_cloud, const vector<double> &normals, const int pixel) const
{
  if (getDepth(pixel) <= 0)
    return false;
  vector<double> point = getPoint(point_cloud, pixel);
  vector<double> normal = getPoint(normals, pixel);
  if (checkPointValidity(point)) {
    if (segment_type_ == 0) {
      double distance = abs(calcPointPlaneDistance(point, plane_));
      //if (pixel == 28885)
      //cout << distance << endl;
      if (distance > STATISTICS_.pixel_fitting_distance_threshold)
	return false;
      // vector<double> segment_normal(plane_.begin(), plane_.begin() + 3);
      // double angle = calcAngle(segment_normal, normal);
      // cout << angle << endl;
      // if (angle > STATISTICS_.pixel_fitting_angle_threshold)
      // 	return false;
    } else {
      vector<double> segment_point = getPoint(segment_point_cloud_, pixel);
      vector<double> segment_normal = getPoint(segment_normals_, pixel);
      double distance = 0;
      for (int c = 0; c < 3; c++)
        distance += (segment_point[c] - point[c]) * segment_normal[c];
      distance = abs(distance);
      if (distance > STATISTICS_.pixel_fitting_distance_threshold)
        return false;
      // double angle = calcAngle(segment_normal, normal);
      // if (angle > STATISTICS_.pixel_fitting_angle_threshold)
      //   return false;
    }
  }
  // double color_likelihood = predictColorLikelihood(image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_));
  // //cout << color_likelihood << endl;
  // if (max_color_likelihood_ - color_likelihood > STATISTICS_.pixel_fitting_color_likelihood_threshold)
  //   return false;
  return true;
}

double Segment::calcPixelFittingCost(const Mat &image, const vector<double> &point_cloud, const vector<double> &normals, const int pixel, const RepresenterPenalties &PENALTIES, const double weight_3D, const bool behind_room_structure_tolerance) const
{
  if (checkPointValidity(getPoint(segment_point_cloud_, pixel)) == false)
    return PENALTIES.huge_pen;
  
  double pixel_fitting_cost = 0;
  double sum_pixel_fitting_weights = 0;
  
  vector<double> point = getPoint(point_cloud, pixel);
  vector<double> normal = getPoint(normals, pixel);
  bool is_behind_room_structure = false;
  if (checkPointValidity(point)) {
    if (segment_type_ == 0) {
      double distance = calcPointPlaneDistance(point, plane_);
      //pixel_fitting_cost += min(point_plane_distance / (2 * STATISTICS_.pixel_fitting_distance_threshold), 1.0) * PENALTIES.point_plane_distance_weight;
      //double input_depth = calcNorm(point);
      //double segment_depth = getDepth(pixel);
      double cost_ratio = (behind_room_structure_tolerance) ? PENALTIES.behind_room_structure_cost_ratio : 1;
      //if (behind_room_structure_tolerance && distance > 0)
      //distance = max(distance - STATISTICS_.background_depth_diff_tolerance, 0.0);
      //pixel_fitting_cost += max(cost_ratio - exp(-pow(distance * cost_ratio, 2) / (2 * STATISTICS_.depth_diff_var)), 0.0) * PENALTIES.point_plane_distance_weight * weight_3D;
      //double max_cost = 0.5 * pow(PENALTIES.max_depth_diff, 2);
      //pixel_fitting_cost += min(0.5 * pow(abs(distance), 2) * cost_ratio, max_cost) / max_cost * PENALTIES.point_plane_distance_weight * weight_3D;
      double max_cost = PENALTIES.max_depth_diff;
      pixel_fitting_cost += min(abs(distance) * cost_ratio, max_cost) / max_cost * PENALTIES.point_plane_distance_weight * weight_3D;
      sum_pixel_fitting_weights += PENALTIES.point_plane_distance_weight * weight_3D;
      
      pixel_fitting_cost += PENALTIES.point_plane_distance_weight * (1 - weight_3D);
      sum_pixel_fitting_weights += PENALTIES.point_plane_distance_weight * (1 - weight_3D);
      
      //if (segment_depth < input_depth - STATISTICS_.depth_conflict_threshold && behind_room_structure_tolerance == false)
      	//pixel_fitting_cost = PENALTIES.point_plane_distance_weight * weight_3D;
      
      
      // if (pixel == 17127)
      //  	cout << calcPointPlaneDistance(point, plane_) << '\t' << pixel_fitting_cost / sum_pixel_fitting_weights << endl;
      
      double angle = calcAngle(vector<double>(plane_.begin(), plane_.begin() + 3), normal);
      pixel_fitting_cost += min(angle / (PENALTIES.max_angle_diff), 1.0) * PENALTIES.point_plane_angle_weight * weight_3D;
      sum_pixel_fitting_weights += PENALTIES.point_plane_angle_weight * weight_3D;

      pixel_fitting_cost += PENALTIES.point_plane_angle_weight * (1 - weight_3D);
      sum_pixel_fitting_weights += PENALTIES.point_plane_angle_weight * (1 - weight_3D);

      // if (pixel == 13828)
      // 	cout << angle << '\t' << pixel_fitting_cost << '\t' << sum_pixel_fitting_weights << endl;
      
      //is_behind_room_structure = behind_room_structure_tolerance && (segment_depth < input_depth);
    } else {
      vector<double> segment_point = getPoint(segment_point_cloud_, pixel);
      vector<double> segment_normal = getPoint(segment_normals_, pixel);
      double distance = 0;
      for (int c = 0; c < 3; c++)
	distance += (segment_point[c] - point[c]) * segment_normal[c];
      distance = abs(distance);
      pixel_fitting_cost += (1 - exp(-pow(distance, 2) / (2 * STATISTICS_.depth_diff_var))) * PENALTIES.point_plane_distance_weight * weight_3D;
      sum_pixel_fitting_weights += PENALTIES.point_plane_distance_weight * weight_3D;

      pixel_fitting_cost += PENALTIES.point_plane_distance_weight * (1 - weight_3D);
      sum_pixel_fitting_weights += PENALTIES.point_plane_distance_weight * (1 - weight_3D);

      //      double input_depth = calcNorm(point);
      //double segment_depth = getDepth(pixel);
      //if (segment_depth < input_depth - STATISTICS_.depth_conflict_threshold && behind_room_structure_tolerance == false)
       	//pixel_fitting_cost = PENALTIES.point_plane_distance_weight * weight_3D;


      // if (pixel == 23314) {
      // 	cout << distance << '\t' << segment_depth << '\t' << input_depth << '\t' << weight_3D << endl;
      //   cout << pixel_fitting_cost << '\t' << sum_pixel_fitting_weights << endl;
      // }
      
      double angle = calcAngle(segment_normal, normal);
      pixel_fitting_cost += min(angle / (PENALTIES.max_angle_diff), 1.0) * PENALTIES.point_plane_angle_weight * weight_3D;
      sum_pixel_fitting_weights += PENALTIES.point_plane_angle_weight * weight_3D;

      pixel_fitting_cost += PENALTIES.point_plane_angle_weight * (1 - weight_3D);
      sum_pixel_fitting_weights += PENALTIES.point_plane_angle_weight * (1 - weight_3D);

      
      // if (pixel == 23314)
      //   cout << pixel_fitting_cost << '\t' << sum_pixel_fitting_weights << endl;
      
      pixel_fitting_cost += PENALTIES.non_plane_weight * weight_3D;
      sum_pixel_fitting_weights += PENALTIES.non_plane_weight * weight_3D;

      
      // if (pixel == 23314)
      //   cout << pixel_fitting_cost << '\t' << sum_pixel_fitting_weights << endl;
      
      //is_behind_room_structure = behind_room_structure_tolerance && (segment_depth < input_depth);
    }
  } else {
    pixel_fitting_cost += PENALTIES.point_plane_distance_weight;
    sum_pixel_fitting_weights += PENALTIES.point_plane_distance_weight;
    pixel_fitting_cost += PENALTIES.point_plane_angle_weight;
    sum_pixel_fitting_weights += PENALTIES.point_plane_angle_weight;
    if (segment_type_ != 0) {
      pixel_fitting_cost += PENALTIES.non_plane_weight;
      sum_pixel_fitting_weights += PENALTIES.non_plane_weight;
    }
  }
  
  
  //double color_likelihood = predictColorLikelihood(image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_));

  Vec3b color = image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_);
  double color_diff = 0;
  for (int c = 0; c < 3; c++)
    color_diff += pow(color[c] - color_model_[c], 2);
  color_diff = sqrt(color_diff);
  //double color_fitting_cost = 1 - exp(-pow(color_diff, 2) / (2 * STATISTICS_.color_diff_var));
  double color_fitting_cost = min(color_diff / PENALTIES.max_color_diff, 1.0);
  
  pixel_fitting_cost += color_fitting_cost * PENALTIES.color_likelihood_weight;
  sum_pixel_fitting_weights += PENALTIES.color_likelihood_weight;

  if (sum_pixel_fitting_weights == 0)
    return 0;
  
  // if (pixel == 27449)
  //   cout << checkPointValidity(point) << '\t' << color_fitting_cost << '\t' << pixel_fitting_cost << '\t' << sum_pixel_fitting_weights << PENALTIES.color_likelihood_weight << endl;

  // if (pixel == 13828) {
  //   cout << image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) << endl;
  //   for (int c = 0; c < 3; c++)
  //     cout << color_model_[c] << '\t';
  //   cout << STATISTICS_.color_diff_var << endl;
  //   cout << endl;
  // }

  // if (pixel == 13828)
  //   cout << color_fitting_cost << '\t' << pixel_fitting_cost << '\t' << sum_pixel_fitting_weights << endl;
  
  //double cost_ratio = is_behind_room_structure ? PENALTIES.behind_room_structure_cost_ratio : 1;
  double cost_ratio = 1;
  return pixel_fitting_cost / sum_pixel_fitting_weights * cost_ratio;
  
  // vector<double> point = getPoint(point_cloud, pixel);
  // double point_plane_distance = calcPointPlaneDistance(point, plane_);
  // pixel_fitting_cost += (1 - point_plane_distance_histogram_->getProbability(point_plane_distance)) * PENALTIES.point_plane_distance_weight;
  // double point_plane_angle = calcAngle(vector<double>(plane_.begin(), plane_.begin() + 3), point);
  // pixel_fitting_cost += point_plane_angle_histogram_->getProbability(point_plane_angle) * PENALTIES.point_plane_angle_weight;
  // //double pixel_center_distance = calcDistance(segment_center_, getVec(1.0 * (pixel % IMAGE_WIDTH_), 1.0 * (pixel / IMAGE_WIDTH_)));
  // //pixel_fitting_cost += pixel_center_distance_histogram_->getProbability(pixel_center_distance) * PENALTIES.pixel_center_distance_weight;
  // double color_likelihood = predictColorLikelihood(image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_));
  // pixel_fitting_cost += color_likelihood_histogram_->getProbability(color_likelihood) * PENALTIES.color_likelihood_weight;
  // return pixel_fitting_cost;
}

void Segment::checkPixelFittingCosts(const Mat &image, const vector<double> &point_cloud, const vector<double> &normals, const RepresenterPenalties &PENALTIES, const int index) const
{
  Mat point_plane_distance_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
  Mat point_plane_angle_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
  Mat color_likelihood_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
  Mat cost_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
  
  for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++) {
    if (checkPointValidity(getPoint(point_cloud, pixel)) == false || checkPointValidity(getPoint(normals, pixel)) == false)
      continue;
    vector<double> point = getPoint(point_cloud, pixel);
    double point_plane_distance = abs(calcPointPlaneDistance(point, plane_));
    //    point_plane_distance_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = point_plane_distance_histogram_->getProbability(point_plane_distance) * 255;
    point_plane_distance_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = min(point_plane_distance / (2 * STATISTICS_.pixel_fitting_distance_threshold), 1.0) * 255;
    vector<double> normal = getPoint(normals, pixel);
    double point_plane_angle = calcAngle(vector<double>(plane_.begin(), plane_.begin() + 3), normal);
    //point_plane_angle = min(point_plane_angle, M_PI - point_plane_angle);
    point_plane_angle_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = min(point_plane_angle / (2 * STATISTICS_.pixel_fitting_angle_threshold), 1.0) * 255;
    double color_likelihood = predictColorLikelihood(image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_));
    assert(color_likelihood <= max_color_likelihood_);
    color_likelihood_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = min((max_color_likelihood_ - color_likelihood) / STATISTICS_.pixel_fitting_color_likelihood_threshold, 1.0) * 255;
    double cost = calcPixelFittingCost(image, point_cloud, normals, pixel, PENALTIES, 1, false);
    cost_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = min(cost * 100, 255.0);
  }
  imwrite("Test/segment_image_" + to_string(index) + ".bmp", segment_mask_.drawMaskImage());
  //imwrite("Test/point_plane_distance_image_" + to_string(index) + ".bmp", point_plane_distance_image);
  //imwrite("Test/point_plane_angle_image_" + to_string(index) + ".bmp", point_plane_angle_image);
  //imwrite("Test/color_likelihood_image_" + to_string(index) + ".bmp", color_likelihood_image);
  //imwrite("Test/cost_image_" + to_string(index) + ".bmp", cost_image);
  //exit(1);
}

vector<int> Segment::projectToOtherViewpoints(const int pixel, const double viewpoint_movement)
{
  vector<int> projected_pixels;
  int x = pixel % IMAGE_WIDTH_;
  int y = pixel / IMAGE_WIDTH_;
  // double u = x - CAMERA_PARAMETERS_[1];
  // double v = y - CAMERA_PARAMETERS_[2];
  // //double depth = 1 / ((plane(0) * u + plane(1) * v + plane(2)) / plane(3));
  
  // double disp = disp_plane_[0] * u + disp_plane_[1] * v + disp_plane_[2];
  //double depth = disp != 0 ? 1 / disp : 0;
  
  double depth = getDepth(pixel);
  if (depth <= 0)
    return projected_pixels;
  
  int delta = round(viewpoint_movement / depth * CAMERA_PARAMETERS_[0]);
  if (x - delta >= 0)
    projected_pixels.push_back(pixel - delta);
  if (x + delta < IMAGE_WIDTH_)
    projected_pixels.push_back(pixel + delta + NUM_PIXELS_);
  if (y - delta >= 0)
    projected_pixels.push_back(pixel - delta * IMAGE_WIDTH_ + NUM_PIXELS_ * 2);
  if (y + delta < IMAGE_HEIGHT_)
    projected_pixels.push_back(pixel + delta * IMAGE_WIDTH_ + NUM_PIXELS_ * 3);
  return projected_pixels;
}

Matrix3d Segment::getUnwarpTransform(const std::vector<double> &point_cloud, const std::vector<double> &CAMERA_PARAMETERS) const
{
  if (segment_type_ != 0) {
    return Matrix3d::Identity();
  }
  Vector3d normal;
  normal << plane_[0], plane_[1], plane_[2];
  
  Vector3d vertical_vec;
  vertical_vec << 0, 1, 0;
  Vector3d warped_horizontal_vec = normal.cross(vertical_vec);
  warped_horizontal_vec.normalize();
  Vector3d warped_vertical_vec = normal.cross(warped_horizontal_vec);
  if (warped_vertical_vec.dot(vertical_vec) < 0) {
    warped_horizontal_vec *= -1;
    warped_vertical_vec *= -1;
  }
  Matrix3d rotation_mat;  
  rotation_mat.col(0) = warped_horizontal_vec;
  rotation_mat.col(1) = warped_vertical_vec;
  rotation_mat.col(2) = normal;
  Matrix3d unwarp_rotation_mat = rotation_mat.inverse();
  
  Matrix3d K_mat;
  K_mat << CAMERA_PARAMETERS[0], 0, CAMERA_PARAMETERS[1],
    0, CAMERA_PARAMETERS[0], CAMERA_PARAMETERS[2],
    0, 0, 1;
  
  //cout << unwarp_rotation_mat << endl;
  
  // Matrix3d unwarp_translation_mat;
  // unwarp_translation_mat << 1, 0, -segment_center_x_,
  //   0, 1, -segment_center_y_,
  //   0, 0, 1;
  
  //cout << unwarp_translation_mat << endl;
  
  Matrix3d unwarp_transform = K_mat * unwarp_rotation_mat * K_mat.inverse();

  ofstream out_str("Test/unwarp_transform.txt");
  out_str << unwarp_transform;
  return unwarp_transform;
}

// double Segment::calcDistance(const vector<double> &point)
// {
//   if (segment_type_ == 0) {
//     double distance = depth_plane_[3];
//     for (int c = 0; c < 3; c++)
//       distance -= depth_plane_[c] * point[c];
//     return abs(distance);
//   } else {
//     return 1;
//   }
// }

void Segment::calcSegmentGrowthMap()
{
  segment_growth_map_.assign(NUM_PIXELS_, -1);
  vector<double> distance_map(NUM_PIXELS_, 1000000);
  
  vector<int> border_pixels;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    if (segment_mask_.at(pixel) == false)
      continue;
    
    segment_growth_map_[pixel] = pixel;
    distance_map[pixel] = 0;
    
    vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_);
    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
      if (segment_mask_.at(*neighbor_pixel_it) == false) {
	border_pixels.push_back(pixel);
	break;
      }
    }
  }
  
  while (border_pixels.size() > 0) {
    vector<int> new_border_pixels;
    for (vector<int>::const_iterator border_pixel_it = border_pixels.begin(); border_pixel_it != border_pixels.end(); border_pixel_it++) {
      int pixel = *border_pixel_it;
      double distance = distance_map[pixel];
      vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_);
      for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	int neighbor_pixel = *neighbor_pixel_it;
	double distance_delta = sqrt(pow(*neighbor_pixel_it % IMAGE_WIDTH_ - pixel % IMAGE_WIDTH_, 2) + pow(*neighbor_pixel_it / IMAGE_WIDTH_ - pixel / IMAGE_WIDTH_, 2));
	if (distance + distance_delta < distance_map[neighbor_pixel]) {
	  segment_growth_map_[neighbor_pixel] = pixel;
	  distance_map[neighbor_pixel] = distance + distance_delta;
	  new_border_pixels.push_back(neighbor_pixel);
	}
      }
    }
    border_pixels = new_border_pixels;
  }
}

bool Segment::checkPairwiseConvexity(const int pixel_1, const int pixel_2)
{
  return segment_growth_map_[pixel_1] != pixel_2;
}

vector<double> Segment::getSegmentPoint(const int pixel) const
{
  return getPoint(segment_point_cloud_, pixel);
}

void Segment::refit(const cv::Mat &image, const std::vector<double> &point_cloud, const vector<double> &normals, const vector<double> &camera_parameters, const ImageMask &fitting_mask, const ImageMask &occluded_mask, const std::vector<double> &pixel_weights, const int segment_type)
{
  segment_type_ = segment_type;
  validity_ = true;
  
  vector<int> fitting_pixels = (fitting_mask - getInvalidPointMask(point_cloud, IMAGE_WIDTH_, IMAGE_HEIGHT_)).getPixels();
  if (segment_type == 0) {
    fitSegmentPlane(image, point_cloud, normals, fitting_mask - getInvalidPointMask(point_cloud, IMAGE_WIDTH_, IMAGE_HEIGHT_), pixel_weights);
    segment_mask_ += occluded_mask;
    // vector<double> points = getPoints(point_cloud, fitting_pixels);
    // plane_ = fitPlane(points);
    // if (plane_[3] > 0)
    //   for (int c = 0; c < 4; c++)
    // 	plane_[c] = -plane_[c];
  } else {
    b_spline_surface_ = BSplineSurface(IMAGE_WIDTH_, IMAGE_HEIGHT_, IMAGE_WIDTH_ / 20, IMAGE_WIDTH_ / 20, segment_type_);
    b_spline_surface_.fitBSplineSurface(point_cloud, fitting_pixels);
    segment_mask_ = fitting_mask + occluded_mask;
  }
  
  calcPointCloud(point_cloud);
  //  segment_mask_ = fitting_mask + occluded_mask;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
    if (segment_mask_.at(pixel) == true && getDepth(pixel) <= 0)
      segment_mask_.set(pixel, false);
  //trainGMM(image, segment_mask_.getPixels());
  fitting_mask_ = segment_mask_ - (segment_mask_ - fitting_mask);
  trainColorModel(image, fitting_mask_.getPixels());
  calcConfidence(point_cloud);
  calcSegmentGrowthMap();
  
  //  calcHistograms(image, point_cloud, normals);
}

//void Segment::setBehindRoomStructure(const bool behind_room_structure)
//{
//  behind_room_structure_ = behind_room_structure;
//}

//bool Segment::getBehindRoomStructure() const
//{
//return behind_room_structure_;
//}
