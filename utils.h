//  utils.h
//  SurfaceStereo
//
//  Created by Chen Liu on 9/30/14.
//  Copyright (c) 2014 Chen Liu. All rights reserved.
//

#ifndef SurfaceStereo_utils_h
#define SurfaceStereo_utils_h

#include <vector>
#include <set>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>

#include "Segment.h"


using namespace std;
using cv::Mat;
using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::VectorXd;
using Eigen::Vector3d;

//IplImage *refineDispImage(Mat &surface_id_image, const vector<double> &coordinates, Mat &disp_image, const int scale = 1);
vector<int> deleteSmallSegments(const vector<int> &segmentation, const int width, const int small_segment_threshold);

vector<double> readPointCloudFromObj(const string filename, const int image_width, const int image_height, const double rotation_angle);
void savePointCloudAsPly(const vector<double> &point_cloud, const char *filename);
void savePointCloudAsMesh(const vector<double> &point_cloud, const char *filename);
vector<int> loadSegmentation(const char *filename);
void saveSegmentation(const vector<int> &segmentation, const char *filename);
Mat drawSegmentationImage(const vector<int> &segmentation, const int width);
Mat drawSegmentationImage(const vector<int> &segmentation, const int width, const Mat &image, const char type);
vector<double> loadPointCloud(const char *filename);
void savePointCloud(const vector<double> &point_cloud, const char *filename);
Mat drawDispImage(const vector<double> &point_cloud, const int width, const MatrixXd &projection_matrix);
Mat drawDispImage(const vector<double> &point_cloud, const int width, const int height);


//normalize point cloud
vector<double> normalizePointCloudByZ(const vector<double> &point_cloud);
//zoom image, point cloud and segmentation
void zoomScene(Mat &image, vector<double> &point_cloud, const double scale_x, const double scale_y);
void cropScene(Mat &image, vector<double> &point_cloud, const int start_x, const int start_y, const int end_x, const int end_y);

vector<double> smoothPointCloud(const vector<double> &point_cloud, const vector<int> &segmentation, const int image_width, const int image_height);
vector<double> inpaintPointCloud(const vector<double> &point_cloud, const int image_width, const int image_height);

bool readPtxFile(const string &filename, cv::Mat &image, vector<double> &point_cloud, vector<double> &camera_parameters);

void estimateCameraParametersPanorama(const vector<double> &point_cloud, const int image_width, const int image_height, double &focal_length, double &cx, double &cy);

vector<double> unprojectPixel(const int pixel, const double depth, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const vector<double> &CAMERA_PARAMETERS, const bool USE_PANORAMA);
int projectPoint(const vector<double> &point, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const vector<double> &CAMERA_PARAMETERS, const bool USE_PANORAMA);
double calcPlaneDepthAtPixel(const vector<double> &plane, const int pixel, const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const vector<double> &CAMERA_PARAMETERS, const bool USE_PANORAMA);

#endif
