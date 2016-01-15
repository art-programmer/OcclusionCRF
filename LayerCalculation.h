#ifndef LayerCalculation_H
#define LayerCalculation_H

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <opencv2/core/core.hpp>

#include "DataStructures.h"
//#include "TRW_S/MRFEnergy.h"
#include "Segment.h"
#include "cv_utils.h"

//std::vector<int> fillLayerOpenGM(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::map<int, Segment> &segments, const RepresenterPenalties &penalties, const DataStatistics statistics, const std::vector<std::set<int> > &pixel_segment_indices_map, const std::vector<double> &min_depths, const std::vector<double> &max_depths, const bool is_background_layer);
//std::vector<int> fillLayer(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::map<int, Segment> &segments, const RepresenterPenalties &penalties, const DataStatistics statistics, const std::vector<std::set<int> > &pixel_segment_indices_map, const std::vector<double> &min_depths, const std::vector<double> &max_depths, const bool is_background_layer);
//std::vector<int> estimateLayer(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::map<int, Segment> &segments, const RepresenterPenalties &penalties, const DataStatistics statistics, const int NUM_LAYERS, const std::vector<int> &current_solution_labels, const bool USE_PANORAMA);

std::vector<std::vector<std::set<int> > > fillLayers(const cv::Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::map<int, Segment> &segments, const RepresenterPenalties &PENALTIES, const DataStatistics STATISTICS, const int NUM_LAYERS, const std::vector<int> &current_solution_labels, const int current_solution_num_surfaces, const std::map<int, std::map<int, bool> > &segment_layer_certainty_map, const bool USE_PANORAMA, const bool TOLERATE_CONFLICTS, const bool APPLY_EROSION_AND_DILATION, const std::string &image_name, const std::vector<std::vector<std::set<int> > > &additional_segment_indices_map = std::vector<std::vector<std::set<int> > >());

std::map<int, std::map<int, bool> > swapLayers(const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const std::map<int, Segment> &segments, const std::vector<int> &current_solution_labels, const int NUM_LAYERS, const DataStatistics STATISTICS, const bool USE_PANORAMA, const bool CONSIDER_ROOM_STRUCTURE_LAYER = false, const std::set<int> &invalid_segments = std::set<int>());
std::map<int, std::map<int, bool> > calcNewSegmentLayers(const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const std::map<int, Segment> &segments, const std::vector<int> &current_solution_labels, const int current_solution_num_surfaces, const std::map<int, cv_utils::ImageMask> &new_segment_masks, const int NUM_LAYERS, const DataStatistics STATISTICS, const bool USE_PANORAMA);

#endif
