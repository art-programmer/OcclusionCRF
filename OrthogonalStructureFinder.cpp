#include "OrthogonalStructureFinder.h"

#include <iostream>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>

// #include "OpenGM/mplp.hxx"
// #include <opengm/inference/trws/trws_trws.hxx>
// #include <opengm/inference/alphaexpansion.hxx>
// #include <opengm/inference/graphcut.hxx>
// #include <opengm/inference/auxiliary/minstcutboost.hxx>
#include "cv_utils.h"
#include <opencv2/core/core.hpp>


using namespace std;
using namespace cv;
using namespace cv_utils;


OrthogonalStructureFinder::OrthogonalStructureFinder(const int image_width, const int image_height, const vector<double> &point_cloud, const std::map<int, Segment> &segments, const vector<int> &visible_segmentation, const vector<bool> &ROI_mask, const DataStatistics statistics, const set<int> &invalid_segments) : point_cloud_(point_cloud), segments_(segments), visible_segmentation_(visible_segmentation), ROI_mask_(ROI_mask), IMAGE_WIDTH_(image_width), IMAGE_HEIGHT_(image_height), NUM_PIXELS_(visible_segmentation.size()), NUM_SURFACES_(segments.size()), statistics_(statistics)
{
  vector<vector<int> > surface_occluding_relations(NUM_SURFACES_, vector<int>(NUM_SURFACES_, 0));
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int segment_id = visible_segmentation_[pixel];
    if (segment_id < 0 || segment_id == NUM_SURFACES_)
      continue;
    
    double depth = segments_.at(segment_id).getDepth(pixel);
    if (depth < 0)
      continue;
    for (int other_segment_id = 0; other_segment_id < NUM_SURFACES_; other_segment_id++) {
      if (other_segment_id == segment_id)
        continue;
      double other_depth = segments_.at(other_segment_id).getDepth(pixel);
      if (other_depth < 0)
        continue;
      
      if (other_depth > depth + statistics_.depth_conflict_threshold)
        surface_occluding_relations[segment_id][other_segment_id]++;
      else if (other_depth < depth - statistics_.depth_conflict_threshold)
        surface_occluding_relations[segment_id][other_segment_id]--;
    }
  }
  segment_pair_convexity_map_.clear();
  for (int segment_id_1 = 0; segment_id_1 < NUM_SURFACES_; segment_id_1++)
    for (int segment_id_2 = segment_id_1; segment_id_2 < NUM_SURFACES_; segment_id_2++)
      if (surface_occluding_relations[segment_id_1][segment_id_2] + surface_occluding_relations[segment_id_2][segment_id_1] < 0)
        segment_pair_convexity_map_[segment_id_1 * NUM_SURFACES_ + segment_id_2] = segment_pair_convexity_map_[segment_id_2 * NUM_SURFACES_ + segment_id_1] = true;
      else
        segment_pair_convexity_map_[segment_id_1 * NUM_SURFACES_ + segment_id_2] = segment_pair_convexity_map_[segment_id_2 * NUM_SURFACES_ + segment_id_1] = false;
  
  
  segment_pair_orthogonality_map_.clear();
  for (int segment_id_1 = 0; segment_id_1 < NUM_SURFACES_; segment_id_1++) {
    for (int segment_id_2 = segment_id_1; segment_id_2 < NUM_SURFACES_; segment_id_2++) {
      if (segments.at(segment_id_1).getSegmentType() != 0 || segments.at(segment_id_2).getSegmentType() != 0)      
        continue;
      vector<double> plane_1 = segments.at(segment_id_1).getPlane();
      vector<double> plane_2 = segments.at(segment_id_2).getPlane();
      double angle = calcAngle(vector<double>(plane_1.begin(), plane_1.begin() + 3), vector<double>(plane_2.begin(), plane_2.begin() + 3));
      if (abs(M_PI / 2 - angle) < statistics_.similar_angle_threshold) {
	segment_pair_orthogonality_map_[segment_id_1 * NUM_SURFACES_ + segment_id_2] = true;
	segment_pair_orthogonality_map_[segment_id_2 * NUM_SURFACES_ + segment_id_1] = true;
      } else {
	segment_pair_orthogonality_map_[segment_id_1 * NUM_SURFACES_ + segment_id_2] = false;
        segment_pair_orthogonality_map_[segment_id_2 * NUM_SURFACES_ + segment_id_1] = false;
      }
    }
  }
  
  
  horizontal_segments_.clear();
  vertical_segments_.clear();
  for (int segment_id = 0; segment_id < NUM_SURFACES_; segment_id++) {
    if (segments.at(segment_id).getSegmentType() != 0 || invalid_segments.count(segment_id) > 0)
      continue;
    vector<double> plane = segments_.at(segment_id).getPlane();
    vector<double> vertical_direction(3, 0);
    vertical_direction[1] = 1;
    double angle = calcAngle(vector<double>(plane.begin(), plane.begin() + 3), vertical_direction);
    if (angle < statistics_.similar_angle_threshold || angle > M_PI - statistics_.similar_angle_threshold)
      horizontal_segments_.push_back(segment_id);
    else if (abs(M_PI / 2 - angle) < statistics_.similar_angle_threshold)
      vertical_segments_.push_back(segment_id);
  }
  // for (vector<int>::const_iterator segment_it = horizontal_segments_.begin(); segment_it != horizontal_segments_.end(); segment_it++)
  //   cout << "horizontal segment: " << *segment_it << endl;
  // for (vector<int>::const_iterator segment_it = vertical_segments_.begin(); segment_it != vertical_segments_.end(); segment_it++)
  //   cout << "vertical segment: " << *segment_it << endl;
}

std::vector<pair<double, vector<int> > > OrthogonalStructureFinder::calcOrthogonalStructures(const int NUM_HORIZONTAL_SURFACES, const int NUM_VERTICAL_SURFACES)
{
  //cout << horizontal_segment_ids_vec.size() << '\t' << vertical_segment_ids_vec.size() << endl;
  vector<pair<double, vector<int> > > orthogonal_hulls;
  vector<vector<int> > horizontal_surfaces_vec = cv_utils::findAllCombinations(horizontal_segments_, NUM_HORIZONTAL_SURFACES);
  vector<vector<int> > vertical_surfaces_vec = cv_utils::findAllCombinations(vertical_segments_, NUM_VERTICAL_SURFACES);
  for (vector<vector<int> >::const_iterator horizontal_surfaces_it = horizontal_surfaces_vec.begin(); horizontal_surfaces_it != horizontal_surfaces_vec.end(); horizontal_surfaces_it++) {
    for (vector<vector<int> >::const_iterator vertical_surfaces_it = vertical_surfaces_vec.begin(); vertical_surfaces_it != vertical_surfaces_vec.end(); vertical_surfaces_it++) {
      vector<int> orthogonal_hull_surfaces = *horizontal_surfaces_it;
      orthogonal_hull_surfaces.insert(orthogonal_hull_surfaces.end(), vertical_surfaces_it->begin(), vertical_surfaces_it->end());
      if (orthogonal_hull_surfaces.size() == 0)
	return orthogonal_hulls;
      
      vector<int> orthogonal_hull(NUM_PIXELS_);
      vector<double> depths(NUM_PIXELS_);
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	int selected_surface_id = -1;
	double selected_depth = -1;
	//cout << selected_surface_id << '\t' << selected_depth << endl;
	for (vector<int>::const_iterator orthogonal_hull_surface_it = orthogonal_hull_surfaces.begin(); orthogonal_hull_surface_it != orthogonal_hull_surfaces.end(); orthogonal_hull_surface_it++) {
	  double depth = segments_.at(*orthogonal_hull_surface_it).getDepth(pixel);
	  if (selected_depth < 0) {
	    selected_surface_id = *orthogonal_hull_surface_it;
	    selected_depth = depth;
	    continue;
	  }
	  if (depth < 0)
	    continue;
	  //cout << *orthogonal_hull_surface_it << '\t' << depth << endl;
	  if (segment_pair_convexity_map_[selected_surface_id * NUM_SURFACES_ + *orthogonal_hull_surface_it]) {
	    if (depth > selected_depth) {
	      selected_surface_id = *orthogonal_hull_surface_it;
	      selected_depth = depth;
	    }
	  } else {
	    if (depth < selected_depth) {
	      selected_surface_id = *orthogonal_hull_surface_it;
	      selected_depth = depth;
	    }
	  }
	}
	orthogonal_hull[pixel] = selected_surface_id;
	depths[pixel] = selected_depth;
      }
      
      int num_ROI_pixels = 0;
      int num_consistent_pixels = 0;
      int num_invalid_pixels = 0;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	int visible_segment_id = visible_segmentation_[pixel];  
	if (visible_segment_id < 0 || visible_segment_id == NUM_SURFACES_)  
	  continue;
	num_ROI_pixels++;
	double visible_depth = segments_.at(visible_segment_id).getDepth(pixel);  
	double depth = depths[pixel];
	if (depth < visible_depth - statistics_.depth_conflict_threshold || depth < 0)
	  num_invalid_pixels++;
	if (visible_segmentation_[pixel] == orthogonal_hull[pixel])
          num_consistent_pixels++;
      }
      
      double score = 1.0 * (num_consistent_pixels - num_invalid_pixels * 10) / num_ROI_pixels;
      
      orthogonal_hulls.push_back(make_pair(score, orthogonal_hull));

      if (score > -1) {
	for (vector<int>::const_iterator orthogonal_hull_surface_it = orthogonal_hull_surfaces.begin(); orthogonal_hull_surface_it != orthogonal_hull_surfaces.end(); orthogonal_hull_surface_it++)
	  cout << *orthogonal_hull_surface_it << '\t';
	cout << "\nscore: " << score << '\t' << num_consistent_pixels << '\t' << num_invalid_pixels << endl;
      }
    }
  }
  //  exit(1);
  
  sort(orthogonal_hulls.begin(), orthogonal_hulls.end());
  if (orthogonal_hulls.size() == 0) {
    //cout << "orthogonal hull not found." << endl;
    return orthogonal_hulls;
  }
  
  vector<int> best_orthogonal_hull = orthogonal_hulls[orthogonal_hulls.size() - 1].second;
  
  return orthogonal_hulls;
}

