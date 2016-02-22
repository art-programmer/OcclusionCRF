//
//  TRWSFusion.cpp
//  SurfaceStereo
//
//  Created by Chen Liu on 11/7/14.
//  Copyright (c) 2014 Chen Liu. All rights reserved.
//

#include "TRWSFusion.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <opencv2/imgproc/imgproc.hpp>

#include "cv_utils.h"

using namespace cv;


TRWSFusion::TRWSFusion(const Mat &image, const vector<double> &point_cloud, const vector<double> &normals, const std::vector<double> &pixel_weights_3D, const RepresenterPenalties &penalties, const DataStatistics &statistics, const bool consider_surface_cost) : image_(image), point_cloud_(point_cloud), normals_(normals), pixel_weights_3D_(pixel_weights_3D), IMAGE_WIDTH_(image.cols), IMAGE_HEIGHT_(image.rows), NUM_PIXELS_(image.cols * image.rows), penalties_(penalties), statistics_(statistics), consider_surface_cost_(consider_surface_cost)
{
  cvtColor(image, image_Lab_, CV_BGR2Lab);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    Vec3b color = image_Lab_.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_);
    color[0] = round(color[0] * 1.0 / 3);
    image_Lab_.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = color;
  }
}

// TRWSFusion::TRWSFusion(TRWSFusion &solver)
//   : NUM_NODES_(solver.NUM_NODES_), IMAGE_WIDTH_(solver.IMAGE_WIDTH_), IMAGE_HEIGHT_(solver.IMAGE_HEIGHT_), NUM_PIXELS_(solver.IMAGE_WIDTH_ * solver.IMAGE_HEIGHT_), proposal_num_layers_(solver.proposal_num_layers_), NUM_LABELS_(solver.NUM_LABELS_), NUM_ITERATIONS_(solver.NUM_ITERATIONS_)
// {

// }

TRWSFusion::~TRWSFusion()
{
}

void TRWSFusion::reset()
{
}

double TRWSFusion::calcDataCost(const int pixel, const int label)
{
  //int segment_id = segmentation_[pixel];
  //bool on_boundary = proposal_distance_to_boundaries_[pixel] != -1;
  bool inside_ROI = proposal_ROI_mask_[pixel];
  vector<int> layer_labels(proposal_num_layers_);
  int label_temp = label;
  for (int layer_index = proposal_num_layers_ - 1; layer_index >= 0; layer_index--) {
    layer_labels[layer_index] = label_temp % (proposal_num_surfaces_ + 1);
    label_temp /= (proposal_num_surfaces_ + 1);
  }
  
  int foremost_non_empty_layer_index = proposal_num_layers_;
  Segment foremost_non_empty_segment;
  for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
    if (layer_labels[layer_index] < proposal_num_surfaces_) {
      foremost_non_empty_layer_index = layer_index;
      foremost_non_empty_segment = proposal_segments_[layer_labels[layer_index]];
      break;
    }
  }
  
  assert(foremost_non_empty_layer_index < proposal_num_layers_);
  
  //double decrease_ratio = pow(penalties_.data_term_layer_decrease_ratio, proposal_num_layers_ - 1 - foremost_non_empty_layer_index);
  //foremost_non_empty_segment_confidence *= (1 - boundary_scores_[pixel] + 0.5);
  //foremost_non_empty_segment_confidence = 1;
  //assert(foremost_non_empty_segment_confidence <= 1 && foremost_non_empty_segment_confidence >= 0);
  //foremost_non_empty_segment_confidence = 1;
  
  int unary_cost = 0;
  //pixel fitting cost
  unary_cost += foremost_non_empty_segment.calcPixelFittingCost(image_Lab_, point_cloud_, normals_, pixel, penalties_, pixel_weights_3D_[pixel], foremost_non_empty_layer_index == proposal_num_layers_ - 1) * penalties_.data_cost_weight;

  // //background empty cost
  // {
  //   if (layer_labels[proposal_num_layers_ - 1] == proposal_num_surfaces_)
  //     unary_cost += penalties_.huge_pen;
  // }
  
  // //same label cost
  // {
  //   int same_label_cost = 0;
  //   set<int> used_surface_ids;
  //   for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
  //     if (layer_labels[layer_index] == proposal_num_surfaces_)
  // 	continue;
  //     if (used_surface_ids.count(layer_labels[layer_index]) > 0)
  // 	same_label_cost += penalties_.huge_pen;
  //     used_surface_ids.insert(layer_labels[layer_index]);
  //   }
  //   unary_cost += same_label_cost;
  // }
  
  // //non-plane segment cost
  // {
  //   int segment_type = foremost_non_empty_segment.getSegmentType();
  //   int segment_type_cost_scale = segment_type == -1 ? 2 : (segment_type == 0 ? 0 : 1);
  //   //    int non_plane_segment_cost = segment_type_cost_scale * (1 - exp(-1 / (2 * pow(penalties_.data_cost_non_plane_ratio, 2)))) * penalties_.depth_inconsistency_pen;
  //   int non_plane_segment_cost = segment_type_cost_scale * penalties_.data_cost_non_plane_ratio * penalties_.depth_inconsistency_pen;
  //   unary_cost += non_plane_segment_cost;
  // }
  
  
  // //layer occupacy cost
  // {
  //   int layer_occupacy_cost = 0;
  //   for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
  //     if (layer_labels[layer_index] == proposal_num_surfaces_)
  // 	continue;
  //     layer_occupacy_cost += penalties_.layer_occupacy_pen;
  //   }
  //   unary_cost += layer_occupacy_cost;
  // }
  
  // //front layer cost
  // {
  //   if (foremost_non_empty_layer_index == 0)
  //     unary_cost += penalties_.front_layer_pen;
  // }
  
  // if (unary_cost < 0 || unary_cost > 1000000) {
  //   cout << "unary cost: " << unary_cost << endl;
  //   exit(1);
  // }
  return unary_cost;
}

double TRWSFusion::calcDepthChangeCost(const double depth_change)
{
  return max(pow(min(depth_change / penalties_.max_depth_change, 1.0), 1) * penalties_.smoothness_pen, penalties_.smoothness_small_constant_pen);
}

double TRWSFusion::calcSmoothnessCost(const int pixel_1, const int pixel_2, const int label_1, const int label_2)
{
  if (label_1 == label_2)
    return 0;
  vector<int> layer_labels_1(proposal_num_layers_);
  int label_temp_1 = label_1;
  for (int layer_index = proposal_num_layers_ - 1; layer_index >= 0; layer_index--) {
    layer_labels_1[layer_index] = label_temp_1 % (proposal_num_surfaces_ + 1);
    label_temp_1 /= (proposal_num_surfaces_ + 1);
  }
  vector<int> layer_labels_2(proposal_num_layers_);
  int label_temp_2 = label_2;
  for (int layer_index = proposal_num_layers_ - 1; layer_index >= 0; layer_index--) {
    layer_labels_2[layer_index] = label_temp_2 % (proposal_num_surfaces_ + 1);
    label_temp_2 /= (proposal_num_surfaces_ + 1);
  }
  
  double pairwise_cost = 0;
  //  double max_boundary_score = max(boundary_scores_[pixel_1], boundary_scores_[pixel_2]);
  bool surface_1_visible = true;
  bool surface_2_visible = true;
  for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
    int surface_id_1 = layer_labels_1[layer_index];
    int surface_id_2 = layer_labels_2[layer_index];
    if (surface_id_1 == surface_id_2) {
      if (surface_id_1 < proposal_num_surfaces_) {
	surface_1_visible = false;
	surface_2_visible = false;
      }
      continue;
    }
    if (surface_id_1 < proposal_num_surfaces_ && surface_id_2 < proposal_num_surfaces_) {
      
      double depth_1_1 = proposal_segments_.at(surface_id_1).getDepth(pixel_1);
      double depth_1_2 = proposal_segments_.at(surface_id_1).getDepth(pixel_2);
      double depth_2_1 = proposal_segments_.at(surface_id_2).getDepth(pixel_1);
      double depth_2_2 = proposal_segments_.at(surface_id_2).getDepth(pixel_2);
      
      
      if (depth_1_1 <= 0 || depth_1_2 <= 0 || depth_2_1 <= 0 || depth_2_2 <= 0)
	return penalties_.large_pen;
      
      double diff_1 = abs(depth_1_1 - depth_2_1);
      double diff_2 = abs(depth_1_2 - depth_2_2);
      double diff_middle = (depth_1_1 - depth_2_1) * (depth_1_2 - depth_2_2) <= 0 ? 0 : 1000000;
      double min_diff = min(min(diff_1, diff_2), diff_middle);
      
      pairwise_cost += calcDepthChangeCost(min_diff);
      
      surface_1_visible = false;
      surface_2_visible = false;
      
    } else if (surface_id_1 < proposal_num_surfaces_ || surface_id_2 < proposal_num_surfaces_) {
      
      double boundary_score = 1;
      bool visible = false;
      if (surface_id_1 < proposal_num_surfaces_ && surface_1_visible) {
	boundary_score = 1; //max_boundary_score;
	visible = true;
	surface_1_visible = false;
      }
      if (surface_id_2 < proposal_num_surfaces_ && surface_2_visible) {
	boundary_score = 1; //max_boundary_score;
	visible = true;
	surface_2_visible = false;
      }
      //if (visible == false)
      pairwise_cost += calcDepthChangeCost(statistics_.depth_change_smoothness_threshold);
    }
  }
  
  
  surface_1_visible = true;
  surface_2_visible = true;
  for (int layer_index = 0; layer_index < proposal_num_layers_ - 1; layer_index++) {
    int surface_id_1 = layer_labels_1[layer_index];
    int surface_id_2 = layer_labels_2[layer_index];
    if (surface_id_1 < proposal_num_surfaces_) {
      if (surface_1_visible == true) {
	if (surface_id_1 != surface_id_2 && proposal_segments_[surface_id_1].checkPairwiseConvexity(pixel_1, pixel_2) == false)
          pairwise_cost += penalties_.smoothness_concave_shape_pen;
  	surface_1_visible = false;

	// if (surface_id_1 != surface_id_2 && proposal_segments_[surface_id_1].checkPairwiseConvexity(pixel_1, pixel_2) == false)
	//   cout << surface_id_1 << '\t' << pixel_1 % IMAGE_WIDTH_ << '\t' << pixel_1 / IMAGE_WIDTH_ << endl;
      }
      // if (surface_id_1 == 2)
      // 	cout << "why" << endl;
    }
    if (surface_id_2 < proposal_num_surfaces_) {
      if (surface_1_visible == true) {
	if (surface_id_1 != surface_id_2 && proposal_segments_[surface_id_2].checkPairwiseConvexity(pixel_2, pixel_1) == false)
  	  pairwise_cost += penalties_.smoothness_concave_shape_pen;
  	surface_2_visible = false;

	// if (surface_id_1 != surface_id_2 && proposal_segments_[surface_id_2].checkPairwiseConvexity(pixel_2, pixel_1) == false)
	//   cout << surface_id_2 << '\t' << pixel_2 % IMAGE_WIDTH_ << '\t' << pixel_2 / IMAGE_WIDTH_ << endl;
      }
      // if (surface_id_2 == 2)
      //   cout << "why" << endl;
    }
  }
  
  
  int visible_surface_1 = -1;
  int visible_surface_2 = -1;
  int visible_layer_index_1 = -1;
  int visible_layer_index_2 = -1;
  for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
    int surface_id_1 = layer_labels_1[layer_index];
    int surface_id_2 = layer_labels_2[layer_index];
    if (surface_id_1 < proposal_num_surfaces_) {
      if (visible_surface_1 == -1) {
        visible_surface_1 = surface_id_1;
        visible_layer_index_1 = layer_index;
      }
      // if (surface_id_1 == 2)
      //        cout << "why" << endl;
    }
    if (surface_id_2 < proposal_num_surfaces_) {
      if (visible_surface_2 == -1) {
        visible_surface_2 = surface_id_2;
        visible_layer_index_2 = layer_index;
      }
      // if (surface_id_2 == 2)
      //   cout << "why" << endl;
    }
  }
  
  if (visible_surface_1 != visible_surface_2) { // && visible_layer_index_1 != visible_layer_index_2) {
    pairwise_cost += max(exp(-pow(cv_utils::calcColorDiff(image_Lab_, pixel_1, pixel_2), 2) / (2 * statistics_.color_diff_var)) * penalties_.smoothness_boundary_pen, penalties_.smoothness_min_boundary_pen);
  } else if (visible_layer_index_1 != visible_layer_index_2) {
    pairwise_cost += penalties_.smoothness_segment_splitted_pen;
  }
  
  double distance_2D = sqrt(pow(pixel_1 % IMAGE_WIDTH_ - pixel_2 % IMAGE_WIDTH_, 2) + pow(pixel_1 / IMAGE_WIDTH_ - pixel_2 / IMAGE_WIDTH_, 2));
  return pairwise_cost / distance_2D;
}

vector<int> TRWSFusion::fuse(const vector<vector<int> > &proposal_labels, const int proposal_num_surfaces, const int proposal_num_layers, const map<int, Segment> &proposal_segments, const vector<int> &previous_solution_indices, const vector<bool> &proposal_ROI_mask)
{
  // cout << proposal_surface_depths_[3][35 * 50 + 32] << '\t' << proposal_surface_depths_[3][35 * 50 + 33] << '\t' << proposal_surface_depths_[4][35 * 50 + 32] << '\t' << proposal_surface_depths_[4][35 * 50 + 33] << endl;
  // cout << calcSmoothnessCostMulti(35 * 50 + 32, 35 * 50 + 33, 6 * 49 + 3 * 7 + 5, 6 * 49 + 4 * 7 + 5) << endl;
  // exit(1);
  cout << "fuse" << endl;
  
  proposal_num_surfaces_ = proposal_num_surfaces;
  proposal_num_layers_ = proposal_num_layers;
  proposal_segments_ = proposal_segments;
  
  //proposal_surface_depths_ = proposal_surface_depths;
  if (proposal_ROI_mask.size() == NUM_PIXELS_)
    proposal_ROI_mask_ = proposal_ROI_mask;
  else
    proposal_ROI_mask_ = vector<bool>(NUM_PIXELS_, true);
  
  const int NUM_NODES = consider_surface_cost_ ? NUM_PIXELS_ + proposal_num_layers_ * proposal_num_surfaces_ : NUM_PIXELS_;
  //const int NUM_NODES = NUM_PIXELS_ + proposal_num_surfaces_;
  //proposal_distance_to_boundaries_ = calcDistanceToBoundaries(proposal_segmentation, IMAGE_WIDTH_, penalties_.segmentation_unconfident_region_width);
  //proposal_distance_to_boundaries_ = proposal_distance_to_boundaries;
  
  
  unique_ptr<MRFEnergy<TypeGeneral> > energy(new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize()));
  //MRFEnergy<TypeGeneral> *energy = new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize());
  vector<MRFEnergy<TypeGeneral>::NodeId> nodes(NUM_NODES);
  
  
  //int pixel_index_offset = proposal_num_layers_ * proposal_num_surfaces_ + proposal_num_layers_;
  //int indicator_index_offset = -NUM_PIXELS_;
  int pixel_index_offset = 0;
  int indicator_index_offset = 0;
  
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    //cout << pixel << endl;
    vector<int> pixel_proposal = proposal_labels[pixel];
    const int NUM_PROPOSALS = pixel_proposal.size();
    if (NUM_PROPOSALS == 0) {
      cout << "empty proposal error: " << pixel << endl;
      exit(1);
    }
    vector<double> cost(NUM_PROPOSALS);
    for (int proposal_index = 0; proposal_index < NUM_PROPOSALS; proposal_index++)
      cost[proposal_index] = calcDataCost(pixel, pixel_proposal[proposal_index]);
    nodes[pixel + pixel_index_offset] = energy->AddNode(TypeGeneral::LocalSize(NUM_PROPOSALS), TypeGeneral::NodeData(&cost[0]));
  }
  
  if (consider_surface_cost_ == true) {
    for (int i = NUM_PIXELS_; i < NUM_PIXELS_ + proposal_num_layers_ * proposal_num_surfaces_; i++) {
      double surface_pen = penalties_.surface_pen;
      //double surface_pen = proposal_segments_.at((i - NUM_PIXELS_) % proposal_num_surfaces_).getBehindRoomStructure() ? penalties_.behind_room_structure_surface_pen : penalties_.surface_pen;
      //double surface_pen = penalties_.surface_pen * (2 - proposal_segments_.at((i - NUM_PIXELS_) % proposal_num_surfaces_).getConfidence());
      vector<int> layer_surface_indicator_proposal = proposal_labels[i];
      const int NUM_PROPOSALS = layer_surface_indicator_proposal.size();
      vector<double> surface_cost(NUM_PROPOSALS);
      for (int proposal_index = 0; proposal_index < NUM_PROPOSALS; proposal_index++) {
        surface_cost[proposal_index] = layer_surface_indicator_proposal[proposal_index] == 1 ? surface_pen : 0;
      }
      nodes[i + indicator_index_offset] = energy->AddNode(TypeGeneral::LocalSize(NUM_PROPOSALS), TypeGeneral::NodeData(&surface_cost[0]));
    }
  }

  bool consider_layer_cost = false;
  if (consider_layer_cost == true) {
    for (int i = NUM_PIXELS_ + proposal_num_layers_ * proposal_num_surfaces_; i < NUM_PIXELS_ + proposal_num_layers_ * proposal_num_surfaces_ + proposal_num_layers_; i++) {
      vector<int> layer_indicator_proposal = proposal_labels[i];
      const int NUM_PROPOSALS = layer_indicator_proposal.size();
      vector<double> layer_cost(NUM_PROPOSALS, 0);
      for (int proposal_index = 0; proposal_index < NUM_PROPOSALS; proposal_index++)
	layer_cost[proposal_index] = layer_indicator_proposal[proposal_index] == 1 ? penalties_.layer_pen : 0;
      nodes[i + indicator_index_offset] = energy->AddNode(TypeGeneral::LocalSize(NUM_PROPOSALS), TypeGeneral::NodeData(&layer_cost[0]));
    }
  }

  // bool consider_label_cost = false;
  // if (consider_label_cost == true) {
  //   for (int i = NUM_PIXELS_; i < NUM_PIXELS_ + pow(proposal_num_surfaces_ + 1, proposal_num_layers_); i++) {
  //     vector<int> label_indicator_proposal = proposal_labels[i];
  //     const int NUM_PROPOSALS = label_indicator_proposal.size();
  //     vector<double> label_cost(NUM_PROPOSALS, 0);
  //     for (int proposal_index = 0; proposal_index < NUM_PROPOSALS; proposal_index++)
  //       label_cost[proposal_index] = label_indicator_proposal[proposal_index] == 1 ? penalties_.label_pen : 0;
  //     nodes[i + indicator_index_offset] = energy->AddNode(TypeGeneral::LocalSize(NUM_PROPOSALS), TypeGeneral::NodeData(&label_cost[0]));
  //   }
  // }
  
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    vector<int> pixel_proposal = proposal_labels[pixel];
    int x = pixel % IMAGE_WIDTH_;
    int y = pixel / IMAGE_WIDTH_;
    vector<int> neighbor_pixels;
    if (x < IMAGE_WIDTH_ - 1)
      neighbor_pixels.push_back(pixel + 1);
    if (y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
    if (x > 0 && y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
    if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
    
    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
      int neighbor_pixel = *neighbor_pixel_it;
      vector<int> neighbor_pixel_proposal = proposal_labels[neighbor_pixel];
      vector<double> cost(pixel_proposal.size() * neighbor_pixel_proposal.size(), 0);
      for (int proposal_index_1 = 0; proposal_index_1 < pixel_proposal.size(); proposal_index_1++)
	for (int proposal_index_2 = 0; proposal_index_2 < neighbor_pixel_proposal.size(); proposal_index_2++)
          //          cost[label_1 + label_2 * NUM_LABELS_] = calcSmoothnessCost(pixel, neighbor_pixel, label_1, label_2);
          cost[proposal_index_1 + proposal_index_2 * pixel_proposal.size()] = calcSmoothnessCost(pixel, neighbor_pixel, pixel_proposal[proposal_index_1], neighbor_pixel_proposal[proposal_index_2]);
      bool has_non_zero_cost = false;
      for (int i = 0; i < cost.size(); i++)
	if (cost[i] > 0)
	  has_non_zero_cost = true;
      if (has_non_zero_cost == true)
	energy->AddEdge(nodes[pixel + pixel_index_offset], nodes[neighbor_pixel + pixel_index_offset], TypeGeneral::EdgeData(TypeGeneral::GENERAL, &cost[0]));

      // if (cost[0] != cost[1] || cost[1] != cost[2] || cost[2] != cost[3])
      // 	for (int i = 0; i < 4; i++)
      // 	  cout << cost[i] << endl;
    }
  }
  
  bool consider_other_viewpoints = true;
  if (consider_other_viewpoints) {
    map<int, map<int, vector<double> > > pairwise_costs;
    vector<vector<set<int> > > layer_pixel_surface_pixel_pairs = calcOverlapPixels(proposal_labels);
    for (int layer_index_1 = 0; layer_index_1 < proposal_num_layers_ - 1; layer_index_1++) {
      vector<map<int, vector<int> > > pixel_surface_proposals_map_vec_1(NUM_PIXELS_);
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
        vector<int> pixel_proposal = proposal_labels[pixel];
        for (vector<int>::const_iterator label_it = pixel_proposal.begin(); label_it != pixel_proposal.end(); label_it++) {
          int surface_id = *label_it / static_cast<int>(pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1 - layer_index_1)) % (proposal_num_surfaces_ + 1);
          if (surface_id < proposal_num_surfaces_)
            pixel_surface_proposals_map_vec_1[pixel][surface_id].push_back(label_it - pixel_proposal.begin());
        }
      }
      vector<set<int> > pixel_surface_pixel_pairs_1 = layer_pixel_surface_pixel_pairs[layer_index_1];
      for (int layer_index_2 = layer_index_1; layer_index_2 < proposal_num_layers_ - 1; layer_index_2++) {
        vector<map<int, vector<int> > > pixel_surface_proposals_map_vec_2(NUM_PIXELS_);
	if (layer_index_2 == layer_index_1)
	  pixel_surface_proposals_map_vec_2 = pixel_surface_proposals_map_vec_1;
	else {
	  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	    vector<int> pixel_proposal = proposal_labels[pixel];
	    for (vector<int>::const_iterator label_it = pixel_proposal.begin(); label_it != pixel_proposal.end(); label_it++) {
	      int surface_id = *label_it / static_cast<int>(pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1 - layer_index_2)) % (proposal_num_surfaces_ + 1);
	      if (surface_id < proposal_num_surfaces_)
		pixel_surface_proposals_map_vec_2[pixel][surface_id].push_back(label_it - pixel_proposal.begin());
	    }
	  }
	}
        vector<set<int> > pixel_surface_pixel_pairs_2 = layer_pixel_surface_pixel_pairs[layer_index_2];
	for (vector<set<int> >::const_iterator pixel_it = pixel_surface_pixel_pairs_1.begin(); pixel_it != pixel_surface_pixel_pairs_1.end(); pixel_it++) {
	  set<int> surface_pixel_pairs_1 = *pixel_it;
	  set<int> surface_pixel_pairs_2 = pixel_surface_pixel_pairs_2[pixel_it - pixel_surface_pixel_pairs_1.begin()];
	  for (set<int>::const_iterator surface_pixel_pair_it_1 = surface_pixel_pairs_1.begin(); surface_pixel_pair_it_1 != surface_pixel_pairs_1.end(); surface_pixel_pair_it_1++) {
	    for (set<int>::const_iterator surface_pixel_pair_it_2 = surface_pixel_pairs_2.begin(); surface_pixel_pair_it_2 != surface_pixel_pairs_2.end(); surface_pixel_pair_it_2++) {
	      // int surface_id_1 = surface_it_1->first;
	      // int pixel_1 = surface_it_1->second;
	      // int surface_id_2 = surface_it_2->first;
	      // int pixel_2 = surface_it_2->second;
              int surface_id_1 = *surface_pixel_pair_it_1 / NUM_PIXELS_;
              int pixel_1 = *surface_pixel_pair_it_1 % NUM_PIXELS_;
              int surface_id_2 = *surface_pixel_pair_it_2 / NUM_PIXELS_;
              int pixel_2 = *surface_pixel_pair_it_2 % NUM_PIXELS_;
              
	      if (pixel_1 == pixel_2 || surface_id_1 == surface_id_2)
		continue;
	      double cost = 0;
	      if (layer_index_2 == layer_index_1) {
		if (surface_id_2 >= surface_id_1)
		  continue;
		if (abs(pixel_1 % IMAGE_WIDTH_ - pixel_2 % IMAGE_WIDTH_) <= 1 && abs(pixel_1 / IMAGE_WIDTH_ - pixel_2 / IMAGE_WIDTH_) <= 1)
		  continue;
		//cout << surface_id_1 << '\t' << surface_id_2 << '\t' << pixel / NUM_PIXELS_ << '\t' << pixel % NUM_PIXELS_ % IMAGE_WIDTH_ << '\t' << pixel % NUM_PIXELS_ / IMAGE_WIDTH_ << endl;
		double depth_diff = abs(proposal_segments_.at(surface_id_1).getDepth(pixel_1) - proposal_segments_.at(surface_id_2).getDepth(pixel_2));
		//cost = min(depth_diff * pow(penalties_.smoothness_term_layer_decrease_ratio, proposal_num_layers_ - 1 - layer_index_1) / statistics_.depth_change_smoothness_threshold * penalties_.smoothness_empty_non_empty_ratio, 1.0) * penalties_.other_viewpoint_depth_change_pen + penalties_.smoothness_small_constant_pen;
		cost = calcDepthChangeCost(depth_diff);
              } else {
		if (proposal_segments_.at(surface_id_1).getDepth(pixel_1) > proposal_segments_.at(surface_id_2).getDepth(pixel_2) + statistics_.depth_conflict_threshold)
		  cost = penalties_.other_viewpoint_depth_conflict_pen;
	      }
	      //double cost = (1 - exp(-pow(depth_diff, 2) / (2 * pow(statistics_.depth_change_smoothness_threshold * penalties_.smoothness_cost_depth_change_ratio, 2)))) * penalties_.other_viewpoint_depth_change_pen;
	      //double cost = log(2 / (1 + exp(-pow(depth_diff, 2) / (2 * pow(statistics_.depth_change_smoothness_threshold * penalties_.smoothness_cost_depth_change_ratio, 2))))) / log(2) * penalties_.other_viewpoint_depth_change_pen;
	      if (cost < 0.000001)
		continue;
	      // if (pixel_1 >= pixel_2) {
	      //              cout << pixel_1 << '\t' << pixel_2 << '\t' << surface_id_1 << '\t' << surface_id_2 << '\t' << proposal_segments_.at(surface_id_1).getDepth(pixel_1) << '\t' << proposal_segments_.at(surface_id_2).getDepth(pixel_2) << '\t' << layer_index_1 << '\t' << layer_index_2 << endl;
		
              // }

	      if (pixel_1 < pixel_2) {
                if (pairwise_costs.count(pixel_1) == 0 || pairwise_costs[pixel_1].count(pixel_2) == 0)
		  pairwise_costs[pixel_1][pixel_2] = vector<double>(proposal_labels[pixel_1].size() * proposal_labels[pixel_2].size(), 0);
	      } else {
		if (pairwise_costs.count(pixel_2) == 0 || pairwise_costs[pixel_2].count(pixel_1) == 0)
                  pairwise_costs[pixel_2][pixel_1] = vector<double>(proposal_labels[pixel_1].size() * proposal_labels[pixel_2].size(), 0);
	      }
              vector<int> surface_proposals_1 = pixel_surface_proposals_map_vec_1[pixel_1][surface_id_1];
	      vector<int> surface_proposals_2 = pixel_surface_proposals_map_vec_2[pixel_2][surface_id_2];
	      for (vector<int>::const_iterator proposal_it_1 = surface_proposals_1.begin(); proposal_it_1 != surface_proposals_1.end(); proposal_it_1++)
		for (vector<int>::const_iterator proposal_it_2 = surface_proposals_2.begin(); proposal_it_2 != surface_proposals_2.end(); proposal_it_2++)
		  if (pixel_1 < pixel_2)
		    pairwise_costs[pixel_1][pixel_2][*proposal_it_1 + *proposal_it_2 * proposal_labels[pixel_1].size()] += cost;
                  else
		    pairwise_costs[pixel_2][pixel_1][*proposal_it_2 + *proposal_it_1 * proposal_labels[pixel_2].size()] += cost;
	    }
	  }
	}
      }
    }
    
    for (map<int, map<int, vector<double> > >::iterator pixel_it_1 = pairwise_costs.begin(); pixel_it_1 != pairwise_costs.end(); pixel_it_1++)
      for (map<int, vector<double> >::iterator pixel_it_2 = pixel_it_1->second.begin(); pixel_it_2 != pixel_it_1->second.end(); pixel_it_2++)
	energy->AddEdge(nodes[pixel_it_1->first + pixel_index_offset], nodes[pixel_it_2->first + pixel_index_offset], TypeGeneral::EdgeData(TypeGeneral::GENERAL, &pixel_it_2->second[0]));
  }
  
  
  if (consider_surface_cost_ == true) {
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      vector<int> pixel_proposal = proposal_labels[pixel];
      for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	for (int surface_id = 0; surface_id < proposal_num_surfaces_; surface_id++) {
          int layer_surface_indicator_index = NUM_PIXELS_ + layer_index * proposal_num_surfaces_ + surface_id;
	  
          vector<int> layer_surface_indicator_proposal = proposal_labels[layer_surface_indicator_index];
	  vector<double> cost(pixel_proposal.size() * layer_surface_indicator_proposal.size(), 0);
	  bool has_non_zero_cost = false;
          for (int proposal_index_1 = 0; proposal_index_1 < pixel_proposal.size(); proposal_index_1++) {
            for (int proposal_index_2 = 0; proposal_index_2 < layer_surface_indicator_proposal.size(); proposal_index_2++) {
	      int label = pixel_proposal[proposal_index_1];
	      int label_surface_id = label / static_cast<int>(pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
              double layer_surface_indicator_conflict_cost = (label_surface_id == surface_id && layer_surface_indicator_proposal[proposal_index_2] == 0) ? penalties_.huge_pen : 0;
	      if (layer_surface_indicator_conflict_cost > 0) {
		cost[proposal_index_1 + proposal_index_2 * pixel_proposal.size()] = layer_surface_indicator_conflict_cost;
		has_non_zero_cost = true;
	      }
	    }
	  }
	  
	  if (has_non_zero_cost == true)
            energy->AddEdge(nodes[pixel + pixel_index_offset], nodes[layer_surface_indicator_index + indicator_index_offset], TypeGeneral::EdgeData(TypeGeneral::GENERAL, &cost[0]));
        }
      }
    }
    
    for (int surface_id = 0; surface_id < proposal_num_surfaces_; surface_id++) {
      for (int layer_index_1 = 0; layer_index_1 < proposal_num_layers_; layer_index_1++) {
	int layer_surface_indicator_index_1 = NUM_PIXELS_ + layer_index_1 * proposal_num_surfaces_ + surface_id;
	vector<int> layer_surface_indicator_proposal_1 = proposal_labels[layer_surface_indicator_index_1];
        for (int layer_index_2 = layer_index_1 + 1; layer_index_2 < proposal_num_layers_; layer_index_2++) {
	  int layer_surface_indicator_index_2 = NUM_PIXELS_ + layer_index_2 * proposal_num_surfaces_ + surface_id;
	  vector<int> layer_surface_indicator_proposal_2 = proposal_labels[layer_surface_indicator_index_2];  
          vector<double> cost(layer_surface_indicator_proposal_1.size() * layer_surface_indicator_proposal_2.size(), 0);
          bool has_non_zero_cost = false;
          for (int proposal_index_1 = 0; proposal_index_1 < layer_surface_indicator_proposal_1.size(); proposal_index_1++) {
            for (int proposal_index_2 = 0; proposal_index_2 < layer_surface_indicator_proposal_2.size(); proposal_index_2++) {
	      if (layer_surface_indicator_proposal_1[proposal_index_1] == 1 && layer_surface_indicator_proposal_2[proposal_index_2] == 1)
		cost[proposal_index_1 + proposal_index_2 * layer_surface_indicator_proposal_1.size()] = penalties_.surface_splitted_pen;
	      has_non_zero_cost = true;
            }
          }
	  
          if (has_non_zero_cost == true)
            energy->AddEdge(nodes[layer_surface_indicator_index_1 + pixel_index_offset], nodes[layer_surface_indicator_index_2 + indicator_index_offset], TypeGeneral::EdgeData(TypeGeneral::GENERAL, &cost[0]));
        }
      }
    }
  }
  
  if (consider_layer_cost == true) {
    // for (int i = NUM_PIXELS_ + proposal_num_layers_ * proposal_num_surfaces_; i < NUM_PIXELS_ + proposal_num_layers_ * proposal_num_surfaces_ + proposal_num_layers_; i++) {
    //   vector<int> layer_indicator_proposal = proposals[i];
    //   const int NUM_PROPOSALS = layer_indicator_proposal.size();
    //   vector<double> layer_cost(NUM_PROPOSALS, 0);
    //   for (int proposal_index = 0; proposal_index < NUM_PROPOSALS; proposal_index++)
    //     layer_cost[proposal_index] = layer_indicator_proposal[proposal_index] == 1 ? penalties_.layer_pen : 0;
    //   nodes[i] = energy->AddNode(TypeGeneral::LocalSize(NUM_PROPOSALS), TypeGeneral::NodeData(&layer_cost[0]));
    // }

    if (true) {
      for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	int layer_indicator_index = NUM_PIXELS_ + proposal_num_layers_ * proposal_num_surfaces_ + layer_index;
	vector<int> layer_indicator_proposal = proposal_labels[layer_indicator_index];
	for (int surface_id = 0; surface_id < proposal_num_surfaces_; surface_id++) {
	  int layer_surface_indicator_index = NUM_PIXELS_ + layer_index * proposal_num_surfaces_ + surface_id;
	  vector<int> layer_surface_indicator_proposal = proposal_labels[layer_surface_indicator_index];
	  vector<double> cost(layer_surface_indicator_proposal.size() * layer_indicator_proposal.size(), 0);
	  bool has_non_zero_cost = false;
	  for (int proposal_index_1 = 0; proposal_index_1 < layer_surface_indicator_proposal.size(); proposal_index_1++) {
	    for (int proposal_index_2 = 0; proposal_index_2 < layer_indicator_proposal.size(); proposal_index_2++) {
	      int label = layer_surface_indicator_proposal[proposal_index_1];
	      int layer_indicator_conflict_cost = (label == 1 && layer_indicator_proposal[proposal_index_2] == 0) ? penalties_.huge_pen : 0;
	      if (layer_indicator_conflict_cost > 0) {
		cost[proposal_index_1 + proposal_index_2 * layer_surface_indicator_proposal.size()] = layer_indicator_conflict_cost;
		has_non_zero_cost = true;
	      }
	    }
	  }
	
	  if (has_non_zero_cost == true)
	    energy->AddEdge(nodes[layer_surface_indicator_index + indicator_index_offset], nodes[layer_indicator_index + indicator_index_offset], TypeGeneral::EdgeData(TypeGeneral::GENERAL, &cost[0]));
	}
      }
    } else {
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	vector<int> pixel_proposal = proposal_labels[pixel];
	for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	  int layer_indicator_index = NUM_PIXELS_ + proposal_num_layers_ * proposal_num_surfaces_ + layer_index;
	  vector<int> layer_indicator_proposal = proposal_labels[layer_indicator_index];
	  vector<double> cost(pixel_proposal.size() * layer_indicator_proposal.size(), 0);
	  bool has_non_zero_cost = false;
	  for (int proposal_index_1 = 0; proposal_index_1 < pixel_proposal.size(); proposal_index_1++) {
	    for (int proposal_index_2 = 0; proposal_index_2 < layer_indicator_proposal.size(); proposal_index_2++) {
	      int label = pixel_proposal[proposal_index_1];
	      int layer_indicator_conflict_cost = (label / static_cast<int>(pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1) < proposal_num_surfaces_ && layer_indicator_proposal[proposal_index_2] == 0) ? penalties_.huge_pen : 0;
	      if (layer_indicator_conflict_cost > 0) {
		cost[proposal_index_1 + proposal_index_2 * pixel_proposal.size()] = layer_indicator_conflict_cost;
		has_non_zero_cost = true;
	      }
	    }
	  }
        
	  if (has_non_zero_cost == true)
	    energy->AddEdge(nodes[pixel + pixel_index_offset], nodes[layer_indicator_index + indicator_index_offset], TypeGeneral::EdgeData(TypeGeneral::GENERAL, &cost[0]));
	}
      }
    }
  }

  // if (consider_label_cost == true) {
  //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //     vector<int> pixel_proposal = proposal_labels[pixel];
  //     for (int label = 0; label < pow(proposal_num_surfaces_ + 1, proposal_num_layers_); label++) {
  //         int label_indicator_index = NUM_PIXELS_ + label;
          
  //         vector<int> label_indicator_proposal = proposal_labels[label_indicator_index];
  //         vector<double> cost(pixel_proposal.size() * label_indicator_proposal.size(), 0);
  //         bool has_non_zero_cost = false;
  //         for (int proposal_index_1 = 0; proposal_index_1 < pixel_proposal.size(); proposal_index_1++) {
  //           for (int proposal_index_2 = 0; proposal_index_2 < label_indicator_proposal.size(); proposal_index_2++) {
  //             int pixel_label = pixel_proposal[proposal_index_1];
  //             double label_indicator_conflict_cost = (pixel_label == label && label_indicator_proposal[proposal_index_2] == 0) ? penalties_.label_indicator_conflict_pen : 0;
  //             if (label_indicator_conflict_cost > 0) {
  //               cost[proposal_index_1 + proposal_index_2 * pixel_proposal.size()] = label_indicator_conflict_cost;
  //               has_non_zero_cost = true;
  //             }
  //           }
  //         }
	  
  //         if (has_non_zero_cost == true)
  //           energy->AddEdge(nodes[pixel + pixel_index_offset], nodes[label_indicator_index + indicator_index_offset], TypeGeneral::EdgeData(TypeGeneral::GENERAL, &cost[0]));
  //     }
  //   }
  // }


  const int NUM_INDICATORS = proposal_num_layers_ * proposal_num_surfaces_; // + proposal_num_layers_;
  vector<int> fixed_indicator_mask(NUM_INDICATORS, -1);
  int num_fixed_indicators = 0;
  if (consider_surface_cost_) {
    map<int, set<int> > surface_layers;
    vector<vector<int> > layer_pixel_single_segment_vec(proposal_num_layers_, vector<int>(NUM_PIXELS_, proposal_num_surfaces_));
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      vector<int> pixel_proposal = proposal_labels[pixel];
      for (int proposal_index = 0; proposal_index < pixel_proposal.size(); proposal_index++) {
	int label = pixel_proposal[proposal_index];
	for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	  int surface_id = label / static_cast<int>(pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
	  if (surface_id < proposal_num_surfaces_) {
	    // if (surface_layers[surface_id].count(layer_index) == 0)
	    //   cout << layer_index << '\t' << surface_id << endl;
	    surface_layers[surface_id].insert(layer_index);
	    if (layer_pixel_single_segment_vec[layer_index][pixel] == proposal_num_surfaces_)
              layer_pixel_single_segment_vec[layer_index][pixel] = surface_id;
	    else if (layer_pixel_single_segment_vec[layer_index][pixel] != -1 && layer_pixel_single_segment_vec[layer_index][pixel] != surface_id)
	      layer_pixel_single_segment_vec[layer_index][pixel] = -1;
          } else if (layer_pixel_single_segment_vec[layer_index][pixel] != -1)
	    layer_pixel_single_segment_vec[layer_index][pixel] = -1;
        }
      }
    }
    // for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
    //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    // 	int surface_id = layer_pixel_single_segment_vec[layer_index][pixel];
    // 	if (surface_id != -1 && surface_id != proposal_num_surfaces_) {
    // 	  int indicator_index = layer_index * proposal_num_surfaces_ + surface_id;
    //       vector<double> fixed_indicator_cost_diff(2, 0);
    //       fixed_indicator_cost_diff[0] = 1000000;
    //       energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&fixed_indicator_cost_diff[0]));
    //       fixed_indicator_mask[indicator_index] = 1;
    //       num_fixed_indicators++;
    // 	}
    //   }
    // }
    
    for (map<int, set<int> >::const_iterator surface_it = surface_layers.begin(); surface_it != surface_layers.end(); surface_it++) {
      set<int> layers = surface_it->second;
      if (layers.size() == proposal_num_layers_)
	continue;
      for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
        if (layers.count(layer_index) > 0)
	  continue;
	int indicator_index = layer_index * proposal_num_surfaces_ + surface_it->first;
	vector<double> fixed_indicator_cost_diff(2, 0);
	fixed_indicator_cost_diff[1] = 1000000;
	energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&fixed_indicator_cost_diff[0]));
	fixed_indicator_mask[indicator_index] = 0;
	num_fixed_indicators++;
      }
      // if (layers.size() == 1) {
      //   int layer_index = *layers.begin();
      //   int indicator_index = layer_index * proposal_num_surfaces_ + surface_it->first;
      //   vector<double> fixed_indicator_cost_diff(2, 0);
      //   fixed_indicator_cost_diff[0] = 1000000;
      //   energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&fixed_indicator_cost_diff[0]));
      //   fixed_indicator_mask[indicator_index] = 1;
      //   num_fixed_indicators++;
      // }  
    }
  }
    
    
  static double previous_energy = -1;
  bool check_previous_energy = true;
  if (check_previous_energy) {
    vector<int> previous_solution_labels(NUM_PIXELS_);
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
      previous_solution_labels[pixel] = proposal_labels[pixel][previous_solution_indices[pixel]];
    vector<int> indicators(proposal_num_surfaces * proposal_num_layers_, 0);
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      int label = previous_solution_labels[pixel];
      for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	int surface_id = label / static_cast<int>(pow(proposal_num_surfaces + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_surfaces + 1);
	if (surface_id < proposal_num_surfaces) {
	  indicators[proposal_num_surfaces * layer_index + surface_id] = 1;
	}
      }
    }
    previous_solution_labels.insert(previous_solution_labels.end(), indicators.begin(), indicators.end());
    double previous_solution_energy = checkSolutionEnergy(previous_solution_labels);
    assert(previous_energy < 0 || abs(previous_solution_energy - previous_energy) < 1);

    bool test_possible_solution = false;
    if (test_possible_solution) {
      vector<int> possible_solution = previous_solution_labels;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	int ori_label = previous_solution_labels[pixel];
	int new_label = 0;
	for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	  int surface_id = ori_label / static_cast<int>(pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
	  if (surface_id != 11)
	    new_label += surface_id * pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1 - layer_index);
          else
	    new_label += 1 * pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1 - layer_index);
        }
	possible_solution[pixel] = new_label;
      }
      energy_ = checkSolutionEnergy(possible_solution);
      return possible_solution;
    }
  }
  //exit(1);
  
  MRFEnergy<TypeGeneral>::Options options;
  options.m_iterMax = 2000;
  options.m_printIter = 200;
  options.m_printMinIter = 100;
  options.m_eps = 0.1;

  //energy->SetAutomaticOrdering();
  //energy->ZeroMessages();
  //energy->AddRandomMessages(0, 0, 0.001);
  
  //double lower_bound;
  
  // static int proposal_index = 0;
  // if (proposal_index == 200) {
  //   solution_.assign(NUM_NODES, 0);
  //   for (int i = 0; i < NUM_NODES; i++)
  //     solution_[i] = proposal_labels[i][0];
  //   checkSolutionEnergy(solution_);
  //   return solution_;
  // }
  // proposal_index++;
  
  energy->Minimize_TRW_S(options, lower_bound_, energy_);
  solution_.assign(NUM_NODES, 0);

  vector<int> fused_labels(NUM_NODES);
  vector<double> confidences(NUM_NODES);
  for (int i = 0; i < NUM_NODES; i++) {
    // if (consider_surface_cost == false && i >= NUM_PIXELS_)
    //   break;
    // if (consider_layer_cost == false && i >= NUM_PIXELS_ + proposal_num_layers_ * proposal_num_surfaces_)
    //   break;
    // if (consider_surface_cost == false && consider_layer_cost == false && i >= NUM_PIXELS_)
    //   break;
    int label = i < NUM_PIXELS_ ? energy->GetSolution(nodes[i + pixel_index_offset]) : energy->GetSolution(nodes[i + indicator_index_offset]);
    //double confidence = i < NUM_PIXELS_ ? energy->GetConfidence(i + pixel_index_offset) : energy->GetConfidence(i + indicator_index_offset);
    solution_[i] = label;
    fused_labels[i] = proposal_labels[i][label];
    //confidences[i] = confidence;

    
    // if (i >= NUM_PIXELS_ && i - NUM_PIXELS_ < proposal_num_layers_ * proposal_num_surfaces_) {
    // //   //if (label == 1)
    //   cout << "layer: " << (i - NUM_PIXELS_) / proposal_num_surfaces_ << "\tsurface: " << (i - NUM_PIXELS_) % proposal_num_surfaces_ << '\t' << fused_labels[i] << '\t' << confidence << endl;
    // }

    
    //cout << confidences[i] << endl;
    //      cout << solution_[i] << endl;
  }
  
  //cout << "energy: " << energy_ << " lower bound: " << lower_bound << endl;

  
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
  //   fused_labels[pixel] = proposal_labels[pixel][1];

  
  checkSolutionEnergy(fused_labels);
  
  //  exit(1);
  // vector<int> test_labels = fused_labels;
  // //test_labels[NUM_PIXELS_ + 26] = 1;
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
  //   if (test_labels[pixel] / (proposal_num_surfaces_ + 1) / (proposal_num_surfaces_ + 1) % (proposal_num_surfaces_ + 1) == 4)
  //     test_labels[pixel] = 4 * pow(proposal_num_surfaces_ + 1, 1) + proposal_num_surfaces_ * pow(proposal_num_surfaces_ + 1, 2) + test_labels[pixel] % (proposal_num_surfaces_ + 1);
  //test_labels[pixel] += -2 * (proposal_num_surfaces_ + 1) + 2 * pow(proposal_num_surfaces_ + 1, 2);
  //test_labels[NUM_PIXELS_ + 36] = 1;
  //test_labels[NUM_PIXELS_ + 37] = 1;
  //  test_labels = getOptimalSolution();
  //  test_labels[29 * 50 + 32] += (4 - 9) * 100 + (2 - 4) * 10;
  //checkSolutionEnergy(test_labels);
  
  const double OPTIMAL_THRESHOLD_SCALE = 1.1;
  const double LOWER_BOUND_DIFF_THRESHOLD = 0.01;
  
  if (energy_ <= lower_bound_ * OPTIMAL_THRESHOLD_SCALE) {
    //delete energy;
    if (energy_ < previous_energy)
      previous_energy = energy_;
    return fused_labels;
  } else {
    energy_ = 100000000;
    lower_bound_ = 100000000;
    return fused_labels;
  }

  if (true) {
    bool optimal_solution_found = false;
    
    if (true) {
      const int NUM_INDICATORS = proposal_num_layers_ * proposal_num_surfaces_; // + proposal_num_layers_;
      int NUM_INCONFIDENT_INDICATORS = NUM_INDICATORS * 0;
      vector<pair<double, int> > confidence_index_pairs;
      for (int i = NUM_PIXELS_; i < NUM_PIXELS_ + NUM_INDICATORS; i++)
	confidence_index_pairs.push_back(make_pair(confidences[i], i - NUM_PIXELS_));
      sort(confidence_index_pairs.begin(), confidence_index_pairs.end());
	
      bool new_indicator_fixed = false;
      for (int i = NUM_INCONFIDENT_INDICATORS; i < NUM_INDICATORS; i++) {
	if (abs(confidence_index_pairs[i].first - penalties_.surface_pen) < 0.0001) {
	  new_indicator_fixed = true;
	    
	  int indicator_index = confidence_index_pairs[i].second;
	  vector<double> fixed_indicator_cost_diff(2, 0);
	  if (fused_labels[NUM_PIXELS_ + indicator_index] == 0)
	    fixed_indicator_cost_diff[1] = 1000000;
	  else
	    fixed_indicator_cost_diff[0] = 1000000;
	  energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&fixed_indicator_cost_diff[0]));
	  fixed_indicator_mask[indicator_index] = fused_labels[NUM_PIXELS_ + indicator_index];
	  num_fixed_indicators++;
	}
      }
      if (new_indicator_fixed == true) {
	for (int surface_id = 0; surface_id < proposal_num_surfaces_; surface_id++) {
	  int not_fixed_layer_index = -1;
	  bool has_non_empty_layer = false;
	  for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	    if (fixed_indicator_mask[layer_index * proposal_num_surfaces_ + surface_id] == -1) {
	      if (not_fixed_layer_index == -1)
		not_fixed_layer_index = layer_index;
	      else {
		not_fixed_layer_index = -1;
		break;
	      }
	    } else if (fixed_indicator_mask[layer_index * proposal_num_surfaces_ + surface_id] == 1) {
	      has_non_empty_layer = true;
	      break;
	    }
	  }
	  if (not_fixed_layer_index != -1 && has_non_empty_layer == false) {
	    int indicator_index = not_fixed_layer_index * proposal_num_surfaces_ + surface_id;
	    vector<double> fixed_indicator_cost_diff(2, 0);
	    fixed_indicator_cost_diff[0] = 1000000;
	    energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&fixed_indicator_cost_diff[0]));

	    fixed_indicator_mask[indicator_index] = 1;
	    num_fixed_indicators++;
	  }
	}
          
	    
	double lower_bound;
	//energy->ZeroMessages();
	energy->Minimize_TRW_S(options, lower_bound, energy_);
	if (energy_ <= lower_bound * OPTIMAL_THRESHOLD_SCALE)
	  optimal_solution_found = true;
      }
    }
      
    while (num_fixed_indicators < NUM_INDICATORS && optimal_solution_found == false) {
      double lowest_energy = -1;
      int lowest_energy_indicator_index = -1;
      int lowest_energy_indicator_value = -1;
      for (int indicator_index = 0; indicator_index < NUM_INDICATORS; indicator_index++) {
	if (fixed_indicator_mask[indicator_index] != -1)
	  continue;
	vector<double> cost_diff(2, 0);
	cost_diff[1] = 1000000;
	energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&cost_diff[0]));
	double lower_bound_0, energy_0;
	//energy->ZeroMessages();
	cout << "try to fix indicator " << indicator_index << " as 0" << endl;
	energy->Minimize_TRW_S(options, lower_bound_0, energy_0);
	  
	if (lowest_energy < 0 || lower_bound_0 < lowest_energy) {
	  lowest_energy = lower_bound_0;
	  lowest_energy_indicator_index = indicator_index;
	  lowest_energy_indicator_value = 0;
	}
	  
	cost_diff[0] = 1000000;
	cost_diff[1] = -1000000;
	energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&cost_diff[0]));
	double lower_bound_1, energy_1;
	//energy->ZeroMessages();
	cout << "try to fix indicator " << indicator_index << " as 1" << endl;
	energy->Minimize_TRW_S(options, lower_bound_1, energy_1);
	  
	if (lowest_energy < 0 || lower_bound_1 < lowest_energy) {
	  lowest_energy = lower_bound_1;
	  lowest_energy_indicator_index = indicator_index;
	  lowest_energy_indicator_value = 1;
	}
	  
	cost_diff[0] = -1000000;
	cost_diff[1] = 0;
	energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&cost_diff[0]));
	  
	if (energy_0 <= lower_bound_0 * OPTIMAL_THRESHOLD_SCALE && lower_bound_0 <= lower_bound_1) {
	  cost_diff[0] = 0;
	  cost_diff[1] = 1000000;
	  energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&cost_diff[0]));
	  //energy->ZeroMessages();
	  energy->Minimize_TRW_S(options, lower_bound_0, energy_0);
	  energy_ = energy_0;
	  optimal_solution_found = true;
	  break;
	}
	if (energy_1 <= lower_bound_1 * OPTIMAL_THRESHOLD_SCALE && lower_bound_1 <= lower_bound_0) {
	  cost_diff[0] = 1000000;
	  cost_diff[1] = 0;
	  energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&cost_diff[0]));
	  //energy->ZeroMessages();
	  energy->Minimize_TRW_S(options, lower_bound_1, energy_1);
	  energy_ = energy_1;
	  optimal_solution_found = true;
	  break;
	}
	// if (abs(lower_bound_0 - lower_bound_1) < min(lower_bound_0, lower_bound_1) * LOWER_BOUND_DIFF_THRESHOLD) {
	//   if (lower_bound_0 < lower_bound_1) {
	//     cout << "fix indicator " << indicator_index << " as 0" << endl;
	//     cost_diff[0] = 0;
	//     cost_diff[1] = 1000000;
	//     energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&cost_diff[0]));
	//   } else {
	//     cout << "fix indicator " << indicator_index << " as 1" << endl;
	//     cost_diff[0] = 1000000;
	//     cost_diff[1] = 0;
	//     energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&cost_diff[0]));
	//   }	  
	// }
      }
      if (optimal_solution_found == true)
	break;
	
      vector<double> fixed_indicator_cost_diff(2, 0);
      if (lowest_energy_indicator_value == 0)
	fixed_indicator_cost_diff[1] = 1000000;
      else
	fixed_indicator_cost_diff[0] = 1000000;
      energy->AddNodeData(nodes[NUM_PIXELS_ + lowest_energy_indicator_index], TypeGeneral::NodeData(&fixed_indicator_cost_diff[0]));
      fixed_indicator_mask[lowest_energy_indicator_index] = lowest_energy_indicator_value;
      num_fixed_indicators++;
      cout << "fix indicator " << lowest_energy_indicator_index << " as " << lowest_energy_indicator_value << endl;
	
      for (int surface_id = 0; surface_id < proposal_num_surfaces_; surface_id++) {
	int not_fixed_layer_index = -1;
	bool has_non_empty_layer = false;
	for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	  if (fixed_indicator_mask[layer_index * proposal_num_surfaces_ + surface_id] == -1) {
	    if (not_fixed_layer_index == -1)
	      not_fixed_layer_index = layer_index;
	    else {
	      not_fixed_layer_index = -1;
	      break;
	    }
	  } else if (fixed_indicator_mask[layer_index * proposal_num_surfaces_ + surface_id] == 1) {
	    has_non_empty_layer = true;
	    break;
	  }
	}
	if (not_fixed_layer_index != -1 && has_non_empty_layer == false) {
	  int indicator_index = not_fixed_layer_index * proposal_num_surfaces_ + surface_id;
	  vector<double> fixed_indicator_cost_diff(2, 0);
	  fixed_indicator_cost_diff[0] = 1000000;
	  energy->AddNodeData(nodes[NUM_PIXELS_ + indicator_index], TypeGeneral::NodeData(&fixed_indicator_cost_diff[0]));
	    
	  fixed_indicator_mask[indicator_index] = 1;
	  num_fixed_indicators++;
	}
      }
    }
    vector<int> fused_labels(NUM_NODES);
    for (int node_index = 0; node_index < NUM_NODES; node_index++) {
      int label = node_index < NUM_PIXELS_ ? energy->GetSolution(nodes[node_index + pixel_index_offset]) : energy->GetSolution(nodes[node_index + indicator_index_offset]);
      fused_labels[node_index] = proposal_labels[node_index][label];
    }
      
      
    energy_ = checkSolutionEnergy(fused_labels);
    // vector<int> test_labels = fused_labels;
    // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
    //   if (test_labels[pixel] / (proposal_num_surfaces_ + 1) / (proposal_num_surfaces_ + 1) % (proposal_num_surfaces_ + 1) == 4)
    //     test_labels[pixel] = 4 * pow(proposal_num_surfaces_ + 1, 1) + proposal_num_surfaces_ * pow(proposal_num_surfaces_ + 1, 2) + test_labels[pixel] % (proposal_num_surfaces_ + 1);
    // checkSolutionEnergy(test_labels);

    if (energy_ < previous_energy)
      previous_energy = energy_;
    return fused_labels;
  }
  
  energy_ = 100000000;
  return fused_labels;
  
  
  if (false) {
    vector<vector<int> > new_proposal_labels = proposal_labels;
    new_proposal_labels[NUM_PIXELS_ + 1 * proposal_num_surfaces_ + 2] = vector<int>(1, 1);
    new_proposal_labels[NUM_PIXELS_ + 1 * proposal_num_surfaces_ + 3] = vector<int>(1, 1);
    new_proposal_labels[NUM_PIXELS_ + 1 * proposal_num_surfaces_ + 4] = vector<int>(1, 1);
    new_proposal_labels[NUM_PIXELS_ + 2 * proposal_num_surfaces_ + 0] = vector<int>(1, 1);
    new_proposal_labels[NUM_PIXELS_ + 2 * proposal_num_surfaces_ + 1] = vector<int>(1, 1);
    new_proposal_labels[NUM_PIXELS_ + 2 * proposal_num_surfaces_ + 5] = vector<int>(1, 1);
    // new_proposals[NUM_PIXELS_ + 1 * proposal_num_surfaces_ + 6] = vector<int>(1, 1);
    // new_proposals[NUM_PIXELS_ + 2 * proposal_num_surfaces_ + 0] = vector<int>(1, 1);
    // new_proposals[NUM_PIXELS_ + 2 * proposal_num_surfaces_ + 1] = vector<int>(1, 1);
    // new_proposals[NUM_PIXELS_ + 2 * proposal_num_surfaces_ + 8] = vector<int>(1, 1);

    fused_labels = vector<int>(); //fuse(new_proposal_labels, proposal_ROI_mask);

    checkSolutionEnergy(fused_labels);
    vector<int> test_labels = fused_labels;
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
      if (test_labels[pixel] / (proposal_num_surfaces_ + 1) % (proposal_num_surfaces_ + 1) == 4)
	test_labels[pixel] = 4 * pow(proposal_num_surfaces_ + 1, 2) + proposal_num_surfaces_ * pow(proposal_num_surfaces_ + 1, 1) + test_labels[pixel] % (proposal_num_surfaces_ + 1);
    checkSolutionEnergy(test_labels);
    return fused_labels;
  }
  
  
  // vector<vector<int> > new_proposals(NUM_NODES_);
  // bool has_inconfidence = false;
  // for (int i = NUM_PIXELS_; i < NUM_NODES_; i++) {
  //   if (consider_surface_cost == false && i >= NUM_PIXELS_)
  //     break;
  //   if (consider_layer_cost == false && i >= NUM_PIXELS_ + proposal_num_layers_ * proposal_num_surfaces_)
  //     break;
  //   if (consider_label_cost == false && i >= NUM_PIXELS_)
  //     break;
  //   if (confidences[i] > confidence_threshold)
  //     new_proposals[i].push_back(proposals[i][solution_[i]]);
  //   else {
  //     new_proposals[i] = proposals[i];
  //     has_inconfidence = true;
  //   }
  // }
  //if (has_inconfidence == true)
  //  return fuse(new_proposals);

  return fused_labels;
}

  
// void TRWSFusion::solve()
// {
//   cout << "solve" << endl;
//   Energy *energy = initializeEnergyFromCalculation();
//   energy->SetFullEdges(1);
//   Energy::Options options;
//   options.method = Energy::Options::CMP;
//   options.iter_max = 100;
//   options.verbose = true;
//   energy->Solve(options);
//   solution_.assign(NUM_NODES_, 0);
//   for (int i = 0; i < NUM_NODES_; i++)
//     solution_[i] = energy->GetSolution(i);
//   cout << "energy: " << energy->ComputeCost() << " lower bound: " << energy->ComputeLowerBound() << endl;
//   delete energy;
// }

// Energy *TRWSFusion::initializeEnergyFromCalculation()
// {
//   Energy *energy = new Energy(NUM_NODES_);
//   const int NUM_PIXELS_ = NUM_PIXELS_;
//   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
//     energy->AddNode(NUM_LABELS_);
//   for (int i = 0; i < NUM_LABELS_; i++)
//     energy->AddNode(2);
  
//   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
//     energy->AddUnaryFactor(pixel, &unary_term_[pixel]);
//   vector<double> surface_cost(2);
//   surface_cost[0] = 0;
//   surface_cost[1] = penalties_.label_pen;
//   for (int i = 0; i < NUM_LABELS_; i++)
//     energy->AddUnaryFactor(NUM_PIXELS_ + i, &surface_cost[0]);
  
  
//   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//     int x = pixel % IMAGE_WIDTH_;
//     int y = pixel / IMAGE_WIDTH_;
//     vector<int> neighbor_pixels;
//     if (x < IMAGE_WIDTH_ - 1)
//       neighbor_pixels.push_back(pixel + 1);
//     if (y < IMAGE_HEIGHT_ - 1)
//       neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
//     if (x > 0 && y < IMAGE_HEIGHT_ - 1)
//       neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
//     if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
//       neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
    
//     for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
//       int neighbor_pixel = *neighbor_pixel_it;
//       vector<double> cost(NUM_LABELS_ * NUM_LABELS_);
//       for (int label_1 = 0; label_1 < NUM_LABELS_; label_1++)
//      for (int label_2 = 0; label_2 < NUM_LABELS_; label_2++)
//        cost[label_1 * NUM_LABELS_ + label_2] = calcSmoothnessCost(pixel, neighbor_pixel, label_1, label_2);
//       energy->AddPairwiseFactor(pixel, neighbor_pixel, &cost[0]);
//     }
//   }
  
//   for (int label = 0; label < NUM_LABELS_; label++) {
//     int surface_indicator_index = NUM_PIXELS_ + label;
//     vector<double> cost(NUM_LABELS_ * 2, 0);
//     cost[label * 2 + 0] = penalties_.surface_indicator_conflict_pen;
//     for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
//       energy->AddPairwiseFactor(pixel, surface_indicator_index, &cost[0]);
//   }
//   return energy;
// }

  
vector<double> TRWSFusion::getEnergyInfo()
{
  vector<double> energy_info(2);
  energy_info[0] = energy_;
  energy_info[1] = lower_bound_;
  return energy_info;
}

vector<int> TRWSFusion::getSolution()
{
  return solution_;
}

// void TRWSFusion::setNoCostMask(const map<long, int> neighbor_pair_label_map)
// {
//   no_cost_mask_ = neighbor_pair_label_map;
// }

double TRWSFusion::checkSolutionEnergy(const vector<int> &solution_for_check)
{
  vector<int> solution = solution_for_check;
  
  if (consider_surface_cost_) {
    vector<int> correct_indicators(proposal_num_surfaces_ * proposal_num_layers_, 0);
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      int label = solution[pixel];
      for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	int surface_id = label / static_cast<int>(pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
	if (surface_id < proposal_num_surfaces_) {
	  correct_indicators[proposal_num_surfaces_ * layer_index + surface_id] = 1;
	}
      }
    }
    bool has_indicator_conflict = false;
    for (int indicator_index = 0; indicator_index < proposal_num_surfaces_ * proposal_num_layers_; indicator_index++) {
      if (solution[indicator_index + NUM_PIXELS_] != correct_indicators[indicator_index]) {
	has_indicator_conflict = true;
	//cout << "correct indicator: " << indicator_index << '\t' << proposal_num_surfaces_ << '\t' << solution[indicator_index + NUM_PIXELS_] << endl;
	solution[indicator_index + NUM_PIXELS_] = correct_indicators[indicator_index];
	//break;
      }
    }
    //assert(has_indicator_conflict == false);
  }
  
  
  // for (int segment_id = 0; segment_id < 4; segment_id++) {
  //   for (int pixel = 8; pixel < 12; pixel++)
  //     cout << proposal_segments_[segment_id].getDepth(pixel) << '\t';
  //   cout << endl;
  // }
  // cout << exp(-pow(calcColorDiff(8, 9), 2) / (2 * color_diff_var_)) * penalties_.smoothness_boundary_pen << '\t' << calcSmoothnessCost(8, 9, solution[8], solution[11]) << '\t' << calcSmoothnessCost(8, 9, solution[8], solution[9]) << endl;
  // cout << exp(-pow(calcColorDiff(9, 10), 2) / (2 * color_diff_var_)) * penalties_.smoothness_boundary_pen << '\t' << calcSmoothnessCost(9, 10, solution[8], solution[11]) << '\t' << calcSmoothnessCost(9, 10, solution[9], solution[10]) << endl;
  // cout << exp(-pow(calcColorDiff(10, 11), 2) / (2 * color_diff_var_)) * penalties_.smoothness_boundary_pen << '\t' << calcSmoothnessCost(10, 11, solution[8], solution[11]) << '\t' << calcSmoothnessCost(10, 11, solution[10], solution[11]) << endl;
  // exit(1);
  
  bool check_energy = false;
  if (check_energy) {
    static int checking_index = 0;
    stringstream energy_filename;
    energy_filename << "Test/energy_" << 0;
    ofstream energy_out_str(energy_filename.str().c_str());
    checking_index++;
    if (true) {
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
	energy_out_str << calcDataCost(pixel, solution[pixel]) << endl;;
    } else {
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	int x = pixel % IMAGE_WIDTH_;
	int y = pixel / IMAGE_WIDTH_;
	vector<int> neighbor_pixels;
	if (x < IMAGE_WIDTH_ - 1)
	  neighbor_pixels.push_back(pixel + 1);
	if (y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
	if (x > 0 && y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
	if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
	
	double cost_sum = 0;
	for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	  int neighbor_pixel = *neighbor_pixel_it;
	  //    	if ((solution[pixel] / ((int)pow(proposal_num_surfaces_ + 1, 0)) % (proposal_num_surfaces_ + 1) == 0 && solution[neighbor_pixel] / ((int)pow(proposal_num_surfaces_ + 1, 0)) % (proposal_num_surfaces_ + 1) == 2)
	  //|| (solution[pixel] / ((int)pow(proposal_num_surfaces_ + 1, 0)) % (proposal_num_surfaces_ + 1) == 2 && solution[neighbor_pixel] / ((int)pow(proposal_num_surfaces_ + 1, 0)) % (proposal_num_surfaces_ + 1) == 0)) {
	  //double cost_1 = log(2 / (1 + boundary_scores_[pixel] * exp(-1 / (2 * pow(penalties_.smoothness_cost_depth_change_ratio, 2))))) / log(2) * penalties_.smoothness_pen + penalties_.smoothness_small_constant_pen;
	  //    	  double cost_2 = log(2 / (1 + boundary_scores_[neighbor_pixel] * exp(-1 / (2 * pow(penalties_.smoothness_cost_depth_change_ratio, 2))))) / log(2) * penalties_.smoothness_pen + penalties_.smoothness_small_constant_pen;
	  //energy_out_str << pixel << '\t' << calcSmoothnessCost(pixel, neighbor_pixel, solution[pixel], solution[neighbor_pixel]) << endl;
	  int label_1 = solution[pixel];
	  int label_2 = solution[neighbor_pixel];
	  
	  //cost_sum += calcSmoothnessCost(pixel, neighbor_pixel, solution[pixel] % (proposal_num_surfaces_ + 1), solution[neighbor_pixel] %(proposal_num_surfaces_ + 1));
	  cost_sum += calcSmoothnessCost(pixel, neighbor_pixel, solution[pixel], solution[neighbor_pixel]);
          //energy_out_str << pixel << '\t' << neighbor_pixel << '\t' << calcSmoothnessCost(pixel, neighbor_pixel, solution[pixel], solution[neighbor_pixel]) << endl;
    	  //        }
	}
	energy_out_str << cost_sum << endl;
      }
    }
    energy_out_str.close();
    exit(1);
  }
  
  // if (checking_index == 2) {
  //   ifstream energy_in_str_1("Test/energy_1");
  //   ifstream energy_in_str_2("Test/energy_2");
  //   ofstream energy_diff_out_str("Test/energy_diff");
  //   cout << "yes" << endl;
  //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //     double energy_1;
  //     energy_in_str_1 >> energy_1;
  //     double energy_2;
  //     energy_in_str_2 >> energy_2;
  //     energy_diff_out_str << pixel << '\t' << energy_1 - energy_2 << endl;
  //   }
  //   energy_diff_out_str.close();  
  //   exit(1);
  // }
  
  double unary_cost = 0;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
    unary_cost += calcDataCost(pixel, solution[pixel]);
  
  double pairwise_cost = 0;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int x = pixel % IMAGE_WIDTH_;
    int y = pixel / IMAGE_WIDTH_;
    vector<int> neighbor_pixels;
    if (x < IMAGE_WIDTH_ - 1)
      neighbor_pixels.push_back(pixel + 1);
    if (y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
    if (x > 0 && y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
    if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
      neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
    
    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
      int neighbor_pixel = *neighbor_pixel_it;
      pairwise_cost += calcSmoothnessCost(pixel, neighbor_pixel, solution[pixel], solution[neighbor_pixel]);
      
      //energy_out_str << pixel << '\t' << neighbor_pixel << '\t' << calcSmoothnessCost(pixel, neighbor_pixel, solution[pixel], solution[neighbor_pixel]) << endl;
      
      // if (solution[pixel] / (int)pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1) == 3 && solution[neighbor_pixel] / (int)pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1) == 4)
      // 	cout << calcSmoothnessCostMulti(pixel, neighbor_pixel, solution[pixel], solution[neighbor_pixel]) << endl;
    }
  }
  
  double other_viewpoint_depth_change_cost = 0;
  bool consider_other_viewpoints = true;
  if (consider_other_viewpoints) {
    
    vector<vector<int> > solution_labels(solution.size());
    for (int i = 0; i < solution.size(); i++)
      solution_labels[i].push_back(solution[i]);
    vector<vector<set<int> > > layer_pixel_surface_pixel_pairs = calcOverlapPixels(solution_labels);
    
    for (int layer_index_1 = 0; layer_index_1 < proposal_num_layers_ - 1; layer_index_1++) {
      vector<set<int> > pixel_surface_pixel_pairs_1 = layer_pixel_surface_pixel_pairs[layer_index_1];
      for (int layer_index_2 = layer_index_1; layer_index_2 < proposal_num_layers_ - 1; layer_index_2++) {
        vector<set<int> > pixel_surface_pixel_pairs_2 = layer_pixel_surface_pixel_pairs[layer_index_2];
        for (vector<set<int> >::const_iterator pixel_it = pixel_surface_pixel_pairs_1.begin(); pixel_it != pixel_surface_pixel_pairs_1.end(); pixel_it++) {
          set<int> surface_pixel_pairs_1 = *pixel_it;
          set<int> surface_pixel_pairs_2 = pixel_surface_pixel_pairs_2[pixel_it - pixel_surface_pixel_pairs_1.begin()];
          for (set<int>::const_iterator surface_pixel_pair_it_1 = surface_pixel_pairs_1.begin(); surface_pixel_pair_it_1 != surface_pixel_pairs_1.end(); surface_pixel_pair_it_1++) {
            for (set<int>::const_iterator surface_pixel_pair_it_2 = surface_pixel_pairs_2.begin(); surface_pixel_pair_it_2 != surface_pixel_pairs_2.end(); surface_pixel_pair_it_2++) {
              // int surface_id_1 = surface_it_1->first;
              // int pixel_1 = surface_it_1->second;
              // int surface_id_2 = surface_it_2->first;
              // int pixel_2 = surface_it_2->second;
              int surface_id_1 = *surface_pixel_pair_it_1 / NUM_PIXELS_;
              int pixel_1 = *surface_pixel_pair_it_1 % NUM_PIXELS_;
              int surface_id_2 = *surface_pixel_pair_it_2 / NUM_PIXELS_;
              int pixel_2 = *surface_pixel_pair_it_2 % NUM_PIXELS_;
	      
	      if (pixel_1 == pixel_2 || surface_id_1 == surface_id_2)
                continue;
              double cost = 0;
              if (layer_index_2 == layer_index_1) {
                if (surface_id_1 >= surface_id_2)
                  continue;
                if (abs(pixel_1 % IMAGE_WIDTH_ - pixel_2 % IMAGE_WIDTH_) <= 1 && abs(pixel_1 / IMAGE_WIDTH_ - pixel_2 / IMAGE_WIDTH_) <= 1)
                  continue;
                double depth_diff = abs(proposal_segments_.at(surface_id_1).getDepth(pixel_1) - proposal_segments_.at(surface_id_2).getDepth(pixel_2));
                //cost = min(depth_diff / statistics_.depth_change_smoothness_threshold * penalties_.smoothness_empty_non_empty_ratio, 1.0) * penalties_.other_viewpoint_depth_change_pen + penalties_.smoothness_small_constant_pen;
		cost = calcDepthChangeCost(depth_diff);
              } else {
                if (proposal_segments_.at(surface_id_1).getDepth(pixel_1) > proposal_segments_.at(surface_id_2).getDepth(pixel_2) + statistics_.depth_conflict_threshold) {
                  cost = penalties_.other_viewpoint_depth_conflict_pen;
		  cout << "other viewpoint cost: " << pixel_1 << '\t' << pixel_2 << '\t' << proposal_segments_.at(surface_id_1).getDepth(pixel_1) << '\t' << proposal_segments_.at(surface_id_2).getDepth(pixel_2) << endl;
		}
              }
	      other_viewpoint_depth_change_cost += cost;
	    }
	  }
        }
      }
    }
  }
  
  double surface_cost = 0;
  double layer_cost = 0;
  if (consider_surface_cost_) {
    for (int i = NUM_PIXELS_; i < NUM_PIXELS_ + proposal_num_layers_ * proposal_num_surfaces_; i++) {
      int layer_surface_indicator = solution[i];
      
      double surface_pen = penalties_.surface_pen;
      //double surface_pen = penalties_.surface_pen * (2 - proposal_segments_.at((i - NUM_PIXELS_) % proposal_num_surfaces_).getConfidence());
      surface_cost += layer_surface_indicator == 1 ? surface_pen : 0;
    }
    
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      int pixel_label = solution[pixel];
      for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	for (int surface_id = 0; surface_id < proposal_num_surfaces_; surface_id++) {
	  int layer_surface_indicator_index = NUM_PIXELS_ + layer_index * proposal_num_surfaces_ + surface_id;
	  
	  int layer_surface_indicator = solution[layer_surface_indicator_index];
	  int label_surface_id = pixel_label / static_cast<int>(pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
	  surface_cost += (label_surface_id == surface_id && layer_surface_indicator == 0) ? penalties_.huge_pen : 0;
	}
      }
    }
    
    for (int surface_id = 0; surface_id < proposal_num_surfaces_; surface_id++) {
      for (int layer_index_1 = 0; layer_index_1 < proposal_num_layers_; layer_index_1++) {
	int layer_surface_indicator_index_1 = NUM_PIXELS_ + layer_index_1 * proposal_num_surfaces_ + surface_id;
	int layer_surface_indicator_1 = solution[layer_surface_indicator_index_1];
	for (int layer_index_2 = layer_index_1 + 1; layer_index_2 < proposal_num_layers_; layer_index_2++) {
	  int layer_surface_indicator_index_2 = NUM_PIXELS_ + layer_index_2 * proposal_num_surfaces_ + surface_id;
	  int layer_surface_indicator_2 = solution[layer_surface_indicator_index_2];
	  surface_cost += layer_surface_indicator_1 == 1 && layer_surface_indicator_2 == 1 ? penalties_.surface_splitted_pen : 0;
	}
      }
    }
  }
  
  
  
  double total_cost = unary_cost + pairwise_cost + other_viewpoint_depth_change_cost + surface_cost;
  cout << "cost: " << total_cost << " = " << unary_cost << " + " << pairwise_cost << " + " << other_viewpoint_depth_change_cost << " + " << surface_cost << endl;
  return total_cost;
}

vector<int> TRWSFusion::getOptimalSolution()
{
  set<int> background_surfaces;
  background_surfaces.insert(0);
  background_surfaces.insert(1);
  background_surfaces.insert(2);
  background_surfaces.insert(6);
  vector<int> background_layer(NUM_PIXELS_);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    double min_depth = 1000000;
    int min_depth_surface = -1;
    for (set<int>::const_iterator surface_it = background_surfaces.begin(); surface_it != background_surfaces.end(); surface_it++) {
      double depth = proposal_segments_.at(*surface_it).getDepth(pixel);
      if (depth > 0 && depth < min_depth) {
	min_depth = depth;
	min_depth_surface = *surface_it;
      }
    }
    background_layer[pixel] = min_depth_surface;
  }

  vector<int> labels(NUM_PIXELS_);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int segment_id = -1; //segmentation_[pixel];
    if (segment_id == 3)
      labels[pixel] = segment_id * (proposal_num_surfaces_ + 1) * (proposal_num_surfaces_ + 1) + proposal_num_surfaces_ * (proposal_num_surfaces_ + 1) + background_layer[pixel];
    else if (segment_id == 4 || segment_id == 5)
      labels[pixel] = proposal_num_surfaces_ * (proposal_num_surfaces_ + 1) * (proposal_num_surfaces_ + 1) + segment_id * (proposal_num_surfaces_ + 1) + background_layer[pixel];
    else
      labels[pixel] = proposal_num_surfaces_ * (proposal_num_surfaces_ + 1) * (proposal_num_surfaces_ + 1) + proposal_num_surfaces_ * (proposal_num_surfaces_ + 1) + background_layer[pixel];
  }
  return labels;
}

vector<vector<set<int> > > TRWSFusion::calcOverlapPixels(const vector<vector<int> > &proposal_labels)
{
  // for (map<int, Segment>::const_iterator segment_it = proposal_segments_.begin(); segment_it != proposal_segments_.end(); segment_it++)
  //   cout << segment_it->first << endl;
  
  vector<vector<set<int> > > layer_pixel_surface_pixel_pairs(proposal_num_layers_, vector<set<int> >(NUM_PIXELS_ * 4));
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    vector<int> pixel_proposal = proposal_labels[pixel];
    for (vector<int>::const_iterator label_it = pixel_proposal.begin(); label_it != pixel_proposal.end(); label_it++) {
      for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
	int surface_id = *label_it / static_cast<int>(pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
        if (surface_id == proposal_num_surfaces_)
	  continue;
	
	// if (layer_index != 2)
	//   cout << *label_it << endl;
        vector<int> projected_pixels = proposal_segments_.at(surface_id).projectToOtherViewpoints(pixel, statistics_.viewpoint_movement);
        for (vector<int>::const_iterator projected_pixel_it = projected_pixels.begin(); projected_pixel_it != projected_pixels.end(); projected_pixel_it++) {
          // if (layer_pixel_surface_pixel_maps[layer_index][*projected_pixel_it].count(surface_id) == 0)
	  //   layer_pixel_surface_pixels_maps[layer_index][*projected_pixel_it][surface_id] = pixel;
          // if (proposal_segments_.at(surface_id).getDepth(pixel) < proposal_segments_.at(surface_id).getDepth(layer_pixel_surface_pixels_maps[layer_index][*projected_pixel_it][surface_id]))
          //   cout << pixel << '\t' << surface_id << endl;
	  layer_pixel_surface_pixel_pairs[layer_index][*projected_pixel_it].insert(surface_id * NUM_PIXELS_ + pixel);
        }
      }
    }
  }
  
  // bool check_projected_image = false;
  // if (check_projected_image) {
  //   Mat projected_pixel_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
  //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //     vector<int> pixel_proposal = proposal_labels[pixel];
  //     for (vector<int>::const_iterator label_it = pixel_proposal.begin(); label_it != pixel_proposal.end(); label_it++) {
  // 	for (int layer_index = 0; layer_index < proposal_num_layers_; layer_index++) {
  // 	  int surface_id = *label_it / static_cast<int>(pow(proposal_num_surfaces_ + 1, proposal_num_layers_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
  // 	  if (surface_id != 10)
  // 	    continue;
  // 	  vector<int> projected_pixels = proposal_segments_.at(surface_id).projectToOtherViewpoints(pixel, statistics_.viewpoint_movement);
  // 	  for (vector<int>::const_iterator projected_pixel_it = projected_pixels.begin(); projected_pixel_it != projected_pixels.end(); projected_pixel_it++) {
  // 	    int projected_pixel = *projected_pixel_it % NUM_PIXELS_;
  // 	    int direction = *projected_pixel_it / NUM_PIXELS_;
  // 	    if (direction != 1)
  // 	      continue;
  // 	    Vec3b color((direction % 3 == 0) * 255, (direction % 3 == 1 || direction == 3) * 255, (direction % 3 == 2) * 255);
  // 	    Vec3b previous_color = projected_pixel_image.at<Vec3b>(projected_pixel / IMAGE_WIDTH_, projected_pixel % IMAGE_WIDTH_);
  // 	    if (previous_color != Vec3b(0, 0, 0) && previous_color != color)
  // 	      color = Vec3b(255, 255, 255);
  // 	    projected_pixel_image.at<Vec3b>(projected_pixel / IMAGE_WIDTH_, projected_pixel % IMAGE_WIDTH_) = color;
  // 	  }
  // 	}
  //     }
  //   }
  //   imwrite("Test/projected_pixel_image_2.bmp", projected_pixel_image);
  //   exit(1);
  // }
  return layer_pixel_surface_pixel_pairs;
}
