#ifndef DataStructures_H
#define DataStructures_H

#include <fstream>
#include <cmath>
#include <opencv2/core/core.hpp>

struct RepresenterPenalties {
  double depth_inconsistency_pen;
  double color_inconsistency_pen;
  double normal_inconsistency_pen;
  double distance_2D_pen;
  double close_parallel_surface_pen;
  double layer_empty_pen;
  //double normal_inconsistency_pen;
  //double segmentation_inconsistency_pen;
  //double segmentation_unconfident_region_inconsistency_pen;
  //double segmentation_unconfident_region_width;
  //double same_label_pen;
  //double depth_conflict_pen;
  //double depth_weak_conflict_pen;
  //double unconfident_region_depth_conflict_pen;
  //double behind_camera_pen;
  //double empty_background_pen;
  //double layer_occupacy_pen;
  
  //double front_layer_pen;
  //double reuse_surface_pen;
  
  double surface_pen;
  double layer_pen;
  double surface_splitted_pen;
  double behind_room_structure_surface_pen;
  //double label_pen;
  //double label_indicator_conflict_pen;
  
  double smoothness_pen;
  double smoothness_small_constant_pen;
  double smoothness_concave_shape_pen;
  double smoothness_segment_splitted_pen;
  double smoothness_spurious_empty_pen;
  double smoothness_boundary_pen;
  double smoothness_min_boundary_pen;
  
  double other_viewpoint_depth_change_pen;
  double other_viewpoint_depth_conflict_pen;
  /* double smoothness_max_depth_change; */
  /* double smoothness_between_empty_pen;   */
  /* double smoothness_pen_back_layer_ratio; */
  /* double smoothness_dangerous_boundary_pen; */
  /* double smoothness_break_smooth_connection_pen; */
  
  //double smoothness_depth_change_threshold;
  //double depth_conflict_threshold;
  
  double data_cost_depth_change_ratio;
  //double data_cost_angle_diff_ratio;
  //double data_cost_color_likelihood_ratio;
  double non_plane_weight;
  double smoothness_empty_non_empty_ratio;
  
  /* double line_pairwise_pen; */
  /* double line_length_no_pen_threshold; */
  /* double line_length_detect_threshold; */
  /* double line_detect_gap; */
  
  double large_pen;
  double huge_pen;
  
  double data_term_layer_decrease_ratio;
  double smoothness_term_layer_decrease_ratio;
  
  double point_plane_distance_weight;
  double point_plane_angle_weight;
  double pixel_center_distance_weight;
  double color_likelihood_weight;
  double data_cost_weight;
  double behind_room_structure_cost_ratio;
  
  double max_depth_change;
  double max_depth_diff;
  double max_angle_diff;
  double max_color_diff;
  
  //  double depth_conflict_ratio_threshold;
  
  //  double plane_fitting_ratio_confidence_threshold;
  //double boundary_score_gradient_weight = 0.2;
};

struct DataStatistics {
  /* double disp_residual = 0; */
  /* double disp_residual_svar = 0; */
  /* double depth_residual = 0; */
  /* double depth_residual_svar = 0; */
  /* double color_likelihood_svar = 0; */
  /* double depth_conflict_threshold = 0; */
  double pixel_fitting_distance_threshold = 0.05;
  double pixel_fitting_angle_threshold = 20 * M_PI / 180;
  double pixel_fitting_color_likelihood_threshold = 30;
  double similar_angle_threshold = 20 * M_PI / 180;
  double depth_diff_var = 0.05;
  double depth_conflict_threshold = 0.05;
  double depth_change_smoothness_threshold = 0.005;
  //double parallel_angle_threshold = 10 * M_PI / 180;
  double viewpoint_movement = 0.1;
  //double color_diff_threshold = 0.5;
  int small_segment_num_pixels_threshold = 20;
  double bspline_surface_num_pixels_threshold;
  double background_depth_diff_tolerance = 0.05;
  double num_grams = 20;
  double fitted_pixel_ratio_threshold;
  double segment_refitting_common_ratio_threshold;
  double color_diff_var;

  double pixel_weight_curvature_ratio = 10;
  double min_pixel_weight = 10;
  
  /* friend std::ostream & operator <<(std::ostream &out_str, const DataStatistics &statistics) */
  /* { */
  /*   out_str << 0.0 << 0.0 << 0.0 << 0.0 << 0.0 << 0.0 << 0.0 << 0.0 << std::endl; */
  /*   /\* out_str << statistics.disp_residual << '\t' << statistics.disp_residual_svar << '\t' << statistics.depth_residual << '\t' << statistics.depth_residual_svar << '\t' << statistics.color_likelihood << '\t' << statistics.color_likelihood_svar << '\t' << statistics.fitting_distance_threshold << '\t' << statistics.fitting_angle_threshold << std::endl; *\/ */
  /*   return out_str; */
  /* } */
  /* friend std::istream & operator >>(std::istream &in_str, DataStatistics &statistics) */
  /* { */
  /*   double temp; */
  /*   in_str >> temp >> temp >> temp >> temp >> temp >> temp >> temp >> temp; */
  /*   /\* in_str >> statistics.disp_residual >> statistics.disp_residual_svar >> statistics.depth_residual >> statistics.depth_residual_svar >> statistics.color_likelihood >> statistics.color_likelihood_svar >> statistics.fitting_distance_threshold >> statistics.fitting_angle_threshold; *\/ */
  /*   /\* statistics.calcThresholds(); *\/ */
  /*   return in_str; */
  /* } */
  
  
  /* void calcThresholds() { */
  /*   depth_conflict_threshold = fitting_distance_threshold; */
  /*   depth_change_smoothness_threshold = fitting_distance_threshold * 0.5; */
  /* } */

  
  static double calcColorDiffVar(cv::Mat &image);
  
};

#endif /* defined(__LayerDepthMap__DataStructures__) */
