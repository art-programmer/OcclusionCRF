#include "LayerCalculation.h"

#include <iostream>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>

// //#include "TRW_S/MRFEnergy.h"

#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/trws/trws_trws.hxx>
#include <opengm/inference/icm.hxx>
#include <opengm/inference/alphaexpansion.hxx>
#include <opengm/inference/alphabetaswap.hxx>
#include <opengm/inference/graphcut.hxx>
#include <opengm/inference/auxiliary/minstcutboost.hxx>
#include <opengm/inference/astar.hxx>

using namespace std;
using namespace cv;
using namespace cv_utils;


// std::vector<int> fillLayerOpenGM(const Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::map<int, Segment> &segments, const RepresenterPenalties &penalties, const DataStatistics statistics, const std::vector<std::set<int> > &pixel_segment_indices_map, const std::vector<double> &min_depths, const std::vector<double> &max_depths, const bool is_background_layer)
// {
//   const int IMAGE_WIDTH = image.cols;
//   const int IMAGE_HEIGHT = image.rows;
//   const int NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;
//   const int NUM_SURFACES = segments.size();
  
//   typedef opengm::GraphicalModel<float, opengm::Adder> Model;
//   size_t *label_nums = new size_t[NUM_PIXELS];
//   for (int pixel = 0; pixel < NUM_PIXELS; pixel++)
//     label_nums[pixel] = NUM_SURFACES + 1;
//   Model gm(opengm::DiscreteSpace<>(label_nums, label_nums + NUM_PIXELS));
  
//   typedef opengm::ExplicitFunction<float> ExplicitFunction;
//   typedef Model::FunctionIdentifier FunctionIdentifier;
  
  
//   vector<bool> existing_segment_mask(NUM_SURFACES, false);
//   for (int pixel = 0; pixel < NUM_PIXELS; pixel++)
//     for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++)
//       existing_segment_mask[*segment_it] = true;
  
//   for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
//     vector<double> data_cost(NUM_SURFACES + 1, 0);
//     bool is_fitted = false;
//     for (int surface_id = 0; surface_id < NUM_SURFACES; surface_id++) {
//       double depth = segments.at(surface_id).getDepth(pixel);
//       if (existing_segment_mask[surface_id] == false) {
//         data_cost[surface_id] = penalties.huge_pen;
// 	continue;
//       }
//       if ((depth < min_depths[pixel] - statistics.depth_conflict_threshold || depth > max_depths[pixel] + statistics.depth_conflict_threshold) && segments.at(surface_id).checkPixelFitting(image, point_cloud, normals, pixel) == false)
//         data_cost[surface_id] = penalties.huge_pen;
//       else if (pixel_segment_indices_map[pixel].size() > 0 && pixel_segment_indices_map[pixel].count(surface_id) == 0)
//         data_cost[surface_id] = penalties.data_cost_weight;
//       if (segments.at(surface_id).checkPixelFitting(image, point_cloud, normals, pixel))
// 	is_fitted = true;
//       //data_cost[surface_id] = penalties.huge_pen;
//     }
//     if (is_background_layer)
//       data_cost[NUM_SURFACES] = penalties.huge_pen;
//     else {
//       if (pixel_segment_indices_map[pixel].size() > 0 || is_fitted)
// 	data_cost[NUM_SURFACES] = penalties.data_cost_weight;
//       else
// 	data_cost[NUM_SURFACES] = 10;
//     }
    
    
//     const size_t shape[] = {NUM_SURFACES + 1};
//     ExplicitFunction f(shape, shape + 1);
//     for (int label = 0; label < NUM_SURFACES + 1; label++)
//       f(label) = static_cast<float>(data_cost[label]);
//     FunctionIdentifier id = gm.addFunction(f);
//     size_t variable_index[] = {pixel};
//     gm.addFactor(id, variable_index, variable_index + 1);
    
//     //    nodes[pixel] = energy->AddNode(TypeGeneral::LocalSize(NUM_SURFACES + 1), TypeGeneral::NodeData(&data_cost[0]));
//   }
  
//   for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
//     vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH, IMAGE_HEIGHT);
//     for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
//       if (*neighbor_pixel_it < pixel)
// 	continue;
//       vector<double> pairwise_cost((NUM_SURFACES + 1) * (NUM_SURFACES + 1), 0);
//       for (int surface_id_1 = 0; surface_id_1 < NUM_SURFACES + 1; surface_id_1++) {
//         for (int surface_id_2 = 0; surface_id_2 < NUM_SURFACES + 1; surface_id_2++) {
//           if (surface_id_2 == surface_id_1)
//             continue;
//           if (surface_id_1 < NUM_SURFACES && surface_id_2 < NUM_SURFACES) {
//             double depth_1_1 = segments.at(surface_id_1).getDepth(pixel);
//             double depth_1_2 = segments.at(surface_id_1).getDepth(*neighbor_pixel_it);
//             double depth_2_1 = segments.at(surface_id_2).getDepth(pixel);
//             double depth_2_2 = segments.at(surface_id_2).getDepth(*neighbor_pixel_it);
	    
//             double diff_1 = abs(depth_1_1 - depth_2_1);
//             double diff_2 = abs(depth_1_2 - depth_2_2);
//             double diff_middle = (depth_1_1 - depth_2_1) * (depth_1_2 - depth_2_2) <= 0 ? 0 : 1000000;
//             double min_diff = min(min(diff_1, diff_2), diff_middle);
//             pairwise_cost[surface_id_1 + surface_id_2 * (NUM_SURFACES + 1)] = max(min(min_diff / statistics.depth_change_smoothness_threshold / penalties.max_depth_change_ratio, 1.0) * penalties.smoothness_pen, penalties.smoothness_small_constant_pen);
//           } else {
//             pairwise_cost[surface_id_1 + surface_id_2 * (NUM_SURFACES + 1)] = penalties.smoothness_empty_non_empty_ratio * penalties.smoothness_pen;
//           }
//         }
//       }
      
//       const size_t shape[] = {
//   	NUM_SURFACES + 1,
//   	NUM_SURFACES + 1
//       };
//       ExplicitFunction f(shape, shape + 2);
//       for (int label_1 = 0; label_1 < NUM_SURFACES + 1; label_1++)
//   	for (int label_2 = 0; label_2 < NUM_SURFACES + 1; label_2++)
//   	  f(label_1, label_2) = static_cast<float>(pairwise_cost[label_1 + label_2 * (NUM_SURFACES + 1)]);
//       FunctionIdentifier id = gm.addFunction(f);
//       size_t variable_indices[] = {pixel, *neighbor_pixel_it};
//       gm.addFactor(id, variable_indices, variable_indices + 2);  
      
//       //energy->AddEdge(nodes[pixel], nodes[*neighbor_pixel_it], TypeGeneral::EdgeData(TypeGeneral::GENERAL, &pairwise_cost[0]));
//     }
//   }
  
//   vector<size_t> labels(NUM_PIXELS);
//   if (false) {
//     opengm::TRWSi_Parameter<Model> parameter(30);
//     opengm::TRWSi<Model, opengm::Minimizer> solver(gm, parameter);
//     opengm::TRWSi<Model, opengm::Minimizer>::VerboseVisitorType verbose_visitor;
//     solver.infer(verbose_visitor);
//     solver.arg(labels);
//     cout << "energy: "<< solver.value() << " lower bound: " << solver.bound() << endl;
//   } else if (false) {
//     // opengm::external::MPLP<Model> solver(gm);
//     // solver.infer();
//     // std::vector<size_t> labels;
//     // solver.arg(surface_ids);
//     // cout << "energy: "<< solver.value() << " lower bound: " << solver.bound() << endl;
//   } else if (false) {
//     typedef opengm::ICM<Model, opengm::Minimizer> IcmType;
//     typedef IcmType::VerboseVisitorType VerboseVisitorType;
//     IcmType solver(gm);
//     VerboseVisitorType verbose_visitor;
//     solver.infer(verbose_visitor);
//     solver.arg(labels);
//     cout << "energy: "<< solver.value() << " lower bound: " << solver.bound() << endl;
//   } {
//     typedef opengm::MinSTCutBoost<size_t, double, opengm::PUSH_RELABEL> MinStCutType;
//     typedef opengm::GraphCut<Model, opengm::Minimizer, MinStCutType> MinGraphCut;
//     typedef opengm::AlphaExpansion<Model, MinGraphCut> MinAlphaExpansion;
//     MinAlphaExpansion solver(gm);
//     // typedef opengm::AlphaBetaSwap<Model, MinGraphCut> MinAlphaBetaSwap;
//     // MinAlphaBetaSwap solver(gm);
//     solver.infer();
//     solver.arg(labels);
//     cout << "value: " << solver.value() << " lower bound: " << solver.bound() << endl;
//   }
//   vector<int> surface_ids(NUM_PIXELS);
//   for (int pixel = 0; pixel < NUM_PIXELS; pixel++)
//     surface_ids[pixel] = labels[pixel];
//   return surface_ids;
// }


// vector<int> estimateLayer(const Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::map<int, Segment> &segments, const RepresenterPenalties &penalties, const DataStatistics statistics, const int NUM_LAYERS, const vector<int> &current_solution_labels, const bool USE_PANORAMA)
// {
//   const int IMAGE_WIDTH = image.cols;
//   const int IMAGE_HEIGHT = image.rows;
//   const int NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;
//   const int NUM_SURFACES = segments.size();
  
//   vector<vector<int> > layer_surface_ids(NUM_LAYERS, vector<int>(NUM_PIXELS, NUM_SURFACES));
//   vector<vector<double> > layer_depths(NUM_LAYERS, vector<double>(NUM_PIXELS, -1));
//   vector<int> visible_layer_indices(NUM_PIXELS);
//   for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
//     int current_solution_label = current_solution_labels[pixel];
//     bool is_visible = true;
//     for (int layer_index = 0; layer_index < NUM_LAYERS; layer_index++) {
//       int surface_id = current_solution_label / static_cast<int>(pow(NUM_SURFACES + 1, NUM_LAYERS - 1 - layer_index)) % (NUM_SURFACES + 1);
//       if (surface_id < NUM_SURFACES) {
//         double depth = segments.at(surface_id).getDepth(pixel);
//         layer_depths[layer_index][pixel] = depth;
//         layer_surface_ids[layer_index][pixel] = surface_id;
	
//         if (is_visible)
//           visible_layer_indices[pixel] = layer_index;
//         is_visible = false;
//       }
//     }
//   }
  
  
//   // cout << calcNorm(getPoint(point_cloud, 8527)) << endl;
//   // cout << segments.at(1).getDepth(8527) << '\t' << segments.at(6).getDepth(8527) << endl;
//   // cout << segments.at(1).checkPixelFitting(image, point_cloud, normals, 8527) << '\t' << segments.at(6).checkPixelFitting(image, point_cloud, normals, 8527) << endl;
//   // exit(1);
//   vector<vector<double> > segment_pair_layer_cost_vec(NUM_SURFACES * NUM_SURFACES, vector<double>((NUM_LAYERS + 1) * (NUM_LAYERS + 1), 0));
//   for (int layer_index = 0; layer_index < NUM_LAYERS; layer_index++) {
//     set<int> segment_indices;
//     for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
//       int surface_id = layer_surface_ids[layer_index][pixel];
//       if (surface_id == NUM_SURFACES)
//         continue;
//       if (visible_layer_indices[pixel] != layer_index || segments.at(surface_id).checkPixelFitting(image, point_cloud, normals, pixel) == false)
//         continue;
//       double segment_depth = segments.at(surface_id).getDepth(pixel);
//       vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH, IMAGE_HEIGHT, USE_PANORAMA);
//       for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
//         int neighbor_surface_id = layer_surface_ids[layer_index][*neighbor_pixel_it];
//         if (neighbor_surface_id == surface_id)
//           continue;
//         if (neighbor_surface_id == NUM_SURFACES) {
//           continue;
//         }
//         if (visible_layer_indices[*neighbor_pixel_it] == layer_index) {
//           if (segments.at(surface_id).checkPixelFitting(image, point_cloud, normals, *neighbor_pixel_it) == true) {
//             for (int layer_index = 0; layer_index <= NUM_LAYERS; layer_index++)
//               for (int neighbor_layer_index = 0; neighbor_layer_index <= NUM_LAYERS; neighbor_layer_index++)
//                 if (neighbor_layer_index != layer_index)
//                   segment_pair_layer_cost_vec[surface_id * NUM_SURFACES + neighbor_surface_id][layer_index * (NUM_LAYERS + 1) + neighbor_layer_index] += penalties.smoothness_pen * penalties.smoothness_empty_non_empty_ratio;
//             continue;
//           }
//           if (segments.at(neighbor_surface_id).checkPixelFitting(image, point_cloud, normals, *neighbor_pixel_it) == false)
//             continue;
//         } else
//           continue;
	
//         double neighbor_segment_neighbor_depth = segments.at(neighbor_surface_id).getDepth(*neighbor_pixel_it);
//         double segment_neighbor_depth = segments.at(surface_id).getDepth(*neighbor_pixel_it);
//         double neighbor_segment_depth = segments.at(neighbor_surface_id).getDepth(pixel);
//         double diff_1 = segment_depth - neighbor_segment_depth;
//         double diff_2 = segment_neighbor_depth - neighbor_segment_neighbor_depth;
//         // double diff_middle = (depth_1_1 - depth_2_1) * (depth_1_2 - depth_2_2) <= 0 ? 0 : 1000000;
//         // double min_diff = min(min(diff_1, diff_2), diff_middle);
//         if (diff_1 < -statistics.depth_conflict_threshold && diff_2 < -statistics.depth_conflict_threshold) {
//           for (int layer_index = 0; layer_index <= NUM_LAYERS; layer_index++)
//             for (int neighbor_layer_index = 0; neighbor_layer_index <= layer_index; neighbor_layer_index++)
//               segment_pair_layer_cost_vec[surface_id * NUM_SURFACES + neighbor_surface_id][layer_index * (NUM_LAYERS + 1) + neighbor_layer_index] += penalties.smoothness_pen;
//         } else if (diff_1 > statistics.depth_conflict_threshold && diff_2 > statistics.depth_conflict_threshold) {
//           if (surface_id == 2 && neighbor_surface_id == 5 && false)
//             cout << visible_layer_indices[*neighbor_pixel_it] << '\t' << pixel << '\t' << *neighbor_pixel_it << '\t' << diff_1 << '\t' << diff_2 << endl;
          
//           for (int layer_index = 0; layer_index <= NUM_LAYERS; layer_index++)
//             for (int neighbor_layer_index = layer_index; neighbor_layer_index <= NUM_LAYERS; neighbor_layer_index++)
//               segment_pair_layer_cost_vec[surface_id * NUM_SURFACES + neighbor_surface_id][layer_index * (NUM_LAYERS + 1) + neighbor_layer_index] += penalties.smoothness_pen;
//         } else
//           for (int layer_index = 0; layer_index <= NUM_LAYERS; layer_index++)
//             for (int neighbor_layer_index = 0; neighbor_layer_index <= NUM_LAYERS; neighbor_layer_index++)
//               if (neighbor_layer_index != layer_index)
//                 segment_pair_layer_cost_vec[surface_id * NUM_SURFACES + neighbor_surface_id][layer_index * (NUM_LAYERS + 1) + neighbor_layer_index] += penalties.smoothness_pen * penalties.smoothness_empty_non_empty_ratio;
//       }
//       if (visible_layer_indices[pixel] == layer_index) {
//         for (int backward_layer_index = layer_index + 1; backward_layer_index < NUM_LAYERS; backward_layer_index++) {
//           for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
//             if (layer_surface_ids[layer_index][*neighbor_pixel_it] == surface_id)
//               continue;
//             int neighbor_surface_id = layer_surface_ids[backward_layer_index][*neighbor_pixel_it];
//             if (neighbor_surface_id == NUM_SURFACES || neighbor_surface_id == surface_id)
//               continue;
	    
//             if (visible_layer_indices[*neighbor_pixel_it] == backward_layer_index) {
//               if (segments.at(surface_id).checkPixelFitting(image, point_cloud, normals, *neighbor_pixel_it) == true) {
//                 for (int layer_index = 0; layer_index <= NUM_LAYERS; layer_index++)
//                   for (int neighbor_layer_index = 0; neighbor_layer_index <= NUM_LAYERS; neighbor_layer_index++)
//                     if (neighbor_layer_index != layer_index)
//                       segment_pair_layer_cost_vec[surface_id * NUM_SURFACES + neighbor_surface_id][layer_index * (NUM_LAYERS + 1) + neighbor_layer_index] += penalties.smoothness_pen * penalties.smoothness_empty_non_empty_ratio;
//                 continue;
//               }
//               // if (checkPointValidity(getPoint(point_cloud, pixel)) == false)
//               //        continue;
//               if (segments.at(neighbor_surface_id).checkPixelFitting(image, point_cloud, normals, *neighbor_pixel_it) == false)
//                 continue;
//             } else
//               continue;
	    
//             double neighbor_segment_neighbor_depth = segments.at(neighbor_surface_id).getDepth(*neighbor_pixel_it);
//             double segment_neighbor_depth = segments.at(surface_id).getDepth(*neighbor_pixel_it);
//             double neighbor_segment_depth = segments.at(neighbor_surface_id).getDepth(pixel);
//             double diff_1 = segment_depth - neighbor_segment_depth;
//             double diff_2 = segment_neighbor_depth - neighbor_segment_neighbor_depth;
//             // double diff_middle = (depth_1_1 - depth_2_1) * (depth_1_2 - depth_2_2) <= 0 ? 0 : 1000000;
//             // double min_diff = min(min(diff_1, diff_2), diff_middle);
//             if (diff_1 < -statistics.depth_conflict_threshold && diff_2 < -statistics.depth_conflict_threshold) {
//               if (surface_id == 6 && neighbor_surface_id == 5 && false)
//                 cout << visible_layer_indices[*neighbor_pixel_it] << '\t' << pixel << '\t' << *neighbor_pixel_it << '\t' << diff_1 << '\t' << diff_2 << endl;
	      
//               for (int layer_index = 0; layer_index <= NUM_LAYERS; layer_index++)
//                 for (int neighbor_layer_index = 0; neighbor_layer_index <= layer_index; neighbor_layer_index++)
//                   segment_pair_layer_cost_vec[surface_id * NUM_SURFACES + neighbor_surface_id][layer_index * (NUM_LAYERS + 1) + neighbor_layer_index] += penalties.smoothness_pen;
//             } else if (diff_1 > statistics.depth_conflict_threshold && diff_2 > statistics.depth_conflict_threshold) {
//               for (int layer_index = 0; layer_index <= NUM_LAYERS; layer_index++)
//                 for (int neighbor_layer_index = layer_index; neighbor_layer_index <= NUM_LAYERS; neighbor_layer_index++)
//                   segment_pair_layer_cost_vec[surface_id * NUM_SURFACES + neighbor_surface_id][layer_index * (NUM_LAYERS + 1) + neighbor_layer_index] += penalties.smoothness_pen;
//             } else
//               for (int layer_index = 0; layer_index <= NUM_LAYERS; layer_index++)
//                 for (int neighbor_layer_index = 0; neighbor_layer_index <= NUM_LAYERS; neighbor_layer_index++)
//                   if (neighbor_layer_index != layer_index)
//                     segment_pair_layer_cost_vec[surface_id * NUM_SURFACES + neighbor_surface_id][layer_index * (NUM_LAYERS + 1) + neighbor_layer_index] += penalties.smoothness_pen * penalties.smoothness_empty_non_empty_ratio;
//           }
//         }
//       }
//     }
//   }
  
//   const double data_cost_weight = penalties.data_cost_weight;
  
//   typedef opengm::GraphicalModel<float, opengm::Adder> Model;
//   size_t *label_nums = new size_t[NUM_SURFACES];
//   for (int pixel = 0; pixel < NUM_SURFACES; pixel++)
//     label_nums[pixel] = NUM_LAYERS + 1;
//   Model gm(opengm::DiscreteSpace<>(label_nums, label_nums + NUM_SURFACES));
  
//   typedef opengm::ExplicitFunction<float> ExplicitFunction;
//   typedef Model::FunctionIdentifier FunctionIdentifier;
  
  
//   srand(time(0));
//   for (int surface_id = 0; surface_id < NUM_SURFACES; surface_id++) {
//     vector<double> data_cost(NUM_LAYERS + 1, 0);
//     for (int layer_index = 0; layer_index <= NUM_LAYERS; layer_index++)
//       data_cost[layer_index] = randomProbability() * data_cost_weight;
//     //data_cost[NUM_LAYERS] = penalties.huge_pen;
//     const size_t shape[] = {NUM_LAYERS + 1};
//     ExplicitFunction f(shape, shape + 1);
//     for (int label = 0; label < NUM_LAYERS + 1; label++)
//       f(label) = static_cast<float>(data_cost[label]);
//     FunctionIdentifier id = gm.addFunction(f);
//     size_t variable_index[] = {surface_id};
//     gm.addFactor(id, variable_index, variable_index + 1);
    
//   }
  
//   for (int surface_id_1 = 0; surface_id_1 < NUM_SURFACES; surface_id_1++) {
//     for (int surface_id_2 = 0; surface_id_2 < NUM_SURFACES; surface_id_2++) {
//       bool has_non_zero_cost = false;
//       for (int layer_index_1 = 0; layer_index_1 <= NUM_LAYERS; layer_index_1++)
//         for (int layer_index_2 = 0; layer_index_2 <= NUM_LAYERS; layer_index_2++)
//           if (segment_pair_layer_cost_vec[surface_id_1 * NUM_SURFACES + surface_id_2][layer_index_1 * (NUM_LAYERS + 1) + layer_index_2] > 0)
//             has_non_zero_cost = true;
//       if (has_non_zero_cost == false)
//         continue;
//       if (false) {
// 	cout << "pairwise cost: " << surface_id_1 << '\t' << surface_id_2 << endl;
// 	for (int layer_index_1 = 0; layer_index_1 <= NUM_LAYERS; layer_index_1++)
// 	  for (int layer_index_2 = 0; layer_index_2 <= NUM_LAYERS; layer_index_2++)
// 	    cout << layer_index_1 << '\t' << layer_index_2 << '\t' << segment_pair_layer_cost_vec[surface_id_1 * NUM_SURFACES + surface_id_2][layer_index_1 * (NUM_LAYERS + 1) + layer_index_2] << endl;
//       }
//     }
//   }
//   //exit(1);
//   for (int surface_id_1 = 0; surface_id_1 < NUM_SURFACES; surface_id_1++) {
//     for (int surface_id_2 = surface_id_1 + 1; surface_id_2 < NUM_SURFACES; surface_id_2++) {
//       vector<double> pairwise_cost((NUM_LAYERS + 1) * (NUM_LAYERS + 1), 0);
//       for (int layer_index_1 = 0; layer_index_1 <= NUM_LAYERS; layer_index_1++)
//         for (int layer_index_2 = 0; layer_index_2 <= NUM_LAYERS; layer_index_2++)
//           pairwise_cost[layer_index_1 + layer_index_2 * (NUM_LAYERS + 1)] = segment_pair_layer_cost_vec[surface_id_1 * NUM_SURFACES + surface_id_2][layer_index_1 * (NUM_LAYERS + 1) + layer_index_2] + segment_pair_layer_cost_vec[surface_id_2 * NUM_SURFACES + surface_id_1][layer_index_2 * (NUM_LAYERS + 1) + layer_index_1];
      
//       const size_t shape[] = {
//         NUM_LAYERS + 1,
//         NUM_LAYERS + 1
//       };
//       ExplicitFunction f(shape, shape + 2);
//       for (int label_1 = 0; label_1 < NUM_LAYERS + 1; label_1++)
//         for (int label_2 = 0; label_2 < NUM_LAYERS + 1; label_2++)
//           f(label_1, label_2) = static_cast<float>(pairwise_cost[label_1 + label_2 * (NUM_LAYERS + 1)]);
//       FunctionIdentifier id = gm.addFunction(f);
//       size_t variable_indices[] = {surface_id_1, surface_id_2};
//       gm.addFactor(id, variable_indices, variable_indices + 2);  
      
//       //      energy->AddEdge(nodes[surface_id_1], nodes[surface_id_2], TypeGeneral::EdgeData(TypeGeneral::GENERAL, &pairwise_cost[0]));
//     }
//   }

//   vector<size_t> labels(NUM_SURFACES);
//   if (false) {
//     opengm::TRWSi_Parameter<Model> parameter(30);
//     opengm::TRWSi<Model, opengm::Minimizer> solver(gm, parameter);
//     opengm::TRWSi<Model, opengm::Minimizer>::VerboseVisitorType verbose_visitor;
//     solver.infer(verbose_visitor);
//     solver.arg(labels);
//     cout << "energy: "<< solver.value() << " lower bound: " << solver.bound() << endl;
//   } else if (false) {
//     typedef opengm::ICM<Model, opengm::Minimizer> IcmType;
//     typedef IcmType::VerboseVisitorType VerboseVisitorType;
//     IcmType solver(gm);
//     VerboseVisitorType verbose_visitor;
//     solver.infer(verbose_visitor);
//     solver.arg(labels);
//     cout << "energy: "<< solver.value() << " lower bound: " << solver.bound() << endl;
//   } else if (true) {
//     typedef opengm::AStar<Model, opengm::Minimizer> AStar;
//     AStar::Parameter parameters;
//     parameters.heuristic_ =  parameters.FASTHEURISTIC;
//     AStar solver(gm);
//     solver.infer();
//     solver.arg(labels);
//     std::cout << "value: " << solver.value() << " lower bound: " << solver.bound() << std::endl;
//   }
  
//   vector<int> segment_layers(NUM_SURFACES);
//   for (int surface_id = 0; surface_id < NUM_SURFACES; surface_id++)
//     segment_layers[surface_id] = labels[surface_id];
//   return segment_layers;
// }

// std::vector<int> LayerFiller::fillLayer(const Mat &image, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::map<int, Segment> &segments, const RepresenterPenalties &penalties, const DataStatistics statistics, const std::vector<std::set<int> > &pixel_segment_indices_map, const std::vector<double> &min_depths, const std::vector<double> &max_depths, const bool is_background_layer)
// {
//   const int IMAGE_WIDTH = image.cols;
//   const int IMAGE_HEIGHT = image.rows;
//   const int NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;
//   const int NUM_SURFACES = segments.size();

//   unique_ptr<MRFEnergy<TypeGeneral> > energy(new MRFEnergy<TypeGeneral>(TypeGeneral::GlobalSize()));
//   vector<MRFEnergy<TypeGeneral>::NodeId> nodes(NUM_PIXELS);

//   vector<bool> existing_segment_mask(NUM_SURFACES, false);
//   for (int pixel = 0; pixel < NUM_PIXELS; pixel++)
//     for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++)
//       existing_segment_mask[*segment_it] = true;

//   for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
//     vector<double> data_cost(NUM_SURFACES + 1, 0);
//     for (int surface_id = 0; surface_id < NUM_SURFACES; surface_id++) {
//       double depth = segments.at(surface_id).getDepth(pixel);
//       if (existing_segment_mask[surface_id] == false)
// 	data_cost[surface_id] = penalties.huge_pen;
//       else if ((depth < min_depths[pixel] - statistics.depth_conflict_threshold || depth > max_depths[pixel] + statistics.depth_conflict_threshold) && segments.at(surface_id).checkPixelFitting(image, point_cloud, normals, pixel) == false)
// 	data_cost[surface_id] = penalties.huge_pen;
//       else if (pixel_segment_indices_map[pixel].size() > 0 && pixel_segment_indices_map[pixel].count(surface_id) == 0)
// 	data_cost[surface_id] = penalties.data_cost_weight;
// 	//data_cost[surface_id] = penalties.huge_pen;
//     }
//     if (is_background_layer)
//       data_cost[NUM_SURFACES] = penalties.huge_pen;
//     else if (pixel_segment_indices_map[pixel].size() > 0)
//       data_cost[NUM_SURFACES] = penalties.data_cost_weight;
//     else
//       data_cost[NUM_SURFACES] = 1;
//     nodes[pixel] = energy->AddNode(TypeGeneral::LocalSize(NUM_SURFACES + 1), TypeGeneral::NodeData(&data_cost[0]));
//   }

//   for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
//     vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH, IMAGE_HEIGHT);
//     for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
//       if (*neighbor_pixel_it < pixel)
// 	continue;
//       vector<double> pairwise_cost((NUM_SURFACES + 1) * (NUM_SURFACES + 1), 0);
//       for (int surface_id_1 = 0; surface_id_1 < NUM_SURFACES + 1; surface_id_1++) {
// 	for (int surface_id_2 = 0; surface_id_2 < NUM_SURFACES + 1; surface_id_2++) {
// 	  if (surface_id_2 == surface_id_1)
// 	    continue;
// 	  if (surface_id_1 < NUM_SURFACES && surface_id_2 < NUM_SURFACES) {
// 	    double depth_1_1 = segments.at(surface_id_1).getDepth(pixel);
// 	    double depth_1_2 = segments.at(surface_id_1).getDepth(*neighbor_pixel_it);
// 	    double depth_2_1 = segments.at(surface_id_2).getDepth(pixel);
// 	    double depth_2_2 = segments.at(surface_id_2).getDepth(*neighbor_pixel_it);

// 	    double diff_1 = abs(depth_1_1 - depth_2_1);
// 	    double diff_2 = abs(depth_1_2 - depth_2_2);
// 	    double diff_middle = (depth_1_1 - depth_2_1) * (depth_1_2 - depth_2_2) <= 0 ? 0 : 1000000;
// 	    double min_diff = min(min(diff_1, diff_2), diff_middle);
// 	    pairwise_cost[surface_id_1 + surface_id_2 * (NUM_SURFACES + 1)] = max(min(min_diff / statistics.depth_change_smoothness_threshold / penalties.max_depth_change_ratio, 1.0) * penalties.smoothness_pen, penalties.smoothness_small_constant_pen);
//           } else {
// 	    pairwise_cost[surface_id_1 + surface_id_2 * (NUM_SURFACES + 1)] = penalties.smoothness_empty_non_empty_ratio * penalties.smoothness_pen;
// 	  }
// 	}
//       }
//       energy->AddEdge(nodes[pixel], nodes[*neighbor_pixel_it], TypeGeneral::EdgeData(TypeGeneral::GENERAL, &pairwise_cost[0]));
//     }
//   }

//   MRFEnergy<TypeGeneral>::Options options;
//   options.m_iterMax = 1000;
//   options.m_printIter = 200;
//   options.m_printMinIter = 100;
//   options.m_eps = 0.001;

//   double lower_bound, lowest_energy;
//   energy->Minimize_TRW_S(options, lower_bound, lowest_energy);

//   vector<int> surface_ids(NUM_PIXELS);
//   for (int pixel = 0; pixel < NUM_PIXELS; pixel++)
//     surface_ids[pixel] = energy->GetSolution(nodes[pixel]);

//   return surface_ids;
// }


vector<vector<set<int> > > fillLayers(const Mat &image, const Mat &image_Lab, const std::vector<double> &point_cloud, const std::vector<double> &normals, const std::map<int, Segment> &segments, const RepresenterPenalties &PENALTIES, const DataStatistics STATISTICS, const int NUM_LAYERS, const std::vector<int> &current_solution_labels, const int current_solution_num_surfaces, const map<int, map<int, bool> > &segment_layer_certainty_map, const bool USE_PANORAMA, const bool TOLERATE_CONFLICTS, const bool APPLY_EROSION_AND_DILATION, const string &image_name, const vector<vector<set<int> > > &additional_segment_indices_map, const bool additional_segments_activity, const set<int> &erosion_layers)
{
  //segments.at(11).checkPixelFitting(image, point_cloud, normals, 28630);
  //exit(1);
  const int IMAGE_WIDTH = image.cols;
  const int IMAGE_HEIGHT = image.rows;
  const int NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;
  const int NUM_SURFACES = segments.size();
  const int NUM_EROSION_ITERATIONS = 2;
  const int NUM_DILATION_ITERATIONS = 1;
  
  vector<vector<set<int> > > layer_pixel_segment_indices_map(NUM_LAYERS, vector<set<int> >(NUM_PIXELS));
  vector<vector<bool> > layer_active_pixel_masks(NUM_LAYERS, vector<bool>(NUM_PIXELS, false));
  vector<int> certain_visible_layer_indices(NUM_PIXELS, NUM_LAYERS);
  vector<int> visible_layer_indices(NUM_PIXELS, NUM_LAYERS);
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    int current_solution_label = current_solution_labels[pixel];
    bool is_visible = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces + 1, NUM_LAYERS - 1 - layer_index)) % (current_solution_num_surfaces + 1);
      if (surface_id == current_solution_num_surfaces)
	continue;
      for (map<int, bool>::const_iterator layer_it = segment_layer_certainty_map.at(surface_id).begin(); layer_it != segment_layer_certainty_map.at(surface_id).end(); layer_it++) {
	layer_pixel_segment_indices_map[layer_it->first][pixel].insert(surface_id);
	if (layer_it->second == true) {
	  //layer_active_pixel_masks[layer_it->first][pixel] = false;
	  if (is_visible)
	    certain_visible_layer_indices[pixel] = min(layer_it->first, certain_visible_layer_indices[pixel]);
        } else
	  layer_active_pixel_masks[layer_it->first][pixel] = true;
	
	layer_pixel_segment_indices_map[layer_it->first][pixel].insert(surface_id);
	
	if (segment_layer_certainty_map.at(surface_id).count(layer_index) == 0) {
	  vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH, IMAGE_HEIGHT, USE_PANORAMA);
	  for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++)
	    if (current_solution_labels[*neighbor_pixel_it] / static_cast<int>(pow(current_solution_num_surfaces + 1, NUM_LAYERS - 1 - layer_index)) % (current_solution_num_surfaces + 1) != surface_id)
	      layer_active_pixel_masks[layer_index][*neighbor_pixel_it] = true;
	}
      }
      is_visible = false;
    }
  }

  if (additional_segment_indices_map.size() > 0) {
    for (int layer_index = 0; layer_index < NUM_LAYERS; layer_index++) {
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
	if (additional_segment_indices_map[layer_index][pixel].size() > 0) {
	  layer_pixel_segment_indices_map[layer_index][pixel].insert(additional_segment_indices_map[layer_index][pixel].begin(), additional_segment_indices_map[layer_index][pixel].end());
	  layer_active_pixel_masks[layer_index][pixel] = additional_segments_activity;
	}
      }
    }
  }

  map<int, map<int, bool> > complete_segment_layer_certainty_map = segment_layer_certainty_map;
  vector<vector<double> > layer_min_depths(NUM_LAYERS, vector<double>(NUM_PIXELS, 0));
  vector<vector<double> > layer_max_depths(NUM_LAYERS, vector<double>(NUM_PIXELS, 1000000));
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    bool is_visible = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS; layer_index++) {
      if (layer_pixel_segment_indices_map[layer_index][pixel].size() == 1) {
	int segment_id = *layer_pixel_segment_indices_map[layer_index][pixel].begin();
        double depth = segments.at(*layer_pixel_segment_indices_map[layer_index][pixel].begin()).getDepth(pixel);
	for (int forward_layer_index = 0; forward_layer_index < layer_index; forward_layer_index++)
	  layer_max_depths[forward_layer_index][pixel] = min(depth, layer_max_depths[forward_layer_index][pixel]);
        for (int backward_layer_index = layer_index + 1; backward_layer_index < NUM_LAYERS; backward_layer_index++)
          layer_min_depths[backward_layer_index][pixel] = max(depth, layer_min_depths[backward_layer_index][pixel]);
	
	if (is_visible)
	  certain_visible_layer_indices[pixel] = min(layer_index, certain_visible_layer_indices[pixel]);
	
	//	complete_segment_layer_certainty_map[*layer_pixel_segment_indices_map[layer_index][pixel].begin()][layer_index] = true;
      }
      if (layer_pixel_segment_indices_map[layer_index][pixel].size() > 0)
	is_visible = false;
    }
  }
  
  vector<vector<bool> > surface_frontal_surface_mask(NUM_SURFACES, vector<bool>(NUM_SURFACES, false));
  vector<vector<bool> > surface_backward_surface_mask(NUM_SURFACES, vector<bool>(NUM_SURFACES, false));
  for (int layer_index = 0; layer_index < NUM_LAYERS; layer_index++) {
    for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
      for (set<int>::const_iterator segment_it = layer_pixel_segment_indices_map[layer_index][pixel].begin(); segment_it != layer_pixel_segment_indices_map[layer_index][pixel].end(); segment_it++) {
	double depth = segments.at(*segment_it).getDepth(pixel);
	if (depth <= 0) {
	  cout << "negative depth" << endl;
	  cout << pixel << '\t' << *segment_it << '\t' << depth << endl;
	  //exit(1);
	  continue;
	}
	for (int other_surface_id = 0; other_surface_id < NUM_SURFACES; other_surface_id++) {
	  if (other_surface_id == *segment_it)
	    continue;
	  double other_depth = segments.at(other_surface_id).getDepth(pixel);
	  if (other_depth <= 0)
	    continue;
	  if (other_depth < depth - STATISTICS.depth_change_smoothness_threshold)
	    surface_frontal_surface_mask[*segment_it][other_surface_id] = true;
	  if (other_depth > depth + STATISTICS.depth_change_smoothness_threshold)
	    surface_backward_surface_mask[*segment_it][other_surface_id] = true;
	}
      }
    }
  }
  for (int segment_id_1 = 0; segment_id_1 < NUM_SURFACES; segment_id_1++)
    for (int segment_id_2 = 0; segment_id_2 < NUM_SURFACES; segment_id_2++)
      if (surface_frontal_surface_mask[segment_id_1][segment_id_2] == false && surface_backward_surface_mask[segment_id_1][segment_id_2] == false)
	surface_frontal_surface_mask[segment_id_1][segment_id_2] = surface_backward_surface_mask[segment_id_1][segment_id_2] = true;
  //for (int segment_id_1 = 0; segment_id_1 < NUM_SURFACES; segment_id_1++)
  //   for (int segment_id_2 = 0; segment_id_2 < NUM_SURFACES; segment_id_2++)
  //     cout << segment_id_1 << '\t' << segment_id_2 << '\t' << surface_frontal_surface_mask[segment_id_1][segment_id_2] << '\t' << surface_backward_surface_mask[segment_id_1][segment_id_2] << endl;
  
  map<int, Vec3b> segment_color_table;
  for (int segment_id = 0; segment_id < NUM_SURFACES; segment_id++)
    segment_color_table[segment_id] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
  segment_color_table[NUM_SURFACES] = Vec3b(0, 0, 0);
  
  if (image_name.size() > 0) {
    for (int layer_index = 0; layer_index < NUM_LAYERS; layer_index++) {
      Mat layer_filling_image = Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
        Vec3b color(0, 0, 0);
        double segment_color_weight = 1.0 / layer_pixel_segment_indices_map[layer_index][pixel].size();
	for (set<int>::const_iterator segment_it = layer_pixel_segment_indices_map[layer_index][pixel].begin(); segment_it != layer_pixel_segment_indices_map[layer_index][pixel].end(); segment_it++) {
          color += segment_color_table[*segment_it] * segment_color_weight;
        }
        layer_filling_image.at<Vec3b>(pixel / IMAGE_WIDTH, pixel % IMAGE_WIDTH) = color;
      }
      imwrite("Test/" + image_name + "_" + to_string(layer_index) + "_initial.bmp", layer_filling_image);
    }
  }
  
  for (int layer_index = 0; layer_index < NUM_LAYERS; layer_index++) {
    vector<set<int> > pixel_segment_indices_map = layer_pixel_segment_indices_map[layer_index];
    vector<bool> active_pixel_mask = layer_active_pixel_masks[layer_index];
    
    for (int iteration = 0; iteration < NUM_DILATION_ITERATIONS; iteration++) {
      vector<set<int> > dilated_pixel_segment_indices_map = pixel_segment_indices_map;
      vector<bool> dilated_active_pixel_mask(NUM_PIXELS, false);
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
	// if (active_pixel_mask[pixel] == false)
	//   continue;
        vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH, IMAGE_HEIGHT, USE_PANORAMA);
	for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
          for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
            if (dilated_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
              continue;
            if (*segment_it == NUM_SURFACES || segments.at(*segment_it).getDepth(*neighbor_pixel_it) > 0) {
              dilated_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
	      if (active_pixel_mask[pixel])
		dilated_active_pixel_mask[*neighbor_pixel_it] = true;
	    }
          }
        }
      }
      pixel_segment_indices_map = dilated_pixel_segment_indices_map;
      active_pixel_mask = dilated_active_pixel_mask;
    }
    
    vector<bool> modified_pixel_mask(NUM_PIXELS, false);
    while (true) {
      bool has_change = false;
      vector<bool> new_active_pixel_mask(NUM_PIXELS, false);
      //cout << ImageMask(active_pixel_mask, IMAGE_WIDTH, IMAGE_HEIGHT).getNumPixels() << endl;
      
      vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
        if (active_pixel_mask[pixel] == false)
          continue;
	//if (layer_index == 3)
	modified_pixel_mask[pixel] = true;
        // if (pixel_segment_indices_map[pixel].size() == 0)
        //   continue;
        vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH, IMAGE_HEIGHT, USE_PANORAMA);
        for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	  for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
            if (*segment_it == NUM_SURFACES)
              continue;
	    if (new_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
	      continue;
	    // if (segments.at(*segment_it).getBehindRoomStructure())
	    //   continue;
	    // if (*segment_it != 12)
	    //   continue;
	    double depth = segments.at(*segment_it).getDepth(*neighbor_pixel_it);

	    if (false && layer_index == 1 && *segment_it == 12)
	      cout << *neighbor_pixel_it << '\t' << depth << '\t' << certain_visible_layer_indices[*neighbor_pixel_it] << '\t' << layer_min_depths[layer_index][*neighbor_pixel_it] << '\t' << layer_max_depths[layer_index][*neighbor_pixel_it] << endl;

	    // if (*segment_it == 12 && *neighbor_pixel_it == 28885)
	    //   cout << certain_visible_layer_indices[*neighbor_pixel_it] << '\t' << segments.at(*segment_it).checkPixelFitting(image_Lab, point_cloud, normals, *neighbor_pixel_it) << endl;
	    
            if (certain_visible_layer_indices[*neighbor_pixel_it] >= layer_index && certain_visible_layer_indices[*neighbor_pixel_it] < NUM_LAYERS) {
	      if (segments.at(*segment_it).checkPixelFitting(image_Lab, point_cloud, normals, *neighbor_pixel_it) == false && (checkPointValidity(getPoint(point_cloud, *neighbor_pixel_it)) || depth <= 0))
		continue;
	    } else {
	      double min_depth = layer_min_depths[layer_index][*neighbor_pixel_it];
	      double max_depth = layer_max_depths[layer_index][*neighbor_pixel_it];
	      if (depth < min_depth - STATISTICS.depth_conflict_threshold || depth > max_depth + STATISTICS.depth_conflict_threshold)
		continue;
	    }
	    
	    if (false)
	      cout << "1: " << *neighbor_pixel_it << endl;
	    
            if (TOLERATE_CONFLICTS == false) {
	      bool has_conflict = false;
	      for (set<int>::const_iterator neighbor_segment_it = pixel_segment_indices_map[*neighbor_pixel_it].begin(); neighbor_segment_it != pixel_segment_indices_map[*neighbor_pixel_it].end(); neighbor_segment_it++) {
		if (complete_segment_layer_certainty_map.count(*neighbor_segment_it) == 0 || complete_segment_layer_certainty_map.at(*neighbor_segment_it).count(layer_index) == 0 || complete_segment_layer_certainty_map.at(*neighbor_segment_it).at(layer_index) == false)
		  continue;
		double neighbor_depth = segments.at(*neighbor_segment_it).getDepth(*neighbor_pixel_it);
                if (surface_frontal_surface_mask[*segment_it][*neighbor_segment_it] == false && depth > neighbor_depth + STATISTICS.depth_conflict_threshold) {
		  //if (*neighbor_pixel_it == 29180)
		  //cout << *neighbor_segment_it << endl;
        	  has_conflict = true;
		  break;
		}
		if (surface_backward_surface_mask[*segment_it][*neighbor_segment_it] == false && depth < neighbor_depth - STATISTICS.depth_conflict_threshold) {
		  //if (*neighbor_pixel_it == 29180)                
		  //cout << *neighbor_segment_it << endl;
                  has_conflict = true;
                  break;
                }
	      }
	      if (has_conflict)
		continue;
	    }
	    
	    if (false)
	      cout << "2: " << *neighbor_pixel_it << endl;
	    
            new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
            has_change = true;
            new_active_pixel_mask[*neighbor_pixel_it] = true;
          }
        }
      }
      if (has_change == false)
        break;
      pixel_segment_indices_map = new_pixel_segment_indices_map;
      active_pixel_mask = new_active_pixel_mask;
    }
    // if (layer_index == 3) {
    //   imwrite("Test/modified_mask.bmp", ImageMask(modified_pixel_mask, IMAGE_WIDTH, IMAGE_HEIGHT).drawMaskImage());
    //   exit(1);
    // }
    
    if (APPLY_EROSION_AND_DILATION) {
      //erode thin parts
      for (int iteration = 0; iteration < NUM_EROSION_ITERATIONS; iteration++) {
	vector<set<int> > eroded_pixel_segment_indices_map = pixel_segment_indices_map;
	for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
	  if (modified_pixel_mask[pixel] == false)
	    continue;
	  vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH, IMAGE_HEIGHT, USE_PANORAMA);
	  for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
	    bool on_boundary = false;
	    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	      if (pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) == 0) {
		on_boundary = true;
		break;
	      }
	    }
	    if (on_boundary)
	      eroded_pixel_segment_indices_map[pixel].erase(*segment_it);
	  }
	}
	pixel_segment_indices_map = eroded_pixel_segment_indices_map;
      }
    }
    
    //add empty pixels
    if (layer_index < NUM_LAYERS - 1) {
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    	int current_solution_surface_id = current_solution_labels[pixel] / static_cast<int>(pow(current_solution_num_surfaces + 1, NUM_LAYERS - 1 - layer_index)) % (current_solution_num_surfaces + 1);
        if (pixel_segment_indices_map[pixel].size() == 0 || (pixel_segment_indices_map[pixel].count(current_solution_surface_id) == 0 && current_solution_surface_id < current_solution_num_surfaces)) {
    	  pixel_segment_indices_map[pixel].insert(NUM_SURFACES);
          modified_pixel_mask[pixel] = true;
    	}
      }
    }
    
    //add original pixels
    for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
      int current_solution_surface_id = current_solution_labels[pixel] / static_cast<int>(pow(current_solution_num_surfaces + 1, NUM_LAYERS - 1 - layer_index)) % (current_solution_num_surfaces + 1);
      if (current_solution_surface_id < current_solution_num_surfaces) {
    	pixel_segment_indices_map[pixel].insert(current_solution_surface_id);
    	// vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH, IMAGE_HEIGHT, USE_PANORAMA);
        // for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++)
    	//   pixel_segment_indices_map[*neighbor_pixel_it].insert(current_solution_surface_id);
      }
    }
    
    
    if (APPLY_EROSION_AND_DILATION) {
      for (int iteration = 0; iteration < NUM_EROSION_ITERATIONS + NUM_DILATION_ITERATIONS; iteration++) {
	vector<set<int> > dilated_pixel_segment_indices_map = pixel_segment_indices_map;
	for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
	  if (modified_pixel_mask[pixel] == false)
	    continue;
	  vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH, IMAGE_HEIGHT, USE_PANORAMA);
	  for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	    for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
	      if (dilated_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
		continue;
	      if (*segment_it == NUM_SURFACES || segments.at(*segment_it).getDepth(*neighbor_pixel_it) > 0) {
		dilated_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
		modified_pixel_mask[*neighbor_pixel_it] = true;
	      }
	    }
	  }
	}
	pixel_segment_indices_map = dilated_pixel_segment_indices_map;
      }
    } else {
      for (int iteration = 0; iteration < 2; iteration++) {
	break;
        vector<set<int> > dilated_pixel_segment_indices_map = pixel_segment_indices_map;
        for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
          vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH, IMAGE_HEIGHT, USE_PANORAMA);
          for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
            for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
              if (dilated_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
                continue;
              if (*segment_it == NUM_SURFACES || segments.at(*segment_it).getDepth(*neighbor_pixel_it) > 0) {
                dilated_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
                modified_pixel_mask[*neighbor_pixel_it] = true;
              }
            }
          }
        }
        pixel_segment_indices_map = dilated_pixel_segment_indices_map;
      }
    }


    if (erosion_layers.count(layer_index) > 0) {
      active_pixel_mask = modified_pixel_mask;
      while (true) {
        bool has_change = false;
        vector<bool> new_active_pixel_mask(NUM_PIXELS, false);
        for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
          if (active_pixel_mask[pixel] == false)
            continue;
          vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH, IMAGE_HEIGHT, USE_PANORAMA);
	  set<int> new_segments;         
	  for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
            if (*segment_it == NUM_SURFACES) {
	      new_segments.insert(*segment_it);
	      continue;
	    }
	    bool on_segment_boundary = false;
	    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	      if (pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) == 0) {
		on_segment_boundary = true;
		break;
	      }
	    }
	    if (on_segment_boundary == false) {
	      new_segments.insert(*segment_it);
              continue;
	    }
	    
            bool validity = false;
	    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
              for (set<int>::const_iterator neighbor_segment_it = pixel_segment_indices_map[*neighbor_pixel_it].begin(); neighbor_segment_it != pixel_segment_indices_map[*neighbor_pixel_it].end(); *neighbor_segment_it++) {
		if (*neighbor_segment_it == *segment_it)
		  continue;
		if (*neighbor_segment_it == NUM_SURFACES) {
		  validity = true;
		  break;
		}
		
		double segment_depth = segments.at(*segment_it).getDepth(pixel);
		double neighbor_segment_depth = segments.at(*neighbor_segment_it).getDepth(pixel);
		double segment_neighbor_depth = segments.at(*segment_it).getDepth(*neighbor_pixel_it);
		double neighbor_segment_neighbor_depth = segments.at(*neighbor_segment_it).getDepth(*neighbor_pixel_it);
		double depth_diff = segment_depth - neighbor_segment_depth;
		double neighbor_depth_diff = segment_neighbor_depth - neighbor_segment_neighbor_depth;
		if (abs(depth_diff) <= STATISTICS.depth_change_smoothness_threshold || abs(neighbor_depth_diff) <= STATISTICS.depth_change_smoothness_threshold || depth_diff * neighbor_depth_diff <= 0) {
		  validity = true;
		  break;
		}
              }
              if (validity)
		break;
	    }
	    if (validity) {
	      new_segments.insert(*segment_it);
              continue;
	    }

	    if (*segment_it == 0 && false) {
	      cout << pixel << endl;
	      for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
                for (set<int>::const_iterator neighbor_segment_it = pixel_segment_indices_map[*neighbor_pixel_it].begin(); neighbor_segment_it != pixel_segment_indices_map[*neighbor_pixel_it].end(); *neighbor_segment_it++) {
                  if (*neighbor_segment_it == *segment_it)
                    continue;
                  if (*neighbor_segment_it == NUM_SURFACES) {
                    break;
                  }
                  double segment_depth = segments.at(*segment_it).getDepth(pixel);
                  double neighbor_segment_depth = segments.at(*neighbor_segment_it).getDepth(pixel);
                  double segment_neighbor_depth = segments.at(*segment_it).getDepth(*neighbor_pixel_it);
                  double neighbor_segment_neighbor_depth = segments.at(*neighbor_segment_it).getDepth(*neighbor_pixel_it);
                  double depth_diff = segment_depth - neighbor_segment_depth;
                  double neighbor_depth_diff = segment_neighbor_depth - neighbor_segment_neighbor_depth;
		  cout << *neighbor_segment_it << '\t' << depth_diff << '\t' << neighbor_depth_diff << endl;
		}
	      }
	    }
	    
	    new_active_pixel_mask[pixel] = true;
	    has_change = true;
	  }
	  pixel_segment_indices_map[pixel] = new_segments;
	}
        if (has_change == false)
          break;
        active_pixel_mask = new_active_pixel_mask;
      }
    }
      
    layer_pixel_segment_indices_map[layer_index] = pixel_segment_indices_map;
  }
  
  // cout << "done" << endl;
  // for (set<int>::const_iterator segment_it = layer_pixel_segment_indices_map[3][25092].begin(); segment_it != layer_pixel_segment_indices_map[3][25092].end(); segment_it++)
  //   cout << "25092: " << *segment_it << endl;
  // for (set<int>::const_iterator segment_it = layer_pixel_segment_indices_map[3][22701].begin(); segment_it != layer_pixel_segment_indices_map[3][22701].end(); segment_it++)
  //   cout << "22701: " << *segment_it << endl;
  //for (set<int>::const_iterator segment_it = layer_pixel_segment_indices_map[1][28330].begin(); segment_it != layer_pixel_segment_indices_map[1][28330].end(); segment_it++)
  //cout << "28330: " << *segment_it << endl;
  //exit(1);
  
  if (image_name.size()) {
    for (int layer_index = 0; layer_index < NUM_LAYERS; layer_index++) {
      Mat layer_filling_image = Mat::zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
        Vec3b color(0, 0, 0);
        double segment_color_weight = 1.0 / layer_pixel_segment_indices_map[layer_index][pixel].size();
	for (set<int>::const_iterator segment_it = layer_pixel_segment_indices_map[layer_index][pixel].begin(); segment_it != layer_pixel_segment_indices_map[layer_index][pixel].end(); segment_it++) {
          color += segment_color_table[*segment_it] * segment_color_weight;
        }
        layer_filling_image.at<Vec3b>(pixel / IMAGE_WIDTH, pixel % IMAGE_WIDTH) = color;
      }
      imwrite("Test/" + image_name + "_" + to_string(layer_index) + ".bmp", layer_filling_image);
    }
  }        
	
  return layer_pixel_segment_indices_map;
}

map<int, map<int, bool> > swapLayers(const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const std::map<int, Segment> &segments, const vector<int> &current_solution_labels, const int NUM_LAYERS, const DataStatistics STATISTICS, const bool USE_PANORAMA, const bool CONSIDER_ROOM_STRUCTURE_LAYER, const bool use_disconnnected_neighbors, const set<int> &invalid_segments)
{
  const int NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;
  const int NUM_SURFACES = segments.size();
  
  vector<ImageMask> segment_masks(NUM_SURFACES, ImageMask(false, IMAGE_WIDTH, IMAGE_HEIGHT));
  vector<set<int> > segment_layers(NUM_SURFACES);
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    int current_solution_label = current_solution_labels[pixel];
    bool is_visible = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(NUM_SURFACES + 1, NUM_LAYERS - 1 - layer_index)) % (NUM_SURFACES + 1);
      if (surface_id < NUM_SURFACES) {
	segment_masks[surface_id].set(pixel, true);
	
	segment_layers[surface_id].insert(layer_index);
      }
    }
  }
  set<int> room_structure_segments;
  for (int segment_id = 0; segment_id < NUM_SURFACES; segment_id++)
    if (segment_layers[segment_id].size() == 1 && segment_layers[segment_id].count(NUM_LAYERS - 1) > 0)
      room_structure_segments.insert(segment_id);
  
  
  vector<vector<int> > surface_frontal_surface_counter(NUM_SURFACES, vector<int>(NUM_SURFACES, 0));
  vector<vector<int> > surface_backward_surface_counter(NUM_SURFACES, vector<int>(NUM_SURFACES, 0));
  vector<vector<int> > surface_overlap_pixel_counter(NUM_SURFACES, vector<int>(NUM_SURFACES, 0));
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    int current_solution_label = current_solution_labels[pixel];
    for (int layer_index = 0; layer_index < NUM_LAYERS; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(NUM_SURFACES + 1, NUM_LAYERS - 1 - layer_index)) % (NUM_SURFACES + 1);
      if (surface_id < NUM_SURFACES) {
	double depth = segments.at(surface_id).getDepth(pixel);
        if (depth <= 0) {
          cout << "negative depth" << endl;
          cout << pixel << '\t' << surface_id << '\t' << depth << endl;
          //exit(1);
          continue;
        }
        for (int other_surface_id = 0; other_surface_id < NUM_SURFACES; other_surface_id++) {
          if (other_surface_id == surface_id)
            continue;
          double other_depth = segments.at(other_surface_id).getDepth(pixel);
          if (other_depth <= 0)
            continue;
          if (other_depth < depth - STATISTICS.depth_change_smoothness_threshold) {
            surface_frontal_surface_counter[surface_id][other_surface_id]++;
	    surface_overlap_pixel_counter[surface_id][other_surface_id]++;
	  }
          if (other_depth > depth + STATISTICS.depth_change_smoothness_threshold) {
            surface_backward_surface_counter[surface_id][other_surface_id]++;
	    surface_overlap_pixel_counter[surface_id][other_surface_id]++;
	  }
        }
	break;
      }
    }
  }

  if (false)
    for (int segment_id_1 = 0; segment_id_1 < NUM_SURFACES; segment_id_1++)
      for (int segment_id_2 = 0; segment_id_2 < NUM_SURFACES; segment_id_2++)
	cout << segment_id_1 << '\t' << segment_id_2 << '\t' << surface_frontal_surface_counter[segment_id_1][segment_id_2] << '\t' << surface_backward_surface_counter[segment_id_1][segment_id_2] << '\t' << surface_overlap_pixel_counter[segment_id_1][segment_id_2] << endl;
  
  
  for (int segment_id = 0; segment_id < NUM_SURFACES; segment_id++)
    segment_masks[segment_id].dilate();
  
  map<int, set<int> > segment_neighbor_segments;
  
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    for (int segment_id_1 = 0; segment_id_1 < NUM_SURFACES; segment_id_1++) {
      //if ((CONSIDER_ROOM_STRUCTURE_LAYER == false && room_structure_segments.count(segment_id_1) > 0))
      //continue;
      for (int segment_id_2 = segment_id_1 + 1; segment_id_2 < NUM_SURFACES; segment_id_2++) {
        //if ((CONSIDER_ROOM_STRUCTURE_LAYER == false && room_structure_segments.count(segment_id_2) > 0))
	//continue;
	if (CONSIDER_ROOM_STRUCTURE_LAYER == true && room_structure_segments.count(segment_id_1) == 0 && room_structure_segments.count(segment_id_2) == 0)
	  continue;
	if (segment_masks[segment_id_1].at(pixel) && segment_masks[segment_id_2].at(pixel)) {
	  segment_neighbor_segments[segment_id_1].insert(segment_id_2);
	  segment_neighbor_segments[segment_id_2].insert(segment_id_1);
	}
      }
    }
  }
  
  if (CONSIDER_ROOM_STRUCTURE_LAYER) {
    for (set<int>::const_iterator segment_it = room_structure_segments.begin(); segment_it != room_structure_segments.end(); segment_it++) {
      for (set<int>::const_iterator other_segment_it = room_structure_segments.begin(); other_segment_it != room_structure_segments.end(); other_segment_it++) {
        if (*other_segment_it != *segment_it) {
  	  segment_neighbor_segments[*segment_it].insert(*other_segment_it);
  	  segment_neighbor_segments[*other_segment_it].insert(*segment_it);
  	}
      }
    }
  }
	  
  const double BACKWARD_RATIO_REWARD_WEIGHT = CONSIDER_ROOM_STRUCTURE_LAYER ? 10 : 10;
  const double SAME_LAYER_REWARD_WEIGHT = BACKWARD_RATIO_REWARD_WEIGHT * 0.02;
  const double RANDOM_GUESS_WEIGHT = 1;
  const double LARGE_PEN = 100;
  
  
  typedef opengm::GraphicalModel<float, opengm::Adder> Model;
  size_t *label_nums = new size_t[NUM_SURFACES];
  for (int segment_id = 0; segment_id < NUM_SURFACES; segment_id++)
    label_nums[segment_id] = NUM_LAYERS;
  Model gm(opengm::DiscreteSpace<>(label_nums, label_nums + NUM_SURFACES));
  
  typedef opengm::ExplicitFunction<float> ExplicitFunction;
  typedef Model::FunctionIdentifier FunctionIdentifier;
  
  srand(time(0));
  for (int segment_id = 0; segment_id < NUM_SURFACES; segment_id++) {
    const size_t shape[] = {NUM_LAYERS};
    ExplicitFunction f(shape, shape + 1);
    if (false) {
      for (int label = 0; label < NUM_LAYERS; label++)
	f(label) = static_cast<float>(randomProbability() * RANDOM_GUESS_WEIGHT);
    } else if (CONSIDER_ROOM_STRUCTURE_LAYER == false) {
      if (room_structure_segments.count(segment_id) == 0) {
	for (int label = 0; label < NUM_LAYERS - 1; label++)
	  f(label) = static_cast<float>(randomProbability() * RANDOM_GUESS_WEIGHT);
	f(NUM_LAYERS - 1) = 0;
        // for (int label = 0; label < NUM_LAYERS - 1; label++)
	//   if (segment_layers[segment_id].count(label) > 0)
	//     f(label) = 0;
        //   else
	//     f(label) = static_cast<float>(randomProbability() * RANDOM_GUESS_WEIGHT);
        // f(NUM_LAYERS - 1) = LARGE_PEN;
      } else
	for (int label = 0; label < NUM_LAYERS - 1; label++)
	  f(label) = LARGE_PEN;
    } else {
      for (int label = 0; label < NUM_LAYERS - 1; label++)
	if (segment_layers[segment_id].count(label) > 0)
          f(label) = 0;
	else
	  f(label) = static_cast<float>(randomProbability() * RANDOM_GUESS_WEIGHT);
      if (room_structure_segments.count(segment_id) == 0)
	f(NUM_LAYERS - 1) = LARGE_PEN;
      else
	f(NUM_LAYERS - 1) = 0;
    }

    if (true)
      for (int label = 0; label < NUM_LAYERS; label++)
	cout << segment_id << '\t' << f(label) << endl;
    
    FunctionIdentifier id = gm.addFunction(f);
    size_t variable_index[] = {segment_id};
    gm.addFactor(id, variable_index, variable_index + 1);
  }
  
  for (map<int, set<int> >::const_iterator segment_it = segment_neighbor_segments.begin(); segment_it != segment_neighbor_segments.end(); segment_it++) {
    for (set<int>::const_iterator neighbor_segment_it = segment_it->second.begin(); neighbor_segment_it != segment_it->second.end(); neighbor_segment_it++) {
      //cout << segment_it->first << '\t' << *neighbor_segment_it << endl;
      if (segment_it->first >= *neighbor_segment_it)
        continue;
      int segment_id_1 = segment_it->first;
      int segment_id_2 = *neighbor_segment_it;
      
      const size_t shape[] = {
        NUM_LAYERS,
        NUM_LAYERS
      };
      ExplicitFunction f(shape, shape + 2);

      if (surface_overlap_pixel_counter[segment_id_1][segment_id_2] == 0 || surface_overlap_pixel_counter[segment_id_2][segment_id_1] == 0)
	continue;
      
      double backward_ratio_1 = 1.0 * surface_backward_surface_counter[segment_id_1][segment_id_2] / surface_overlap_pixel_counter[segment_id_1][segment_id_2];
      double backward_ratio_2 = 1.0 * surface_backward_surface_counter[segment_id_2][segment_id_1] / surface_overlap_pixel_counter[segment_id_2][segment_id_1];
      
      if (invalid_segments.count(segment_id_2) > 0)
        backward_ratio_1 = 0;
      if (invalid_segments.count(segment_id_1) > 0)
	backward_ratio_2 = 0;
      
      for (int layer_index_1 = 0; layer_index_1 < NUM_LAYERS; layer_index_1++)
        for (int layer_index_2 = 0; layer_index_2 < NUM_LAYERS; layer_index_2++)
          if (layer_index_1 < layer_index_2)
	    f(layer_index_1, layer_index_2) = static_cast<float>(-(backward_ratio_1 - backward_ratio_2) / 2 * BACKWARD_RATIO_REWARD_WEIGHT) + static_cast<float>(SAME_LAYER_REWARD_WEIGHT);
          else if (layer_index_1 > layer_index_2)
	    f(layer_index_1, layer_index_2) = static_cast<float>(-(backward_ratio_2 - backward_ratio_1) / 2 * BACKWARD_RATIO_REWARD_WEIGHT) + static_cast<float>(SAME_LAYER_REWARD_WEIGHT);

      if (true)
	cout << segment_id_1 << '\t' << segment_id_2 << '\t' << backward_ratio_1 << '\t' << backward_ratio_2 << endl;
      
      if ((segment_id_1 == 3 || segment_id_2 == 3) && true) {
      	cout << segment_id_1 << '\t' << segment_id_2 << endl;
      	for (int layer_index_1 = 0; layer_index_1 < NUM_LAYERS; layer_index_1++) {
      	  for (int layer_index_2 = 0; layer_index_2 < NUM_LAYERS; layer_index_2++)
      	    cout << f(layer_index_1, layer_index_2) << '\t';
      	  cout << endl;
      	}
      	//exit(1);
      }
      FunctionIdentifier id = gm.addFunction(f);
      size_t variable_indices[] = {segment_id_1, segment_id_2};
      gm.addFactor(id, variable_indices, variable_indices + 2);
    }
  }
  
  
  vector<size_t> labels(NUM_SURFACES);
  if (false) {
    opengm::TRWSi_Parameter<Model> parameter(30);
    opengm::TRWSi<Model, opengm::Minimizer> solver(gm, parameter);
    opengm::TRWSi<Model, opengm::Minimizer>::VerboseVisitorType verbose_visitor;
    solver.infer(verbose_visitor);
    solver.arg(labels);
    cout << "energy: "<< solver.value() << " lower bound: " << solver.bound() << endl;
  } else if (false) {
    typedef opengm::ICM<Model, opengm::Minimizer> IcmType;
    typedef IcmType::VerboseVisitorType VerboseVisitorType;
    IcmType solver(gm);
    VerboseVisitorType verbose_visitor;
    solver.infer(verbose_visitor);
    solver.arg(labels);
    cout << "energy: "<< solver.value() << " lower bound: " << solver.bound() << endl;
  } else if (true) {
    typedef opengm::AStar<Model, opengm::Minimizer> AStar;
    AStar::Parameter parameters;
    parameters.heuristic_ =  parameters.FASTHEURISTIC;
    AStar solver(gm);
    solver.infer();
    solver.arg(labels);
    std::cout << "value: " << solver.value() << " lower bound: " << solver.bound() << std::endl;
  }
  
  map<int, map<int, bool> > segment_layer_certainty_map;
  for (int segment_id = 0; segment_id < NUM_SURFACES; segment_id++) {
    if (segment_layers[segment_id].count(labels[segment_id]) > 0)
      segment_layer_certainty_map[segment_id][*segment_layers[segment_id].begin()] = true;
    else if (CONSIDER_ROOM_STRUCTURE_LAYER && labels[segment_id] < NUM_LAYERS - 1)
      segment_layer_certainty_map[segment_id][labels[segment_id]] = true;
    else
      segment_layer_certainty_map[segment_id][labels[segment_id]] = false;
  }
  return segment_layer_certainty_map;
}

map<int, map<int, bool> > calcNewSegmentLayers(const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const std::map<int, Segment> &segments, const vector<int> &current_solution_labels, const int current_solution_num_surfaces, const map<int, ImageMask> &new_segment_masks, const int NUM_LAYERS, const DataStatistics STATISTICS, const bool USE_PANORAMA)
{
  const int NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;
  const int NUM_SURFACES = segments.size();
  
  vector<ImageMask> segment_masks(NUM_SURFACES, ImageMask(false, IMAGE_WIDTH, IMAGE_HEIGHT));
  vector<set<int> > segment_layers(current_solution_num_surfaces);
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    int current_solution_label = current_solution_labels[pixel];
    bool is_visible = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces + 1, NUM_LAYERS - 1 - layer_index)) % (current_solution_num_surfaces + 1);
      if (surface_id < current_solution_num_surfaces) {
        segment_masks[surface_id].set(pixel, true);
	
        segment_layers[surface_id].insert(layer_index);
      }
    }
  }
  
  vector<vector<int> > surface_frontal_surface_counter(NUM_SURFACES, vector<int>(NUM_SURFACES, 0));
  vector<vector<int> > surface_backward_surface_counter(NUM_SURFACES, vector<int>(NUM_SURFACES, 0));
  vector<vector<int> > surface_overlap_pixel_counter(NUM_SURFACES, vector<int>(NUM_SURFACES, 0));
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    int current_solution_label = current_solution_labels[pixel];
    for (int layer_index = 0; layer_index < NUM_LAYERS; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces + 1, NUM_LAYERS - 1 - layer_index)) % (current_solution_num_surfaces + 1);
      if (surface_id < current_solution_num_surfaces) {
        double depth = segments.at(surface_id).getDepth(pixel);
        if (depth <= 0) {
          cout << "negative depth" << endl;
          cout << pixel << '\t' << surface_id << '\t' << depth << endl;
          //exit(1);
          continue;
        }
        for (int other_surface_id = 0; other_surface_id < NUM_SURFACES; other_surface_id++) {
          if (other_surface_id == surface_id)
            continue;
          double other_depth = segments.at(other_surface_id).getDepth(pixel);
          if (other_depth <= 0)
            continue;
          if (other_depth < depth - STATISTICS.depth_change_smoothness_threshold)
            surface_frontal_surface_counter[surface_id][other_surface_id]++;
          if (other_depth > depth + STATISTICS.depth_change_smoothness_threshold)
            surface_backward_surface_counter[surface_id][other_surface_id]++;
          surface_overlap_pixel_counter[surface_id][other_surface_id]++;
        }
	break;
      }
    }
  }
  
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    for (map<int, ImageMask>::const_iterator segment_it = new_segment_masks.begin(); segment_it != new_segment_masks.end(); segment_it++) {
      if (segment_it->second.at(pixel) == true) {
	int surface_id = segment_it->first;
	double depth = segments.at(surface_id).getDepth(pixel);
        if (depth <= 0) {
          cout << "negative depth" << endl;
          cout << pixel << '\t' << surface_id << '\t' << depth << endl;
          //exit(1);
          continue;
        }
        for (int other_surface_id = 0; other_surface_id < NUM_SURFACES; other_surface_id++) {
          if (other_surface_id == surface_id)
            continue;
          double other_depth = segments.at(other_surface_id).getDepth(pixel);
          if (other_depth <= 0)
            continue;
          if (other_depth < depth - STATISTICS.depth_change_smoothness_threshold)
            surface_frontal_surface_counter[surface_id][other_surface_id]++;
          if (other_depth > depth + STATISTICS.depth_change_smoothness_threshold)
            surface_backward_surface_counter[surface_id][other_surface_id]++;
          surface_overlap_pixel_counter[surface_id][other_surface_id]++;
        }
      }
    }
  }
  
  // for (int segment_id_1 = 0; segment_id_1 < NUM_SURFACES; segment_id_1++)
  //   for (int segment_id_2 = 0; segment_id_2 < NUM_SURFACES; segment_id_2++)
  //     cout << segment_id_1 << '\t' << segment_id_2 << '\t' << surface_frontal_surface_counter[segment_id_1][segment_id_2] << '\t' << surface_backward_surface_counter[segment_id_1][segment_id_2] << endl;
  
  for (map<int, ImageMask>::const_iterator segment_it = new_segment_masks.begin(); segment_it != new_segment_masks.end(); segment_it++)
    segment_masks[segment_it->first] = segment_it->second;
  for (int segment_id = 0; segment_id < NUM_SURFACES; segment_id++)
    segment_masks[segment_id].dilate();
  
  map<int, set<int> > segment_neighbor_segments;
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    for (int segment_id_1 = 0; segment_id_1 < NUM_SURFACES; segment_id_1++) {
      for (int segment_id_2 = segment_id_1 + 1; segment_id_2 < NUM_SURFACES; segment_id_2++) {
        if (new_segment_masks.count(segment_id_1) == 0 && new_segment_masks.count(segment_id_2) == 0)
          continue;
        if (segment_masks[segment_id_1].at(pixel) && segment_masks[segment_id_2].at(pixel)) {
          segment_neighbor_segments[segment_id_1].insert(segment_id_2);
          segment_neighbor_segments[segment_id_2].insert(segment_id_1);
        }
      }
    }
  }
  
  const double BACKWARD_RATIO_REWARD_WEIGHT = 100;
  const double SAME_LAYER_REWARD_WEIGHT = BACKWARD_RATIO_REWARD_WEIGHT * 0.02;
  const double BACKWARD_PREFERENCE_WEIGHT = 1;
  const double LARGE_PEN = 100;
  
  typedef opengm::GraphicalModel<float, opengm::Adder> Model;
  size_t *label_nums = new size_t[NUM_SURFACES];
  for (int segment_id = 0; segment_id < NUM_SURFACES; segment_id++)
    label_nums[segment_id] = NUM_LAYERS;
  Model gm(opengm::DiscreteSpace<>(label_nums, label_nums + NUM_SURFACES));
  
  typedef opengm::ExplicitFunction<float> ExplicitFunction;
  typedef Model::FunctionIdentifier FunctionIdentifier;
  
  srand(time(0));
  for (int segment_id = 0; segment_id < NUM_SURFACES; segment_id++) {
    const size_t shape[] = {NUM_LAYERS};
    ExplicitFunction f(shape, shape + 1);
    if (new_segment_masks.count(segment_id) == 0) {
      for (int label = 0; label < NUM_LAYERS; label++)
	if (segment_layers[segment_id].count(label) > 0)
	  f(label) = 0;
	else
	  f(label) = LARGE_PEN;
    } else {
      for (int label = 0; label < NUM_LAYERS; label++)
        f(label) = static_cast<float>((NUM_LAYERS - 1 - label) * BACKWARD_PREFERENCE_WEIGHT);
      // for (int label = 0; label < NUM_LAYERS - 1; label++)
        //   f(label) = static_cast<float>(randomProbability() * RANDOM_GUESS_WEIGHT);
	// f(NUM_LAYERS - 1) = LARGE_PEN;
    }
    
    FunctionIdentifier id = gm.addFunction(f);
    size_t variable_index[] = {segment_id};
    gm.addFactor(id, variable_index, variable_index + 1);
  }
  
  for (map<int, set<int> >::const_iterator segment_it = segment_neighbor_segments.begin(); segment_it != segment_neighbor_segments.end(); segment_it++) {
    for (set<int>::const_iterator neighbor_segment_it = segment_it->second.begin(); neighbor_segment_it != segment_it->second.end(); neighbor_segment_it++) {
      //cout << segment_it->first << '\t' << *neighbor_segment_it << endl;
      if (segment_it->first >= *neighbor_segment_it)
        continue;
      int segment_id_1 = segment_it->first;
      int segment_id_2 = *neighbor_segment_it;
      
      const size_t shape[] = {
        NUM_LAYERS,
        NUM_LAYERS
      };
      ExplicitFunction f(shape, shape + 2);
      
      double backward_ratio_1 = 1.0 * surface_backward_surface_counter[segment_id_1][segment_id_2] / surface_overlap_pixel_counter[segment_id_1][segment_id_2];
      double backward_ratio_2 = 1.0 * surface_backward_surface_counter[segment_id_2][segment_id_1] / surface_overlap_pixel_counter[segment_id_2][segment_id_1];
      
      for (int layer_index_1 = 0; layer_index_1 < NUM_LAYERS; layer_index_1++)
        for (int layer_index_2 = 0; layer_index_2 < NUM_LAYERS; layer_index_2++)
          if (layer_index_1 < layer_index_2)
            f(layer_index_1, layer_index_2) = static_cast<float>(-(backward_ratio_1 - backward_ratio_2) / 2 * BACKWARD_RATIO_REWARD_WEIGHT) + static_cast<float>(SAME_LAYER_REWARD_WEIGHT);
          else if (layer_index_1 > layer_index_2)
            f(layer_index_1, layer_index_2) = static_cast<float>(-(backward_ratio_2 - backward_ratio_1) / 2 * BACKWARD_RATIO_REWARD_WEIGHT) + static_cast<float>(SAME_LAYER_REWARD_WEIGHT);
      
      if ((segment_id_1 == 14 || segment_id_2 == 14) && false) {
        for (int layer_index_1 = 0; layer_index_1 < NUM_LAYERS; layer_index_1++) {
          for (int layer_index_2 = 0; layer_index_2 < NUM_LAYERS; layer_index_2++)
            cout << f(layer_index_1, layer_index_2) << '\t';
          cout << endl;
        }
      }
      FunctionIdentifier id = gm.addFunction(f);
      size_t variable_indices[] = {segment_id_1, segment_id_2};
      gm.addFactor(id, variable_indices, variable_indices + 2);
    }
  }
  
  
  vector<size_t> labels(NUM_SURFACES);
  if (false) {
    opengm::TRWSi_Parameter<Model> parameter(30);
    opengm::TRWSi<Model, opengm::Minimizer> solver(gm, parameter);
    opengm::TRWSi<Model, opengm::Minimizer>::VerboseVisitorType verbose_visitor;
    solver.infer(verbose_visitor);
    solver.arg(labels);
    cout << "energy: "<< solver.value() << " lower bound: " << solver.bound() << endl;
  } else if (false) {
    typedef opengm::ICM<Model, opengm::Minimizer> IcmType;
    typedef IcmType::VerboseVisitorType VerboseVisitorType;
    IcmType solver(gm);
    VerboseVisitorType verbose_visitor;
    solver.infer(verbose_visitor);
    solver.arg(labels);
    cout << "energy: "<< solver.value() << " lower bound: " << solver.bound() << endl;
  } else if (true) {
    typedef opengm::AStar<Model, opengm::Minimizer> AStar;
    AStar::Parameter parameters;
    parameters.heuristic_ =  parameters.FASTHEURISTIC;
    AStar solver(gm);
    solver.infer();
    solver.arg(labels);
    std::cout << "value: " << solver.value() << " lower bound: " << solver.bound() << std::endl;
  }
  
  map<int, map<int, bool> > segment_layer_certainty_map;
  for (int segment_id = 0; segment_id < NUM_SURFACES; segment_id++) {
    segment_layer_certainty_map[segment_id][labels[segment_id]] = true;
    //cout << segment_id << '\t' << labels[segment_id] << endl;
  }
  return segment_layer_certainty_map;
}

set<int> findBehindRoomStructureSegments(const int IMAGE_WIDTH, const int IMAGE_HEIGHT, const std::map<int, Segment> &segments, const set<int> &new_segment_indices, const vector<set<int> > &pixel_segment_indices_map, const DataStatistics STATISTICS, const bool USE_PANORAMA)
{
  const int NUM_SURFACES = segments.size();
  const int NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;


  map<int, vector<int> > segment_pixels_vec;
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
      if (*segment_it == NUM_SURFACES) {
        cout << "has empty pixel in room structure layer" << endl;
        exit(1);
      }
      
      segment_pixels_vec[*segment_it].push_back(pixel);
    }
  }
  
  map<int, ImageMask> segment_masks;
  for (map<int, vector<int> >::const_iterator segment_it = segment_pixels_vec.begin(); segment_it != segment_pixels_vec.end(); segment_it++)
    segment_masks[segment_it->first] = ImageMask(segment_it->second, IMAGE_WIDTH, IMAGE_HEIGHT);

  set<int> behind_room_structure_segment_indices;
  const double COMMON_REGION_RATIO_THRESHOLD = 0.9;
  const double FRONTAL_PIXELS_RATIO_THRESHOLD = 0.05;
  
  for (map<int, ImageMask>::const_iterator segment_it = segment_masks.begin(); segment_it != segment_masks.end(); segment_it++) {
    ImageMask completed_segment_mask = completeContour(segment_it->second, ImageMask(false, IMAGE_WIDTH, IMAGE_HEIGHT), IMAGE_WIDTH, IMAGE_HEIGHT, USE_PANORAMA, 4, IMAGE_WIDTH / 10);

    bool check_mask_images = false;
    if (check_mask_images) {
      imwrite("Test/mask_image_" + to_string(segment_it->first) + ".bmp", segment_it->second.drawMaskImage());
      imwrite("Test/mask_image_" + to_string(segment_it->first) + "_completed.bmp", completed_segment_mask.drawMaskImage());
    }
    //exit(1);
    
    for (set<int>::const_iterator other_segment_it = new_segment_indices.begin(); other_segment_it != new_segment_indices.end(); other_segment_it++) {
      if (*other_segment_it == segment_it->first || segment_masks.count(*other_segment_it) == 0)
	continue;
      if (behind_room_structure_segment_indices.count(*other_segment_it) > 0)
	continue;

      ImageMask common_region_mask = segment_masks[*other_segment_it] - (segment_masks[*other_segment_it] - completed_segment_mask);
      vector<int> common_region_pixels = common_region_mask.getPixels();
      if (common_region_pixels.size() < segment_masks[*other_segment_it].getNumPixels() * COMMON_REGION_RATIO_THRESHOLD)
	continue;

      int num_frontal_pixels = 0;
      for (vector<int>::const_iterator pixel_it = common_region_pixels.begin(); pixel_it != common_region_pixels.end(); pixel_it++) {
	double depth = segments.at(segment_it->first).getDepth(*pixel_it);
	if (depth <= 0) {
	  num_frontal_pixels++;
	  continue;
	}
	double other_depth = segments.at(*other_segment_it).getDepth(*pixel_it);
	if (other_depth <= 0)
	  continue;
	if (other_depth < depth - STATISTICS.depth_change_smoothness_threshold)
	  num_frontal_pixels++;
      }
      if (num_frontal_pixels > segment_masks[*other_segment_it].getNumPixels() * FRONTAL_PIXELS_RATIO_THRESHOLD)
	continue;
      
      behind_room_structure_segment_indices.insert(*other_segment_it);
    }
  }
  
  // vector<vector<int> > surface_frontal_surface_counter(NUM_SURFACES, vector<int>(NUM_SURFACES, 0));
  // vector<vector<int> > surface_backward_surface_counter(NUM_SURFACES, vector<int>(NUM_SURFACES, 0));
  // vector<vector<int> > surface_overlap_pixel_counter(NUM_SURFACES, vector<int>(NUM_SURFACES, 0));
  // for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
  //   for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
  //     double depth = segments.at(*segment_it).getDepth(pixel);
  //     if (depth <= 0) {
  // 	cout << "negative depth" << endl;
  // 	cout << pixel << '\t' << *segment_it << '\t' << depth << endl;
  // 	//exit(1);
  // 	continue;
  //     }
  //     for (int other_surface_id = 0; other_surface_id < NUM_SURFACES; other_surface_id++) {
  // 	if (other_surface_id == *segment_it)
  // 	  continue;
  // 	double other_depth = segments.at(other_surface_id).getDepth(pixel);
  // 	if (other_depth <= 0)
  // 	  continue;
  // 	if (other_depth < depth - STATISTICS.depth_change_smoothness_threshold)
  // 	  surface_frontal_surface_counter[*segment_it][other_surface_id]++;
  // 	if (other_depth > depth + STATISTICS.depth_change_smoothness_threshold)
  // 	  surface_backward_surface_counter[*segment_it][other_surface_id]++;
  // 	surface_overlap_pixel_counter[*segment_it][other_surface_id]++;
  //     }
  //   }
  // }
  
  // map<int, Mat> segment_points;
  // for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
  //   for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
  //     //if (new_segment_indices.count(*segment_it) == 0)
  //     //continue;
  //     if (*segment_it == NUM_SURFACES) {
  // 	cout << "has empty pixel in room structure layer" << endl;
  // 	exit(1);
  //     }
      
  //     vector<double> segment_point = segments.at(*segment_it).getSegmentPoint(pixel);
  //     if (checkPointValidity(segment_point)) {
  //       Mat point(1, 3, CV_64FC1);
  // 	for (int c = 0; c < 3; c++)
  // 	  point.at<double>(0, c) = segment_point[c];
  // 	if (segment_points.count(*segment_it) == 0)
  //         segment_points[*segment_it] = point;
  // 	else
  // 	  segment_points[*segment_it].push_back(point);
  //     }
  //   }
  // }

  // const double IN_RANGE_THRESHOLD = 1.1;
  // const double BEHIND_PIXEL_RATIO = 0.1;
  
  // set<int> behind_room_structure_segment_indices;
  // for (map<int, Mat>::const_iterator segment_it = segment_points.begin(); segment_it != segment_points.end(); segment_it++) {
  //   if (segments.at(segment_it->first).getSegmentType() != 0)
  //     continue;
  //   PCA segment_pca_plane(segment_it->second, Mat(), CV_PCA_DATA_AS_ROW);
  //   Mat projected_points = segment_pca_plane.project(segment_it->second);
  //   vector<double> range(6);
  //   for (int c = 0; c < 3; c++) {
  //     range[c * 2 + 0] = 1000000;
  //     range[c * 2 + 1] = -1000000;
  //   }
  //   for (int point_index = 0; point_index < projected_points.rows; point_index++) {
  //     for (int c = 0; c < 3; c++) {
  // 	range[c * 2 + 0] = min(projected_points.at<double>(point_index, c), range[c * 2 + 0]);
  // 	range[c * 2 + 1] = max(projected_points.at<double>(point_index, c), range[c * 2 + 1]);
  //     }
  //   }
  //   for (int c = 0; c < 3; c++) {
  //     double mean = (range[c * 2 + 0] + range[c * 2 + 1]) / 2;
  //     double diff = range[c * 2 + 1] - range[c * 2 + 0];
  //     range[c * 2 + 0] = mean - diff * IN_RANGE_THRESHOLD / 2;
  //     range[c * 2 + 1] = mean + diff * IN_RANGE_THRESHOLD / 2;
  //   }
  //   //cout << segment_it->first << endl;
  //   //for (int c = 0; c < 3; c++)
  //   //cout << range[c * 2 + 0] << '\t' << range[c * 2 + 1] << endl;
    
  //   for (set<int>::const_iterator other_segment_it = new_segment_indices.begin(); other_segment_it != new_segment_indices.end(); other_segment_it++) {
  //     if (*other_segment_it == segment_it->first || segment_points.count(*other_segment_it) == 0)
  // 	continue;
  //     if (behind_room_structure_segment_indices.count(*other_segment_it) > 0)
  // 	continue;
  //     Mat other_projected_points = segment_pca_plane.project(segment_points[*other_segment_it]);
  //     bool in_range = true;
  //     for (int other_point_index = 0; other_point_index < other_projected_points.rows; other_point_index++) {
  //       for (int c = 0; c < 2; c++) {
  //         if (other_projected_points.at<double>(other_point_index, c) < range[c * 2 + 0]) {
  // 	    if (*other_segment_it == 1 && false) {
  // 	      cout << other_projected_points.row(other_point_index) << endl;
  // 	      exit(1);
  // 	    }
  // 	    in_range = false;
  // 	    break;
  // 	  }
  // 	  if (other_projected_points.at<double>(other_point_index, c) > range[c * 2 + 1]) {
  // 	    if (*other_segment_it == 1 && false) {
  //             cout << other_projected_points.row(other_point_index) << endl;
  //             exit(1);
  //           }
  //           in_range = false;
  //           break;
  //         }
  // 	}
  // 	if (in_range == false)
  // 	  break;
  //     }
      
  //     if (in_range == false)
  // 	continue;
      
  //     if (surface_frontal_surface_counter[*other_segment_it][segment_it->first] <= surface_overlap_pixel_counter[*other_segment_it][segment_it->first] * BEHIND_PIXEL_RATIO || surface_backward_surface_counter[*other_segment_it][segment_it->first] > 0)
  // 	continue;

  //     cout << segment_it->first << '\t' << *other_segment_it << endl;
  //     behind_room_structure_segment_indices.insert(*other_segment_it);
  //   }
  // }
  return behind_room_structure_segment_indices;
}
