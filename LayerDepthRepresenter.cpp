#include "LayerDepthRepresenter.h"
#include <iostream>
#include <fstream>
#include <ctime>

#include <Eigen/Sparse>

#include "cv_utils.h"
#include "TRWSFusion.h"
//#include "FusionSpaceLayerIndicator.h"
//#include "TRWSSolver.h"
//#include "ConcaveHullFinder.h"
//#include "SubRegionFinder.h"
//#include "GraphRepresenter.h"
//#include "BSpline.h"
//#include "LayerEstimator.h"
#include "ProposalDesigner.h"
#include "ProposalGenerator.h"
#include "ImageMask.h"

//#include "BinaryProposalDesigner.h"
//#include "PatchMatcher.h"

//#include "Superpixel.h"

//#include "SegmentationRefiner.h"
//#include "ContourCompleter.h"

//#include "PointCloudSegmenter.h"

//#include "OpenGMSolver.h"

//#include "LinearProgrammingSolver.h"

using namespace cv;
using namespace Eigen;
using namespace cv_utils;

LayerDepthRepresenter::LayerDepthRepresenter(const Mat &image, const vector<double> &point_cloud, const RepresenterPenalties &penalties, const DataStatistics &statistics, const int scene_index, const Mat &ori_image, const vector<double> &ori_point_cloud, const bool first_time, const int num_layers, const bool use_panorama) : image_(image), point_cloud_(point_cloud), IMAGE_WIDTH_(image.cols), IMAGE_HEIGHT_(image.rows), NUM_PIXELS_(IMAGE_WIDTH_ * IMAGE_HEIGHT_), PENALTIES_(penalties), STATISTICS_(statistics), SCENE_INDEX_(scene_index), ori_image_(ori_image), ori_point_cloud_(ori_point_cloud), FIRST_TIME_(first_time), num_layers_(num_layers), USE_PANORAMA_(use_panorama)
{
  //  surface_models_ = fitSurfaceModels(point_cloud, segmentation);
  //  assert(num_surfaces_ > 1);
  ROI_mask_ = vector<bool>(NUM_PIXELS_, true);
  
  //  segmentation_ = initial_segmentation_;
  
  sub_sampling_ratio_ = 1;
  //  focal_length_ = 50;
  if (true) {
    camera_parameters_.assign(3, 0);
    estimateCameraParameters(point_cloud_, IMAGE_WIDTH_, IMAGE_HEIGHT_, camera_parameters_, USE_PANORAMA_);
    cout << "camera parameters: " << camera_parameters_[0] << '\t' << camera_parameters_[1] << '\t' << camera_parameters_[2] << endl;
    //exit(1);
    //focal_length_ *= sub_sampling_ratio_;
    //cout << "focal length: " << focal_length_ << endl;
  }
  
  // cout << focal_length_ << endl;
  // exit(1);
  //exit(1);
  
  optimizeLayerRepresentation();
  
  //  initializeLabels();
  
  //  optimizeUsingLinearProgramming();
  
  //initializeLabels();
  //optimizeLayerRepresentation();
}

LayerDepthRepresenter::~LayerDepthRepresenter()
{
}

void LayerDepthRepresenter::optimizeLayerRepresentation()
{
  srand(time(NULL));
  
  //num_layers_ = min(estimateNumLayers(IMAGE_WIDTH_, IMAGE_HEIGHT_, initial_segmentation_, surface_depths_), 3);
  
  // PointCloudSegmenter segmenter(point_cloud_, image_, SCENE_INDEX_);
  // segmenter.getNormals();
  // segmenter.getCurvatures();
  // exit(1);
  
  // stringstream normal_image_filename;
  // normal_image_filename << "Cache/scene_" << SCENE_INDEX_ << "/normal_image.bmp";
  // Mat normal_image = imread(normal_image_filename.str());
  // normals_ = vector<double>(IMAGE_WIDTH_ * IMAGE_HEIGHT_ * 3);
  // for (int y = 0; y < IMAGE_HEIGHT_; y++) {
  //   for (int x = 0; x < IMAGE_WIDTH_; x++) {
  //     int pixel = y * IMAGE_WIDTH_ + x;
  //     Vec3b color = normal_image.at<Vec3b>(y, x);
  //     vector<double> normal(3);
  //     for (int c = 0; c < 3; c++)
  //       normal[c] = 1.0 * color[c] / 128 - 1;
  //     double norm = 0;
  //     for (int c = 0; c < 3; c++)
  //       norm += pow(normal[c], 2);
  //     norm = sqrt(norm);
  //     for (int c = 0; c < 3; c++)
  // 	normal[c] /= norm;
  
  //     for (int c = 0; c < 3; c++)
  // 	normals_[pixel * 3 + c] = normal[c];
  //   }
  // }
  // stringstream curvature_image_filename;
  // curvature_image_filename << "Cache/scene_" << SCENE_INDEX_ << "/curvature_image.bmp";
  // Mat curvature_image = imread(curvature_image_filename.str(), 0);
  // vector<double> curvatures(IMAGE_WIDTH_ * IMAGE_HEIGHT_);
  // for (int y = 0; y < IMAGE_HEIGHT_; y++) {
  //   for (int x = 0; x < IMAGE_WIDTH_; x++) {
  //     int pixel = y * IMAGE_WIDTH_ + x;
  //     uchar gray_value = curvature_image.at<uchar>(y, x);
  //     curvatures[pixel] = 1.0 * gray_value / 255;
  //   }
  // }
  
  // calcSuperpixels(image_, point_cloud_, normals);
  // exit(1);
  
  normals_ = calcNormals(point_cloud_, IMAGE_WIDTH_, IMAGE_HEIGHT_);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    if (calcDotProduct(getPoint(point_cloud_, pixel), getPoint(normals_, pixel)) > 0)
      for (int c = 0; c < 3; c++)
	normals_[pixel * 3 + c] = -normals_[pixel * 3 + c];
  }
  
  vector<double> pixel_weights_3D(NUM_PIXELS_, 1);
  vector<double> curvatures = cv_utils::calcCurvatures(point_cloud_, IMAGE_WIDTH_, IMAGE_HEIGHT_);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    pixel_weights_3D[pixel] = max(1 - curvatures[pixel] * STATISTICS_.pixel_weight_curvature_ratio, STATISTICS_.min_pixel_weight);
    if (cv_utils::checkPointValidity(cv_utils::getPoint(point_cloud_, pixel)) == false)
      pixel_weights_3D[pixel] = 0;
  }
  
  
  map<int, vector<double> > iteration_statistics_map;
  map<int, string> iteration_proposal_type_map;
  stringstream iteration_info_filename;
  iteration_info_filename << "Cache/scene_" << SCENE_INDEX_ << "/iteration_info.txt";
  ifstream iteration_info_in_str(iteration_info_filename.str());
  int num_single_surface_expansion_proposals = 0;
  double total_running_time = 0;
  if (iteration_info_in_str) {
    while (true) {
      int iteration;
      string proposal_type;
      vector<double> statistics(3);
      // double energy;
      // double lower_bound;
      // double time;
      iteration_info_in_str >> iteration >> proposal_type >> statistics[0] >> statistics[1] >> statistics[2];
      if (iteration_info_in_str.eof() == true)
        break;
      iteration_statistics_map[iteration] = statistics;
      iteration_proposal_type_map[iteration] = proposal_type;
      if (proposal_type == "single_surface_expansion_proposal")
	num_single_surface_expansion_proposals++;
      cout << iteration << '\t' << proposal_type << '\t' << statistics[0] << '\t' << statistics[1] << '\t' << statistics[2] << endl;
      total_running_time += statistics[2];
    }
    iteration_info_in_str.close();
  }
  cout << "total running time: " << total_running_time << endl;
  
  
  int iteration_start_index = 0;
  double previous_energy = -1;
  int previous_energy_iteration = -1;
  for (map<int, vector<double> >::const_iterator iteration_it = iteration_statistics_map.begin(); iteration_it != iteration_statistics_map.end(); iteration_it++) {
    iteration_start_index = iteration_it->first + 1;
    if (previous_energy < 0 || iteration_it->second[0] < previous_energy) {
      previous_energy_iteration = iteration_it->first;
      previous_energy = iteration_it->second[0];
    }
  }
  
  
  //STATISTICS_ = calcInputStatistics(previous_energy_iteration == -1);
  
  
  unique_ptr<ProposalDesigner> proposal_designer(new ProposalDesigner(image_, point_cloud_, normals_, pixel_weights_3D, camera_parameters_, num_layers_, PENALTIES_, STATISTICS_, SCENE_INDEX_, USE_PANORAMA_));
  //unique_ptr<ProposalGenerator> proposal_designer(new ProposalGenerator(image_, point_cloud_, normals_, pixel_weights_3D, camera_parameters_, num_layers_, PENALTIES_, STATISTICS_, SCENE_INDEX_, USE_PANORAMA_));
  
  unique_ptr<TRWSFusion> TRW_solver(new TRWSFusion(image_, point_cloud_, normals_, pixel_weights_3D, PENALTIES_, STATISTICS_));
  //unique_ptr<FusionSpaceLayerIndicator> TRW_solver(new FusionSpaceLayerIndicator(image_, point_cloud_, normals_, PENALTIES_, STATISTICS_));
  const int NUM_ITERATIONS = 12;
  
  //previous_energy_iteration = -1;
  
  // iteration_start_index = previous_energy_iteration + 1;
  //previous_energy_iteration = 19;
  int solution_num_surfaces = -1;
  if (previous_energy_iteration >= 0) {
    //for (int iteration = 0; iteration <= previous_energy_iteration; iteration++) {
      //vector<int> solution_labels;
      //int solution_num_surfaces;
      //map<int, Segment> solution_segments;
      //readLayers(IMAGE_WIDTH_, IMAGE_HEIGHT_, camera_parameters_, PENALTIES_, STATISTICS_, num_layers_, solution_labels, solution_num_surfaces, solution_segments, SCENE_INDEX_, iteration) == true;
      //writeLayers(image_, IMAGE_WIDTH_, IMAGE_HEIGHT_, point_cloud_, camera_parameters_, num_layers_, solution_labels, solution_num_surfaces, solution_segments, SCENE_INDEX_, iteration, ori_image_, ori_point_cloud_);
      
      //       readLayers(solution_labels, solution_num_surfaces, solution_segments, iteration);
      // stringstream disp_image_filename;
      // disp_image_filename << "Test/disp_image_" << iteration << ".bmp";
      // writeDispImageFromSegments(solution_labels, solution_num_surfaces, solution_segments, num_layers_, IMAGE_WIDTH_, IMAGE_HEIGHT_, disp_image_filename.str());
      //writeLayers(solution_labels, solution_num_surfaces, solution_segments, iteration);
      //exit(1);
    //}
    //exit(1);
    
    vector<int> previous_solution_labels;
    int previous_solution_num_surfaces;
    map<int, Segment> previous_solution_segments;

    cout << previous_energy_iteration << endl;
    bool read_success = readLayers(IMAGE_WIDTH_, IMAGE_HEIGHT_, camera_parameters_, PENALTIES_, STATISTICS_, num_layers_, previous_solution_labels, previous_solution_num_surfaces, previous_solution_segments, SCENE_INDEX_, previous_energy_iteration, USE_PANORAMA_);
    //bool read_success = readLayers(previous_solution_labels, previous_solution_num_surfaces, previous_solution_segments, previous_energy_iteration) == true;
    cout << "done" << endl;
    assert(read_success);
    stringstream disp_image_filename;
    disp_image_filename << "Test/previous_solution_disp_image.bmp";
    //writeDispImageFromSegments(previous_solution_labels, previous_solution_num_surfaces, previous_solution_segments, num_layers_, IMAGE_WIDTH_, IMAGE_HEIGHT_, disp_image_filename.str());
    
    writeLayers(image_, point_cloud_, camera_parameters_, num_layers_, previous_solution_labels, previous_solution_num_surfaces, previous_solution_segments, SCENE_INDEX_, 10000);
    
    //writeLayers(image_, point_cloud_, camera_parameters_, num_layers_, current_solution_labels, proposal_num_surfaces, proposal_segments, SCENE_INDEX_, iteration);
    // exit(1);
    
    proposal_designer->setCurrentSolution(previous_solution_labels, previous_solution_num_surfaces, previous_solution_segments);

    solution_num_surfaces = proposal_designer->getNumSurfaces();
  }
  
  // vector<int> test_solution_labels;
  // int test_solution_num_surfaces;
  // map<int, Segment> test_solution_segments;
  // readLayers(IMAGE_WIDTH_, IMAGE_HEIGHT_, camera_parameters_, PENALTIES_, STATISTICS_, num_layers_, test_solution_labels, test_solution_num_surfaces, test_solution_segments, SCENE_INDEX_, 5, USE_PANORAMA_);
  
  
  int best_solution_iteration = previous_energy_iteration;
  //cout << iteration_start_index << '\t' << NUM_ITERATIONS << endl;
  for (int iteration = iteration_start_index; iteration < NUM_ITERATIONS + num_single_surface_expansion_proposals; iteration++) {
    cout << "proposal: " << iteration << endl;
    iteration_start_index = iteration + 1;
    
    // {
    //   vector<int> solution_labels;
    //   int solution_num_surfaces;
    //   map<int, Segment> solution_segments;
    //   if (readLayers(solution_labels, solution_num_surfaces, solution_segments, iteration) == true) {
    // 	proposal_designer->setCurrentSolution(solution_labels, solution_num_surfaces, solution_segments);
    // 	continue;
    //   }
    // }
    
    const clock_t begin_time = clock();
    vector<vector<int> > proposal_labels;
    //    vector<int> proposal_segmentation;
    int proposal_num_surfaces;
    map<int, Segment> proposal_segments;
    vector<int> proposal_distance_to_boundaries;
    //if (previous_solution.size() > 0)
    //proposal_designer.setCurrentSolution(previous_solution, previous_solution_num_surfaces, previous_solution_surface_depths);
    string proposal_type;
    if (proposal_designer->getProposal(iteration, proposal_labels, proposal_num_surfaces, proposal_segments, proposal_type) == false)
      break;
    
    // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    //   if (find(proposal_labels[pixel].begin(), proposal_labels[pixel].end(), test_solution_labels[pixel]) == proposal_labels[pixel].end()) {
    // 	cout << pixel << endl;
    //   }
    //}
    
    if (proposal_type == "single_surface_expansion_proposal")
      num_single_surface_expansion_proposals++;
    
    vector<int> previous_solution_indices = proposal_designer->getCurrentSolutionIndices();
    vector<int> current_solution_labels = TRW_solver->fuse(proposal_labels, proposal_num_surfaces, num_layers_, proposal_segments, previous_solution_indices);
    //double current_solution_energy = TRW_solver->getEnergy();
    vector<double> energy_info = TRW_solver->getEnergyInfo();
    vector<double> statistics = energy_info;
    statistics.push_back(static_cast<double>(clock() - begin_time) / CLOCKS_PER_SEC);
    iteration_statistics_map[iteration] = statistics;
    iteration_proposal_type_map[iteration] = proposal_type;
    
    writeLayers(ori_image_, point_cloud_, camera_parameters_, num_layers_, current_solution_labels, proposal_num_surfaces, proposal_segments, SCENE_INDEX_, iteration);
    //exit(1);    
    
    if (energy_info[0] >= previous_energy && previous_energy >= 0) {
      cout << "energy increases" << endl;
      continue;
    }
    previous_energy = energy_info[0];
    
    int current_solution_num_surfaces = proposal_num_surfaces;
    map<int, Segment> current_solution_segments = proposal_segments;
    
    
    // cout << proposal_segments[25].getDepth(19647) << endl;
    // exit(1);
    
    // if (iteration == 6) {
    //   cout << current_solution_segments[25].getDepth(19647) << endl;
    //   //exit(1);
    // }
    proposal_designer->setCurrentSolution(current_solution_labels, current_solution_num_surfaces, current_solution_segments);
    
    
    ofstream iteration_info_out_str(iteration_info_filename.str());
    for (map<int, vector<double> >::const_iterator iteration_it = iteration_statistics_map.begin(); iteration_it != iteration_statistics_map.end(); iteration_it++) {
      iteration_info_out_str << iteration_it->first << '\t' << iteration_proposal_type_map[iteration_it->first] << '\t' << iteration_it->second[0] << '\t' << iteration_it->second[1] << '\t' << iteration_it->second[2] << endl;
    }
    iteration_info_out_str.close();
    
    best_solution_iteration = iteration;

    solution_num_surfaces = current_solution_num_surfaces;
  }

  for (int iteration = iteration_start_index; iteration < NUM_ITERATIONS + solution_num_surfaces; iteration++) {
    cout << "proposal: " << iteration << endl;
    
    const clock_t begin_time = clock();
    vector<vector<int> > proposal_labels;
    int proposal_num_surfaces;
    map<int, Segment> proposal_segments;
    vector<int> proposal_distance_to_boundaries;
    string proposal_type;
    if (proposal_designer->getRefinementProposal(iteration, proposal_labels, proposal_num_surfaces, proposal_segments, proposal_type) == false)
      break;
    
    vector<int> previous_solution_indices = proposal_designer->getCurrentSolutionIndices();
    vector<int> current_solution_labels = TRW_solver->fuse(proposal_labels, proposal_num_surfaces, num_layers_, proposal_segments, previous_solution_indices);
    
    vector<double> energy_info = TRW_solver->getEnergyInfo();
    vector<double> statistics = energy_info;
    statistics.push_back(static_cast<double>(clock() - begin_time) / CLOCKS_PER_SEC);
    iteration_statistics_map[iteration] = statistics;
    iteration_proposal_type_map[iteration] = proposal_type;
    
    writeLayers(image_, point_cloud_, camera_parameters_, num_layers_, current_solution_labels, proposal_num_surfaces, proposal_segments, SCENE_INDEX_, iteration);
      //exit(1);    
    
    if (energy_info[0] >= previous_energy && previous_energy >= 0) {
      cout << "energy increases" << endl;
      continue;
    }
    previous_energy = energy_info[0];
    
    int current_solution_num_surfaces = proposal_num_surfaces;
    map<int, Segment> current_solution_segments = proposal_segments;
    
    proposal_designer->setCurrentSolution(current_solution_labels, current_solution_num_surfaces, current_solution_segments); 
    
    ofstream iteration_info_out_str(iteration_info_filename.str());
    for (map<int, vector<double> >::const_iterator iteration_it = iteration_statistics_map.begin(); iteration_it != iteration_statistics_map.end(); iteration_it++) {
      iteration_info_out_str << iteration_it->first << '\t' << iteration_proposal_type_map[iteration_it->first] << '\t' << iteration_it->second[0] << '\t' << iteration_it->second[1] << '\t' << iteration_it->second[2] << endl;
    }
    iteration_info_out_str.close();
    
    best_solution_iteration = iteration;
  }
  
  // if (false) {
  //   num_layers_++;
  //   if (best_solution_iteration < NUM_ITERATIONS) {
  //     vector<vector<int> > proposal_labels;
  //     vector<int> proposal_segmentation;
  //     int proposal_num_surfaces;
  //     map<int, Segment> proposal_segments;
  //     vector<int> proposal_distance_to_boundaries;
  //     string proposal_type;
  //     if (proposal_designer->getLastProposal(proposal_labels, proposal_num_surfaces, proposal_segments, proposal_type) == true) {
  // 	vector<int> previous_solution_indices = proposal_designer->getCurrentSolutionIndices();
  // 	vector<int> current_solution_labels = TRW_solver->fuse(proposal_labels, proposal_num_surfaces, num_layers_, proposal_segments, previous_solution_indices);
  // 	double current_solution_energy = TRW_solver->getEnergy();
  // 	iteration_energy_map[NUM_ITERATIONS] = current_solution_energy;
  // 	iteration_proposal_type_map[NUM_ITERATIONS] = proposal_type;
  
  // 	writeLayers(image_, IMAGE_WIDTH_, IMAGE_HEIGHT_, point_cloud_, camera_parameters_, num_layers_, current_solution_labels, proposal_num_surfaces, proposal_segments, SCENE_INDEX_, NUM_ITERATIONS);
  // 	writeDispImageFromSegments(current_solution_labels, proposal_num_surfaces, proposal_segments, num_layers_, IMAGE_WIDTH_, IMAGE_HEIGHT_, "Test/final_disp_image.bmp");
  //       if (current_solution_energy >= previous_energy && previous_energy >= 0) {
  // 	  cout << "energy increases" << endl;
  // 	  num_layers_--;
  // 	} else
  // 	  best_solution_iteration = NUM_ITERATIONS;
  //     }
  //   }
  // }
  
  
  ofstream iteration_info_out_str(iteration_info_filename.str());
  for (map<int, vector<double> >::const_iterator iteration_it = iteration_statistics_map.begin(); iteration_it != iteration_statistics_map.end(); iteration_it++) {
    iteration_info_out_str << iteration_it->first << '\t' << iteration_proposal_type_map[iteration_it->first] << '\t' << iteration_it->second[0] << '\t' << iteration_it->second[1] << '\t' << iteration_it->second[2] << endl;
  }
  iteration_info_out_str.close();
  
  generateLayerImageHTML(SCENE_INDEX_, iteration_statistics_map, iteration_proposal_type_map);

  //exit(1);
  
  vector<int> best_solution_labels;
  int best_solution_num_surfaces;
  map<int, Segment> best_solution_segments;

  bool read_success = readLayers(IMAGE_WIDTH_, IMAGE_HEIGHT_, camera_parameters_, PENALTIES_, STATISTICS_, num_layers_, best_solution_labels, best_solution_num_surfaces, best_solution_segments, SCENE_INDEX_, best_solution_iteration, USE_PANORAMA_);
  
  //bool read_success = readLayers(solution_labels, solution_num_surfaces, solution_segments, best_solution_iteration);
  assert(read_success);
  
  writeLayers(image_, IMAGE_WIDTH_, IMAGE_HEIGHT_, point_cloud_, camera_parameters_, num_layers_, best_solution_labels, best_solution_num_surfaces, best_solution_segments, SCENE_INDEX_, 10000, ori_image_, ori_point_cloud_);
  //  writeLayers(image_, IMAGE_WIDTH_, IMAGE_HEIGHT_, point_cloud_, camera_parameters_, num_layers_, best_solution_labels, best_solution_num_surfaces, best_solution_segments, SCENE_INDEX_, 10000);
  //exit(1);
  
  if (false) {
    vector<int> refined_solution_labels;
    int refined_solution_num_surfaces;
    map<int, Segment> refined_solution_segments;
    if (readLayers(IMAGE_WIDTH_, IMAGE_HEIGHT_, camera_parameters_, PENALTIES_, STATISTICS_, num_layers_, refined_solution_labels, refined_solution_num_surfaces, refined_solution_segments, SCENE_INDEX_, 10000, USE_PANORAMA_) == false || true) {
      //refineSolution(solution_labels, solution_num_surfaces, solution_segments, refined_solution_labels);
      //solution_labels = refined_solution_labels;
      //writeLayers(image_, IMAGE_WIDTH_, IMAGE_HEIGHT_, point_cloud_, camera_parameters_, num_layers_, solution_labels, solution_num_surfaces, solution_segments, SCENE_INDEX_, 10000, ori_image_, ori_point_cloud_);
      //exit(1);
    }
  }

  //exit(1);
  //writeDispImageFromSegments(solution_labels, solution_num_surfaces, solution_segments, num_layers_, IMAGE_WIDTH_, IMAGE_HEIGHT_, "Test/final_disp_image.bmp");
  //writeLayers(solution_labels, solution_num_surfaces, solution_segments, 100);
  writeRenderingInfo(best_solution_labels, best_solution_num_surfaces, best_solution_segments);
  //drawLayersMulti(initial_segmentation_, labels, "Results");
  
  return;
}

void LayerDepthRepresenter::writeSegmentationImage(const vector<int> &segmentation, const string filename)
{
  Mat segmentation_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
  map<int, int> segment_center_x;
  map<int, int> segment_center_y;
  map<int, int> segment_pixel_counter;
  for (int i = 0; i < NUM_PIXELS_; i++) {
    if (ROI_mask_[i] == false)
      continue;
    int x = i % IMAGE_WIDTH_;
    int y = i / IMAGE_WIDTH_;

    int surface_id = segmentation[i];
    int surface_color = surface_colors_[surface_id];
    segmentation_image.at<Vec3b>(y, x) = Vec3b(surface_color % 256, surface_color % 256, surface_color % 256);

    segment_center_x[surface_id] += x;
    segment_center_y[surface_id] += y;
    segment_pixel_counter[surface_id]++;
  }
  for (map<int, int>::const_iterator segment_it = segment_pixel_counter.begin(); segment_it != segment_pixel_counter.end(); segment_it++) {
    Point origin(segment_center_x[segment_it->first] / segment_it->second, segment_center_y[segment_it->first] / segment_it->second);
    char *text = new char[10];
    sprintf(text, "%d", segment_it->first);
    putText(segmentation_image, text, origin, FONT_HERSHEY_PLAIN, 0.6, Scalar(0, 0, 255, 1));
  }
  //  stringstream segmentation_image_filename;
  //  segmentation_image_filename << "Results/segmentation_image.bmp";
  imwrite(filename.c_str(), segmentation_image);
}

void LayerDepthRepresenter::writeDispImage(const vector<int> &segmentation, const string filename)
{
  Mat disp_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int x = pixel % IMAGE_WIDTH_;
    int y = pixel / IMAGE_WIDTH_;
    //if (surface_depths_[1][pixel] > surface_depths_[5][pixel] && surface_depths_[5][pixel] > 0)
    //      test_image.at<uchar>(y, x) = 255;
    
    //for (int surface_id = 0; surface_id < num_surfaces_; surface_id++) {
    //      if (surface_id == segmentation[pixel]) {
    double depth = segmentation[pixel] < num_surfaces_ ? surface_depths_[segmentation[pixel]][pixel] : -1;
    if (depth > 0)
      disp_image.at<uchar>(y, x) = min(max(disp_image_numerator_ / depth, 0.0), 255.0);
    else
      disp_image.at<uchar>(y, x) = 255;
  }
  //  imwrite("Temp/test_image.bmp", test_image);
  imwrite(filename, disp_image);
}

void writeLayers(const Mat &image, const int image_width, const int image_height, const vector<double> &point_cloud, const vector<double> &camera_parameters, const int num_layers, const vector<int> &solution, const int solution_num_surfaces, const map<int, Segment> &solution_segments, const int scene_index, const int result_index, const Mat &ori_image, const vector<double> &ori_point_cloud)
{
  //cout << "write layers" << endl;
  
  //const int depth_map_sub_sampling_ratio = 4;
  const int NUM_PIXELS = image_width * image_height;
  vector<map<int, int> > layer_surface_x_sum(num_layers);
  vector<map<int, int> > layer_surface_y_sum(num_layers);
  vector<map<int, int> > layer_surface_counter(num_layers);
  vector<vector<int> > layer_surface_ids(num_layers, vector<int>(NUM_PIXELS, 0));
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    vector<int> layer_labels(num_layers);
    int label_temp = solution[pixel];
    for (int layer_index = num_layers - 1; layer_index >= 0; layer_index--) {
      layer_labels[layer_index] = label_temp % (solution_num_surfaces + 1);
      label_temp /= (solution_num_surfaces + 1);
    }
    for (int layer_index = 0; layer_index < num_layers; layer_index++) {
      int surface_id = layer_labels[layer_index];
      layer_surface_ids[layer_index][pixel] = surface_id;
      if (surface_id < solution_num_surfaces) {
	layer_surface_x_sum[layer_index][surface_id] += pixel % image_width;
	layer_surface_y_sum[layer_index][surface_id] += pixel / image_width;
	layer_surface_counter[layer_index][surface_id] += 1;
      }
    }
  }
  vector<map<int, int> > layer_surface_centers(num_layers);
  for (int layer_index = 0; layer_index < num_layers; layer_index++)
    for (map<int, int>::const_iterator counter_it = layer_surface_counter[layer_index].begin(); counter_it != layer_surface_counter[layer_index].end(); counter_it++)
      layer_surface_centers[layer_index][counter_it->first] = (layer_surface_y_sum[layer_index][counter_it->first] / counter_it->second) * image_width + (layer_surface_x_sum[layer_index][counter_it->first] / counter_it->second);
  
  //map<int, int> surface_colors = calcSurfaceColors(image_, proposal_segmentation);
  vector<Mat> layer_images;
  if (true) {
    map<int, Vec3b> color_table;
    vector<bool> visible_mask(ori_image.cols * ori_image.rows, true);
    
    map<int, Vec3b> layer_color_table;
    layer_color_table[2] = Vec3b(0, 0, 255);
    layer_color_table[1] = Vec3b(0, 255, 0);
    layer_color_table[0] = Vec3b(255, 0, 0);
    layer_color_table[3] = Vec3b(255, 0, 255);
    const double BLENDING_ALPHA = 0.5;
    
    
    for (int layer_index = 0; layer_index < num_layers; layer_index++) {
      // Mat layer_image = Mat::zeros(image_height, image_width, CV_8UC3);
      // vector<int> surface_ids = layer_surface_ids[layer_index];
      // for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
      //   int x = pixel % image_width;
      //   int y = pixel / image_width;
      
      //   int surface_id = surface_ids[pixel];
      //   if (surface_id != solution_num_surfaces) {
      //     if (visible_mask[pixel] == true) {
      //       layer_image.at<Vec3b>(y, x) = image.at<Vec3b>(y, x);
      //       visible_mask[pixel] = false;
      //     } else
      // 	    layer_image.at<Vec3b>(y, x) = Vec3b(0, 0, 0);;
      //   } else {
      // 	  layer_image.at<Vec3b>(y, x) = Vec3b(255, 255, 255);;
      //   }          
      // }
      
      //const double UPSAMPLING_RATIO = 1.0 * ori_image.cols / image.cols;
      
      Mat layer_image = Mat::zeros(ori_image.rows, ori_image.cols, CV_8UC3);
      vector<int> surface_ids = layer_surface_ids[layer_index];
      for (int ori_pixel = 0; ori_pixel < ori_image.cols * ori_image.rows; ori_pixel++) {
        int ori_x = ori_pixel % ori_image.cols;
        int ori_y = ori_pixel / ori_image.cols;
	int x = min(static_cast<int>(round(1.0 * ori_x / ori_image.cols * image_width)), image_width - 1);
	int y = min(static_cast<int>(round(1.0 * ori_y / ori_image.rows * image_height)), image_height - 1);
	int pixel = y * image_width + x;
        int surface_id = surface_ids[pixel];
        if (surface_id != solution_num_surfaces) {
          if (visible_mask[ori_pixel] == true) {
            layer_image.at<Vec3b>(ori_y, ori_x) = ori_image.at<Vec3b>(ori_y, ori_x);
            visible_mask[ori_pixel] = false;
          } else
	    layer_image.at<Vec3b>(ori_y, ori_x) = Vec3b(0, 0, 0);;
        } else {
	  layer_image.at<Vec3b>(ori_y, ori_x) = Vec3b(255, 255, 255);;
        }          
      }
      
      // Mat line_image = Mat::zeros(image_height * LINE_UPSAMPLING_RATIO, image_width * LINE_UPSAMPLING_RATIO, CV_8UC3);
      // resize(ori_image, line_image, line_image.size());
      vector<int> line_mask(ori_image.cols * ori_image.rows, -1);
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
        int surface_id = surface_ids[pixel];
        if (surface_id == solution_num_surfaces)
	  continue;
	
        int x = pixel % image_width;
        int y = pixel / image_width;
	vector<int> neighbor_pixels;
	if (x > 0)
	  neighbor_pixels.push_back(pixel - 1);
	if (x < image_width - 1)
	  neighbor_pixels.push_back(pixel + 1);
	if (y > 0)
	  neighbor_pixels.push_back(pixel - image_width);
	if (y < image_height - 1)
	  neighbor_pixels.push_back(pixel + image_width);
	if (x > 0 && y > 0)
	  neighbor_pixels.push_back(pixel - 1 - image_width);
	if (x > 0 && y < image_height - 1)
	  neighbor_pixels.push_back(pixel - 1 + image_width);
	if (x < image_width - 1 && y > 0)
	  neighbor_pixels.push_back(pixel + 1 - image_width);
	if (x < image_width - 1 && y < image_height - 1)
	  neighbor_pixels.push_back(pixel + 1 + image_width);
	bool on_boundary = false;
	int neighbor_segment_id = -1;
	for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	  if (surface_ids[*neighbor_pixel_it] != surface_id) {
	    on_boundary = true;
	    neighbor_segment_id = surface_ids[*neighbor_pixel_it];
	    break;
	  }
	}
	
	if (on_boundary == false)
	  continue;
	
	if (color_table.count(surface_id) == 0)
	  color_table[surface_id] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
	
	int ori_x = min(static_cast<int>(round(1.0 * x / image_width * ori_image.cols)), ori_image.cols - 1);
	int ori_y = min(static_cast<int>(round(1.0 * y / image_height * ori_image.rows)), ori_image.rows - 1);
	line_mask[ori_y * ori_image.cols + ori_x] = surface_id;
	
	//line_mask[min(static_cast<int>(round(1.0 * (*pixel_it / (image_width * LINE_UPSAMPLING_RATIO)) / LINE_UPSAMPLING_RATIO)), image_height - 1) * image_width + min(static_cast<int>(round(1.0 * (*pixel_it % (image_width * LINE_UPSAMPLING_RATIO)) / LINE_UPSAMPLING_RATIO)), image_width - 1)] = true;
	// vector<int> upsampling_pixels;
	// for (int delta_x = - (LINE_WINDOW_SIZE - 1) / 2; delta_x <= (LINE_WINDOW_SIZE - 1) / 2; delta_x++)
	//   for (int delta_y = - (LINE_WINDOW_SIZE - 1) / 2; delta_y <= (LINE_WINDOW_SIZE - 1) / 2; delta_y++)
	//     if (upsampling_x + delta_x >= 0 && upsampling_x + delta_x < image_width * LINE_UPSAMPLING_RATIO && upsampling_y + delta_y >= 0 && upsampling_y + delta_y < image_height * LINE_UPSAMPLING_RATIO)
	//       upsampling_pixels.push_back((upsampling_y + delta_y) * image_width * LINE_UPSAMPLING_RATIO + upsampling_x + delta_x);
	// for (vector<int>::const_iterator pixel_it = upsampling_pixels.begin(); pixel_it != upsampling_pixels.end(); pixel_it++) {
	//   line_image.at<Vec3b>(*pixel_it / (image_width * LINE_UPSAMPLING_RATIO), *pixel_it % (image_width * LINE_UPSAMPLING_RATIO)) = color_table[surface_id];
	//   line_mask[min(static_cast<int>(round(1.0 * (*pixel_it / (image_width * LINE_UPSAMPLING_RATIO)) / LINE_UPSAMPLING_RATIO)), image_height - 1) * image_width + min(static_cast<int>(round(1.0 * (*pixel_it % (image_width * LINE_UPSAMPLING_RATIO)) / LINE_UPSAMPLING_RATIO)), image_width - 1)] = true;
	// }
	// if (on_boundary == true) {
	//   if (color_table.count(surface_id) == 0)
	//     color_table[surface_id] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
	//   layer_image.at<Vec3b>(y, x) = layer_image.at<Vec3b>(y, x) * 0.3 + color_table[surface_id] * 0.7;
	// }
      }
      const int LINE_WIDTH = 1;
      for (int iteration = 0; iteration < LINE_WIDTH; iteration++) {
	vector<int> new_line_mask = line_mask;
	for (int pixel = 0; pixel < ori_image.cols * ori_image.rows; pixel++) {
	  if (line_mask[pixel] == -1)
	    continue;
          int x = pixel % ori_image.cols;
          int y = pixel / ori_image.cols;
          vector<int> neighbor_pixels;
          if (x > 0)
            neighbor_pixels.push_back(pixel - 1);
          if (x < ori_image.cols - 1)
            neighbor_pixels.push_back(pixel + 1);
          if (y > 0)
            neighbor_pixels.push_back(pixel - ori_image.cols);
          if (y < ori_image.rows - 1)
            neighbor_pixels.push_back(pixel + ori_image.cols);
          // if (x > 0 && y > 0)
          //   neighbor_pixels.push_back(pixel - 1 - ori_image.cols);
          // if (x > 0 && y < ori_image.rows - 1)
          //   neighbor_pixels.push_back(pixel - 1 + ori_image.cols);
          // if (x < ori_image.cols - 1 && y > 0)
          //   neighbor_pixels.push_back(pixel + 1 - ori_image.cols);
          // if (x < ori_image.cols - 1 && y < ori_image.rows - 1)
          //   neighbor_pixels.push_back(pixel + 1 + ori_image.cols);
          for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++)
	    if (new_line_mask[*neighbor_pixel_it] == -1)
	      new_line_mask[*neighbor_pixel_it] = line_mask[pixel];
	}
	line_mask = new_line_mask;
      }
      
      for (int ori_pixel = 0; ori_pixel < ori_image.cols * ori_image.rows; ori_pixel++) {
	if (line_mask[ori_pixel] == -1)
	  continue;
        int ori_x = ori_pixel % ori_image.cols;
        int ori_y = ori_pixel / ori_image.cols;
	layer_image.at<Vec3b>(ori_y, ori_x) = color_table[line_mask[ori_pixel]];
      }
      
      
      Mat layer_mask_image = ori_image.clone();
      for (int ori_pixel = 0; ori_pixel < ori_image.cols * ori_image.rows; ori_pixel++) {
        int ori_x = ori_pixel % ori_image.cols;
        int ori_y = ori_pixel / ori_image.cols;
        int x = min(static_cast<int>(round(1.0 * ori_x / ori_image.cols * image_width)), image_width - 1);
        int y = min(static_cast<int>(round(1.0 * ori_y / ori_image.rows * image_height)), image_height - 1);
        int pixel = y * image_width + x;
        int surface_id = surface_ids[pixel];
        if (surface_id != solution_num_surfaces) {
          Vec3b image_color = layer_mask_image.at<Vec3b>(ori_y, ori_x);
          Vec3b layer_color = layer_color_table[layer_index];
          Vec3b blended_color;
          for (int c = 0; c < 3; c++)
            blended_color[c] = min(image_color[c] * BLENDING_ALPHA + layer_color[c] * (1 - BLENDING_ALPHA), 255.0);
	  layer_mask_image.at<Vec3b>(ori_y, ori_x) = blended_color;
        }
	if (line_mask[ori_pixel] != -1)
	  layer_mask_image.at<Vec3b>(ori_y, ori_x) = Vec3b(0, 0, 0);
      }
      
      stringstream layer_mask_image_filename;
      layer_mask_image_filename << "Results/scene_" << scene_index << "/" << "layer_mask_image" << scene_index << "_" << layer_index << ".bmp";
      imwrite(layer_mask_image_filename.str().c_str(), layer_mask_image);
      
      //resize(layer_image, layer_image, image.size());
      
      // map<int, int> surface_centers = layer_surface_centers[layer_index];
      // for (map<int, int>::const_iterator surface_it = surface_centers.begin(); surface_it != surface_centers.end(); surface_it++) {
      //   char *text = new char[10];
      //   sprintf(text, "%d", surface_it->first);
      // 	if (solution_segments.at(surface_it->first).getType() == 0)
      // 	  putText(layer_image, text, Point(surface_it->second % image_width, surface_it->second / image_width), FONT_HERSHEY_PLAIN, 0.6, color_table[surface_it->first]);
      //   else
      // 	  putText(layer_image, text, Point(surface_it->second % image_width, surface_it->second / image_width), FONT_HERSHEY_PLAIN & FONT_ITALIC, 0.6, color_table[surface_it->first]);
      // }
      
      layer_images.push_back(layer_image);
    }
  } else {
    map<int, int> color_table;
    double segment_color_weight = 1;
    double ori_image_color_weight = 0.3;
    vector<bool> visible_mask(NUM_PIXELS, true);
    for (int layer_index = 0; layer_index < num_layers; layer_index++) {
      Mat layer_image = Mat::zeros(image_height, image_width, CV_8UC3);
      vector<double> depths(NUM_PIXELS);
      vector<int> surface_ids = layer_surface_ids[layer_index];
      map<int, int> surface_centers = layer_surface_centers[layer_index];
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
	int x = pixel % image_width;
	int y = pixel / image_width;
	
	int surface_id = surface_ids[pixel];
	if (surface_id != solution_num_surfaces) {
	  if (color_table.count(surface_id) == 0)
	    color_table[surface_id] = rand() % static_cast<int>(pow(256, 3));
	  int surface_color = color_table[surface_id];
	  Vec3b color;
	  color[0] = color[1] = color[2] = surface_color % 256;
	  // if (solution_segments.at(surface_id).getSegmentType() == 0)
	  //   color[0] = color[1] = color[2] = surface_color % 256;
	  // else {
	  //   color[0] = surface_color / pow(256, 2);
	  //   color[1] = surface_color / 256 % 256;
	  //   color[2] = surface_color % 256;
	  // }
	  if (visible_mask[pixel] == true) {
	    //color += image_.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_);
	    visible_mask[pixel] = false;
	  }
	  layer_image.at<Vec3b>(y, x) = color;
	  depths[pixel] = solution_segments.at(surface_id).getDepth(1.0 * (pixel % image_width) / image_width, 1.0 * (pixel / image_width) / image_height);
	} else {
	  Vec3b color;
	  color[0] = 255;
	  color[1] = color[2] = 0;
	  layer_image.at<Vec3b>(y, x) = color;
	  depths[pixel] = -1;
	}
      }
      for (map<int, int>::const_iterator surface_it = surface_centers.begin(); surface_it != surface_centers.end(); surface_it++) {
	char *text = new char[10];
	sprintf(text, "%d", surface_it->first);
	putText(layer_image, text, Point(surface_it->second % image_width, surface_it->second / image_width), FONT_HERSHEY_PLAIN, 0.6, Scalar(0, 0, 255, 1));
      }
      
      layer_images.push_back(layer_image);
    }
  }
  
  for (int layer_index = 0; layer_index < num_layers; layer_index++) {
    vector<int> surface_ids = layer_surface_ids[layer_index]; 
    Mat layer_image_raw = Mat::zeros(image_height, image_width, CV_8UC1);
    for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
      int x = pixel % image_width;
      int y = pixel / image_width;
      int surface_id = surface_ids[pixel];
      layer_image_raw.at<uchar>(y, x) = surface_id;
    }
    
    stringstream layer_image_raw_filename;
    
    layer_image_raw_filename << "Cache/scene_" << scene_index << "/" << "layer_image_raw_" << result_index << "_" << layer_index << ".bmp";
    imwrite(layer_image_raw_filename.str().c_str(), layer_image_raw);
  }
  
  
    // stringstream layer_disp_image_filename;
    // layer_disp_image_filename << "Results/scene_" << SCENE_INDEX_ << "/" << "layer_disp_image_" << iteration << "_" << layer_index << ".bmp";
    // imwrite(layer_disp_image_filename.str().c_str(), layer_disp_image);
  
  //  layer_alpha_map[layer_index] = 255 - (255 - 30) / (num_layers - 1) * (num_layers - 1 - layer_index);
  
  
  const int IMAGE_PADDING = 0;
  const int BORDER_WIDTH = 16;
  Mat multi_layer_image(ori_image.rows + BORDER_WIDTH * 2, (ori_image.cols + BORDER_WIDTH * 2 + IMAGE_PADDING) * (num_layers), CV_8UC3);
  multi_layer_image.setTo(Scalar(255, 255, 255));
  for (int layer_index = 0; layer_index < num_layers; layer_index++) {
    Mat layer_image_with_border = Mat::zeros(ori_image.rows + BORDER_WIDTH * 2, ori_image.cols + BORDER_WIDTH * 2, CV_8UC3);
    Mat layer_image_region(layer_image_with_border, Rect(BORDER_WIDTH, BORDER_WIDTH, ori_image.cols, ori_image.rows));
    layer_images[layer_index].copyTo(layer_image_region);
    
    Mat region(multi_layer_image, Rect((layer_image_with_border.cols + IMAGE_PADDING) * layer_index, 0, layer_image_with_border.cols, layer_image_with_border.rows));
    layer_image_with_border.copyTo(region);
  }
  //resize(multi_layer_image, multi_layer_image, Size(1000, 200));
  // const int IMAGE_PADDING = 10;
  // Mat multi_layer_image(image_height, (image_width + IMAGE_PADDING) * num_layers, CV_8UC3);
  // multi_layer_image.setTo(Scalar(255, 255, 255));
  // for (int layer_index = 0; layer_index < num_layers; layer_index++) {
  //   Mat region(multi_layer_image, Rect((image_width + IMAGE_PADDING) * layer_index, 0, image_width, image_height));
  //   layer_images[layer_index].copyTo(region);
  // }
  stringstream multi_layer_image_filename;
  multi_layer_image_filename << "Results/scene_" << scene_index << "/" << "multi_layer_image_" << result_index << ".bmp";
  imwrite(multi_layer_image_filename.str().c_str(), multi_layer_image);
  
  
  Mat input_multi_layer_image(ori_image.rows + BORDER_WIDTH * 2, (ori_image.cols + BORDER_WIDTH * 2 + IMAGE_PADDING) * (num_layers + 2), CV_8UC3);
  input_multi_layer_image.setTo(Scalar(0, 0, 0));
  
  Mat image_region(input_multi_layer_image, Rect(BORDER_WIDTH, BORDER_WIDTH, ori_image.cols, ori_image.rows));
  ori_image.copyTo(image_region);
  Mat disp_image_region(input_multi_layer_image, Rect(BORDER_WIDTH + (ori_image.cols + BORDER_WIDTH * 2), BORDER_WIDTH, ori_image.cols, ori_image.rows));
  //imwrite("Test/ori_disp_image.bmp", drawDispImage(ori_point_cloud, ori_image.cols, ori_image.rows));
  
  Mat disp_image = imread("Test/ori_disp_image.bmp");
  //  cout << disp_image.size() << '\t' << disp_image_region.size() << endl;
  //cout << disp_image.depth() << '\t' << ori_image.depth() << endl;
  //imshow("disp_image", disp_image);
  //waitKey();
  disp_image.copyTo(disp_image_region);
  Mat multi_layer_image_region(input_multi_layer_image, Rect((ori_image.cols + BORDER_WIDTH * 2) * 2, 0, multi_layer_image.cols, multi_layer_image.rows));
  multi_layer_image.copyTo(multi_layer_image_region);
  
  stringstream input_multi_layer_image_filename;
  input_multi_layer_image_filename << "Results/scene_" << scene_index << "/" << "input_multi_layer_image_" << scene_index << ".bmp";
  imwrite(input_multi_layer_image_filename.str().c_str(), input_multi_layer_image);
  
  
  stringstream segments_filename;
  segments_filename << "Cache/scene_" << scene_index << "/segments_" << result_index << ".txt";
  ofstream segments_out_str(segments_filename.str());
  segments_out_str << solution_num_surfaces << endl;
  for (map<int, Segment>::const_iterator surface_it = solution_segments.begin(); surface_it != solution_segments.end(); surface_it++) {
    segments_out_str << surface_it->first << endl;
    segments_out_str << surface_it->second << endl;
  }
  segments_out_str.close();

  bool use_GMM_models = false;
  if (use_GMM_models) {
    stringstream segment_GMMs_filename;
    segment_GMMs_filename << "Cache/scene_" << scene_index << "/segment_GMMs_" << result_index << ".xml";
    FileStorage segment_GMMs_fs(segment_GMMs_filename.str(), FileStorage::WRITE);
    for (map<int, Segment>::const_iterator surface_it = solution_segments.begin(); surface_it != solution_segments.end(); surface_it++) {
      stringstream segment_name;
      segment_name << "Segment" << surface_it->first;
      segment_GMMs_fs << segment_name.str() << "{";
      surface_it->second.getGMM()->write(segment_GMMs_fs);
      segment_GMMs_fs << "}";
    }
    segment_GMMs_fs.release();
  }  
  
  //write .ply files
  bool write_ply_files = false;
  if (write_ply_files == true) {
    vector<vector<int> > layer_visible_pixel_segment_map(num_layers, vector<int>(NUM_PIXELS, -1));
    for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
      for (int layer_index = 0; layer_index < num_layers; layer_index++) {
	int surface_id = layer_surface_ids[layer_index][pixel];
	if (surface_id < solution_num_surfaces) {
	  layer_visible_pixel_segment_map[layer_index][pixel] = surface_id;
	  break;
	}
      }
    }
    for (int layer_index = 0; layer_index < num_layers; layer_index++) {
      vector<int> surface_ids = layer_surface_ids[layer_index];
      int num_points = 0;
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++)
        if (surface_ids[pixel] < solution_num_surfaces)
    	  num_points++;

      map<int, vector<int> > segment_fitted_pixels;
      for (map<int, Segment>::const_iterator segment_it = solution_segments.begin(); segment_it != solution_segments.end(); segment_it++) {
        vector<int> fitted_pixels = segment_it->second.getSegmentPixels();
	vector<int> new_fitted_pixels;
	for (vector<int>::const_iterator pixel_it = fitted_pixels.begin(); pixel_it != fitted_pixels.end(); pixel_it++)
	  if (layer_visible_pixel_segment_map[layer_index][*pixel_it] == segment_it->first)
	    new_fitted_pixels.push_back(*pixel_it);
        segment_fitted_pixels[segment_it->first] = new_fitted_pixels;
      }	

      int num_fitted_pixels = 0;
      for (map<int, vector<int> >::const_iterator segment_it = segment_fitted_pixels.begin(); segment_it != segment_fitted_pixels.end(); segment_it++)
	num_fitted_pixels += segment_it->second.size();
      
      vector<double> point_cloud_range(6);
      for (int c = 0; c < 3; c++) {
	point_cloud_range[c * 2] = 1000000;
	point_cloud_range[c * 2 + 1] = -1000000;
      }
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
	vector<double> point(point_cloud.begin() + pixel * 3, point_cloud.begin() + (pixel + 1) * 3);
	for (int c = 0; c < 3; c++) {
	  if (point[c] < point_cloud_range[c * 2])
	    point_cloud_range[c * 2] = point[c];
          if (point[c] > point_cloud_range[c * 2 + 1])
            point_cloud_range[c * 2 + 1] = point[c];
	}
      }
      // map<int, vector<vector<double> > > segment_plane_vertices;
      // for (map<int, Segment>::const_iterator segment_it = solution_segments.begin(); segment_it != solution_segments.end(); segment_it++) {
      // 	if (segment_fitted_pixels.count(segment_it->first) == 0)
      // 	  continue;
      //   vector<double> plane = segment_it->second.getDepthPlane();
      // 	int max_direction = -1;
      // 	double max_direction_value = -1000000;
      // 	for (int c = 0; c < 3; c++) {
      // 	  if (abs(plane[c]) > max_direction_value) {
      // 	    max_direction = c;
      // 	    max_direction_value = abs(plane[c]);
      // 	  }
      // 	}
      // 	vector<vector<double> > plane_vertices(4, vector<double>(3));
      // 	bool first_direction = true;
      // 	for (int c = 0; c < 3; c++) {
      // 	  if (c != max_direction) {
      // 	    if (first_direction == true) {
      // 	      plane_vertices[0][c] = point_cloud_range[c * 2];
      // 	      plane_vertices[1][c] = point_cloud_range[c * 2];
      //         plane_vertices[2][c] = point_cloud_range[c * 2 + 1];
      //         plane_vertices[3][c] = point_cloud_range[c * 2 + 1];
      // 	      first_direction = false;
      // 	    } else {
      //         plane_vertices[0][c] = point_cloud_range[c * 2];
      //         plane_vertices[1][c] = point_cloud_range[c * 2 + 1];
      //         plane_vertices[2][c] = point_cloud_range[c * 2];
      //         plane_vertices[3][c] = point_cloud_range[c * 2 + 1];
      // 	    }
      // 	  }
      // 	}
      // 	for (int vertex_index = 0; vertex_index < 4; vertex_index++) {
      // 	  double missing_value = plane[3];
      // 	  for (int c = 0; c < 3; c++)
      // 	    if (c != max_direction)
      // 	      missing_value -= plane[c] * plane_vertices[vertex_index][c];
      // 	  missing_value = plane[max_direction] == 0 ? 0 : missing_value / plane[max_direction];
      // 	  plane_vertices[vertex_index][max_direction] = missing_value;
      // 	}
      // 	segment_plane_vertices[segment_it->first] = plane_vertices;
      // }

      map<int, vector<vector<double> > > segment_plane_vertices;
      for (map<int, Segment>::const_iterator segment_it = solution_segments.begin(); segment_it != solution_segments.end(); segment_it++)
	segment_plane_vertices[segment_it->first] = vector<vector<double> >(4, vector<double>(3));
      stringstream layer_ply_filename;
      layer_ply_filename << "Results/scene_" << scene_index << "/" << "layer_ply_" << result_index << "_" << layer_index << ".ply";
      ofstream out_str(layer_ply_filename.str());
      
      out_str << "ply" << endl;
      out_str << "format ascii 1.0" << endl;
      out_str << "element vertex " << num_points << endl;
      out_str << "property float x" << endl;
      out_str << "property float y" << endl;
      out_str << "property float z" << endl;
      out_str << "end_header" << endl;
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
        int surface_id = surface_ids[pixel];
        if (surface_id == solution_num_surfaces)
    	  continue;
    	double depth = solution_segments.at(surface_id).getDepth(1.0 * (pixel % image_width) / image_width, 1.0 * (pixel / image_width) / image_height);
	double x = pixel % image_width - camera_parameters[1];
	double y = pixel / image_width - camera_parameters[2];
	double X = -x / camera_parameters[0] * depth;
	double Y = -y / camera_parameters[0] * depth;
	out_str << X << ' ' << Y << ' ' << depth << endl;
      }
      out_str.close();

      stringstream layer_segment_fitting_filename;
      layer_segment_fitting_filename << "Results/scene_" << scene_index << "/" << "layer_segment_fitting_" << result_index << "_" << layer_index << ".ply";
      out_str.open(layer_segment_fitting_filename.str());
      
      out_str << "ply" << endl;
      out_str << "format ascii 1.0" << endl;
      out_str << "element vertex " << num_fitted_pixels + solution_segments.size() * 4 << endl;
      out_str << "property float x" << endl;
      out_str << "property float y" << endl;
      out_str << "property float z" << endl;
      out_str << "property uchar red" << endl;
      out_str << "property uchar green" << endl;
      out_str << "property uchar blue" << endl;
      out_str << "property uchar alpha" << endl;
      out_str << "element face " << segment_fitted_pixels.size() * 2 << endl;
      out_str << "property list uchar int vertex_indices" << endl;
      out_str << "end_header" << endl;
      map<int, int> color_table;
      for (map<int, vector<int> >::const_iterator segment_it = segment_fitted_pixels.begin(); segment_it != segment_fitted_pixels.end(); segment_it++) {
        int r = rand() % 256;
        int g = rand() % 256;
        int b = rand() % 256;
        color_table[segment_it->first] = r * 256 * 256 + g * 256 + b;
        for (vector<int>::const_iterator pixel_it = segment_it->second.begin(); pixel_it != segment_it->second.end(); pixel_it++) {
          int pixel = *pixel_it;
	  
          // double depth = solution_segments.at(segment_it->first).getDepth(pixel);
          // double x = pixel % IMAGE_WIDTH_ - IMAGE_WIDTH_ / 2;
          // double y = pixel / IMAGE_WIDTH_ - IMAGE_HEIGHT_ / 2;
          // double X = -x / focal_length_ * depth;
          // double Y = -y / focal_length_ * depth;
	  double X = -point_cloud[pixel * 3 + 0];
	  double Y = -point_cloud[pixel * 3 + 1];
	  double Z = point_cloud[pixel * 3 + 2];
          out_str << X << ' ' << Y << ' ' << Z << ' ' << r << ' ' << g << ' ' << b << " 255" << endl;
        }
      }
      for (map<int, vector<vector<double> > >::const_iterator segment_it = segment_plane_vertices.begin(); segment_it != segment_plane_vertices.end(); segment_it++) {
        int color = color_table[segment_it->first];
        int r = color / (256 * 256);
        int g = color / 256 % 256;
        int b = color % 256;
        for (vector<vector<double> >::const_iterator vertex_it = segment_it->second.begin(); vertex_it != segment_it->second.end(); vertex_it++)
          out_str << vertex_it->at(0) << ' ' << vertex_it->at(1) << ' ' << vertex_it->at(2) << ' ' << r << ' ' << g << ' ' << b << " 10" << endl;
      }
      for (int i = 0; i < segment_fitted_pixels.size(); i++) {
	out_str << "3 " << num_fitted_pixels + i * 4 + 0 << ' ' << num_fitted_pixels + i * 4 + 1 << ' ' << num_fitted_pixels + i * 4 + 2 << endl;
	out_str << "3 " << num_fitted_pixels + i * 4 + 0 << ' ' << num_fitted_pixels + i * 4 + 2 << ' ' << num_fitted_pixels + i * 4 + 3 << endl;
      }
      out_str.close();
    }
  }
  
  
  //write mesh .ply files
  bool write_mesh_ply_files = false;
  if (write_mesh_ply_files == true) {
    map<int, int> layer_color_table;
    //  layer_alpha_map[layer_index] = 255 - (255 - 30) / (num_layers - 1) * (num_layers - 1 - layer_index);
    layer_color_table[0] = 255 * 256 * 256;
    layer_color_table[1] = 255 * 256;
    layer_color_table[2] = 255;
    layer_color_table[3] = 255 * 256 + 255;
    vector<vector<int> > layer_triangle_pixels_vec(num_layers);
    int num_non_empty_pixels = 0;
    vector<vector<int> > layer_pixel_index_map(num_layers);
    for (int layer_index = 0; layer_index < num_layers; layer_index++) {
      vector<int> surface_ids = layer_surface_ids[layer_index];
      vector<int> triangle_pixels_vec;
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
        int x = pixel % image_width;
	int y = pixel / image_width;
	if (x == image_width - 1 || y == image_height - 1)
	  continue;
	vector<int> cell_pixels;
	if (surface_ids[pixel] < solution_num_surfaces)
	  cell_pixels.push_back(pixel);
	if (surface_ids[pixel + 1] < solution_num_surfaces)
          cell_pixels.push_back(pixel + 1);
        if (surface_ids[pixel + 1 + image_width] < solution_num_surfaces)
          cell_pixels.push_back(pixel + 1 + image_width);
        if (surface_ids[pixel + image_width] < solution_num_surfaces)
          cell_pixels.push_back(pixel + image_width);
	
	if (cell_pixels.size() == 3)
	  triangle_pixels_vec.insert(triangle_pixels_vec.end(), cell_pixels.begin(), cell_pixels.end());
        if (cell_pixels.size() == 4) {
	  triangle_pixels_vec.insert(triangle_pixels_vec.end(), cell_pixels.begin(), cell_pixels.begin() + 3);
	  triangle_pixels_vec.push_back(cell_pixels[0]);
	  triangle_pixels_vec.push_back(cell_pixels[2]);
	  triangle_pixels_vec.push_back(cell_pixels[3]);
	}
      }
      layer_triangle_pixels_vec[layer_index] = triangle_pixels_vec;
      
      vector<int> pixel_index_map(NUM_PIXELS, -1);
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++)
	if (surface_ids[pixel] < solution_num_surfaces)
	  pixel_index_map[pixel] = num_non_empty_pixels++;
      layer_pixel_index_map[layer_index] = pixel_index_map;
    }
    
    {
      stringstream layer_segment_mesh_filename;
      layer_segment_mesh_filename << "Results/scene_" << scene_index << "/" << "layered_mesh.ply";
      ofstream out_str(layer_segment_mesh_filename.str());

      out_str << "ply" << endl;
      out_str << "format ascii 1.0" << endl;
      out_str << "element vertex " << num_non_empty_pixels << endl;
      out_str << "property float x" << endl;
      out_str << "property float y" << endl;
      out_str << "property float z" << endl;
      out_str << "property uchar red" << endl;
      out_str << "property uchar green" << endl;
      out_str << "property uchar blue" << endl;
      //      out_str << "property uchar alpha" << endl;
      int num_triangles = 0;
      for (int layer_index = 0; layer_index < num_layers; layer_index++)
	num_triangles += layer_triangle_pixels_vec[layer_index].size() / 3;
      out_str << "element face " << num_triangles << endl;
      out_str << "property list uchar int vertex_indices" << endl;
      out_str << "end_header" << endl;
      map<int, int> color_table;
      for (int layer_index = 0; layer_index < num_layers; layer_index++) {
        for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
	  int segment_id = layer_surface_ids[layer_index][pixel];
	  if (segment_id == solution_num_surfaces)
	    continue;
	  // if (color_table.count(segment_id) == 0)
	  //   color_table[segment_id] = rand() % (256 * 256 * 256);
	
	  int x = pixel % image_width;
	  int y = pixel / image_width;
	  vector<int> neighbor_pixels;
	  if (x > 0)
	    neighbor_pixels.push_back(pixel - 1);
	  if (x < image_width - 1)
	    neighbor_pixels.push_back(pixel + 1);
	  if (y > 0)
	    neighbor_pixels.push_back(pixel - image_width);
	  if (y < image_height - 1)
	    neighbor_pixels.push_back(pixel + image_width);
	  if (x > 0 && y > 0)
	    neighbor_pixels.push_back(pixel - 1 - image_width);
	  if (x > 0 && y < image_height - 1)
	    neighbor_pixels.push_back(pixel - 1 + image_width);
	  if (x < image_width - 1 && y > 0)
	    neighbor_pixels.push_back(pixel + 1 - image_width);
	  if (x < image_width - 1 && y < image_height - 1)
	    neighbor_pixels.push_back(pixel + 1 + image_width);
	  bool on_boundary = false;
	  for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	    if (layer_surface_ids[layer_index][*neighbor_pixel_it] != segment_id) {
	      on_boundary = true;
	      break;
	    }
	  }
	
	  double depth = solution_segments.at(segment_id).getDepth(1.0 * (pixel % image_width) / image_width, 1.0 * (pixel / image_width) / image_height);
	  double u = pixel % image_width - camera_parameters[1];
	  double v = pixel / image_width - camera_parameters[2];
	  double X = -u / camera_parameters[0] * depth;
	  double Y = -v / camera_parameters[0] * depth;
	  // double X = -point_cloud[pixel * 3 + 0];
	  // double Y = -point_cloud[pixel * 3 + 1];
	  // double Z = point_cloud[pixel * 3 + 2];
	  //int color = on_boundary ? 255 * 256 * 256 + 255 * 256 + 255 : layer_color_table[layer_index];
	  int color = layer_color_table[layer_index];
	  //	  out_str << X << ' ' << Y << ' ' << depth << endl;
	  out_str << X << ' ' << Y << ' ' << depth << ' ' << color / (256 * 256) << ' ' << color / 256 % 256 << ' ' << color % 256 << endl;
        }
      }
      
      for (int layer_index = 0; layer_index < num_layers; layer_index++) { 
	for (int triangle_index = 0; triangle_index < layer_triangle_pixels_vec[layer_index].size() / 3; triangle_index++) {
	  out_str << "3";
	  for (int c = 0; c < 3; c++) {
	    int pixel = layer_triangle_pixels_vec[layer_index][triangle_index * 3 + c];
	    int pixel_index = layer_pixel_index_map[layer_index][pixel];
	    
	    out_str << ' ' << pixel_index;
	  }
	  out_str << endl;
	}
      }
      out_str.close();
    }
  }
}

void writeLayers(const Mat &image, const vector<double> &point_cloud, const vector<double> &camera_parameters, const int num_layers, const vector<int> &solution, const int solution_num_surfaces, const map<int, Segment> &solution_segments, const int scene_index, const int result_index)
{
  //cout << "write layers" << endl;
  const int image_width = image.cols;
  const int image_height = image.rows;
  
  //const int depth_map_sub_sampling_ratio = 4;
  const int NUM_PIXELS = image_width * image_height;
  vector<map<int, int> > layer_surface_x_sum(num_layers);
  vector<map<int, int> > layer_surface_y_sum(num_layers);
  vector<map<int, int> > layer_surface_counter(num_layers);
  vector<vector<int> > layer_surface_ids(num_layers, vector<int>(NUM_PIXELS, 0));
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
    vector<int> layer_labels(num_layers);
    int label_temp = solution[pixel];
    for (int layer_index = num_layers - 1; layer_index >= 0; layer_index--) {
      layer_labels[layer_index] = label_temp % (solution_num_surfaces + 1);
      label_temp /= (solution_num_surfaces + 1);
    }
    for (int layer_index = 0; layer_index < num_layers; layer_index++) {
      int surface_id = layer_labels[layer_index];
      layer_surface_ids[layer_index][pixel] = surface_id;
      if (surface_id < solution_num_surfaces) {
        layer_surface_x_sum[layer_index][surface_id] += pixel % image_width;
        layer_surface_y_sum[layer_index][surface_id] += pixel / image_width;
        layer_surface_counter[layer_index][surface_id] += 1;
      }
    }
  }
  vector<map<int, int> > layer_surface_centers(num_layers);
  for (int layer_index = 0; layer_index < num_layers; layer_index++)
    for (map<int, int>::const_iterator counter_it = layer_surface_counter[layer_index].begin(); counter_it != layer_surface_counter[layer_index].end(); counter_it++)
      layer_surface_centers[layer_index][counter_it->first] = (layer_surface_y_sum[layer_index][counter_it->first] / counter_it->second) * image_width + (layer_surface_x_sum[layer_index][counter_it->first] / counter_it->second);
  
  //map<int, int> surface_colors = calcSurfaceColors(image_, proposal_segmentation);
  
  
  // map<int, Vec3b> layer_color_table;
  // layer_color_table[2] = Vec3b(0, 0, 255);
  // layer_color_table[1] = Vec3b(0, 255, 0);
  // layer_color_table[0] = Vec3b(255, 0, 0);
  // layer_color_table[3] = Vec3b(255, 0, 255);
  const double BLENDING_ALPHA = 0.5;
  map<int, Vec3b> color_table;  
  
  vector<Mat> layer_mask_images;
  for (int layer_index = 0; layer_index < num_layers; layer_index++) {
    Mat layer_mask_image = image.clone();
    vector<int> surface_ids = layer_surface_ids[layer_index];
    for (int pixel = 0; pixel < image_width * image_height; pixel++) {
      int x = pixel % image_width;
      int y = pixel / image_width;
      int surface_id = surface_ids[pixel];
      if (surface_id != solution_num_surfaces) {
	Vec3b image_color = image.at<Vec3b>(y, x);
	if (color_table.count(surface_id) == 0)
	  color_table[surface_id] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
	Vec3b segment_color = color_table[surface_id];
	Vec3b blended_color;
	for (int c = 0; c < 3; c++)
	  blended_color[c] = min(image_color[c] * BLENDING_ALPHA + segment_color[c] * (1 - BLENDING_ALPHA), 255.0);
	layer_mask_image.at<Vec3b>(y, x) = blended_color;
      }

      vector<int> neighbor_pixels = findNeighbors(pixel, image_width, image_height);
      bool on_border = false;
      for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
        if (surface_ids[*neighbor_pixel_it] != surface_id) {
          on_border = true;
          break;
        }
      }
      if (on_border)
        layer_mask_image.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
    }
    for (map<int, int>::const_iterator segment_it = layer_surface_centers[layer_index].begin(); segment_it != layer_surface_centers[layer_index].end(); segment_it++) {
      Point origin(segment_it->second % image_width, segment_it->second / image_width);
      //if (solution_segments.at(segment_it->first).getBehindRoomStructure() == true)
      //putText(layer_mask_image, to_string(segment_it->first), origin, FONT_HERSHEY_PLAIN, 0.6, Scalar(0, 255, 0, 1));
      
      // if (solution_segments.at(segment_it->first).getSegmentType() == 0)
      // 	putText(layer_mask_image, to_string(segment_it->first), origin, FONT_HERSHEY_PLAIN, 0.6, Scalar(0, 0, 255, 1));
      // else
      // 	putText(layer_mask_image, to_string(segment_it->first), origin, FONT_HERSHEY_PLAIN, 0.6, Scalar(255, 0, 0, 1));
    }
    layer_mask_images.push_back(layer_mask_image);
    imwrite("Test/layer_mask_image_" + to_string(layer_index) + ".bmp", layer_mask_image);
  }
  
  
  for (int layer_index = 0; layer_index < num_layers; layer_index++) {
    vector<int> surface_ids = layer_surface_ids[layer_index]; 
    Mat layer_image_raw = Mat::zeros(image_height, image_width, CV_8UC1);
    for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
      int x = pixel % image_width;
      int y = pixel / image_width;
      int surface_id = surface_ids[pixel];
      layer_image_raw.at<uchar>(y, x) = surface_id;
    }
    
    stringstream layer_image_raw_filename;
    
    layer_image_raw_filename << "Cache/scene_" << scene_index << "/" << "layer_image_raw_" << result_index << "_" << layer_index << ".bmp";
    imwrite(layer_image_raw_filename.str().c_str(), layer_image_raw);
  }
  
  
  vector<Mat> disp_images(num_layers);
  for (int layer_index = 0; layer_index < num_layers; layer_index++) {
    Mat disp_image(image_height, image_width, CV_8UC3);
    vector<int> surface_ids = layer_surface_ids[layer_index];
    for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
      if (surface_ids[pixel] == solution_num_surfaces)
	disp_image.at<Vec3b>(pixel / image_width, pixel % image_width) = Vec3b(255, 0, 0);
      else {
	double disp = 1.0 / solution_segments.at(surface_ids[pixel]).getDepth(pixel);
	assert(disp > 0);
	double color = min(disp * 300, 255.0);
	disp_image.at<Vec3b>(pixel / image_width, pixel % image_width) = Vec3b(color, color, color);
      }
    }
    disp_images[layer_index] = disp_image;
    imwrite("Test/layer_disp_image_" + to_string(layer_index) + ".bmp", disp_image);
  }
  
  Mat disp_image(image_height, image_width, CV_8UC3);
  for (int pixel = 0; pixel < image_width * image_height; pixel++) {
    int label = solution[pixel];
    for (int layer_index = 0; layer_index < num_layers; layer_index++) {
      int surface_id = label / static_cast<int>(pow(solution_num_surfaces + 1, num_layers - 1 - layer_index)) % (solution_num_surfaces + 1);
      if (surface_id < solution_num_surfaces) {
	disp_image.at<Vec3b>(pixel / image_width, pixel % image_width) = disp_images[layer_index].at<Vec3b>(pixel / image_width, pixel % image_width);
	break;
      }
    }
  }
  imwrite("Test/disp_image.bmp", disp_image);
  
  
  const int IMAGE_PADDING = 0;
  const int BORDER_WIDTH = 4;
  Mat multi_layer_image(image.rows + BORDER_WIDTH * 2, (image.cols + BORDER_WIDTH * 2 + IMAGE_PADDING) * (num_layers), CV_8UC3);
  multi_layer_image.setTo(Scalar(255, 255, 255));
  for (int layer_index = 0; layer_index < num_layers; layer_index++) {
    Mat layer_image_with_border = Mat::zeros(image.rows + BORDER_WIDTH * 2, image.cols + BORDER_WIDTH * 2, CV_8UC3);
    Mat layer_image_region(layer_image_with_border, Rect(BORDER_WIDTH, BORDER_WIDTH, image.cols, image.rows));
    layer_mask_images[layer_index].copyTo(layer_image_region);
    Mat region(multi_layer_image, Rect((layer_image_with_border.cols + IMAGE_PADDING) * layer_index, 0, layer_image_with_border.cols, layer_image_with_border.rows));
    layer_image_with_border.copyTo(region);
  }
  //resize(multi_layer_image, multi_layer_image, Size(1000, 200));
  // const int IMAGE_PADDING = 10;
  // Mat multi_layer_image(image_height, (image_width + IMAGE_PADDING) * num_layers, CV_8UC3);
  // multi_layer_image.setTo(Scalar(255, 255, 255));
  // for (int layer_index = 0; layer_index < num_layers; layer_index++) {
  //   Mat region(multi_layer_image, Rect((image_width + IMAGE_PADDING) * layer_index, 0, image_width, image_height));
  //   layer_images[layer_index].copyTo(region);
  // }
  stringstream multi_layer_image_filename;
  multi_layer_image_filename << "Results/scene_" << scene_index << "/" << "multi_layer_image_" << result_index << ".bmp";
  imwrite(multi_layer_image_filename.str().c_str(), multi_layer_image);
  
  Mat multi_disp_image(image.rows + BORDER_WIDTH * 2, (image.cols + BORDER_WIDTH * 2 + IMAGE_PADDING) * (num_layers), CV_8UC3);
  multi_disp_image.setTo(Scalar(255, 255, 255));
  for (int layer_index = 0; layer_index < num_layers; layer_index++) {
    Mat disp_image_with_border = Mat::zeros(image.rows + BORDER_WIDTH * 2, image.cols + BORDER_WIDTH * 2, CV_8UC3);
    Mat disp_image_region(disp_image_with_border, Rect(BORDER_WIDTH, BORDER_WIDTH, image.cols, image.rows));
    disp_images[layer_index].copyTo(disp_image_region);
    Mat region(multi_disp_image, Rect((disp_image_with_border.cols + IMAGE_PADDING) * layer_index, 0, disp_image_with_border.cols, disp_image_with_border.rows));
    disp_image_with_border.copyTo(region);
  }
  stringstream multi_disp_image_filename;
  multi_disp_image_filename << "Results/scene_" << scene_index << "/" << "multi_disp_image_" << result_index << ".bmp";
  imwrite(multi_disp_image_filename.str().c_str(), multi_disp_image);
  
  
  stringstream segments_filename;
  segments_filename << "Cache/scene_" << scene_index << "/segments_" << result_index << ".txt";
  ofstream segments_out_str(segments_filename.str());
  segments_out_str << solution_num_surfaces << endl;
  for (map<int, Segment>::const_iterator surface_it = solution_segments.begin(); surface_it != solution_segments.end(); surface_it++) {
    segments_out_str << surface_it->first << endl;
    segments_out_str << surface_it->second << endl;
  }
  segments_out_str.close();

  bool use_GMM_models = false;
  if (use_GMM_models) {
    stringstream segment_GMMs_filename;
    segment_GMMs_filename << "Cache/scene_" << scene_index << "/segment_GMMs_" << result_index << ".xml";
    FileStorage segment_GMMs_fs(segment_GMMs_filename.str(), FileStorage::WRITE);
    for (map<int, Segment>::const_iterator surface_it = solution_segments.begin(); surface_it != solution_segments.end(); surface_it++) {
      stringstream segment_name;
      segment_name << "Segment" << surface_it->first;
      segment_GMMs_fs << segment_name.str() << "{";
      surface_it->second.getGMM()->write(segment_GMMs_fs);
      segment_GMMs_fs << "}";
    }
    segment_GMMs_fs.release();
  }
  
  //write .ply files
  bool write_ply_files = false;
  if (write_ply_files == true) {
    vector<vector<int> > layer_visible_pixel_segment_map(num_layers, vector<int>(NUM_PIXELS, -1));
    for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
      for (int layer_index = 0; layer_index < num_layers; layer_index++) {
        int surface_id = layer_surface_ids[layer_index][pixel];
        if (surface_id < solution_num_surfaces) {
          layer_visible_pixel_segment_map[layer_index][pixel] = surface_id;
          break;
        }
      }
    }
    for (int layer_index = 0; layer_index < num_layers; layer_index++) {
      vector<int> surface_ids = layer_surface_ids[layer_index];
      int num_points = 0;
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++)
        if (surface_ids[pixel] < solution_num_surfaces)
          num_points++;
      
      map<int, vector<int> > segment_fitted_pixels;
      for (map<int, Segment>::const_iterator segment_it = solution_segments.begin(); segment_it != solution_segments.end(); segment_it++) {
        vector<int> fitted_pixels = segment_it->second.getSegmentPixels();
        vector<int> new_fitted_pixels;
        for (vector<int>::const_iterator pixel_it = fitted_pixels.begin(); pixel_it != fitted_pixels.end(); pixel_it++)
          if (layer_visible_pixel_segment_map[layer_index][*pixel_it] == segment_it->first)
            new_fitted_pixels.push_back(*pixel_it);
        segment_fitted_pixels[segment_it->first] = new_fitted_pixels;
      } 
      
      int num_fitted_pixels = 0;
      for (map<int, vector<int> >::const_iterator segment_it = segment_fitted_pixels.begin(); segment_it != segment_fitted_pixels.end(); segment_it++)
        num_fitted_pixels += segment_it->second.size();
      
      vector<double> point_cloud_range(6);
      for (int c = 0; c < 3; c++) {
        point_cloud_range[c * 2] = 1000000;
        point_cloud_range[c * 2 + 1] = -1000000;
      }
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
        vector<double> point(point_cloud.begin() + pixel * 3, point_cloud.begin() + (pixel + 1) * 3);
        for (int c = 0; c < 3; c++) {
          if (point[c] < point_cloud_range[c * 2])
            point_cloud_range[c * 2] = point[c];
          if (point[c] > point_cloud_range[c * 2 + 1])
            point_cloud_range[c * 2 + 1] = point[c];
        }
      }
      // map<int, vector<vector<double> > > segment_plane_vertices;
      // for (map<int, Segment>::const_iterator segment_it = solution_segments.begin(); segment_it != solution_segments.end(); segment_it++) {
      //        if (segment_fitted_pixels.count(segment_it->first) == 0)
      //          continue;
      //   vector<double> plane = segment_it->second.getDepthPlane();
      //        int max_direction = -1;
      //        double max_direction_value = -1000000;
      //        for (int c = 0; c < 3; c++) {
      //          if (abs(plane[c]) > max_direction_value) {
      //            max_direction = c;
      //            max_direction_value = abs(plane[c]);
      //          }
      //        }
      //        vector<vector<double> > plane_vertices(4, vector<double>(3));
      //        bool first_direction = true;
      //        for (int c = 0; c < 3; c++) {
      //          if (c != max_direction) {
      //            if (first_direction == true) {
      //              plane_vertices[0][c] = point_cloud_range[c * 2];
      //              plane_vertices[1][c] = point_cloud_range[c * 2];
      //         plane_vertices[2][c] = point_cloud_range[c * 2 + 1];
      //         plane_vertices[3][c] = point_cloud_range[c * 2 + 1];
      //              first_direction = false;
      //            } else {
      //         plane_vertices[0][c] = point_cloud_range[c * 2];
      //         plane_vertices[1][c] = point_cloud_range[c * 2 + 1];
      //         plane_vertices[2][c] = point_cloud_range[c * 2];
      //         plane_vertices[3][c] = point_cloud_range[c * 2 + 1];
      //            }
      //          }
      //        }
      //        for (int vertex_index = 0; vertex_index < 4; vertex_index++) {
      //          double missing_value = plane[3];
      //          for (int c = 0; c < 3; c++)
      //            if (c != max_direction)
      //              missing_value -= plane[c] * plane_vertices[vertex_index][c];
      //          missing_value = plane[max_direction] == 0 ? 0 : missing_value / plane[max_direction];
      //          plane_vertices[vertex_index][max_direction] = missing_value;
      //        }
      //        segment_plane_vertices[segment_it->first] = plane_vertices;
      // }
      
      map<int, vector<vector<double> > > segment_plane_vertices;
      for (map<int, Segment>::const_iterator segment_it = solution_segments.begin(); segment_it != solution_segments.end(); segment_it++)
        segment_plane_vertices[segment_it->first] = vector<vector<double> >(4, vector<double>(3));
      stringstream layer_ply_filename;
      layer_ply_filename << "Results/scene_" << scene_index << "/" << "layer_ply_" << result_index << "_" << layer_index << ".ply";
      ofstream out_str(layer_ply_filename.str());
      
      out_str << "ply" << endl;
      out_str << "format ascii 1.0" << endl;
      out_str << "element vertex " << num_points << endl;
      out_str << "property float x" << endl;
      out_str << "property float y" << endl;
      out_str << "property float z" << endl;
      out_str << "end_header" << endl;
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
        int surface_id = surface_ids[pixel];
        if (surface_id == solution_num_surfaces)
          continue;
        double depth = solution_segments.at(surface_id).getDepth(1.0 * (pixel % image_width) / image_width, 1.0 * (pixel / image_width) / image_height);
        double x = pixel % image_width - camera_parameters[1];
        double y = pixel / image_width - camera_parameters[2];
        double X = -x / camera_parameters[0] * depth;
        double Y = -y / camera_parameters[0] * depth;
        out_str << X << ' ' << Y << ' ' << depth << endl;
      }
      out_str.close();
      
      stringstream layer_segment_fitting_filename;
      layer_segment_fitting_filename << "Results/scene_" << scene_index << "/" << "layer_segment_fitting_" << result_index << "_" << layer_index << ".ply";
      out_str.open(layer_segment_fitting_filename.str());
      
      out_str << "ply" << endl;
      out_str << "format ascii 1.0" << endl;
      out_str << "element vertex " << num_fitted_pixels + solution_segments.size() * 4 << endl;
      out_str << "property float x" << endl;
      out_str << "property float y" << endl;
      out_str << "property float z" << endl;
      out_str << "property uchar red" << endl;
      out_str << "property uchar green" << endl;
      out_str << "property uchar blue" << endl;
      out_str << "property uchar alpha" << endl;
      out_str << "element face " << segment_fitted_pixels.size() * 2 << endl;
      out_str << "property list uchar int vertex_indices" << endl;
      out_str << "end_header" << endl;
      map<int, int> color_table;
      for (map<int, vector<int> >::const_iterator segment_it = segment_fitted_pixels.begin(); segment_it != segment_fitted_pixels.end(); segment_it++) {
        int r = rand() % 256;
        int g = rand() % 256;
        int b = rand() % 256;
        color_table[segment_it->first] = r * 256 * 256 + g * 256 + b;
        for (vector<int>::const_iterator pixel_it = segment_it->second.begin(); pixel_it != segment_it->second.end(); pixel_it++) {
          int pixel = *pixel_it;
	  
          // double depth = solution_segments.at(segment_it->first).getDepth(pixel);
          // double x = pixel % IMAGE_WIDTH_ - IMAGE_WIDTH_ / 2;
          // double y = pixel / IMAGE_WIDTH_ - IMAGE_HEIGHT_ / 2;
          // double X = -x / focal_length_ * depth;
          // double Y = -y / focal_length_ * depth;
          double X = -point_cloud[pixel * 3 + 0];
          double Y = -point_cloud[pixel * 3 + 1];
          double Z = point_cloud[pixel * 3 + 2];
          out_str << X << ' ' << Y << ' ' << Z << ' ' << r << ' ' << g << ' ' << b << " 255" << endl;
        }
      }
      for (map<int, vector<vector<double> > >::const_iterator segment_it = segment_plane_vertices.begin(); segment_it != segment_plane_vertices.end(); segment_it++) {
        int color = color_table[segment_it->first];
        int r = color / (256 * 256);
        int g = color / 256 % 256;
        int b = color % 256;
        for (vector<vector<double> >::const_iterator vertex_it = segment_it->second.begin(); vertex_it != segment_it->second.end(); vertex_it++)
          out_str << vertex_it->at(0) << ' ' << vertex_it->at(1) << ' ' << vertex_it->at(2) << ' ' << r << ' ' << g << ' ' << b << " 10" << endl;
      }
      for (int i = 0; i < segment_fitted_pixels.size(); i++) {
        out_str << "3 " << num_fitted_pixels + i * 4 + 0 << ' ' << num_fitted_pixels + i * 4 + 1 << ' ' << num_fitted_pixels + i * 4 + 2 << endl;
        out_str << "3 " << num_fitted_pixels + i * 4 + 0 << ' ' << num_fitted_pixels + i * 4 + 2 << ' ' << num_fitted_pixels + i * 4 + 3 << endl;
      }
      out_str.close();
    }
  }
  
  
  //write mesh .ply files
  bool write_mesh_ply_files = true;
  if (write_mesh_ply_files == true) {
    map<int, int> layer_color_table;
    //  layer_alpha_map[layer_index] = 255 - (255 - 30) / (num_layers - 1) * (num_layers - 1 - layer_index);
    layer_color_table[0] = 255 * 256 * 256;
    layer_color_table[1] = 255 * 256;
    layer_color_table[2] = 255;
    layer_color_table[3] = 255 * 256 + 255;
    vector<vector<int> > layer_triangle_pixels_vec(num_layers);
    int num_non_empty_pixels = 0;
    vector<vector<int> > layer_pixel_index_map(num_layers);
    for (int layer_index = 0; layer_index < num_layers; layer_index++) {
      vector<int> surface_ids = layer_surface_ids[layer_index];
      vector<int> triangle_pixels_vec;
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
        int x = pixel % image_width;
        int y = pixel / image_width;
        if (x == image_width - 1 || y == image_height - 1)
          continue;
        vector<int> cell_pixels;
        if (surface_ids[pixel] < solution_num_surfaces)
          cell_pixels.push_back(pixel);
        if (surface_ids[pixel + 1] < solution_num_surfaces)
          cell_pixels.push_back(pixel + 1);
        if (surface_ids[pixel + 1 + image_width] < solution_num_surfaces)
          cell_pixels.push_back(pixel + 1 + image_width);
        if (surface_ids[pixel + image_width] < solution_num_surfaces)
          cell_pixels.push_back(pixel + image_width);
	
        if (cell_pixels.size() == 3)
          triangle_pixels_vec.insert(triangle_pixels_vec.end(), cell_pixels.begin(), cell_pixels.end());
        if (cell_pixels.size() == 4) {
          triangle_pixels_vec.insert(triangle_pixels_vec.end(), cell_pixels.begin(), cell_pixels.begin() + 3);
          triangle_pixels_vec.push_back(cell_pixels[0]);
          triangle_pixels_vec.push_back(cell_pixels[2]);
          triangle_pixels_vec.push_back(cell_pixels[3]);
        }
      }
      layer_triangle_pixels_vec[layer_index] = triangle_pixels_vec;
      
      vector<int> pixel_index_map(NUM_PIXELS, -1);
      for (int pixel = 0; pixel < NUM_PIXELS; pixel++)
        if (surface_ids[pixel] < solution_num_surfaces)
          pixel_index_map[pixel] = num_non_empty_pixels++;
      layer_pixel_index_map[layer_index] = pixel_index_map;
    }
    
    {
      stringstream layer_segment_mesh_filename;
      layer_segment_mesh_filename << "Results/scene_" << scene_index << "/" << "layered_mesh.ply";
      ofstream out_str(layer_segment_mesh_filename.str());
      
      out_str << "ply" << endl;
      out_str << "format ascii 1.0" << endl;
      out_str << "element vertex " << num_non_empty_pixels << endl;
      out_str << "property float x" << endl;
      out_str << "property float y" << endl;
      out_str << "property float z" << endl;
      out_str << "property uchar red" << endl;
      out_str << "property uchar green" << endl;
      out_str << "property uchar blue" << endl;
      //      out_str << "property uchar alpha" << endl;
      int num_triangles = 0;
      for (int layer_index = 0; layer_index < num_layers; layer_index++)
        num_triangles += layer_triangle_pixels_vec[layer_index].size() / 3;
      out_str << "element face " << num_triangles << endl;
      out_str << "property list uchar int vertex_indices" << endl;
      out_str << "end_header" << endl;
      map<int, int> color_table;
      for (int layer_index = 0; layer_index < num_layers; layer_index++) {
        for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
          int segment_id = layer_surface_ids[layer_index][pixel];
          if (segment_id == solution_num_surfaces)
            continue;
          // if (color_table.count(segment_id) == 0)
          //   color_table[segment_id] = rand() % (256 * 256 * 256);
	  
          int x = pixel % image_width;
          int y = pixel / image_width;
          vector<int> neighbor_pixels;
          if (x > 0)
            neighbor_pixels.push_back(pixel - 1);
          if (x < image_width - 1)
            neighbor_pixels.push_back(pixel + 1);
          if (y > 0)
            neighbor_pixels.push_back(pixel - image_width);
          if (y < image_height - 1)
            neighbor_pixels.push_back(pixel + image_width);
          if (x > 0 && y > 0)
            neighbor_pixels.push_back(pixel - 1 - image_width);
          if (x > 0 && y < image_height - 1)
            neighbor_pixels.push_back(pixel - 1 + image_width);
          if (x < image_width - 1 && y > 0)
            neighbor_pixels.push_back(pixel + 1 - image_width);
          if (x < image_width - 1 && y < image_height - 1)
            neighbor_pixels.push_back(pixel + 1 + image_width);
          bool on_boundary = false;
	  for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
            if (layer_surface_ids[layer_index][*neighbor_pixel_it] != segment_id) {
              on_boundary = true;
              break;
            }
          }
	  
          double depth = solution_segments.at(segment_id).getDepth(1.0 * (pixel % image_width) / image_width, 1.0 * (pixel / image_width) / image_height);
          double u = pixel % image_width - camera_parameters[1];
          double v = pixel / image_width - camera_parameters[2];
          //double X = -u / camera_parameters[0] * depth;
          //double Y = -v / camera_parameters[0] * depth;
	  double X_Z_ratio = u / camera_parameters[0];
	  double Y_Z_ratio = v / camera_parameters[0];
	  double Z = depth / sqrt(pow(X_Z_ratio, 2) + pow(Y_Z_ratio, 2) + 1);
	  double X = X_Z_ratio * Z;
	  double Y = Y_Z_ratio * Z;
          
          // double X = -point_cloud[pixel * 3 + 0];
          // double Y = -point_cloud[pixel * 3 + 1];
          // double Z = point_cloud[pixel * 3 + 2];
          //int color = on_boundary ? 255 * 256 * 256 + 255 * 256 + 255 : layer_color_table[layer_index];
          int color = layer_color_table[layer_index];
          //      out_str << X << ' ' << Y << ' ' << depth << endl;
          out_str << X << ' ' << Y << ' ' << Z << ' ' << color / (256 * 256) << ' ' << color / 256 % 256 << ' ' << color % 256 << endl;
        }
      }
      
      for (int layer_index = 0; layer_index < num_layers; layer_index++) { 
        for (int triangle_index = 0; triangle_index < layer_triangle_pixels_vec[layer_index].size() / 3; triangle_index++) {
          out_str << "3";
          for (int c = 0; c < 3; c++) {
            int pixel = layer_triangle_pixels_vec[layer_index][triangle_index * 3 + c];
            int pixel_index = layer_pixel_index_map[layer_index][pixel];
	    
            out_str << ' ' << pixel_index;
          }
          out_str << endl;
        }
      }
      out_str.close();
    }
  }
}

void LayerDepthRepresenter::writeRenderingInfo(const vector<int> &solution_labels, const int solution_num_surfaces, const map<int, Segment> &solution_segments)
{
  const int ORI_IMAGE_WIDTH = ori_image_.cols;
  const int ORI_IMAGE_HEIGHT = ori_image_.rows;
  const int ORI_NUM_PIXELS = ORI_IMAGE_WIDTH * ORI_IMAGE_HEIGHT;
  
  
  // stringstream upsampled_solution_labels_filename;
  // upsampled_solution_labels_filename << "Cache/scene_" << SCENE_INDEX_ << "/upsampled_solution_labels.txt";
  // ifstream upsampled_solution_labels_in_str(upsampled_solution_labels_filename.str());
  // if (upsampled_solution_labels_in_str && false) {
  //   int num_labels;
  //   upsampled_solution_labels_in_str >> num_labels;
  //   new_solution_labels.assign(num_labels, 0);
  //   for (int label_index = 0; label_index < num_labels; label_index++)
  //     upsampled_solution_labels_in_str >> new_solution_labels[label_index];
  //   upsampled_solution_labels_in_str.close();
  // } else {
  
  
  
  // vector<bool> segment_mask(ORI_NUM_PIXELS, false);
  // for (int pixel = 0; pixel < ORI_NUM_PIXELS; pixel++) {
  //   int solution_label = new_solution_labels[pixel];
  //   for (int layer_index = 0; layer_index < num_layers_; layer_index++) {
  //     int surface_id = solution_label / static_cast<int>(pow(solution_num_surfaces + 1, num_layers_ - 1 - layer_index)) % (solution_num_surfaces + 1);
  //     if (surface_id == 4) {
  //       segment_mask[pixel] = true;
  //     }
  //   }
  // }
  // ImageMask mask(segment_mask, ORI_IMAGE_WIDTH, ORI_IMAGE_HEIGHT);
  // imwrite("Test/mask_image.bmp", mask.drawMaskImage());
  // exit(1);
  
  
  vector<int> new_solution_labels;
  int new_solution_num_surfaces;
  map<int, Segment> new_solution_segments;
  upsampleSolution(solution_labels, solution_num_surfaces, solution_segments, new_solution_labels, new_solution_num_surfaces, new_solution_segments);
  
  // vector<int> new_solution_labels = solution_labels;
  // int new_solution_num_surfaces = solution_num_surfaces;
  // map<int, Segment> new_solution_segments = solution_segments;
  
  for (int layer_index = 0; layer_index < num_layers_; layer_index++) {
    
    stringstream layer_depth_values_filename;
    layer_depth_values_filename << "Results/scene_" << SCENE_INDEX_ << "/" << "depth_values_" << layer_index;
    vector<double> depths((ORI_IMAGE_WIDTH + 1) * (ORI_IMAGE_HEIGHT + 1), 0);
    vector<int> counter((ORI_IMAGE_WIDTH + 1) * (ORI_IMAGE_HEIGHT + 1), 0);
    for (int pixel = 0; pixel < ORI_IMAGE_WIDTH * ORI_IMAGE_HEIGHT; pixel++) {
      int surface_id = new_solution_labels[pixel] / static_cast<int>(pow(new_solution_num_surfaces + 1, num_layers_ - 1 - layer_index)) % (new_solution_num_surfaces + 1);
      if (surface_id < new_solution_num_surfaces) {
	vector<int> corner_pixels(4);
	corner_pixels[0] = (pixel / ORI_IMAGE_WIDTH) * (ORI_IMAGE_WIDTH + 1) + (pixel % ORI_IMAGE_WIDTH);
	corner_pixels[1] = (pixel / ORI_IMAGE_WIDTH + 1) * (ORI_IMAGE_WIDTH + 1) + (pixel % ORI_IMAGE_WIDTH);
	corner_pixels[2] = (pixel / ORI_IMAGE_WIDTH) * (ORI_IMAGE_WIDTH + 1) + (pixel % ORI_IMAGE_WIDTH + 1);
	corner_pixels[3] = (pixel / ORI_IMAGE_WIDTH + 1) * (ORI_IMAGE_WIDTH + 1) + (pixel % ORI_IMAGE_WIDTH + 1);
	for (vector<int>::const_iterator corner_pixel_it = corner_pixels.begin(); corner_pixel_it != corner_pixels.end(); corner_pixel_it++) {
	  //      double depth = new_solution_segments.at(surface_id).getDepth(1.0 * (*corner_pixel_it % (ORI_IMAGE_WIDTH + 1)) / (ORI_IMAGE_WIDTH + 1), 1.0 * (*corner_pixel_it / (ORI_IMAGE_WIDTH + 1)) / (ORI_IMAGE_HEIGHT + 1));
	  double depth = solution_segments.at(surface_id).getDepth(1.0 * (*corner_pixel_it % (ORI_IMAGE_WIDTH + 1)) / (ORI_IMAGE_WIDTH + 1), 1.0 * (*corner_pixel_it / (ORI_IMAGE_WIDTH + 1)) / (ORI_IMAGE_HEIGHT + 1));
          if (depth < 10) {
	    depths[*corner_pixel_it] += depth;
	    counter[*corner_pixel_it]++;
	  }
	}
      }
    }
    for (int pixel = 0; pixel < (ORI_IMAGE_WIDTH + 1) * (ORI_IMAGE_HEIGHT + 1); pixel++) {
      if (counter[pixel] == 0)
	depths[pixel] = -1;
      else
	depths[pixel] /= counter[pixel];
    }
    
    // layer_depth_values_filename.fill('0');
    // layer_depth_values_filename.width(3);
    // layer_depth_values_filename << SCENE_INDEX_;
    // layer_depth_values_filename.width(1);
    // layer_depth_values_filename << "_" << layer_index;
    
    // int new_depth_width = ORI_I + 1;
    // int new_depth_height = ori_image_.rows + 1;
    // vector<int> sub_sampled_surface_ids;
    // vector<double> sub_sampled_depths;
    // sub_sampled_depths = subSampleDepthMap(depths, ORI_IMAGE_WIDTH, ORI_IMAGE_HEIGHT, ORI_IMAGE_WIDTH + 1, ORI_IMAGE_HEIGHT + 1);
    
    //    Mat test_image(ORI_IMAGE_HEIGHT + 1, ORI_IMAGE_WIDTH + 1, CV_8UC1);
    
    ofstream depth_values_out_str(layer_depth_values_filename.str());
    depth_values_out_str << ORI_IMAGE_WIDTH + 1 << ' ' << ORI_IMAGE_HEIGHT + 1 << endl;
    for (int pixel = 0; pixel < depths.size(); pixel++) {
      double depth = depths[pixel];
      depth_values_out_str << depth << endl;
      // if (depth > 0)
      //      test_image.at<uchar>(pixel / (ORI_IMAGE_WIDTH + 1), pixel % (ORI_IMAGE_WIDTH + 1)) = 255;
      // else
      //      test_image.at<uchar>(pixel / (ORI_IMAGE_WIDTH + 1), pixel % (ORI_IMAGE_WIDTH + 1)) = 0;
    }
    // imwrite("Test/depth_mask.bmp", test_image);
    // exit(1);
    depth_values_out_str.close();
  }
  
  vector<vector<int> > layer_surface_ids(num_layers_, vector<int>(ORI_NUM_PIXELS, 0));
  vector<vector<int> > layer_visible_pixels(num_layers_);
  for (int pixel = 0; pixel < ORI_NUM_PIXELS; pixel++) {
    int label = new_solution_labels[pixel];
    bool is_visible = true;
    for (int layer_index = 0; layer_index < num_layers_; layer_index++) {
      int surface_id = label / static_cast<int>(pow(new_solution_num_surfaces + 1, num_layers_ - 1 - layer_index)) % (new_solution_num_surfaces + 1);
      layer_surface_ids[layer_index][pixel] = surface_id;
      if (is_visible && surface_id < new_solution_num_surfaces) {
	layer_visible_pixels[layer_index].push_back(pixel);
	is_visible = false;
      }
    }
  }
  
  // bool dilate_surface_ids = true;
  // bool dilate_ori_surface_ids = false;
  // if (dilate_surface_ids == true) {
  //   for (int iteration = 0; iteration < 1; iteration++) {
  //     for (int layer_index = 0; layer_index < num_layers_; layer_index++) {
  // 	vector<int> surface_ids = layer_surface_ids[layer_index];
  // 	vector<int> new_surface_ids = surface_ids;
  // 	for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  // 	  if (surface_ids[pixel] != solution_num_surfaces)
  // 	    continue;
  // 	  int x = pixel % IMAGE_WIDTH_;
  // 	  int y = pixel / IMAGE_WIDTH_;
  // 	  vector<int> neighbor_pixels;
  // 	  if (x > 0)
  // 	    neighbor_pixels.push_back(pixel - 1);
  // 	  if (x < IMAGE_WIDTH_ - 1)
  // 	    neighbor_pixels.push_back(pixel + 1);
  // 	  if (y > 0)
  // 	    neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
  // 	  if (y < IMAGE_HEIGHT_ - 1)
  // 	    neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
  // 	  if (x > 0 && y > 0)
  // 	    neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
  // 	  if (x > 0 && y < IMAGE_HEIGHT_ - 1)
  // 	    neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
  // 	  if (x < IMAGE_WIDTH_ - 1 && y > 0)
  // 	    neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
  // 	  if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
  // 	    neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
  // 	  map<int, int> surface_id_counter;
  // 	  for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++)
  // 	    surface_id_counter[surface_ids[*neighbor_pixel_it]]++;
  // 	  int max_count = 0;
  // 	  int max_count_surface_id = -1;
  // 	  for (map<int, int>::const_iterator surface_it = surface_id_counter.begin(); surface_it != surface_id_counter.end(); surface_it++) {
  // 	    if (surface_it->second > max_count && surface_it->first != solution_num_surfaces) {
  // 	      max_count_surface_id = surface_it->first;
  // 	      max_count = surface_it->second;
  // 	    }
  // 	  }
  // 	  if (max_count_surface_id != -1)
  // 	    new_surface_ids[pixel] = max_count_surface_id;
  // 	}
  // 	layer_surface_ids[layer_index] = new_surface_ids;
  //     }
  //   }
  // }
  
  vector<vector<int> > segment_pixels_vec(new_solution_num_surfaces);
  vector<vector<int> > hole_pixels_vec(new_solution_num_surfaces);
  for (int pixel = 0; pixel < ORI_NUM_PIXELS; pixel++) {
    //int x = pixel % ORI_IMAGE_WIDTH;
    //int y = pixel / ORI_IMAGE_WIDTH;
    //int label = solution_labels[pixel];
    
    int visible_layer_index = -1;
    for (int layer_index = 0; layer_index < num_layers_; layer_index++) {
      //      int surface_id = label / static_cast<int>(pow(solution_num_surfaces + 1, num_layers_ - 1 - layer_index)) % (solution_num_surfaces + 1);
      int surface_id = layer_surface_ids[layer_index][pixel];
      if (surface_id == new_solution_num_surfaces)
        continue;
      segment_pixels_vec[surface_id].push_back(pixel);
      if (visible_layer_index == -1)
        visible_layer_index = layer_index;
      else
	hole_pixels_vec[surface_id].push_back(pixel);
    }
  }
  
  // cout << layer_surface_ids[0][376 * ORI_IMAGE_WIDTH + 146] << endl;
  // cout << layer_surface_ids[0][377 * ORI_IMAGE_WIDTH + 146] << endl;
  // cout << layer_surface_ids[0][376 * ORI_IMAGE_WIDTH + 145] << endl;
  // cout << layer_surface_ids[0][377 * ORI_IMAGE_WIDTH + 145] << endl;
  // exit(1);
  
  // Mat blurred_image;
  // GaussianBlur(ori_image_, blurred_image, cv::Size(3, 3), 0, 0);
  // Mat blurred_hsv_image;
  // blurred_image.convertTo(blurred_hsv_image, CV_32FC3, 1.0 / 255);
  // cvtColor(blurred_hsv_image, blurred_hsv_image, CV_BGR2HSV);
  
  const int NUM_EROSION_ITERATIONS = 2;
  const double COLOR_LIKELIHOOD_THRESHOLD = 1; //STATISTICS_.fitting_color_likelihood_threshold;
  for (int segment_id = 0; segment_id < new_solution_num_surfaces; segment_id++) {
    if (hole_pixels_vec[segment_id].size() == 0)
      continue;
    vector<bool> known_region_mask(ORI_NUM_PIXELS, false);
    for (vector<int>::const_iterator pixel_it = segment_pixels_vec[segment_id].begin(); pixel_it != segment_pixels_vec[segment_id].end(); pixel_it++)
      known_region_mask[*pixel_it] = true;
    for (vector<int>::const_iterator pixel_it = hole_pixels_vec[segment_id].begin(); pixel_it != hole_pixels_vec[segment_id].end(); pixel_it++)
      known_region_mask[*pixel_it] = false;    
    
    for (int iteration = 0; iteration < NUM_EROSION_ITERATIONS; iteration++) {
      //cout << iteration << endl;
      vector<int> known_region_pixels;
      for (int pixel = 0; pixel < ORI_NUM_PIXELS; pixel++)
        if (known_region_mask[pixel] == true)
          known_region_pixels.push_back(pixel);
      
      set<int> new_hole_pixel_indices;
      for (vector<int>::const_iterator pixel_it = known_region_pixels.begin(); pixel_it != known_region_pixels.end(); pixel_it++) {
        
        int pixel = *pixel_it;
	vector<int> neighbor_pixels;
	int x = pixel % ORI_IMAGE_WIDTH;
	int y = pixel / ORI_IMAGE_WIDTH;
	if (x > 0)
	  neighbor_pixels.push_back(pixel - 1);
	if (x < ORI_IMAGE_WIDTH - 1)
	  neighbor_pixels.push_back(pixel + 1);
	if (y > 0)
	  neighbor_pixels.push_back(pixel - ORI_IMAGE_WIDTH);
	if (y < ORI_IMAGE_HEIGHT - 1)
	  neighbor_pixels.push_back(pixel + ORI_IMAGE_WIDTH);
	if (x > 0 && y > 0)
	  neighbor_pixels.push_back(pixel - 1 - ORI_IMAGE_WIDTH);
	if (x > 0 && y < ORI_IMAGE_HEIGHT - 1)
	  neighbor_pixels.push_back(pixel - 1 + ORI_IMAGE_WIDTH);
	if (x < ORI_IMAGE_WIDTH - 1 && y > 0)
	  neighbor_pixels.push_back(pixel + 1 - ORI_IMAGE_WIDTH);
	if (x < ORI_IMAGE_WIDTH - 1 && y < ORI_IMAGE_HEIGHT - 1)
	  neighbor_pixels.push_back(pixel + 1 + ORI_IMAGE_WIDTH);
	bool on_boundary = false;
	for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	  if (known_region_mask[*neighbor_pixel_it] == false) {
	    on_boundary = true;
	    break;
	  }
	}
	if (on_boundary == false)
	  continue;
	// if (new_solution_segments.at(segment_id).predictColorLikelihood(pixel, blurred_hsv_image.at<Vec3b>(y, x)) < COLOR_LIKELIHOOD_THRESHOLD)
	new_hole_pixel_indices.insert(pixel_it - known_region_pixels.begin());
      }
      if (new_hole_pixel_indices.size() == 0 || new_hole_pixel_indices.size() == segment_pixels_vec[segment_id].size())
	break;
      for (set<int>::const_iterator index_it = new_hole_pixel_indices.begin(); index_it != new_hole_pixel_indices.end(); index_it++) {
	hole_pixels_vec[segment_id].push_back(known_region_pixels[*index_it]);
	known_region_mask[known_region_pixels[*index_it]] = false;
      }
    }
  }
  
  // for (int segment_id = 0; segment_id < solution_num_surfaces; segment_id++) {
  //   if (segment_pixels_vec[segment_id].size() == 0)
  //     continue;
  //   stringstream segment_mask_image_filename;
  //   segment_mask_image_filename << "Test/segment_mask_image_" << segment_id << ".bmp";
  //   Mat segment_mask_image = Mat::zeros(ORI_IMAGE_HEIGHT, ORI_IMAGE_WIDTH, CV_8UC1);
  //   for (vector<int>::const_iterator pixel_it = segment_pixels_vec[segment_id].begin(); pixel_it != segment_pixels_vec[segment_id].end(); pixel_it++)
  //     segment_mask_image.at<uchar>(*pixel_it / ORI_IMAGE_WIDTH, *pixel_it % ORI_IMAGE_WIDTH) = 255;
  //   imwrite(segment_mask_image_filename.str(), segment_mask_image);
  // }
  // for (int segment_id = 0; segment_id < solution_num_surfaces; segment_id++) {
  //   if (hole_pixels_vec[segment_id].size() == 0)
  //     continue;
  //   stringstream hole_mask_image_filename;
  //   hole_mask_image_filename << "Test/hole_mask_image_" << segment_id << ".bmp";
  //   Mat hole_mask_image = Mat::zeros(ORI_IMAGE_HEIGHT, ORI_IMAGE_WIDTH, CV_8UC1);
  //   for (vector<int>::const_iterator pixel_it = hole_pixels_vec[segment_id].begin(); pixel_it != hole_pixels_vec[segment_id].end(); pixel_it++)
  //     hole_mask_image.at<uchar>(*pixel_it / ORI_IMAGE_WIDTH, *pixel_it % ORI_IMAGE_WIDTH) = 255;
  //   imwrite(hole_mask_image_filename.str(), hole_mask_image);
  // }
  // exit(1);
  
  
  // for (map<int, Segment>::iterator segment_it = new_solution_segments.begin(); segment_it != new_solution_segments.end(); segment_it++)
  //   cout << segment_it->first << '\t' << hole_pixels_vec[segment_it->first].size() << '\t' << segment_pixels_vec[segment_it->first].size() << '\t' << segment_it->second.getType() << endl;
  
  map<int, Mat> completed_images;
  vector<double> upsampled_camera_parameters(3);
  estimateCameraParameters(ori_point_cloud_, ORI_IMAGE_WIDTH, ORI_IMAGE_HEIGHT, upsampled_camera_parameters, USE_PANORAMA_);
  //PatchMatcher patch_matcher(ori_image_);
  for (int segment_id = 0; segment_id < new_solution_num_surfaces; segment_id++) {
    //cout << segment_id << '\t' << hole_pixels_vec[segment_id].size() << '\t' << segment_pixels_vec[segment_id].size() << '\t' << new_solution_segments[segment_id].getType() << endl;    
    // if (segment_id != 1)
    //   continue;
    
    stringstream completed_image_filename;
    completed_image_filename << "Cache/scene_" << SCENE_INDEX_ << "/completed_image_" << segment_id << ".bmp";
    if (imread(completed_image_filename.str()).empty() || true) {
      if (segment_pixels_vec[segment_id].size() == 0)
	continue;
      //if (segment_id != 6)
      //continue;
      cout << "inpaint segment: " << segment_id << endl;
      Mat mask_image = Mat::ones(ori_image_.rows, ori_image_.cols, CV_8UC1) * 255;
      
      vector<bool> source_mask(ori_image_.rows * ori_image_.cols, false);
      vector<bool> target_mask(ori_image_.rows * ori_image_.cols, false);
      vector<bool> invalid_source_mask(ori_image_.rows * ori_image_.cols, false);
      for (vector<int>::const_iterator pixel_it = segment_pixels_vec[segment_id].begin(); pixel_it != segment_pixels_vec[segment_id].end(); pixel_it++) {
	target_mask[*pixel_it] = true;
	if (checkPointValidity(getPoint(ori_point_cloud_, *pixel_it)))
	  source_mask[*pixel_it] = true;
	else
	  invalid_source_mask[*pixel_it] = true;
      }
      for (vector<int>::const_iterator pixel_it = hole_pixels_vec[segment_id].begin(); pixel_it != hole_pixels_vec[segment_id].end(); pixel_it++) {
	source_mask[*pixel_it] = false;
	invalid_source_mask[*pixel_it] = false;
      }

      imwrite("Test/source_mask.bmp", ImageMask(source_mask, ori_image_.cols, ori_image_.rows).drawImageWithMask(ori_image_, false, Vec3b(0, 0, 0), true));
      imwrite("Test/target_mask.bmp", ImageMask(target_mask, ori_image_.cols, ori_image_.rows).drawImageWithMask(ori_image_, false, Vec3b(0, 0, 0), true));
      //imwrite("Test/invalid_source_mask.bmp", ImageMask(invalid_source_mask, ori_image_.cols, ori_image_.rows).drawImageWithMask(ori_image_, false, Vec3b(0, 0, 0), true));
      //if (segment_id == 3)
      //exit(1);
      
      // imwrite("Test/image_for_completion.bmp", ori_image_);
      // ofstream source_mask_out_str("Test/source_mask");
      // source_mask_out_str << ImageMask(source_mask, ori_image_.cols, ori_image_.rows);
      // source_mask_out_str.close();
      // ofstream target_mask_out_str("Test/target_mask");
      // target_mask_out_str << ImageMask(target_mask, ori_image_.cols, ori_image_.rows);
      // target_mask_out_str.close();
      //      exit(1);

      // MatrixXd unwarp_transform = solution_segments.at(segment_id).getUnwarpTransform(ori_point_cloud_, upsampled_camera_parameters);
      // int min_x = 1000000, max_x = -1000000, min_y = 1000000, max_y = -1000000;
      // for (int pixel = 0; pixel < ori_image_.cols * ori_image_.rows; pixel++) {
      //   if (target_mask[pixel] == false)
      //     continue;
      //   Vector3d pixel_vec;
      //   pixel_vec << pixel % ori_image_.cols, pixel / ori_image_.cols, 1;
      //   Vector3d unwarped_pixel_vec = unwarp_transform * pixel_vec;
      //   if (unwarped_pixel_vec[2] == 0)
      //     continue;
      //   int x = round(unwarped_pixel_vec[0] / unwarped_pixel_vec[2]);
      //   int y = round(unwarped_pixel_vec[1] / unwarped_pixel_vec[2]);
      //   if (x < min_x)
      //     min_x = x;
      //   if (x > max_x)
      //     max_x = x;
      //   if (y < min_y)
      //     min_y = y;
      //   if (y > max_y)
      //     max_y = y;
      // }
      // cout << min_x << '\t' << max_x << '\t' << min_y << '\t' << max_y << endl;
      // exit(1);
      
      Mat completed_image = completeImage(ori_image_, source_mask, target_mask, 5, solution_segments.at(segment_id).getUnwarpTransform(ori_point_cloud_, upsampled_camera_parameters));
      
      for (int pixel = 0; pixel < ori_image_.cols * ori_image_.rows; pixel++)
        if (invalid_source_mask[pixel])
          completed_image.at<Vec3b>(pixel / ori_image_.cols, pixel % ori_image_.cols) = ori_image_.at<Vec3b>(pixel / ori_image_.cols, pixel % ori_image_.cols);

      imwrite(completed_image_filename.str(), completed_image);
      completed_images[segment_id] = completed_image.clone();      
    } else
      completed_images[segment_id] = imread(completed_image_filename.str());
  }
  
  vector<Mat> texture_images(num_layers_);
  vector<Mat> static_texture_images(num_layers_);
  for (int layer_index = 0; layer_index < num_layers_; layer_index++) {
    Mat texture_image = Mat::zeros(ori_image_.size(), CV_8UC3);
    vector<int> surface_ids = layer_surface_ids[layer_index];
    for (int pixel = 0; pixel < ORI_IMAGE_WIDTH * ORI_IMAGE_HEIGHT; pixel++) {
      int x = pixel % ORI_IMAGE_WIDTH;
      int y = pixel / ORI_IMAGE_WIDTH;
      int surface_id = surface_ids[pixel];
      //cout << pixel << '\t' << scaled_pixel << '\t' << surface_id << endl;
      if (surface_id < new_solution_num_surfaces)
	texture_image.at<Vec3b>(y, x) = completed_images[surface_id].at<Vec3b>(y, x);
      else {
	// texture_image.at<Vec3b>(pixel / ORI_IMAGE_WIDTH, pixel % ORI_IMAGE_WIDTH) = Vec3b(255, 0, 0);
	// continue;
	
	vector<int> neighbor_pixels;
	if (x > 0)
	  neighbor_pixels.push_back(pixel - 1);
	if (x < ORI_IMAGE_WIDTH - 1)
	  neighbor_pixels.push_back(pixel + 1);
	if (y > 0)
	  neighbor_pixels.push_back(pixel - ORI_IMAGE_WIDTH);
	if (y < ORI_IMAGE_HEIGHT - 1)
	  neighbor_pixels.push_back(pixel + ORI_IMAGE_WIDTH);
	if (x > 0 && y > 0)
	  neighbor_pixels.push_back(pixel - 1 - ORI_IMAGE_WIDTH);
	if (x > 0 && y < ORI_IMAGE_HEIGHT - 1)
	  neighbor_pixels.push_back(pixel - 1 + ORI_IMAGE_WIDTH);
	if (x < ORI_IMAGE_WIDTH - 1 && y > 0)
	  neighbor_pixels.push_back(pixel + 1 - ORI_IMAGE_WIDTH);
	if (x < ORI_IMAGE_WIDTH - 1 && y < ORI_IMAGE_HEIGHT - 1)
	  neighbor_pixels.push_back(pixel + 1 + ORI_IMAGE_WIDTH);
	int num_valid_colors = 0;
	double b_sum = 0;
	double g_sum = 0;
	double r_sum = 0;
	for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	  if (surface_ids[*neighbor_pixel_it] != new_solution_num_surfaces) {
	    Vec3b color = completed_images[surface_ids[*neighbor_pixel_it]].at<Vec3b>(*neighbor_pixel_it / ORI_IMAGE_WIDTH, *neighbor_pixel_it % ORI_IMAGE_WIDTH);
	    b_sum += color[0];
	    g_sum += color[1];
	    r_sum += color[2];
            //color_sum += Vec3b(255, 255, 255);
	    num_valid_colors++;
	  }
	}
	if (num_valid_colors == 0)
	  texture_image.at<Vec3b>(pixel / ORI_IMAGE_WIDTH, pixel % ORI_IMAGE_WIDTH) = Vec3b(255, 0, 0);
	else
	  texture_image.at<Vec3b>(pixel / ORI_IMAGE_WIDTH, pixel % ORI_IMAGE_WIDTH) = Vec3b(b_sum / num_valid_colors, g_sum / num_valid_colors, r_sum / num_valid_colors);
      }
    }

    stringstream texture_image_filename;
    texture_image_filename << "Results/scene_" << SCENE_INDEX_ << "/texture_image_" << layer_index << ".bmp";
    imwrite(texture_image_filename.str().c_str(), texture_image);
    texture_images[layer_index] = texture_image.clone();
    
    vector<int> visible_pixels = layer_visible_pixels[layer_index];
    for (vector<int>::const_iterator pixel_it = visible_pixels.begin(); pixel_it != visible_pixels.end(); pixel_it++) {
      //break;
      texture_image.at<Vec3b>(*pixel_it / ORI_IMAGE_WIDTH, *pixel_it % ORI_IMAGE_WIDTH) = ori_image_.at<Vec3b>(*pixel_it / ORI_IMAGE_WIDTH, *pixel_it % ORI_IMAGE_WIDTH);
    }
    stringstream static_texture_image_filename;
    static_texture_image_filename << "Results/scene_" << SCENE_INDEX_ << "/static_texture_image_" << layer_index << ".bmp";
    imwrite(static_texture_image_filename.str().c_str(), texture_image);

    static_texture_images[layer_index] = texture_image.clone();
  }

  Mat object_removal_image = Mat::zeros(ORI_IMAGE_HEIGHT, ORI_IMAGE_WIDTH, CV_8UC3);
  set<int> removed_segments;
  // removed_segments.insert(13);
  // removed_segments.insert(17);
  // removed_segments.insert(20);
  // removed_segments.insert(21);
  // removed_segments.insert(23);
  for (int ori_pixel = 0; ori_pixel < ORI_IMAGE_WIDTH * ORI_IMAGE_HEIGHT; ori_pixel++) {
    for (int layer_index = 0; layer_index < num_layers_; layer_index++) {    
      int surface_id = new_solution_labels[ori_pixel] / static_cast<int>(pow(new_solution_num_surfaces + 1, num_layers_ - 1 - layer_index)) % (new_solution_num_surfaces + 1);
      if (surface_id < solution_num_surfaces && layer_index < 1)
        removed_segments.insert(surface_id);
    }
  }

  vector<set<int> > pixel_segment_indices_map(ORI_NUM_PIXELS);
  
  for (int ori_pixel = 0; ori_pixel < ORI_IMAGE_WIDTH * ORI_IMAGE_HEIGHT; ori_pixel++) {
    for (int layer_index = 0; layer_index < num_layers_; layer_index++) {    
      int surface_id = new_solution_labels[ori_pixel] / static_cast<int>(pow(new_solution_num_surfaces + 1, num_layers_ - 1 - layer_index)) % (new_solution_num_surfaces + 1);
      if (removed_segments.count(surface_id) > 0)
	pixel_segment_indices_map[ori_pixel].insert(surface_id);
    }
  }
  
  const int NUM_DILATION_ITERATIONS = NUM_EROSION_ITERATIONS;
  
  for (int iteration = 0; iteration < NUM_DILATION_ITERATIONS; iteration++) {
    vector<set<int> > dilated_pixel_segment_indices_map = pixel_segment_indices_map;
    for (int ori_pixel = 0; ori_pixel < ORI_NUM_PIXELS; ori_pixel++) {
      // if (pixel_segment_indices_map[pixel].size() == 0)
      //   continue;
      vector<int> neighbor_pixels;
      int x = ori_pixel % ORI_IMAGE_WIDTH;
      int y = ori_pixel / ORI_IMAGE_WIDTH;
      if (x > 0)
	neighbor_pixels.push_back(ori_pixel - 1);
      if (x < ORI_IMAGE_WIDTH - 1)
	neighbor_pixels.push_back(ori_pixel + 1);
      if (y > 0)
	neighbor_pixels.push_back(ori_pixel - ORI_IMAGE_WIDTH);
      if (y < ORI_IMAGE_HEIGHT - 1)
	neighbor_pixels.push_back(ori_pixel + ORI_IMAGE_WIDTH);
      if (x > 0 && y > 0)
	neighbor_pixels.push_back(ori_pixel - 1 - ORI_IMAGE_WIDTH);
      if (x > 0 && y < ORI_IMAGE_HEIGHT - 1)
	neighbor_pixels.push_back(ori_pixel - 1 + ORI_IMAGE_WIDTH);
      if (x < ORI_IMAGE_WIDTH - 1 && y > 0)
	neighbor_pixels.push_back(ori_pixel + 1 - ORI_IMAGE_WIDTH);
      if (x < ORI_IMAGE_WIDTH - 1 && y < ORI_IMAGE_HEIGHT - 1)
	neighbor_pixels.push_back(ori_pixel + 1 + ORI_IMAGE_WIDTH);
      for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	// if (pixel_segment_indices_map[*neighbor_pixel_it].size() > 0)
	//   continue;
	for (set<int>::const_iterator segment_it = pixel_segment_indices_map[ori_pixel].begin(); segment_it != pixel_segment_indices_map[ori_pixel].end(); segment_it++) {
	  if (dilated_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
	    continue;
	  dilated_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
	}
      }
    }
    pixel_segment_indices_map = dilated_pixel_segment_indices_map;
  }

  
  for (int ori_pixel = 0; ori_pixel < ORI_IMAGE_WIDTH * ORI_IMAGE_HEIGHT; ori_pixel++) {
    for (int layer_index = 0; layer_index < num_layers_; layer_index++) {    
      int surface_id = new_solution_labels[ori_pixel] / static_cast<int>(pow(new_solution_num_surfaces + 1, num_layers_ - 1 - layer_index)) % (new_solution_num_surfaces + 1);
      if (removed_segments.count(surface_id) > 0)
	continue;
      if (surface_id < new_solution_num_surfaces) {
	if (pixel_segment_indices_map[ori_pixel].size() == 0)
	  object_removal_image.at<Vec3b>(ori_pixel / ORI_IMAGE_WIDTH, ori_pixel % ORI_IMAGE_WIDTH) = static_texture_images[layer_index].at<Vec3b>(ori_pixel / ORI_IMAGE_WIDTH, ori_pixel % ORI_IMAGE_WIDTH);
        else
	  object_removal_image.at<Vec3b>(ori_pixel / ORI_IMAGE_WIDTH, ori_pixel % ORI_IMAGE_WIDTH) = texture_images[layer_index].at<Vec3b>(ori_pixel / ORI_IMAGE_WIDTH, ori_pixel % ORI_IMAGE_WIDTH);
	break;
      }
    }
  }
  stringstream object_removal_image_filename;
  object_removal_image_filename << "Results/scene_" << SCENE_INDEX_ << "/" << "object_removal_image.bmp";
  imwrite(object_removal_image_filename.str(), object_removal_image);

  
  stringstream rendering_info_filename;
  rendering_info_filename << "Results/scene_" << SCENE_INDEX_ << "/" << "rendering_info";
  // camera_parameters_filename.fill('0');
  // camera_parameters_filename.width(3);
  // camera_parameters_filename << SCENE_INDEX_;
  ofstream rendering_info_out_str(rendering_info_filename.str());
  rendering_info_out_str << 512 << endl;
  rendering_info_out_str << ori_image_.cols << '\t' << ori_image_.rows << endl;
  rendering_info_out_str << num_layers_ << endl;
  rendering_info_out_str.close();
}

vector<double> LayerDepthRepresenter::subSampleDepthMap(const vector<double> &depths, const int ori_width, const int ori_height, const int new_width, const int new_height)
{
  const int NUM_PIXELS = new_width * new_height;
  vector<double> new_depth_values(NUM_PIXELS);

  for (int y = 0; y < new_height; ++y) {
    for (int x = 0; x < new_width; ++x) {
      double ori_x = 1.0 * x * ori_width / new_width;
      double ori_y = 1.0 * y * ori_height / new_height;
      new_depth_values[y * new_width + x] = interpolateDepthValue(depths, ori_width, ori_height, ori_x, ori_y);
      // if (ori_x > 154 && ori_x < 156 && ori_y > 11 && ori_y < 13)
      //        cout << new_depth_values[y * new_width + x] << endl;
    }
  }
  //exit(1);
  return new_depth_values;
}

// bool LayerDepthRepresenter::readLayers(vector<int> &solution, int &solution_num_surfaces, map<int, Segment> &solution_segments, const int iteration)
// {
//   if (FIRST_TIME_)
//     return false;

//   stringstream segments_filename;
//   segments_filename << "Cache/scene_" << SCENE_INDEX_ << "/segments_" << iteration << ".txt";
//   ifstream segments_in_str(segments_filename.str());
//   if (!segments_in_str)
//     return false;
  
//   segments_in_str >> solution_num_surfaces;
//   for (int i = 0; i < solution_num_surfaces; i++) {
//     int segment_id;
//     segments_in_str >> segment_id;
//     assert(segment_id == i);
//     Segment segment(IMAGE_WIDTH_, IMAGE_HEIGHT_, camera_parameters_, PENALTIES_, STATISTICS_);
//     segments_in_str >> segment;
//     //cout << segment << endl;
//     solution_segments[i] = segment;

//     //cout << segment_id << '\t' << segment.getConfidence() << endl;
//   }
//   segments_in_str.close();
//   //exit(1);
//   if (solution_num_surfaces == 0)
//     return false;

//   stringstream segment_GMMs_filename;
//   segment_GMMs_filename << "Cache/scene_" << SCENE_INDEX_ << "/segment_GMMs_" << iteration << ".xml";
//   FileStorage segment_GMMs_fs(segment_GMMs_filename.str(), FileStorage::READ);
//   for (map<int, Segment>::iterator surface_it = solution_segments.begin(); surface_it != solution_segments.end(); surface_it++) {
//     stringstream segment_name;
//     segment_name << "Segment" << surface_it->first;
//     FileNode segment_GMM_file_node = segment_GMMs_fs[segment_name.str()];
//     surface_it->second.setGMM(segment_GMM_file_node);

//     //cout << surface_it->second.predictColorLikelihood(static_cast<int>(6.5320558718089686e+01 + 0.5) * IMAGE_WIDTH_ + static_cast<int>(3.6378882105214679e+01 + 0.5), Vec3b(2.1218500883957713e+01, 1.8951972423041440e+01, 3.3314066835337023e+01)) << endl;
      
//     //exit(1);
//   }
//   segment_GMMs_fs.release();

  
//   solution = vector<int>(NUM_PIXELS_, 0);
//   for (int layer_index = 0; layer_index < num_layers_; layer_index++) {
//     stringstream layer_image_filename;
//     layer_image_filename << "Cache/scene_" << SCENE_INDEX_ << "/" << "layer_image_raw_" << iteration << "_" << layer_index << ".bmp";
//     Mat layer_image = imread(layer_image_filename.str().c_str(), 0);
//     if (layer_image.empty())
//       return false;
    
//     for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
//       int x = pixel % IMAGE_WIDTH_;
//       int y = pixel / IMAGE_WIDTH_;

//       int surface_id = layer_image.at<uchar>(y, x);
//       solution[pixel] += surface_id * pow(solution_num_surfaces + 1, num_layers_ - 1 - layer_index);
//     }
//   }
//   return true;
// }

bool readLayers(const int image_width, const int image_height, const vector<double> &camera_parameters, const RepresenterPenalties &penalties, const DataStatistics &statistics, const int num_layers, vector<int> &solution, int &solution_num_surfaces, map<int, Segment> &solution_segments, const int scene_index, const int result_index, const bool use_panorama)
{
  // if (FIRST_TIME_)
  //   return false;
  
  const int NUM_PIXELS = image_width * image_height;
  stringstream segments_filename;
  segments_filename << "Cache/scene_" << scene_index << "/segments_" << result_index << ".txt";
  ifstream segments_in_str(segments_filename.str());
  if (!segments_in_str)
    return false;
  
  segments_in_str >> solution_num_surfaces;
  for (int i = 0; i < solution_num_surfaces; i++) {
    int segment_id;
    segments_in_str >> segment_id;
    assert(segment_id == i);
    Segment segment(image_width, image_height, camera_parameters, statistics, use_panorama);
    segments_in_str >> segment;
    //cout << segment << endl;
    solution_segments[i] = segment;
    
    //cout << segment_id << '\t' << segment.getConfidence() << endl;
  }
  segments_in_str.close();
  //exit(1);
  if (solution_num_surfaces == 0)
    return false;

  bool use_GMM_models = false;
  if (use_GMM_models) {
    stringstream segment_GMMs_filename;
    segment_GMMs_filename << "Cache/scene_" << scene_index << "/segment_GMMs_" << result_index << ".xml";
    FileStorage segment_GMMs_fs(segment_GMMs_filename.str(), FileStorage::READ);
    for (map<int, Segment>::iterator surface_it = solution_segments.begin(); surface_it != solution_segments.end(); surface_it++) {
      stringstream segment_name;
      segment_name << "Segment" << surface_it->first;
      FileNode segment_GMM_file_node = segment_GMMs_fs[segment_name.str()];
      surface_it->second.setGMM(segment_GMM_file_node);
      
      //cout << surface_it->second.predictColorLikelihood(static_cast<int>(6.5320558718089686e+01 + 0.5) * IMAGE_WIDTH_ + static_cast<int>(3.6378882105214679e+01 + 0.5), Vec3b(2.1218500883957713e+01, 1.8951972423041440e+01, 3.3314066835337023e+01)) << endl;
    
      //exit(1);
    }
    segment_GMMs_fs.release();
  }
  
  solution = vector<int>(NUM_PIXELS, 0);
  for (int layer_index = 0; layer_index < num_layers; layer_index++) {
    stringstream layer_image_filename;
    layer_image_filename << "Cache/scene_" << scene_index << "/" << "layer_image_raw_" << result_index << "_" << layer_index << ".bmp";
    Mat layer_image = imread(layer_image_filename.str().c_str(), 0);
    if (layer_image.empty())
      return false;
    
    for (int pixel = 0; pixel < NUM_PIXELS; pixel++) {
      int x = pixel % image_width;
      int y = pixel / image_width;

      int surface_id = layer_image.at<uchar>(y, x);
      solution[pixel] += surface_id * pow(solution_num_surfaces + 1, num_layers - 1 - layer_index);
    }
  }
  return true;
}

double LayerDepthRepresenter::interpolateDepthValue(const vector<double> &depths, const int depth_width, const int depth_height, const double x, const double y) const
{
  const double INTERPOLATION_GAP_THRESHOLD = 0.99;
  double interpolated_depth_value = -1;
  
  int index_1 = static_cast<int>(y) * depth_width + static_cast<int>(x);
  int index_2 = static_cast<int>(y) * depth_width + static_cast<int>(x + 1);
  int index_3 = static_cast<int>(y + 1) * depth_width + static_cast<int>(x);
  int index_4 = static_cast<int>(y + 1) * depth_width + static_cast<int>(x + 1);
  double depth_value_1 = (x >= 0 && y >= 0) ? depths[index_1] : -1;
  double depth_value_2 = (x < depth_width - 1 && y >= 0) ? depths[index_2] : -1;
  double depth_value_3 = (x >= 0 && y < depth_height - 1) ? depths[index_3] : -1;
  double depth_value_4 = (x < depth_width - 1 && y < depth_height - 1) ? depths[index_4] : -1;

  int num_valid_depth_values = 0;
  if (depth_value_1 > 0)
    num_valid_depth_values++;
  if (depth_value_2 > 0)
    num_valid_depth_values++;
  if (depth_value_3 > 0)
    num_valid_depth_values++;
  if (depth_value_4 > 0)
    num_valid_depth_values++;

  switch (num_valid_depth_values) {
  case 4: {
    double area_1 = (static_cast<int>(x + 1) - x) * (static_cast<int>(y + 1) - y);
    double area_2 = (x - static_cast<int>(x)) * (static_cast<int>(y + 1) - y);
    double area_3 = (static_cast<int>(x + 1) - x) * (y - static_cast<int>(y));
    double area_4 = (x - static_cast<int>(x)) * (y - static_cast<int>(y));
    interpolated_depth_value = depth_value_1 * area_1 + depth_value_2 * area_2 + depth_value_3 * area_3 + depth_value_4 * area_4;
    break;
  }
  case 3: {
    if (depth_value_1 > 0 && depth_value_2 > 0 && depth_value_3 > 0) {
      double area_2 = 0.5 * (x - static_cast<int>(x));
      double area_3 = 0.5 * (y - static_cast<int>(y));
      if (area_2 + area_3 <= INTERPOLATION_GAP_THRESHOLD) {
	double area_1 = 0.5 - area_2 - area_3;
	interpolated_depth_value = depth_value_1 * area_1 + depth_value_2 * area_2 + depth_value_3 * area_3;
	interpolated_depth_value *= 2;
      }
    } else if (depth_value_2 > 0 && depth_value_3 > 0 && depth_value_4 > 0) {
      double area_2 = 0.5 * (static_cast<int>(y + 1) - y);
      double area_3 = 0.5 * (static_cast<int>(x + 1) - x);
      if (area_2 + area_3 <= INTERPOLATION_GAP_THRESHOLD) {
	double area_4 = 0.5 - area_2 - area_3;
	interpolated_depth_value = depth_value_2 * area_2 + depth_value_3 * area_3 + depth_value_4 * area_4;
	interpolated_depth_value *= 2;
      }
    } else if (depth_value_1 > 0 && depth_value_3 > 0 && depth_value_4 > 0) {
      double area_1 = 0.5 * (static_cast<int>(y + 1) - y);
      double area_4 = 0.5 * (x - static_cast<int>(x));
      if (area_1 + area_4 <= INTERPOLATION_GAP_THRESHOLD) {
	double area_3 = 0.5 - area_1 - area_4;
	interpolated_depth_value = depth_value_1 * area_1 + depth_value_3 * area_3 + depth_value_4 * area_4;
	interpolated_depth_value *= 2;
      }
    } else if (depth_value_1 > 0 && depth_value_2 > 0 && depth_value_4 > 0) {
      double area_1 = 0.5 * (static_cast<int>(x + 1) - x);
      double area_4 = 0.5 * (y - static_cast<int>(y));
      if (area_1 + area_4 <= INTERPOLATION_GAP_THRESHOLD) {
	double area_2 = 0.5 - area_1 - area_4;
	interpolated_depth_value = depth_value_1 * area_1 + depth_value_2 * area_2 + depth_value_4 * area_4;
	interpolated_depth_value *= 2;
      }
    }
    break;
  }
  case 2: {
    if (depth_value_1 > 0 && depth_value_2 > 0) {
      if (y - static_cast<int>(y) <= INTERPOLATION_GAP_THRESHOLD) {
	double length_1 = (static_cast<int>(x + 1) - x);
	double length_2 = (x - static_cast<int>(x));
	interpolated_depth_value = depth_value_1 * length_1 + depth_value_2 * length_2;
      }
    } else if (depth_value_1 > 0 && depth_value_3 > 0) {
      if (x - static_cast<int>(x) <= INTERPOLATION_GAP_THRESHOLD) {
	double length_1 = (static_cast<int>(y + 1) - y);
	double length_3 = (y - static_cast<int>(y));
	interpolated_depth_value = depth_value_1 * length_1 + depth_value_3 * length_3;
      }
    } else if (depth_value_2 > 0 && depth_value_4 > 0) {
      if (static_cast<int>(x + 1) - x <= INTERPOLATION_GAP_THRESHOLD) {
	double length_2 = (static_cast<int>(y + 1) - y);
	double length_4 = (y - static_cast<int>(y));
	interpolated_depth_value = depth_value_2 * length_2 + depth_value_4 * length_4;
      }
    } else if (depth_value_3 > 0 && depth_value_4 > 0) {
      if (static_cast<int>(y + 1) - y <= INTERPOLATION_GAP_THRESHOLD) {
	double length_3 = (static_cast<int>(x + 1) - x);
	double length_4 = (x - static_cast<int>(x));
	interpolated_depth_value = depth_value_3 * length_3 + depth_value_4 * length_4;
      }
    } else if (depth_value_1 > 0 && depth_value_4 > 0) {
      if (abs((x - static_cast<int>(x)) - (y - static_cast<int>(y))) <= INTERPOLATION_GAP_THRESHOLD) {
	double length_1 = 1 - ((x - static_cast<int>(x)) + (y - static_cast<int>(y))) / 2;
	double length_4 = ((x - static_cast<int>(x)) + (y - static_cast<int>(y))) / 2;
	interpolated_depth_value = depth_value_1 * length_1 + depth_value_4 * length_4;
      }
    } else if (depth_value_2 > 0 && depth_value_3 > 0) {
      if (abs((x - static_cast<int>(x)) + (y - static_cast<int>(y)) - 1) <= INTERPOLATION_GAP_THRESHOLD) {
	double length_2 = 0.5 + ((x - static_cast<int>(x)) - (y - static_cast<int>(y))) / 2;
	double length_3 = 0.5 - ((x - static_cast<int>(x)) - (y - static_cast<int>(y))) / 2;
	interpolated_depth_value = depth_value_2 * length_2 + depth_value_3 * length_3;
      }
    }
    break;
  }
  case 1: {
    if (depth_value_1 > 0) {
      if ((x - static_cast<int>(x)) <= INTERPOLATION_GAP_THRESHOLD && (y - static_cast<int>(y)) <= INTERPOLATION_GAP_THRESHOLD)
	interpolated_depth_value = depth_value_1;
    } else if (depth_value_2 > 0) {
      if ((static_cast<int>(x + 1) - x) <= INTERPOLATION_GAP_THRESHOLD && (y - static_cast<int>(y)) <= INTERPOLATION_GAP_THRESHOLD)
	interpolated_depth_value = depth_value_2;
    } else if (depth_value_3 > 0) {
      if ((x - static_cast<int>(x)) <= INTERPOLATION_GAP_THRESHOLD && (static_cast<int>(y + 1) - y) <= INTERPOLATION_GAP_THRESHOLD)
	interpolated_depth_value = depth_value_3;
    } else if (depth_value_4 > 0) {
      if ((static_cast<int>(x + 1) - x) <= INTERPOLATION_GAP_THRESHOLD && (y - static_cast<int>(y + 1) - y) <= INTERPOLATION_GAP_THRESHOLD)
	interpolated_depth_value = depth_value_4;
    }
    break;
  }
  case 0:
    break;
  }
  return interpolated_depth_value;
}

void LayerDepthRepresenter::generateLayerImageHTML(const int scene_index, const map<int, vector<double> > &iteration_statistics_map, const map<int, string> &iteration_proposal_type_map)
{
  stringstream html_filename;
  html_filename << "Results/scene_" << scene_index << "/layer_images.html";
  ofstream html_out_str(html_filename.str());
  html_out_str << "<!DOCTYPE html><html><head></head><body>" << endl;
  double previous_energy = -1;
  for (map<int, string>::const_iterator iteration_it = iteration_proposal_type_map.begin(); iteration_it != iteration_proposal_type_map.end(); iteration_it++) {
    html_out_str << "<h3>iteration " << iteration_it->first << ": " << iteration_it->second << "</h3>" << endl;
    double energy = iteration_statistics_map.at(iteration_it->first)[0];
    if (previous_energy < 0 || energy < previous_energy) {
      stringstream image_filename;
      image_filename << "multi_layer_image_" << iteration_it->first << ".bmp";
      html_out_str << "<img src=\"" << image_filename.str() << "\" alt=\"" << image_filename.str() << "\" width=\"100%\" height=\"100%\">" << endl;
      previous_energy = energy;
    } else
      html_out_str << "<p>Energy increases.</p>";
  }
  html_out_str << "</body></html>";
  html_out_str.close();
}

void LayerDepthRepresenter::upsampleSolution(const vector<int> &solution_labels, const int solution_num_surfaces, const map<int, Segment> &solution_segments, vector<int> &upsampled_solution_labels, int &upsampled_solution_num_surfaces, map<int, Segment> &upsampled_solution_segments)
{
  // upsampled_solution_labels = solution_labels;
  // upsampled_solution_num_surfaces = solution_num_surfaces;
  // upsampled_solution_segments = solution_segments;
  
  
  // Mat segmentation_image(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
  // map<int, int> color_table;
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //   int solution_label = solution_labels[pixel];
  //   for (int layer_index = 0; layer_index < num_layers_; layer_index++) {
  //     int surface_id = solution_label / static_cast<int>(pow(solution_num_surfaces + 1, num_layers_ - 1 - layer_index)) % (solution_num_surfaces + 1);
  //     if (surface_id < solution_num_surfaces) {
  // 	if (color_table.count(surface_id) == 0)
  // 	  color_table[surface_id] = rand() % 256;
  // 	segmentation_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = color_table[surface_id];
  // 	break;
  //     }
  //   }
  // }
  // imwrite("Test/final_segmentation_image.bmp", segmentation_image);
  
  
  const int ORI_IMAGE_WIDTH = ori_image_.cols;
  const int ORI_IMAGE_HEIGHT = ori_image_.rows;
  const int ORI_NUM_PIXELS = ORI_IMAGE_WIDTH * ORI_IMAGE_HEIGHT;
  const int NUM_DILATION_ITERATIONS = 0;
  
  //const double UPSAMPLING_RATIO = 1.0 * ORI_IMAGE_WIDTH / IMAGE_WIDTH_;
  
  
  vector<double> ori_camera_parameters(3);
  estimateCameraParameters(ori_point_cloud_, ORI_IMAGE_WIDTH, ORI_IMAGE_HEIGHT, ori_camera_parameters, USE_PANORAMA_);
  //vector<double> ori_normals = calcNormals(ori_point_cloud_, ORI_IMAGE_WIDTH, ORI_IMAGE_HEIGHT);
  
  
  // if (readLayers(ORI_IMAGE_WIDTH, ORI_IMAGE_HEIGHT, ori_camera_parameters, PENALTIES_, STATISTICS_, num_layers_, upsampled_solution_labels, upsampled_solution_num_surfaces, upsampled_solution_segments, SCENE_INDEX_, 20000) == true) {
  //   return;
  // }
  // cout << "upsample solution" << endl;
  
  
  // unique_ptr<ProposalDesigner> proposal_designer(new ProposalDesigner(image_, point_cloud_, normals_, camera_parameters_, num_layers_, PENALTIES_, STATISTICS_, SCENE_INDEX_));
  
  // proposal_designer->setCurrentSolution(solution_labels, solution_num_surfaces, solution_segments);
  
  // vector<vector<int> > proposal_labels;
  // int proposal_num_surfaces;
  // map<int, Segment> proposal_segments;
  // string proposal_type;
  // proposal_designer->getUpsamplingProposal(ori_image_, ori_point_cloud_, ori_normals, ori_camera_parameters, proposal_labels, proposal_num_surfaces, proposal_segments, NUM_DILATION_ITERATIONS);
  // //vector<int> previous_solution_indices = proposal_designer->getCurrentSolutionIndices();
  
  // RepresenterPenalties upsampling_penalties = PENALTIES_;
  // upsampling_penalties.depth_inconsistency_pen = PENALTIES_.color_inconsistency_pen;
  // upsampling_penalties.smoothness_boundary_pen = PENALTIES_.smoothness_pen;
  // upsampling_penalties.smoothness_pen = PENALTIES_.smoothness_boundary_pen;
  // //upsampling_penalties.color_inconsistency_pen = PENALTIES_.depth_inconsistency_pen;
  // unique_ptr<TRWSFusion> TRW_solver(new TRWSFusion(ori_image_, ori_point_cloud_, ori_normals, upsampling_penalties, STATISTICS_, false));
  
  // upsampled_solution_labels = TRW_solver->fuse(proposal_labels, proposal_num_surfaces, num_layers_, proposal_segments, vector<int>(ORI_NUM_PIXELS, 0));
  // upsampled_solution_segments = proposal_segments;
  // upsampled_solution_num_surfaces = upsampled_solution_segments.size();
  
  // vector<int> new_solution_labels(ORI_IMAGE_WIDTH * ORI_IMAGE_HEIGHT);
  // for (int pixel = 0; pixel < ORI_IMAGE_WIDTH * ORI_IMAGE_HEIGHT; pixel++) {
  //   int scaled_pixel = min(static_cast<int>(round(static_cast<double>(pixel / ORI_IMAGE_WIDTH) / ORI_IMAGE_HEIGHT * IMAGE_HEIGHT_)), IMAGE_HEIGHT_ - 1) * IMAGE_WIDTH_ + min(static_cast<int>(round(static_cast<double>(pixel % ORI_IMAGE_WIDTH) / ORI_IMAGE_WIDTH * IMAGE_WIDTH_)), IMAGE_WIDTH_ - 1);
  //   new_solution_labels[pixel] = solution_labels[scaled_pixel];
  //   //if (solution_labels[scaled_pixel] != new_solution_labels[pixel])
  //   //cout << pixel << '\t' << solution_labels[scaled_pixel] << '\t' << new_solution_labels[pixel] << endl;
  // }
  
  // //writeLayers(ori_image_, ORI_IMAGE_WIDTH, ORI_IMAGE_HEIGHT, ori_point_cloud_, ori_camera_parameters, num_layers_, upsampled_solution_labels, upsampled_solution_num_surfaces, upsampled_solution_segments, SCENE_INDEX_, 100, ori_image_);
  
  
  // // stringstream upsampled_solution_labels_filename;
  // // upsampled_solution_labels_filename << "Cache/scene_" << SCENE_INDEX_ << "/upsampled_solution_labels.txt";
  // // ofstream upsampled_solution_labels_out_str(upsampled_solution_labels_filename.str());
  // // upsampled_solution_labels_out_str << new_solution_labels.size() << endl;
  // // for (vector<int>::const_iterator label_it = new_solution_labels.begin(); label_it != new_solution_labels.end(); label_it++)
  // //   upsampled_solution_labels_out_str << *label_it << endl;
  // // upsampled_solution_labels_out_str.close();
  
  // // writeDispImageFromSegments(new_solution_labels, solution_num_surfaces, new_solution_segments, num_layers_, ORI_IMAGE_WIDTH, ORI_IMAGE_HEIGHT, "Test/final_disp_image.bmp");
  
  // return;
  
  
  map<int, map<int, vector<int> > > segment_layer_visible_pixels;
  vector<int> visible_layer_indices(NUM_PIXELS_);
  vector<int> visible_surface_ids(NUM_PIXELS_);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int solution_label = solution_labels[pixel];
    for (int layer_index = 0; layer_index < num_layers_; layer_index++) {
      int surface_id = solution_label / static_cast<int>(pow(solution_num_surfaces + 1, num_layers_ - 1 - layer_index)) % (solution_num_surfaces + 1);
      if (surface_id < solution_num_surfaces) {
        segment_layer_visible_pixels[surface_id][layer_index].push_back(pixel);
	visible_layer_indices[pixel] = layer_index;
	visible_surface_ids[pixel] = surface_id;
        break;
      }
    }
  }
  vector<vector<bool> > layer_confident_pixel_mask(num_layers_, vector<bool>(NUM_PIXELS_, true));
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_);
    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
      if (visible_layer_indices[pixel] < visible_layer_indices[*neighbor_pixel_it]) {
	layer_confident_pixel_mask[visible_layer_indices[pixel]][pixel] = false;
	layer_confident_pixel_mask[visible_layer_indices[pixel]][*neighbor_pixel_it] = false;
	break;
      }
    }
  }

  const int UNCONFIDENT_BOUNDARY_WIDTH = 1;
  for (int layer_index = 0; layer_index < num_layers_; layer_index++) {
    vector<bool> confident_pixel_mask = layer_confident_pixel_mask[layer_index];
    for (int i = 1; i < UNCONFIDENT_BOUNDARY_WIDTH; i++) {
      vector<bool> new_confident_pixel_mask = confident_pixel_mask;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	if (confident_pixel_mask[pixel] == true)
	  continue;
	vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_);
	for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++)
	  new_confident_pixel_mask[*neighbor_pixel_it] = false;
      }
      confident_pixel_mask = new_confident_pixel_mask;
    }
    layer_confident_pixel_mask[layer_index] = confident_pixel_mask;
  }

  Mat blurred_image = image_.clone();
  //GaussianBlur(image_, blurred_image, cv::Size(3, 3), 0, 0);
  for (int layer_index = 0; layer_index < num_layers_; layer_index++) {
    vector<bool> confident_pixel_mask = layer_confident_pixel_mask[layer_index];
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
      if (solution_segments.at(visible_surface_ids[pixel]).checkPixelFitting(blurred_image, point_cloud_, normals_, pixel) == false)
	confident_pixel_mask[pixel] = false;
    layer_confident_pixel_mask[layer_index] = confident_pixel_mask;
  }
  
  // Mat confident_pixel_image = image_.clone();
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
  //   if (confident_pixel_mask[pixel] == false)
  //     confident_pixel_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = Vec3b(0, 0, 0);
  // resize(confident_pixel_image, confident_pixel_image, ori_image_.size());
  // imwrite("Test/confident_pixel_image.bmp", confident_pixel_image);
  // exit(1);
  
  
  // map<int, Segment> segments;
  // map<int, int> segment_layer_map;
  // map<int, int> segment_index_map;
  // int segment_index = 0;
  // vector<int> visible_segmentation(NUM_PIXELS_);
  // for (map<int, map<int, vector<int> > >::const_iterator segment_it = segment_layer_visible_pixels.begin(); segment_it != segment_layer_visible_pixels.end(); segment_it++) {
  //   for (map<int, vector<int> >::const_iterator layer_it = segment_it->second.begin(); layer_it != segment_it->second.end(); layer_it++) {
  //     Segment segment = solution_segments.at(segment_it->first);
  //     //      segment.refitSegmentKeepingGeometry(image_, point_cloud_, normals_, layer_it->second);
  //     segments[segment_index] = segment;
  //     segment_layer_map[segment_index] = layer_it->first;
  //     segment_index_map[segment_index] = segment_it->first;
  //     for (vector<int>::const_iterator pixel_it = layer_it->second.begin(); pixel_it != layer_it->second.end(); pixel_it++)
  // 	visible_segmentation[*pixel_it] = segment_index;
  //     segment_index++;
  //   }
  // }
  
  // map<int, int> color_table;
  // map<int, int> reverse_color_table;
  
  // stringstream segmentation_image_low_res_filename;
  // segmentation_image_low_res_filename << "Results/scene_" << SCENE_INDEX_ << "/segmentation_image_low_res.bmp";
  // if (imread(segmentation_image_low_res_filename.str()).empty()) {
  //   exit(1);
  //   Mat segmentation_image(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
  //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //     int segment_id = visible_segmentation[pixel];
  //     if (color_table.count(segment_id) == 0) {
  // 	int color = rand() % 256;
  // 	while (reverse_color_table.count(color) > 0)
  // 	  color = rand() % 256;
  // 	color_table[segment_id] = color;
  // 	reverse_color_table[color] = segment_id;
  //     }
  //     segmentation_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = color_table[segment_id];
  //   }
  //   imwrite(segmentation_image_low_res_filename.str(), segmentation_image);
  // } else {
  //   Mat segmentation_image = imread(segmentation_image_low_res_filename.str(), 0);
  //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //     int segment_id = visible_segmentation[pixel];
  //     int color = segmentation_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_);
  //     color_table[segment_id] = color;
  //     reverse_color_table[color] = segment_id;
  //   }
  // }
  
  
  // stringstream segmentation_image_high_res_filename;
  // segmentation_image_high_res_filename << "Results/scene_" << SCENE_INDEX_ << "/segmentation_image_high_res.bmp";
  // Mat segmentation_high_res_image = imread(segmentation_image_high_res_filename.str(), 0);
  
  // vector<int> segmentation_high_res(ORI_NUM_PIXELS);
  // for (int ori_pixel = 0; ori_pixel < ORI_NUM_PIXELS; ori_pixel++) {
  //   int color = segmentation_high_res_image.at<uchar>(ori_pixel / ORI_IMAGE_WIDTH, ori_pixel % ORI_IMAGE_WIDTH);
  //   segmentation_high_res[ori_pixel] = reverse_color_table[color];
  // }
  
  
  // for (map<int, int>::const_iterator color_it = reverse_color_table.begin(); color_it != reverse_color_table.end(); color_it++)
  //   cout << color_it->first << '\t' << color_it->second << '\t' << segment_layer_map[color_it->second] << endl;
  // exit(1);
  
  // map<int, vector<int> > segment_visible_pixels;
  // for (int ori_pixel = 0; ori_pixel < ORI_NUM_PIXELS; ori_pixel++)
  //   segment_visible_pixels[segment_index_map[segmentation_high_res[ori_pixel]]].push_back(ori_pixel);
  
  // map<int, Segment> solution_segments_high_res;
  // for (map<int, Segment>::const_iterator segment_it = solution_segments.begin(); segment_it != solution_segments.end(); segment_it++) {
  //   cout << segment_it->first << endl;
  //   solution_segments_high_res[segment_it->first] = Segment(ori_image_, ori_point_cloud_, ori_normals, ori_camera_parameters, segment_visible_pixels[segment_it->first], PENALTIES_, STATISTICS_);
  // }
  
  
  vector<vector<double> > distance_map(ORI_NUM_PIXELS, vector<double>(9, 1000000));
  for (int ori_pixel = 0; ori_pixel < ORI_NUM_PIXELS; ori_pixel++) {
    int x = ori_pixel % ORI_IMAGE_WIDTH;
    int y = ori_pixel / ORI_IMAGE_WIDTH;
    for (int delta_x = -1; delta_x <= 1; delta_x++) {
      for (int delta_y = -1; delta_y <= 1; delta_y++) {
	if (x + delta_x >= 0 && x + delta_x < ORI_IMAGE_WIDTH && y + delta_y >= 0 && y + delta_y < ORI_IMAGE_HEIGHT) {
	  Vec3b color_1 = ori_image_.at<Vec3b>(y, x);
	  Vec3b color_2 = ori_image_.at<Vec3b>(y + delta_y, x + delta_x);
	  double distance = 0;
	  for (int c = 0; c < 3; c++)
	    distance += pow(color_1[c] - color_2[c], 2);
	  distance = sqrt(distance / 3);
	  distance_map[ori_pixel][(delta_y + 1) * 3 + (delta_x + 1)] = distance;
	}
      }
    }
  }
  //cout << ori_image_.at<Vec3b>(639, 214) << '\t' << ori_image_.at<Vec3b>(640, 214) << endl;
  //cout << distance_map[639214][7] << '\t' << distance_map[639214][5] << endl;
  //exit(1);
  const double DISTANCE_2D_WEIGHT = 0 * IMAGE_WIDTH_ / ORI_IMAGE_WIDTH;
  //cout << calcGeodesicDistance(distance_map, ORI_IMAGE_WIDTH, ORI_IMAGE_HEIGHT, 639214, 645220, DISTANCE_2D_WEIGHT) << endl;
  //  cout << calcGeodesicDistance(distance_map, ORI_IMAGE_WIDTH, ORI_IMAGE_HEIGHT, 628585, 627577, DISTANCE_2D_WEIGHT) << endl;
  //exit(1);
  
  vector<int> solution_labels_high_res(ORI_NUM_PIXELS, 0);
  for (int ori_pixel = 0; ori_pixel < ORI_NUM_PIXELS; ori_pixel++) {
    //cout << ori_pixel << endl;
    //    if (ori_pixel % ORI_IMAGE_WIDTH < 250 || ori_pixel % ORI_IMAGE_WIDTH > 250 || ori_pixel / ORI_IMAGE_WIDTH < 630 || ori_pixel / ORI_IMAGE_WIDTH > 630)
    //continue;
    //cout << ori_pixel << endl;
    double x = 1.0 * (ori_pixel % ORI_IMAGE_WIDTH) / ORI_IMAGE_WIDTH * IMAGE_WIDTH_;
    double y = 1.0 * (ori_pixel / ORI_IMAGE_WIDTH) / ORI_IMAGE_HEIGHT * IMAGE_HEIGHT_;
    vector<int> xs;
    xs.push_back(max(static_cast<int>(x - 1), 0));
    xs.push_back(min(max(static_cast<int>(x), 0), IMAGE_WIDTH_ - 1));
    xs.push_back(min(static_cast<int>(x) + 1, IMAGE_WIDTH_ - 1));
    xs.push_back(min(static_cast<int>(x) + 2, IMAGE_WIDTH_ - 1));
    vector<int> ys;
    ys.push_back(max(static_cast<int>(y - 1), 0));
    ys.push_back(min(max(static_cast<int>(y), 0), IMAGE_HEIGHT_ - 1));
    ys.push_back(min(static_cast<int>(y) + 1, IMAGE_HEIGHT_ - 1));
    ys.push_back(min(static_cast<int>(y) + 2, IMAGE_HEIGHT_ - 1));
    vector<int> vertex_pixels;
    for (vector<int>::const_iterator y_it = ys.begin(); y_it != ys.end(); y_it++)
      for (vector<int>::const_iterator x_it = xs.begin(); x_it != xs.end(); x_it++)
	vertex_pixels.push_back(*y_it * IMAGE_WIDTH_ + *x_it);

    
    // if (ori_pixel != 189 * ORI_IMAGE_WIDTH + 394)
    //   continue;

    if (false) {
      vector<int> segment_indices;
      for (vector<int>::const_iterator pixel_it = vertex_pixels.begin(); pixel_it != vertex_pixels.end(); pixel_it++)
	segment_indices.push_back(solution_labels[*pixel_it] / static_cast<int>(pow(solution_num_surfaces + 1, num_layers_ - 1 - 3)) % (solution_num_surfaces + 1));
    
      vector<int> ori_vertex_pixels;
      for (vector<int>::const_iterator pixel_it = vertex_pixels.begin(); pixel_it != vertex_pixels.end(); pixel_it++) {
	double ori_vertex_pixel = min(static_cast<int>(round(1.0 * (*pixel_it / IMAGE_WIDTH_) / IMAGE_HEIGHT_ * ORI_IMAGE_HEIGHT)), ORI_IMAGE_HEIGHT - 1) * ORI_IMAGE_WIDTH + min(static_cast<int>(round(1.0 * (*pixel_it % IMAGE_WIDTH_) / IMAGE_WIDTH_ * ORI_IMAGE_WIDTH)), ORI_IMAGE_WIDTH - 1);
	ori_vertex_pixels.push_back(ori_vertex_pixel);
      }
        
      vector<double> distances = calcGeodesicDistances(distance_map, ORI_IMAGE_WIDTH, ORI_IMAGE_HEIGHT, ori_pixel, ori_vertex_pixels, DISTANCE_2D_WEIGHT);
      double min_distance = 1000000;
      int min_distance_index = -1;
      for (vector<double>::const_iterator distance_it = distances.begin(); distance_it != distances.end(); distance_it++) {
	if (*distance_it < min_distance) {
	  cout << *distance_it << '\t' << ori_vertex_pixels[distance_it - distances.begin()] % ORI_IMAGE_WIDTH << '\t' << ori_vertex_pixels[distance_it - distances.begin()] / ORI_IMAGE_WIDTH << '\t' << segment_indices[distance_it - distances.begin()] << endl;
	}
      }
      exit(1);
    }
    
    // bool has_confident_pixels = false;
    // for (vector<int>::const_iterator pixel_it = vertex_pixels.begin(); pixel_it != vertex_pixels.end(); pixel_it++) {
    //   if (confident_pixel_mask[*pixel_it] == true) {
    // 	has_confident_pixels = true;
    // 	break;
    //   }
    // }
    // int dilation_iteration = 1;
    // while (has_confident_pixels == false) {
    //   xs.push_back(max(static_cast<int>(x - 1 - dilation_iteration), 0));
    //   xs.push_back(min(static_cast<int>(x) + 2 + dilation_iteration, IMAGE_WIDTH_ - 1));
    //   ys.push_back(max(static_cast<int>(y - 1 - dilation_iteration), 0));
    //   ys.push_back(min(static_cast<int>(y) + 2 + dilation_iteration, IMAGE_HEIGHT_ - 1));
    
    //   vertex_pixels.clear();
    //   for (vector<int>::const_iterator y_it = ys.begin(); y_it != ys.end(); y_it++)
    //     for (vector<int>::const_iterator x_it = xs.begin(); x_it != xs.end(); x_it++)
    //       vertex_pixels.push_back(*y_it * IMAGE_WIDTH_ + *x_it);
    //   for (vector<int>::const_iterator pixel_it = vertex_pixels.begin(); pixel_it != vertex_pixels.end(); pixel_it++) {
    //     if (confident_pixel_mask[*pixel_it] == true) {
    //       has_confident_pixels = true;
    //       break;
    //     }
    //   }
    //   dilation_iteration++;
    // }
    
    for (int layer_index = 0; layer_index < num_layers_; layer_index++) {
      // vector<pair<double, int> > distance_segment_id_pairs;
      // distance_segment_id_pairs.push_back(make_pair(sqrt(pow(x_1 - x, 2) + pow(x_1 - y, 2)), solution_labels[y_1 * IMAGE_WIDTH_ + x_1] / static_cast<int>(pow(solution_num_surfaces + 1, num_layers_ - 1 - layer_index)) % (solution_num_surfaces + 1)));
      vector<int> segment_indices;
      for (vector<int>::const_iterator pixel_it = vertex_pixels.begin(); pixel_it != vertex_pixels.end(); pixel_it++)
        segment_indices.push_back(solution_labels[*pixel_it] / static_cast<int>(pow(solution_num_surfaces + 1, num_layers_ - 1 - layer_index)) % (solution_num_surfaces + 1));

      set<int> segment_indices_set(segment_indices.begin(), segment_indices.end());
      map<int, map<int, int> > surface_occluding_relations;
      for (vector<int>::const_iterator segment_it_1 = segment_indices.begin(); segment_it_1 != segment_indices.end(); segment_it_1++) {
	if (*segment_it_1 == solution_num_surfaces)
	  continue;
	int vertex_pixel = vertex_pixels[segment_it_1 - segment_indices.begin()];
        for (set<int>::const_iterator segment_it_2 = segment_indices_set.begin(); segment_it_2 != segment_indices_set.end(); segment_it_2++) {
	  if (*segment_it_2 == solution_num_surfaces || *segment_it_2 == *segment_it_1)
            continue;
	  //	  cout << vertex_pixel << '\t' << *segment_it_1 << '\t' << *segment_it_2 << '\t' << solution_segments.at(*segment_it_1).getDepth(vertex_pixel) << '\t' << solution_segments.at(*segment_it_2).getDepth(vertex_pixel) << endl;
	  if (solution_segments.at(*segment_it_2).getDepth(vertex_pixel) > solution_segments.at(*segment_it_1).getDepth(vertex_pixel) || solution_segments.at(*segment_it_2).getDepth(vertex_pixel) < 0)
	      surface_occluding_relations[*segment_it_1][*segment_it_2]++;
          else
	    surface_occluding_relations[*segment_it_1][*segment_it_2]--;
	}
      }
      
      int selected_segment_index = -1;
      int selected_vertex = -1;
      if (true) {
	for (vector<int>::const_iterator segment_it = segment_indices.begin(); segment_it != segment_indices.end(); segment_it++) {
	  if (selected_segment_index == -1 || selected_segment_index == solution_num_surfaces) {
	    selected_segment_index = *segment_it;
	    selected_vertex = segment_it - segment_indices.begin();
	    //cout << "0 " << selected_segment_index << endl;
	  } else if (*segment_it != solution_num_surfaces && *segment_it != selected_segment_index) {
	    if (surface_occluding_relations[*segment_it][selected_segment_index] + surface_occluding_relations[selected_segment_index][*segment_it] > 0) {
	      if (solution_segments.at(*segment_it).getDepth(x / IMAGE_WIDTH_, y / IMAGE_HEIGHT_) < solution_segments.at(selected_segment_index).getDepth(x / IMAGE_WIDTH_, y / IMAGE_HEIGHT_) ||
		  solution_segments.at(selected_segment_index).getDepth(x / IMAGE_WIDTH_, y / IMAGE_HEIGHT_) < 0) {
		selected_segment_index = *segment_it;
		selected_vertex = segment_it - segment_indices.begin();
		//cout << "1 " << selected_segment_index << endl;
	      }
	    } else if (surface_occluding_relations[*segment_it][selected_segment_index] + surface_occluding_relations[selected_segment_index][*segment_it] < 0) {
	      if (solution_segments.at(*segment_it).getDepth(x / IMAGE_WIDTH_, y / IMAGE_HEIGHT_) > solution_segments.at(selected_segment_index).getDepth(x / IMAGE_WIDTH_, y / IMAGE_HEIGHT_)) {
		selected_segment_index = *segment_it;
		selected_vertex = segment_it - segment_indices.begin();
		//cout << "2 " << selected_segment_index << endl;
	      }
	    } else {
	      if (sqrt(pow(x - vertex_pixels[segment_it - segment_indices.begin()] % IMAGE_WIDTH_, 2) + pow(y - vertex_pixels[segment_it - segment_indices.begin()] / IMAGE_WIDTH_, 2)) < sqrt(pow(x - vertex_pixels[selected_vertex] % IMAGE_WIDTH_, 2) + pow(y - vertex_pixels[selected_vertex] / IMAGE_WIDTH_, 2))) {
		selected_segment_index = *segment_it;
		selected_vertex = segment_it - segment_indices.begin();
		//	      cout << "3 " << selected_segment_index << endl;
	      } 
	    }
	  }
	}
      } else {
	vector<int> sorted_segment_indices = segment_indices;
	if (unique(sorted_segment_indices.begin(), sorted_segment_indices.end()) - sorted_segment_indices.begin() == 1) {
	  selected_segment_index = *segment_indices.begin();
	  selected_vertex = 0;
	} else {
	  vector<int> ori_vertex_pixels;
	  for (vector<int>::const_iterator pixel_it = vertex_pixels.begin(); pixel_it != vertex_pixels.end(); pixel_it++) {
	    double ori_vertex_pixel = min(static_cast<int>(round(1.0 * (*pixel_it / IMAGE_WIDTH_) / IMAGE_HEIGHT_ * ORI_IMAGE_HEIGHT)), ORI_IMAGE_HEIGHT - 1) * ORI_IMAGE_WIDTH + min(static_cast<int>(round(1.0 * (*pixel_it % IMAGE_WIDTH_) / IMAGE_WIDTH_ * ORI_IMAGE_WIDTH)), ORI_IMAGE_WIDTH - 1);
	    ori_vertex_pixels.push_back(ori_vertex_pixel);
	  }
	
	  vector<double> distances = calcGeodesicDistances(distance_map, ORI_IMAGE_WIDTH, ORI_IMAGE_HEIGHT, ori_pixel, ori_vertex_pixels, DISTANCE_2D_WEIGHT);
	  double min_distance = 1000000;
	  int min_distance_index = -1;
	  for (vector<double>::const_iterator distance_it = distances.begin(); distance_it != distances.end(); distance_it++) {
	    if (*distance_it < min_distance) {
	      selected_segment_index = segment_indices[distance_it - distances.begin()];
	      selected_vertex = distance_it - distances.begin();
	      min_distance = *distance_it;
	    }
	  }
	  // if (layer_index == 3) {
          //   cout << selected_segment_index << endl;
          //   exit(1);
          // }
        }
      }

      bool has_empty_neighbor = false;
      for (vector<int>::const_iterator segment_it = segment_indices.begin(); segment_it != segment_indices.end(); segment_it++) {
	int vertex_pixel = vertex_pixels[segment_it - segment_indices.begin()];
        if (*segment_it == solution_num_surfaces && visible_layer_indices[vertex_pixel] > layer_index) {
	  has_empty_neighbor = true;
	  break;
	}
      }
      if (selected_segment_index < solution_num_surfaces && has_empty_neighbor) {
	set<int> selected_segment_pixels;
        set<int> empty_pixels;
        int window_size = 2;
        while (selected_segment_pixels.size() == 0 || empty_pixels.size() == 0) {
          for (int delta_x = -window_size; delta_x <= window_size; delta_x++) {
            for (int delta_y = -window_size; delta_y <= window_size; delta_y++) {
              int window_pixel = min(max(static_cast<int>(round(y)) + delta_y, 0), IMAGE_HEIGHT_ - 1) * IMAGE_WIDTH_ + min(max(static_cast<int>(round(x)) + delta_x, 0), IMAGE_WIDTH_ - 1);
              if (layer_confident_pixel_mask[layer_index][window_pixel] == false)
		continue;
              //double distance = sqrt(pow(round(x) + delta_x - x, 2) + pow(round(y) + delta_y - y, 2));
              int window_surface_id = solution_labels[window_pixel] / static_cast<int>(pow(solution_num_surfaces + 1, num_layers_ - 1 - layer_index)) % (solution_num_surfaces + 1);
              if (window_surface_id == selected_segment_index) {
		selected_segment_pixels.insert(window_pixel);
              }
              if (window_surface_id == solution_num_surfaces) {
                empty_pixels.insert(window_pixel);
              }
            }
          }
          window_size++;
	  if (window_size == 5)
	    break;
        }
	if (window_size == 5) {
	  if (empty_pixels.size() >= selected_segment_pixels.size()) {
	    selected_segment_index = solution_num_surfaces;
	  }
	} else {
	  vector<int> ori_window_pixels;
	  for (set<int>::const_iterator selected_segment_pixel_it = selected_segment_pixels.begin(); selected_segment_pixel_it != selected_segment_pixels.end(); selected_segment_pixel_it++) {
	    double ori_selected_segment_pixel = min(static_cast<int>(round(1.0 * (*selected_segment_pixel_it / IMAGE_WIDTH_) / IMAGE_HEIGHT_ * ORI_IMAGE_HEIGHT)), ORI_IMAGE_HEIGHT - 1) * ORI_IMAGE_WIDTH + min(static_cast<int>(round(1.0 * (*selected_segment_pixel_it % IMAGE_WIDTH_) / IMAGE_WIDTH_ * ORI_IMAGE_WIDTH)), ORI_IMAGE_WIDTH - 1);
	    ori_window_pixels.push_back(ori_selected_segment_pixel);
	  }
	  for (set<int>::const_iterator empty_pixel_it = empty_pixels.begin(); empty_pixel_it != empty_pixels.end(); empty_pixel_it++) {
	    double ori_empty_pixel = min(static_cast<int>(round(1.0 * (*empty_pixel_it / IMAGE_WIDTH_) / IMAGE_HEIGHT_ * ORI_IMAGE_HEIGHT)), ORI_IMAGE_HEIGHT - 1) * ORI_IMAGE_WIDTH + min(static_cast<int>(round(1.0 * (*empty_pixel_it % IMAGE_WIDTH_) / IMAGE_WIDTH_ * ORI_IMAGE_WIDTH)), ORI_IMAGE_WIDTH - 1);
	    ori_window_pixels.push_back(ori_empty_pixel);
	  }
	  vector<double> distances = calcGeodesicDistances(distance_map, ORI_IMAGE_WIDTH, ORI_IMAGE_HEIGHT, ori_pixel, ori_window_pixels, DISTANCE_2D_WEIGHT);
          double min_distance = 1000000;
	  int min_distance_index = -1;
	  for (vector<double>::const_iterator distance_it = distances.begin(); distance_it != distances.end(); distance_it++) {
	    if (*distance_it < min_distance) {
	      min_distance_index = distance_it - distances.begin();
	      min_distance = *distance_it;
	    }
	  }
	  if (min_distance_index >= selected_segment_pixels.size())
	    selected_segment_index = solution_num_surfaces;
	}
	
        // int selected_segment_pixel = -1;
	// double selected_segment_pixel_distance = 1000000;
	// int empty_pixel = -1;
	// double empty_pixel_distance = 1000000;
	// int window_size = 2;
	// while (selected_segment_pixel == -1 || empty_pixel == -1) {
	//   for (int delta_x = -window_size; delta_x <= window_size; delta_x++) {
	//     for (int delta_y = -window_size; delta_y <= window_size; delta_y++) {
	//       int window_pixel = min(max(static_cast<int>(round(y)) + delta_y, 0), IMAGE_HEIGHT_ - 1) * IMAGE_WIDTH_ + min(max(static_cast<int>(round(x)) + delta_x, 0), IMAGE_WIDTH_ - 1);
	//       if (confident_pixel_mask[window_pixel] == false)
	// 	continue;
	//       double distance = sqrt(pow(round(x) + delta_x - x, 2) + pow(round(y) + delta_y - y, 2));
	//       int window_surface_id = solution_labels[window_pixel] / static_cast<int>(pow(solution_num_surfaces + 1, num_layers_ - 1 - layer_index)) % (solution_num_surfaces + 1);
	//       if (window_surface_id == selected_segment_index && distance < selected_segment_pixel_distance) {
	// 	selected_segment_pixel = window_pixel;
	// 	selected_segment_pixel_distance = distance;
	//       }
	//       if (window_surface_id == solution_num_surfaces && distance < empty_pixel_distance) {
        //         empty_pixel = window_pixel;
        //         empty_pixel_distance = distance;
        //       }
	//     }
	//   }
	//   window_size++;
	// }
	
	// double ori_selected_segment_pixel = min(static_cast<int>(round(1.0 * (selected_segment_pixel / IMAGE_WIDTH_) / IMAGE_HEIGHT_ * ORI_IMAGE_HEIGHT)), ORI_IMAGE_HEIGHT - 1) * ORI_IMAGE_WIDTH + min(static_cast<int>(round(1.0 * (selected_segment_pixel % IMAGE_WIDTH_) / IMAGE_WIDTH_ * ORI_IMAGE_WIDTH)), ORI_IMAGE_WIDTH - 1);
	// double ori_empty_pixel = min(static_cast<int>(round(1.0 * (empty_pixel / IMAGE_WIDTH_) / IMAGE_HEIGHT_ * ORI_IMAGE_HEIGHT)), ORI_IMAGE_HEIGHT - 1) * ORI_IMAGE_WIDTH + min(static_cast<int>(round(1.0 * (empty_pixel % IMAGE_WIDTH_) / IMAGE_WIDTH_ * ORI_IMAGE_WIDTH)), ORI_IMAGE_WIDTH - 1);
	
	// //cout << ori_pixel << '\t' << ori_empty_pixel << '\t' << ori_selected_segment_pixel << '\t' << calcGeodesicDistance(distance_map, ORI_IMAGE_WIDTH, ORI_IMAGE_HEIGHT, ori_pixel, ori_empty_pixel, DISTANCE_2D_WEIGHT) << '\t' << calcGeodesicDistance(distance_map, ORI_IMAGE_WIDTH, ORI_IMAGE_HEIGHT, ori_pixel, ori_selected_segment_pixel, DISTANCE_2D_WEIGHT) << endl;
	// //exit(1);
	// if (calcGeodesicDistance(distance_map, ORI_IMAGE_WIDTH, ORI_IMAGE_HEIGHT, ori_empty_pixel, ori_pixel, DISTANCE_2D_WEIGHT) < calcGeodesicDistance(distance_map, ORI_IMAGE_WIDTH, ORI_IMAGE_HEIGHT, ori_selected_segment_pixel, ori_pixel, DISTANCE_2D_WEIGHT))
	//   selected_segment_index = solution_num_surfaces;
      }
      //cout << selected_segment_index << endl;
      solution_labels_high_res[ori_pixel] += selected_segment_index * pow(solution_num_surfaces + 1, num_layers_ - 1 - layer_index);
      // if (segment_indices.size() > 1 && find(segment_indices.begin(), segment_indices.end(), solution_num_surfaces) < segment_indices.end())
      // 	confident_pixel_mask[ori_pixel] = false;
    }
  }
  
  // for (int ori_pixel = 0; ori_pixel < ORI_NUM_PIXELS; ori_pixel++) {
  //   if (confident_pixel_mask[ori_pixel] == true)
  //     continue;
  //   int solution_label = solution_labels_high_res[ori_pixel];
  //   int layer_index = segment_layer_map[segmentation_high_res[ori_pixel]];
  //   int segment_id = segment_index_map[segmentation_high_res[ori_pixel]];
  //   for (int target_layer_index = 0; target_layer_index < layer_index; target_layer_index++) {
  //     int surface_id = solution_label / static_cast<int>(pow(solution_num_surfaces + 1, num_layers_ - 1 - target_layer_index)) % (solution_num_surfaces + 1);
  //     if (target_layer_index == layer_index)
  // 	solution_label += (segment_id - surface_id) * pow(solution_num_surfaces + 1, num_layers_ - 1 - target_layer_index);
  //     else
  //       solution_label += (solution_num_surfaces - surface_id) * pow(solution_num_surfaces + 1, num_layers_ - 1 - target_layer_index);
  //   }
  //   solution_labels_high_res[ori_pixel] = solution_label;
  // }
  
  //writeLayers(ori_image_, ORI_IMAGE_WIDTH, ORI_IMAGE_HEIGHT, ori_point_cloud_, ori_camera_parameters, num_layers_, solution_labels_high_res, solution_num_surfaces, solution_segments, SCENE_INDEX_, 20000, ori_image_, ori_point_cloud_);
  writeLayers(ori_image_, ori_point_cloud_, ori_camera_parameters, num_layers_, solution_labels_high_res, solution_num_surfaces, solution_segments, SCENE_INDEX_, 20000);
  upsampled_solution_labels = solution_labels_high_res;
  upsampled_solution_num_surfaces = solution_num_surfaces;
  
  
  //  SegmentationRefiner refiner(ori_image_, ori_point_cloud_, camera_parameters_, PENALTIES_, STATISTICS_, segments, segment_layer_map);
  //vector<int> refined_segmentation = refiner.getRefinedSegmentation();
  // new_solution_labels = solution_labels;
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //   int refined_segment_index = refined_segmentation[pixel];
  //   int new_segment_index = segment_index_map[refined_segment_index];
  //   int refined_layer_index = segment_layer_map[refined_segment_index];
  //   int solution_label = solution_labels[pixel];
  //   int ori_segment_index = solution_label / static_cast<int>(pow(solution_num_surfaces + 1, num_layers_ - 1 - refined_layer_index)) % (solution_num_surfaces + 1);
  
  //   if (new_segment_index != ori_segment_index) {
  //     int new_solution_label = solution_label;
  //     new_solution_label += (new_segment_index - ori_segment_index) * pow(solution_num_surfaces + 1, num_layers_ - 1 - refined_layer_index);
  //     for (int layer_index = 0; layer_index < refined_layer_index; layer_index++) {
  // 	int surface_id = solution_label / static_cast<int>(pow(solution_num_surfaces + 1, num_layers_ - 1 - layer_index)) % (solution_num_surfaces + 1);
  //       if (surface_id < solution_num_surfaces)
  // 	  new_solution_label += (solution_num_surfaces - surface_id) * pow(solution_num_surfaces + 1, num_layers_ - 1 - layer_index);
  //     }
  //     new_solution_labels[pixel] = new_solution_label;
  //   }
  // }
}
