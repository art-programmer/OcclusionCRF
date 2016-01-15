#include "ProposalDesigner.h"

#include <iostream>

#include <opencv2/highgui/highgui.hpp>

#include "cv_utils.h"
#include "OrthogonalStructureFinder.h"
#include "StructureFinder.h"
#include "LayerCalculation.h"
//#include "LayerEstimator.h"

//#include "PointCloudSegmenter.h"


using namespace std;
using namespace cv;
using namespace cv_utils;


ProposalDesigner::ProposalDesigner(const Mat &image, const vector<double> &point_cloud, const vector<double> &normals, const std::vector<double> &pixel_weights_3D, const vector<double> &camera_parameters, const int num_layers, const RepresenterPenalties penalties, const DataStatistics statistics, const int scene_index, const bool use_panorama) : image_(image), point_cloud_(point_cloud), normals_(normals), pixel_weights_3D_(pixel_weights_3D), IMAGE_WIDTH_(image.cols), IMAGE_HEIGHT_(image.rows), CAMERA_PARAMETERS_(camera_parameters), penalties_(penalties), statistics_(statistics), NUM_PIXELS_(image.cols * image.rows), NUM_LAYERS_(num_layers), SCENE_INDEX_(scene_index), NUM_ALL_PROPOSAL_ITERATIONS_(3), USE_PANORAMA_(use_panorama), ROOM_STRUCTURE_LAYER_INDEX_(num_layers - 1), NUM_PROPOSAL_TYPES_(5)
{
  //layer_inpainter_ = unique_ptr<LayerInpainter>(new LayerInpainter(image_, segmentation_, surface_depths_, penalties_, false, true));
  //layer_estimator_ = unique_ptr<LayerEstimator>(new LayerEstimator(image_, point_cloud_, segmentation_, surface_depths_, NUM_LAYERS_, penalties_, surface_colors_, SCENE_INDEX_));
  //  segment_graph_ = layer_estimator_->getSegmentGraph();
  
  //  calcSegmentations();
  
  Mat blurred_image;
  GaussianBlur(image_, blurred_image, cv::Size(3, 3), 0, 0);
  blurred_hsv_image_ = blurred_image.clone();
  //  blurred_image.convertTo(blurred_hsv_image_, CV_32FC3, 1.0 / 255);
  //cvtColor(blurred_hsv_image_, blurred_hsv_image_, CV_BGR2HSV);
  
  initializeCurrentSolution();
  
  proposal_type_indices_ = vector<int>(NUM_PROPOSAL_TYPES_);
  for (int c = 0; c < NUM_PROPOSAL_TYPES_; c++)
    proposal_type_indices_[c] = c;
  proposal_type_index_ptr_ = -1;
  all_proposal_iteration_ = 0;
}

ProposalDesigner::~ProposalDesigner()
{
}

void ProposalDesigner::setCurrentSolution(const vector<int> &current_solution_labels, const int current_solution_num_surfaces, const std::map<int, Segment> &current_solution_segments)
{
  //cout << "set current solution" << endl;
  
  //current_solution_ = current_solution;
  //current_solution_num_surfaces_ = current_solution_num_surfaces;
  //current_solution_surface_depths_ = current_solution_surface_depths;
  
  // for (map<int, Segment>::const_iterator segment_it = current_solution_segments.begin(); segment_it != current_solution_segments.end(); segment_it++)
  //   cout << segment_it->first << endl;
  
  // cout << current_solution_labels[43580] % (current_solution_num_surfaces + 1) << endl;
  // for (map<int, Segment>::const_iterator segment_it = current_solution_segments.begin(); segment_it != current_solution_segments.end(); segment_it++)
  //   cout << segment_it->second.getDepth(43580) << endl;
  // exit(1);
  
  map<int, int> surface_id_map;
  int new_surface_id = 0;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels[pixel];
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces + 1);
      if (surface_id == current_solution_num_surfaces)
        continue;
      if (surface_id_map.count(surface_id) == 0) {
        surface_id_map[surface_id] = new_surface_id;
        new_surface_id++;
      }
    }
    if (surface_id_map.size() == current_solution_num_surfaces)
      break;
  }
  surface_id_map[current_solution_num_surfaces] = new_surface_id;
  
  current_solution_segments_.clear();
  for (map<int, Segment>::const_iterator segment_it = current_solution_segments.begin(); segment_it != current_solution_segments.end(); segment_it++) {
    if (surface_id_map.count(segment_it->first) > 0) {
      current_solution_segments_[surface_id_map[segment_it->first]] = segment_it->second;
      //current_solution_segments_[surface_id_map[segment_it->first]].setVisiblePixels(segment_visible_pixels[surface_id_map[segment_it->first]]);
    }
  }
  
  
  vector<int> new_current_solution_labels(NUM_PIXELS_);
  int new_current_solution_num_surfaces = new_surface_id;
  
  
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels[pixel];
    if (checkLabelValidity(pixel, current_solution_label, current_solution_num_surfaces, current_solution_segments) == false) {
      cout << "invalid current label at pixel: " << pixel << endl;
      exit(1);
    }
    int new_label = 0;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces + 1);
      // if (pixel == 0)
      //   cout << surface_id << endl;
      new_label += surface_id_map[surface_id] * pow(new_current_solution_num_surfaces + 1, NUM_LAYERS_ - 1 - layer_index);
    }
    new_current_solution_labels[pixel] = new_label;
  }
  
  // vector<int> segment_layer_labels(new_current_solution_num_surfaces);
  // for (map<int, int>::const_iterator segment_it = surface_id_map.begin(); segment_it != surface_id_map.end(); segment_it++)
  //   segment_layer_labels[segment_it->second] = current_solution_labels[NUM_PIXELS_ + segment_it->first];
  // new_current_solution_labels.insert(new_current_solution_labels.end(), segment_layer_labels.begin(), segment_layer_labels.end());
  
  
  current_solution_labels_ = new_current_solution_labels;
  current_solution_num_surfaces_ = new_current_solution_num_surfaces;
  
  // writeDispImageFromSegments(current_solution_labels_, current_solution_num_surfaces_, current_solution_segments_, NUM_LAYERS_, IMAGE_WIDTH_, IMAGE_HEIGHT_, "Test/disp_image_0_new.bmp");
  // Mat disp_image_1 = imread("Test/disp_image_0.bmp", 0);
  // Mat disp_image_2 = imread("Test/disp_image_0_new.bmp", 0);
  // for (int y = 0; y < IMAGE_HEIGHT_; y++) {
  //   for (int x = 0; x < IMAGE_WIDTH_; x++) {
  //     int disp_1 = disp_image_1.at<uchar>(y, x);
  //     int disp_2 = disp_image_2.at<uchar>(y, x);
  //     if (disp_1 != disp_2)
  // 	cout << disp_1 - disp_2 << endl;
  //   }
  // }
  // exit(1);
}

bool ProposalDesigner::getProposal(int &iteration, vector<vector<int> > &proposal_labels, int &proposal_num_surfaces, map<int, Segment> &proposal_segments, string &proposal_type)
{
  // if (iteration != 2) {
  //   cout << current_solution_num_surfaces_ << '\t' << current_solution_segments_.size() << endl;
  //   exit(1);
  // }
  srand(time(0));
  proposal_iteration_ = iteration;
  bool test = true;
  if (test) {
    //generateLayerSwapProposal();
    //generateSegmentRefittingProposal();
    generateDesiredProposal();
    proposal_labels = proposal_labels_;
    proposal_num_surfaces = proposal_num_surfaces_;
    proposal_segments = proposal_segments_;
    proposal_type = proposal_type_;
    return true;
  }
  
  if (false) {
    bool generate_success = false;
    if (iteration == 0)
      generate_success = generateSegmentAddingProposal();
    else if (iteration == 1)
      generate_success = generateConcaveHullProposal(true);
    else {
      int proposal_type_index = iteration % 3;
      if (proposal_type_index == 0)
	generate_success = generateSegmentAddingProposal();
      // else if (proposal_type_index == 1)
      //   generate_success = generateConcaveHullProposal(true);
      else if (proposal_type_index == 1)
	generate_success = generateSegmentRefittingProposal();
      else
	//generate_success = generate
	generate_success = generateRandomMoveProposal();
    }
    proposal_labels = proposal_labels_;
    proposal_num_surfaces = proposal_num_surfaces_;
    proposal_segments = proposal_segments_;
    proposal_type = proposal_type_;
    return true;
  }
  
  if (false) {
    int num_attempts = 0;
    while (true) {
      bool generate_success = false;
      switch ((iteration + num_attempts) % 3) {
      case 0:
	generate_success = generateSegmentAddingProposal();
	break;
      case 1:
	generate_success = generatePixelGrowthProposal();
	break;
      case 2:
	generate_success = generateSegmentRefittingProposal();
	break;
      }
      if (generate_success)
	break;
      num_attempts++;
    }
    proposal_labels = proposal_labels_;
    proposal_num_surfaces = proposal_num_surfaces_;
    proposal_segments = proposal_segments_;
    proposal_type = proposal_type_;
    return true;
  }
  
  
  if (true) {
    int num_attempts = 0;
    while (true) {
      bool generate_success = false;
      switch ((iteration + num_attempts) % NUM_PROPOSAL_TYPES_) {
      case 0:
        generate_success = generateSegmentAddingProposal();
        break;
      case 1:
	generate_success = generateConcaveHullProposal();
	break;
      case 2:
        generate_success = generateSegmentRefittingProposal();
        break;
      case 3:
        generate_success = generateLayerSwapProposal();
        break;
      case 4:
        generate_success = generateBackwardMergingProposal();
        break;
      }
      if (generate_success)
        break;
      num_attempts++;
    }
    proposal_labels = proposal_labels_;
    proposal_num_surfaces = proposal_num_surfaces_;
    proposal_segments = proposal_segments_;
    proposal_type = proposal_type_;
    return true;
  }
  
  
  
  if (proposal_type_index_ptr_ < 0 || proposal_type_index_ptr_ >= proposal_type_indices_.size()) {
    random_shuffle(proposal_type_indices_.begin(), proposal_type_indices_.end());
    proposal_type_index_ptr_ = 0;
    all_proposal_iteration_++;
    if (all_proposal_iteration_ > NUM_ALL_PROPOSAL_ITERATIONS_)
      return false;
  }
  
  // if (iteration < segmentations_.size() + 2) {
  //   if (generateSegmentationProposal(iteration % segmentations_.size()) == false)
  //     return false;
  bool first_attempt = true;
  if (iteration == 0) {
    bool generate_success = generateSegmentAddingProposal(0);
    assert(generate_success);
  } else if (iteration == 1) {
    bool generate_success = generateConcaveHullProposal(true);
    assert(generate_success);
  // } else if (iteration == 2) {
  //   bool generate_success = generateSegmentAddingProposal(1);
  //   assert(generate_success);
  } else {
    while (true) {
      bool generate_success = false;
      if (single_surface_candidate_pixels_.size() > 0) {
	generate_success = generateSingleSurfaceExpansionProposal();
	// if (first_attempt && single_surface_candidate_pixels_.size() > 0)
	//   iteration--;
      } else {
	// double random_probability = randomProbability();
	// int proposal_type_index = random_probability * NUM_PROPOSAL_TYPES;
	// proposal_type_index = min(proposal_type_index, NUM_PROPOSAL_TYPES - 1);
	
	//	int proposal_type_index = rand() % NUM_PROPOSAL_TYPES;
	int proposal_type_index = proposal_type_indices_[proposal_type_index_ptr_];
	proposal_type_index_ptr_++;
	
	switch (proposal_type_index) {
	case 0:
	  generate_success = generateSegmentRefittingProposal();
	  break;
	case 1:
	  generate_success = generateConcaveHullProposal(true);
	  break;
	case 2:
	  generate_success = generateLayerSwapProposal();
          break;
	  // if (randomProbability() < 1.0 / NUM_PROPOSAL_TYPES / pow(1 - 1.0 / NUM_PROPOSAL_TYPES, 3))
	  //   if (generateLayerSwapProposal() == true)
	  //     break;
	case 3:
	  generate_success = generateBackwardMergingProposal();
          break;
	case 4:
	  generate_success = generateSegmentAddingProposal();
	  break;
	case 5:
	  generate_success = generateSingleSurfaceExpansionProposal();
          break;
	  // case 6:
	//   generate_success = generateBoundaryRefinementProposal();
	//   break;
	//case 6:
	  //generate_success = generateStructureExpansionProposal();
	  //break;
	default:
	  return false;
	  // case 8:
	  //   generate_success = generateBSplineSurfaceProposal();
	  //   break;
	  // case 3:
	  //   generate_success = generateSurfaceDilationProposal();
	  //   break;
	  
	  // case 7:
	  //   generate_success = generateInpaintingProposal();
	  //   break;
	}
      }      
      if (generate_success == true)
	break;
      first_attempt = false;
    }
  }
  
  proposal_labels = proposal_labels_;
  proposal_num_surfaces = proposal_num_surfaces_;
  proposal_segments = proposal_segments_;
  proposal_type = proposal_type_;
  return true;
}

bool ProposalDesigner::getLastProposal(vector<vector<int> > &proposal_labels, int &proposal_num_surfaces, map<int, Segment> &proposal_segments, string &proposal_type)
{
  NUM_LAYERS_++;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
    current_solution_labels_[pixel] += current_solution_num_surfaces_ * pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1);
  bool generate_success = generateSegmentAddingProposal();
  assert(generate_success);
  
  proposal_labels = proposal_labels_;
  proposal_num_surfaces = proposal_num_surfaces_;
  proposal_segments = proposal_segments_;
  proposal_type = proposal_type_;
  return true;
}

// string ProposalDesigner::getProposalType()
// {
//   switch (proposal_type_) {
//   case 0:
//     return "alpha expansion";
//   case 1:
//     return "connected region";
//   case 2:
//     return "convex structure";
//   }
// }

vector<int> ProposalDesigner::getInitialLabels()
{
  //generateEmptyRepresentationProposal();
  return proposal_labels_[0];
}

void ProposalDesigner::convertProposalLabelsFormat()
{
  int dim_1 = proposal_labels_.size();
  assert(dim_1 > 0);
  int dim_2 = proposal_labels_[0].size();
  assert(dim_2 > 0);
  vector<vector<int> > new_proposal_labels(dim_2, vector<int>(dim_1));
  for (int i = 0; i < dim_1; i++)
    for (int j = 0; j < dim_2; j++)
      new_proposal_labels[j][i] = proposal_labels_[i][j];
  proposal_labels_ = new_proposal_labels;
}

void ProposalDesigner::addIndicatorVariables(const int num_indicator_variables)
{
  //int num = num_indicator_variables == -1 ? pow(NUM_SURFACES_ + 1, NUM_LAYERS_) : num_indicator_variables;
  int num = num_indicator_variables == -1 ? NUM_LAYERS_ * proposal_num_surfaces_ : num_indicator_variables;
  vector<int> indicator_labels(2);
  indicator_labels[0] = 0;
  indicator_labels[1] = 1;
  for (int i = 0; i < num; i++)
    proposal_labels_.push_back(indicator_labels);
}

void ProposalDesigner::addSegmentLayerProposals(const bool restrict_segment_in_one_layer)
{
  //int num = num_indicator_variables == -1 ? pow(NUM_SURFACES_ + 1, NUM_LAYERS_) : num_indicator_variables;
  map<int, vector<bool> > segment_layer_mask_map;
  for (int segment_id = 0; segment_id < proposal_num_surfaces_; segment_id++)
    segment_layer_mask_map[segment_id] = vector<bool>(NUM_LAYERS_, false);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    vector<int> pixel_proposals = proposal_labels_[pixel];
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++) {
      for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
        int surface_id = *label_it / static_cast<int>(pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
	if (surface_id < proposal_num_surfaces_)
	  segment_layer_mask_map[surface_id][layer_index] = true;
      }
    }
  }
  
  vector<vector<int> > segment_layer_proposals(proposal_num_surfaces_);
  for (map<int, vector<bool> >::const_iterator segment_it = segment_layer_mask_map.begin(); segment_it != segment_layer_mask_map.end(); segment_it++) {
    vector<bool> layer_mask = segment_it->second;
    if (restrict_segment_in_one_layer == true) {
      segment_layer_proposals[segment_it->first].push_back(0);
      for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
        if (layer_mask[layer_index] == true)
	  segment_layer_proposals[segment_it->first].push_back(pow(2, NUM_LAYERS_ - 1 - layer_index));
    } else {
      for (int proposal = 0; proposal < static_cast<int>(pow(2, NUM_LAYERS_)); proposal++) {
	bool has_conflict = false;
	for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
	  if (layer_mask[layer_index] == 0 && proposal / static_cast<int>(pow(2, NUM_LAYERS_ - 1 - layer_index) + 0.5) % 2 == 1) {
	    has_conflict = true;
	    break;
	  }
	}
	if (has_conflict == false)
          segment_layer_proposals[segment_it->first].push_back(proposal);
      }
    }
  }
  for (int segment_id = 0; segment_id < current_solution_num_surfaces_; segment_id++)
    if (find(segment_layer_proposals[segment_id].begin(), segment_layer_proposals[segment_id].end(), current_solution_labels_[NUM_PIXELS_ + segment_id]) == segment_layer_proposals[segment_id].end())
      segment_layer_proposals[segment_id].push_back(current_solution_labels_[NUM_PIXELS_ + segment_id]);
  
  proposal_labels_.insert(proposal_labels_.end(), segment_layer_proposals.begin(), segment_layer_proposals.end());
}

bool ProposalDesigner::checkLabelValidity(const int pixel, const int label, const int num_surfaces, const map<int, Segment> &segments)
{
  double previous_depth = 0;
  //  bool inside_ROI = ROI_mask_[pixel];
  
  bool has_depth_conflict = false;
  bool has_same_label = false;
  bool empty_background = false;
  bool segmentation_inconsistency = false;
  bool background_inconsistency = false;
  bool has_layer_estimation_conflict = false;
  bool sub_region_extended = false;
  
  int foremost_non_empty_surface_id = -1;
  vector<bool> used_surface_id_mask(num_surfaces, false);
  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    int surface_id = label / static_cast<int>(pow(num_surfaces + 1, NUM_LAYERS_ - 1 - layer_index)) % (num_surfaces + 1);
    if (surface_id == num_surfaces) {
      continue;
    }
    double depth = segments.at(surface_id).getDepth(pixel);
    //if (pixel == 43580)
    //      cout << depth << endl;
    
    // if (layer_index == NUM_LAYERS_ - 1 && ((estimated_background_surfaces.count(segment_id) > 0 && surface_id != segment_id) || estimated_background_surfaces.count(surface_id) == 0)) {
    //   background_inconsistency = true;
    //   break;
    // }
    // if (confident_segment_layer_map.count(surface_id) > 0 && confident_segment_layer_map[surface_id] != layer_index) {
    //   has_layer_estimation_conflict = true;
    //   break;
    // }
    if (used_surface_id_mask[surface_id] == true) {
      has_same_label = true;
      break;
    }
    used_surface_id_mask[surface_id] = true;
    // if (segment_sub_region_mask[surface_id][pixel] == false) {
    //   sub_region_extended = true;
    //   break;
    // }
    if (foremost_non_empty_surface_id == -1) {
      foremost_non_empty_surface_id = surface_id;
      //      previous_depth = depth;
      
      // if (foremost_non_empty_surface_id != segment_id) {
      //   segmentation_inconsistency = true;
      //   break;
      // }
    }
    if (depth < previous_depth - statistics_.depth_conflict_threshold) {
      // if (pixel == 20803)
      // 	cout << "depth conflict: " << depth << '\t' << previous_depth << endl;
      has_depth_conflict = true;
      break;
    }
    previous_depth = depth;
  }
  //if (label / static_cast<int>(pow(num_surfaces + 1, 0)) % (num_surfaces + 1) == num_surfaces && label / static_cast<int>(pow(num_surfaces + 1, 1)) % (num_surfaces + 1) == num_surfaces)
  if (label / static_cast<int>(pow(num_surfaces + 1, 0)) % (num_surfaces + 1) == num_surfaces)
    empty_background = true;
  
  if (has_depth_conflict == false && has_same_label == false && empty_background == false) // && background_inconsistency == false && has_layer_estimation_conflict == false && sub_region_extended == false && segmentation_inconsistency == false)
    return true;
  else
    return false;
}

void ProposalDesigner::writeSegmentationImage(const vector<int> &segmentation, const string filename)
{ 
  Mat segmentation_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
  map<int, int> segment_center_x;
  map<int, int> segment_center_y;
  map<int, int> segment_pixel_counter;
  map<int, int> color_table;
  for (int i = 0; i < NUM_PIXELS_; i++) {
    int x = i % IMAGE_WIDTH_;
    int y = i / IMAGE_WIDTH_;
    
    int surface_id = segmentation[i];
    if (color_table.count(surface_id) == 0)
      color_table[surface_id] = rand() % (256 * 256 * 256);
    int surface_color = color_table[surface_id];
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

void ProposalDesigner::writeSegmentationImage(const Mat &segmentation_image, const string filename)
{
  map<int, int> segment_center_x;
  map<int, int> segment_center_y;
  map<int, int> segment_pixel_counter;
  for (int i = 0; i < NUM_PIXELS_; i++) {
    int x = i % IMAGE_WIDTH_;
    int y = i / IMAGE_WIDTH_;
    
    Vec3b color = segmentation_image.at<Vec3b>(y, x);
    int surface_id = color[0] * 256 * 256 + color[1] * 256 + color[2];
    
    segment_center_x[surface_id] += x;
    segment_center_y[surface_id] += y;
    segment_pixel_counter[surface_id]++;
  }
  Mat image = segmentation_image.clone();
  for (map<int, int>::const_iterator segment_it = segment_pixel_counter.begin(); segment_it != segment_pixel_counter.end(); segment_it++) {
    Point origin(segment_center_x[segment_it->first] / segment_it->second, segment_center_y[segment_it->first] / segment_it->second);
    char *text = new char[10];
    sprintf(text, "%d", segment_it->first);
    putText(image, text, origin, FONT_HERSHEY_PLAIN, 0.6, Scalar(0, 0, 255, 1));
  }
  //  stringstream segmentation_image_filename;
  //  segmentation_image_filename << "Results/segmentation_image.bmp";
  imwrite(filename.c_str(), image);
}

bool ProposalDesigner::generateSegmentRefittingProposal()
{
  cout << "generate segment refitting proposal" << endl;
  proposal_type_ = "segment_refitting_proposal";
  
  vector<set<int> > layer_segments(NUM_LAYERS_);
  map<int, map<int, vector<int> > > segment_layer_visible_pixels;
  map<int, map<int, vector<int> > > segment_layer_occluded_pixels;
  vector<bool> background_segment_mask(NUM_LAYERS_, false);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    bool is_visible = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
        if (is_visible == true) {
	  segment_layer_visible_pixels[surface_id][layer_index].push_back(pixel);
          is_visible = false;
        } else
	  segment_layer_occluded_pixels[surface_id][layer_index].push_back(pixel);
        layer_segments[layer_index].insert(surface_id);
        if (layer_index == NUM_LAYERS_ - 1)
          background_segment_mask[surface_id] = true;
      }
    }
  }
  
  static bool use_plane_or_bspline = false;
  use_plane_or_bspline = !use_plane_or_bspline;
  
  proposal_segments_ = current_solution_segments_;
  int new_proposal_segment_index = current_solution_num_surfaces_;
  
  vector<vector<set<int> > > layer_pixel_segment_indices_map(NUM_LAYERS_, vector<set<int> >(NUM_PIXELS_));
  static int desired_segment_type = 2;
  desired_segment_type = desired_segment_type == 0 ? 2 : 0;
  
  for (int segment_type = 0; segment_type <= 2; segment_type += 2) {
    for (map<int, Segment>::const_iterator segment_it = current_solution_segments_.begin(); segment_it != current_solution_segments_.end(); segment_it++) {
      //      if (segment_it->first != 11 || segment_type == 0)
      //continue;
      for (map<int, vector<int> >::const_iterator layer_it = segment_layer_visible_pixels[segment_it->first].begin(); layer_it != segment_layer_visible_pixels[segment_it->first].end(); layer_it++) {
	if (layer_it->first == ROOM_STRUCTURE_LAYER_INDEX_)
	  continue;
	ImageMask fitting_mask = ImageMask(layer_it->second, IMAGE_WIDTH_, IMAGE_HEIGHT_) - getInvalidPointMask(point_cloud_, IMAGE_WIDTH_, IMAGE_HEIGHT_);
	int previous_segment_type = current_solution_segments_[segment_it->first].getSegmentType();
	
        vector<int> fitting_pixels = fitting_mask.getPixels();
        if (fitting_pixels.size() < statistics_.small_segment_num_pixels_threshold || (segment_type > 0 && fitting_pixels.size() > statistics_.bspline_surface_num_pixels_threshold && previous_segment_type == 0)) {
	  continue;
        }
	
	//ImageMask previous_fitting_mask = current_solution_segments_[segment_it->first].getFittingMask();
        ImageMask previous_fitting_mask = current_solution_segments_[segment_it->first].getFittingMask();
	previous_fitting_mask.dilate();
        ImageMask common_region_mask = previous_fitting_mask - (previous_fitting_mask - fitting_mask);
	ImageMask union_region_mask = previous_fitting_mask + fitting_mask;
        if (common_region_mask.getNumPixels() > union_region_mask.getNumPixels() * statistics_.segment_refitting_common_ratio_threshold && segment_type == current_solution_segments_[segment_it->first].getSegmentType()) {
	  continue;
        }
	
	//imwrite("Test/fitting_mask_" + to_string(segment_it->first) + "_previous.bmp", previous_fitting_mask.drawMaskImage());
	//imwrite("Test/fitting_mask_" + to_string(segment_it->first) + "_new.bmp", fitting_mask.drawMaskImage());
        Segment refitted_segment = Segment(IMAGE_WIDTH_, IMAGE_HEIGHT_, CAMERA_PARAMETERS_, statistics_, USE_PANORAMA_);
	refitted_segment.refit(image_, point_cloud_, normals_, CAMERA_PARAMETERS_, fitting_mask, ImageMask(segment_layer_occluded_pixels[segment_it->first][layer_it->first], IMAGE_WIDTH_, IMAGE_HEIGHT_), segment_type);
	if (refitted_segment.getValidity() == false)
	  continue;
	proposal_segments_[new_proposal_segment_index] = refitted_segment;
	
	cout << "segment refitting: " << '\t' << segment_it->first << '\t' << new_proposal_segment_index << '\t' << segment_type << endl;
	// vector<int> segment_pixels;
	// segment_pixels.insert(segment_pixels.end(), layer_it->second.begin(), layer_it->second.end());
	// segment_pixels.insert(segment_pixels.end(), segment_layer_occluded_pixels[segment_it->first][layer_it->first].begin(), segment_layer_occluded_pixels[segment_it->first][layer_it->first].end());
	vector<int> segment_pixels = refitted_segment.getSegmentPixels();
	for (vector<int>::const_iterator pixel_it = segment_pixels.begin(); pixel_it != segment_pixels.end(); pixel_it++) {
	  layer_pixel_segment_indices_map[layer_it->first][*pixel_it].insert(new_proposal_segment_index);
	}
	//cout << segment_type << '\t' << segment_it->first << '\t' << new_proposal_segment_index << endl;
	new_proposal_segment_index++;
      }
    }
  }
  
  // imwrite("Test/mask_3.bmp", proposal_segments_[3].getMask().drawMaskImage());
  // imwrite("Test/mask_5.bmp", proposal_segments_[5].getMask().drawMaskImage());
  // imwrite("Test/mask_11.bmp", proposal_segments_[11].getMask().drawMaskImage());
  // imwrite("Test/mask_26.bmp", proposal_segments_[26].getMask().drawMaskImage());
  // imwrite("Test/mask_28.bmp", proposal_segments_[28].getMask().drawMaskImage());
  // imwrite("Test/mask_45.bmp", proposal_segments_[45].getMask().drawMaskImage());
  // vector<double> plane = proposal_segments_[26].getPlane();
  // for (int c = 0; c < 4; c++)
  //   cout << plane[c] << endl;
  // Mat depth_image(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //   if (proposal_segments_[11].getDepth(pixel) <= 0)
  //     depth_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = 0;
  //   else
  //     depth_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = min(300 / proposal_segments_[11].getDepth(pixel), 255.0);
  // }
  // imwrite("Test/depth_image_2.bmp", depth_image);
  //exit(1);
  
  
  // for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
  //   for (set<int>::const_iterator segment_it = layer_segments[layer_index].begin(); segment_it != layer_segments[layer_index].end(); segment_it++)
  //     cout << layer_index << '\t' << *segment_it << endl;
  
  map<int, map<int, bool> > segment_layer_certainty_map;
  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
    for (set<int>::const_iterator segment_it = layer_segments[layer_index].begin(); segment_it != layer_segments[layer_index].end(); segment_it++)
      segment_layer_certainty_map[*segment_it][layer_index] = true;
  
  layer_pixel_segment_indices_map = fillLayers(blurred_hsv_image_, point_cloud_, normals_, proposal_segments_, penalties_, statistics_, NUM_LAYERS_, current_solution_labels_, current_solution_num_surfaces_, segment_layer_certainty_map, USE_PANORAMA_, false, false, "segment_refitting_" + to_string(proposal_iteration_), layer_pixel_segment_indices_map);
  
  proposal_num_surfaces_ = proposal_segments_.size();
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  
  //  int max_num_proposals = 0;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    map<int, set<int> > pixel_layer_surfaces_map;
    //vector<set<int> > layer_surface_ids_map = pixel_layer_surface_ids_map[pixel];
    //vector<int> layer_segment_index_map = pixel_layer_segment_index_map[pixel];
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
	pixel_layer_surfaces_map[layer_index].insert(surface_id);
      } else {
	pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);
      }
    }
    
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
      pixel_layer_surfaces_map[layer_index].insert(layer_pixel_segment_indices_map[layer_index][pixel].begin(), layer_pixel_segment_indices_map[layer_index][pixel].end());
    
    //    for (int layer_index = 0; layer_index < ROOM_STRUCTURE_LAYER_INDEX_; layer_index++)
    //      pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);
    
    //pixel_layer_surfaces_map[0].erase(11);
    
    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);
    
    // if (pixel == 11084) {
    //   for (set<int>::const_iterator segment_it = segment_new_segments_map[1].begin(); segment_it != segment_new_segments_map[1].end(); segment_it++)
    // 	cout << *segment_it << endl;
    //   for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++) {
    //  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    //    int proposal_surface_id = *label_it / static_cast<int>(pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
    //    cout << proposal_surface_id << '\t';
    //  }
    //  cout << endl;
    //   }
    //   exit(1);
    // }
    
    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
        valid_pixel_proposals.push_back(*label_it);
    
    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      // for (int proposal_index = 0; proposal_index < pixel_proposal.size(); proposal_index++)
      //   cout << pixel_proposal[proposal_index] << endl;
      exit(1);
    }      
    
    proposal_labels_[pixel] = valid_pixel_proposals;
    
    
    if (current_solution_num_surfaces_ > 0 && false) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
        cout << "has no current solution label at pixel: " << pixel << endl;
        exit(1);
      }
    }
    
    // if (valid_pixel_proposals.size() > max_num_proposals) {
    //   cout << "max number of proposals: " << pixel % IMAGE_WIDTH_ << '\t' << pixel / IMAGE_WIDTH_ << '\t' << valid_pixel_proposals.size() << endl;
    //   max_num_proposals = valid_pixel_proposals.size();
    // }    
  }
  
  addIndicatorVariables();
  return true;
}

bool ProposalDesigner::generateSingleSurfaceExpansionProposal(const int denoted_expansion_segment_id)
{
  cout << "generate single surface expansion proposal" << endl;
  proposal_type_ = "single_surface_expansion_proposal";
  
  if (single_surface_candidate_pixels_.size() == 0) {
    single_surface_candidate_pixels_.assign(NUM_PIXELS_ * 2, -1);
    for (int pixel = 0; pixel < NUM_PIXELS_ * 2; pixel++)
      single_surface_candidate_pixels_[pixel] = pixel;
  }
  
  int expansion_segment_id = denoted_expansion_segment_id;
  int expansion_type = rand() % 2;
  if (current_solution_segments_.count(expansion_segment_id) == 0) {
    //    int random_pixel = rand() % NUM_PIXELS_;
    int random_pixel = single_surface_candidate_pixels_[rand() % single_surface_candidate_pixels_.size()];
    int current_solution_label = current_solution_labels_[random_pixel % NUM_PIXELS_];
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
	expansion_segment_id = surface_id;
	break;
      }
    }
    expansion_type = random_pixel / NUM_PIXELS_;
    
    // map<int, double> segment_confidence_map;
    // double confidence_sum = 0;
    // for (map<int, Segment>::const_iterator segment_it = current_solution_segments_.begin(); segment_it != current_solution_segments_.end(); segment_it++) {
    //   double confidence = segment_it->second.getConfidence();
    //   segment_confidence_map[segment_it->first] = confidence;
    //   confidence_sum += confidence;
    // }
    
    // double selected_confidence = randomProbability() * confidence_sum;
    // confidence_sum = 0;
    
    // for (map<int, double>::const_iterator segment_it = segment_confidence_map.begin(); segment_it != segment_confidence_map.end(); segment_it++) {
    //   confidence_sum += segment_it->second;
    //   if (confidence_sum >= selected_confidence) {
    // 	expansion_segment_id = segment_it->first;
    // 	break;
    //   }
    // }
    // assert(expansion_segment_id != -1);
  }
  
  
  // int num_attempts = 0;
  // while (true) {
  //   if (num_attempts >= current_solution_num_surfaces_)
  //     return false;
  //   num_attempts++;
  //   expansion_segment_id = segment_id == -1 ? rand() % current_solution_num_surfaces_ : segment_id;
  
  //   if (current_solution_segments_.count(expansion_segment_id) == 0)
  //     continue;
  
  //   if (current_solution_segments_[expansion_segment_id].getConfidence() < 0.5)
  //     continue;
  //   break;
  // }
  
  map<int, int> expansion_segment_layer_counter;
  bool is_occluded = false;
  vector<bool> expansion_segment_visible_pixel_mask(NUM_PIXELS_, false);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    bool is_visible = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id == expansion_segment_id) {
        expansion_segment_layer_counter[layer_index]++;
	if (is_visible == false) {
	  is_occluded = true;
	  break;
	} else
	  expansion_segment_visible_pixel_mask[pixel] = true;
      }
      if (surface_id < current_solution_num_surfaces_)
	is_visible = false;
    }
  }
  vector<int> new_single_surface_candidate_pixels;
  for (vector<int>::const_iterator pixel_it = single_surface_candidate_pixels_.begin(); pixel_it != single_surface_candidate_pixels_.end(); pixel_it++)
    if (*pixel_it / NUM_PIXELS_ != expansion_type || expansion_segment_visible_pixel_mask[*pixel_it % NUM_PIXELS_] == false)
      new_single_surface_candidate_pixels.push_back(*pixel_it);
  single_surface_candidate_pixels_ = new_single_surface_candidate_pixels;
  
  
  if (expansion_segment_layer_counter.size() > 1 && expansion_type == 1)
    return false;
  if (is_occluded && expansion_type == 0)
    return false;
  
  // if (expansion_segment_layer_counter.size() > 1)
  //   expansion_type = 0;
  // else if (is_occluded == true)
  //   expansion_type = 1;
  // else
  //   expansion_type = randomProbability() < 0.5 ? 0 : 1;
  
  //  int expansion_segment_layer_index = expansion_segment_layer_counter.begin()->first;
  
  int expansion_segment_layer_index = -1;
  int max_layer_count = 0;
  for (map<int, int>::const_iterator layer_it = expansion_segment_layer_counter.begin(); layer_it != expansion_segment_layer_counter.end(); layer_it++) {
    if (layer_it->second > max_layer_count) {
      expansion_segment_layer_index = layer_it->first;
      max_layer_count = layer_it->second;
    }
  }
  if (expansion_segment_layer_index == -1)
    return false;
  
  
  cout << "segment: " << expansion_segment_id << "\texpansion type: " << expansion_type << endl;
  
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    
    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      pixel_layer_surfaces_map[layer_index].insert(surface_id);
    }
    if (expansion_type == 0) {
      for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
	pixel_layer_surfaces_map[layer_index].insert(expansion_segment_id);
      for (int layer_index = 0; layer_index < ROOM_STRUCTURE_LAYER_INDEX_; layer_index++)
	pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);
    } else {
      pixel_layer_surfaces_map[expansion_segment_layer_index].insert(expansion_segment_id);
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - expansion_segment_layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_ && surface_id != expansion_segment_id)
	for (int target_layer_index = 0; target_layer_index < expansion_segment_layer_index; target_layer_index++)
	  pixel_layer_surfaces_map[target_layer_index].insert(surface_id);
      for (int target_layer_index = 0; target_layer_index < expansion_segment_layer_index; target_layer_index++)
	pixel_layer_surfaces_map[target_layer_index].insert(proposal_num_surfaces_);
    }
    
    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);
    
    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
	valid_pixel_proposals.push_back(*label_it);
    
    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      exit(1);
    }      
    
    proposal_labels_[pixel] = valid_pixel_proposals;
    
    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
	cout << "has no current solution label at pixel: " << pixel << endl;
	exit(1);
      }
    }
    
    
    // if (pixel == 132 * IMAGE_WIDTH_ + 57) {
    //   for (vector<int>::const_iterator label_it = valid_pixel_proposals.begin(); label_it != valid_pixel_proposals.end(); label_it++) {
    // 	for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    // 	  int proposal_surface_id = *label_it / static_cast<int>(pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
    // 	  cout << proposal_surface_id << '\t';
    // 	}
    // 	cout << endl;
    //   }
    //   exit(1);
    // }
  }
  
  //addSegmentLayerProposals(true);
  addIndicatorVariables();
  
  return true;
}

bool ProposalDesigner::generateLayerSwapProposal()
{
  cout << "generate layer swap proposal" << endl;
  proposal_type_ = "layer_swap_proposal";
  
  // cout << current_solution_segments_[5].calcPixelFittingCost(blurred_hsv_image_, point_cloud_, normals_, 23314, penalties_, 1, false) << '\t' << current_solution_segments_[4].calcPixelFittingCost(blurred_hsv_image_, point_cloud_, normals_, 23314, penalties_, 1, false) << endl;
  // exit(1);
  
  vector<set<int> > layer_segments(NUM_LAYERS_);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
        layer_segments[layer_index].insert(surface_id);
      }
    }
  }
  
  map<int, map<int, bool> > segment_layer_certainty_map = swapLayers(IMAGE_WIDTH_, IMAGE_HEIGHT_, current_solution_segments_, current_solution_labels_, NUM_LAYERS_, statistics_, USE_PANORAMA_);
  
  for (map<int, map<int, bool> >::const_iterator segment_it = segment_layer_certainty_map.begin(); segment_it != segment_layer_certainty_map.end(); segment_it++)
    for (map<int, bool>::const_iterator layer_it = segment_it->second.begin(); layer_it != segment_it->second.end(); layer_it++)
      cout << "layer swap: " << segment_it->first << '\t' << layer_it->first << '\t' << layer_it->second << endl;
  //exit(1);
  
  // //vector<int> layer_map;
  // //for (int layer_index = 0; layer_index < ROOM_STRUCTURE_LAYER_INDEX_; layer_index++)
  // //layer_map.push_back(layer_index);
  // //random_shuffle(layer_map.begin(), layer_map.end());
  // for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
  //   for (set<int>::const_iterator segment_it = layer_segments[layer_index].begin(); segment_it != layer_segments[layer_index].end(); segment_it++)
  //     if (layer_index < ROOM_STRUCTURE_LAYER_INDEX_) {
  // 	segment_layer_certainty_map[*segment_it][rand() % ROOM_STRUCTURE_LAYER_INDEX_] = false;
  
  // 	// if (*segment_it == 4)
  // 	//   segment_layer_certainty_map[*segment_it][2] = false;
  // 	// else if (*segment_it == 5)
  // 	//   segment_layer_certainty_map[*segment_it][1] = false;
  // 	// else
  //       //   segment_layer_certainty_map[*segment_it][layer_index] = false;
  //     } else
  // 	segment_layer_certainty_map[*segment_it][layer_index] = true;

  segment_layer_certainty_map.clear();
  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
    for (set<int>::const_iterator segment_it = layer_segments[layer_index].begin(); segment_it != layer_segments[layer_index].end(); segment_it++)
      if (layer_index < NUM_LAYERS_ - 1)
	segment_layer_certainty_map[*segment_it][layer_index] = true;
      else
	segment_layer_certainty_map[*segment_it][layer_index] = false;
  
  vector<vector<set<int> > > layer_pixel_segment_indices_map = fillLayers(blurred_hsv_image_, point_cloud_, normals_, current_solution_segments_, penalties_, statistics_, NUM_LAYERS_, current_solution_labels_, current_solution_num_surfaces_, segment_layer_certainty_map, USE_PANORAMA_, false, false, "layer_swap_" + to_string(proposal_iteration_));
  
  
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];    
    
    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      pixel_layer_surfaces_map[layer_index].insert(surface_id);
    }
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
      pixel_layer_surfaces_map[layer_index].insert(layer_pixel_segment_indices_map[layer_index][pixel].begin(), layer_pixel_segment_indices_map[layer_index][pixel].end());
    
    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);    
    
    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
        valid_pixel_proposals.push_back(*label_it);
    
    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      // for (int proposal_index = 0; proposal_index < pixel_proposal.size(); proposal_index++)
      //   cout << pixel_proposal[proposal_index] << endl;
      exit(1);
    }      
    
    proposal_labels_[pixel] = valid_pixel_proposals;
    
    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
        cout << "has no current solution label at pixel: " << pixel << endl;
        exit(1);
      }
    }
    
    
    // if (pixel == 134 * IMAGE_WIDTH_ + 155) {
    //   cout << current_solution_segments_[12].getDepth(pixel) << '\t' << current_solution_segments_[1].getDepth(pixel) << endl;
    //   for (vector<int>::const_iterator label_it = valid_pixel_proposals.begin(); label_it != valid_pixel_proposals.end(); label_it++) {
    //     for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    //       int proposal_surface_id = *label_it / static_cast<int>(pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
    //       cout << proposal_surface_id << '\t';
    //     }
    //     cout << endl;
    //   }
    //   exit(1);
    // }
  }
  
  //addSegmentLayerProposals(true);
  addIndicatorVariables();
  
  return true;
}

bool ProposalDesigner::generateConcaveHullProposal(const bool consider_background)
{
  cout << "generate concave hull proposal" << endl;
  proposal_type_ = "concave_hull_proposal";
  
  vector<int> visible_segmentation(NUM_PIXELS_, -1);
  vector<set<int> > segment_layers(current_solution_num_surfaces_);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    bool is_visible = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
        double depth = current_solution_segments_[surface_id].getDepth(pixel);
        if (is_visible) {
          visible_segmentation[pixel] = surface_id;
          is_visible = false;
        }
	segment_layers[surface_id].insert(layer_index);
      }
    }
  }
  
  const double NUM_FITTING_PIXELS_THRESHOLD_RATIO = 0.8;
  map<int, int> segment_num_pixels;
  map<int, int> segment_num_fitting_pixels;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int segment_id = visible_segmentation[pixel];
    segment_num_pixels[segment_id]++;
    if (current_solution_segments_.at(segment_id).checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, pixel))
      segment_num_fitting_pixels[segment_id]++;
  }
  set<int> invalid_segments;
  for (map<int, int>::const_iterator segment_it = segment_num_pixels.begin(); segment_it != segment_num_pixels.end(); segment_it++)
    if (segment_num_fitting_pixels[segment_it->first] < segment_it->second * NUM_FITTING_PIXELS_THRESHOLD_RATIO) {
      invalid_segments.insert(segment_it->first);
      cout << "invalid segments: " << segment_it->first << endl;
    }
  
  map<int, map<int, bool> > segment_layer_certainty_map;
  if (false) {
    OrthogonalStructureFinder orthogonal_structure_finder(IMAGE_WIDTH_, IMAGE_HEIGHT_, point_cloud_, current_solution_segments_, visible_segmentation, vector<bool>(NUM_PIXELS_, true), statistics_, invalid_segments);
    vector<pair<double, vector<int> > > orthogonal_hulls;
    const int MAX_NUM_HORIZONTAL_SURFACES = 2;
    const int MAX_NUM_VERTICAL_SURFACES = 3;
    for (int num_horizontal_surfaces = 0; num_horizontal_surfaces <= MAX_NUM_HORIZONTAL_SURFACES; num_horizontal_surfaces++) {
      for (int num_vertical_surfaces = 0; num_vertical_surfaces <= MAX_NUM_VERTICAL_SURFACES; num_vertical_surfaces++) {
	if (num_horizontal_surfaces == 0 && num_vertical_surfaces == 0)
	  continue;
	vector<pair<double, vector<int> > > orthogonal_hulls_subset = orthogonal_structure_finder.calcOrthogonalStructures(num_horizontal_surfaces, num_vertical_surfaces);
	orthogonal_hulls.insert(orthogonal_hulls.end(), orthogonal_hulls_subset.begin(), orthogonal_hulls_subset.end());
      }
    }
    if (orthogonal_hulls.size() == 0) {
      cout << "Cannot find orthogonal hull." << endl;
      return false;
    }
    vector<int> orthogonal_hull;
    double best_score = -1000000;
    for (vector<pair<double, vector<int> > >::const_iterator orthogonal_hull_it = orthogonal_hulls.begin(); orthogonal_hull_it != orthogonal_hulls.end(); orthogonal_hull_it++) {
      if (orthogonal_hull_it->first > best_score) {
	orthogonal_hull = orthogonal_hull_it->second;
	best_score = orthogonal_hull_it->first;
      }
    }
    
    // double sum_score = 0;
    // for (vector<pair<double, vector<int> > >::const_iterator orthogonal_hull_it = orthogonal_hulls.begin(); orthogonal_hull_it != orthogonal_hulls.end(); orthogonal_hull_it++)
    //   sum_score += orthogonal_hull_it->first;
    // vector<int> orthogonal_hull;
    // double selected_score = randomProbability() * sum_score;
    // sum_score = 0;
    // for (vector<pair<double, vector<int> > >::const_iterator orthogonal_hull_it = orthogonal_hulls.begin(); orthogonal_hull_it != orthogonal_hulls.end(); orthogonal_hull_it++) {
    //   cout << orthogonal_hull_it->first << '\t' << selected_score << endl;
    //   orthogonal_hull = orthogonal_hull_it->second;
    //   sum_score += orthogonal_hull_it->first;
    //   if (sum_score > selected_score)
    //     break;
    // }
    
    map<int, Vec3b> color_table;
    Mat ori_region_image(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
    Mat orthogonal_hull_image(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      int segment_id = visible_segmentation[pixel];
      if (color_table.count(segment_id) == 0) {
	int gray_value = rand() % 256;
	color_table[segment_id] = Vec3b(gray_value, gray_value, gray_value);
      }
      ori_region_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = color_table[segment_id];
      int orthogonal_hull_segment_id = orthogonal_hull[pixel];
      if (orthogonal_hull_segment_id < 0)
	continue;
      if (color_table.count(orthogonal_hull_segment_id) == 0) {
	int gray_value = rand() % 256;
	color_table[orthogonal_hull_segment_id] = Vec3b(gray_value, gray_value, gray_value);
      }
      orthogonal_hull_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = color_table[orthogonal_hull_segment_id];
    }
    
    imwrite("Test/background_ori_region_image_.bmp", ori_region_image);
    imwrite("Test/background_orthogonal_hull_image.bmp", orthogonal_hull_image);
    
    set<int> orthogonal_hull_surfaces;
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
      orthogonal_hull_surfaces.insert(orthogonal_hull[pixel]);
    
    for (int segment_id = 0; segment_id < current_solution_num_surfaces_; segment_id++) {
      if (orthogonal_hull_surfaces.count(segment_id) > 0)
	segment_layer_certainty_map[segment_id][ROOM_STRUCTURE_LAYER_INDEX_] = true;
      else {
	set<int> layers = segment_layers[segment_id];
	layers.erase(ROOM_STRUCTURE_LAYER_INDEX_);
	if (layers.size() == 0)
	  layers.insert(ROOM_STRUCTURE_LAYER_INDEX_ - 1);
	for (set<int>::const_iterator layer_it = layers.begin(); layer_it != layers.end(); layer_it++)
	  segment_layer_certainty_map[segment_id][*layer_it] = true;
      }
    }
  }
  
  // for (int segment_id = 0; segment_id < current_solution_num_surfaces_; segment_id++) {
  //   if (segment_layers[segment_id].count(ROOM_STRUCTURE_LAYER_INDEX_) > 0) {
  //     segment_layer_certainty_map[segment_id][ROOM_STRUCTURE_LAYER_INDEX_] = false;
  //     segment_layer_certainty_map[segment_id][ROOM_STRUCTURE_LAYER_INDEX_ - 1] = true;
  //   } else {
  //     set<int> layers = segment_layers[segment_id];
  //     for (set<int>::const_iterator layer_it = layers.begin(); layer_it != layers.end(); layer_it++)
  // 	segment_layer_certainty_map[segment_id][*layer_it] = true;
  //   }
  // }
  
  segment_layer_certainty_map = swapLayers(IMAGE_WIDTH_, IMAGE_HEIGHT_, current_solution_segments_, current_solution_labels_, NUM_LAYERS_, statistics_, USE_PANORAMA_, true, invalid_segments);

  for (map<int, map<int, bool> >::const_iterator segment_it = segment_layer_certainty_map.begin(); segment_it != segment_layer_certainty_map.end(); segment_it++)
    for (map<int, bool>::const_iterator layer_it = segment_it->second.begin(); layer_it != segment_it->second.end(); layer_it++)
      cout << "concave hull: " << segment_it->first << '\t' << layer_it->first << '\t' << layer_it->second << endl;

  vector<vector<set<int> > > layer_pixel_segment_indices_map = fillLayers(blurred_hsv_image_, point_cloud_, normals_, current_solution_segments_, penalties_, statistics_, NUM_LAYERS_, current_solution_labels_, current_solution_num_surfaces_, segment_layer_certainty_map, USE_PANORAMA_, false, false, "concave_hull_" + to_string(proposal_iteration_));
  
  
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  int max_num_proposals = 0;  
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    
    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      pixel_layer_surfaces_map[layer_index].insert(surface_id);
    }
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
      pixel_layer_surfaces_map[layer_index].insert(layer_pixel_segment_indices_map[layer_index][pixel].begin(), layer_pixel_segment_indices_map[layer_index][pixel].end());

    //for (int target_layer_index = 0; target_layer_index < ROOM_STRUCTURE_LAYER_INDEX_; target_layer_index++)
    //pixel_layer_surfaces_map[target_layer_index].insert(proposal_num_surfaces_);
    
    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);
    
    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
        valid_pixel_proposals.push_back(*label_it);
    
    // if (pixel == 20803) {
    //   for (map<int, set<int> >::const_iterator layer_it = pixel_layer_surfaces_map.begin(); layer_it != pixel_layer_surfaces_map.end(); layer_it++)
    //     for (set<int>::const_iterator segment_it = layer_it->second.begin(); segment_it != layer_it->second.end(); segment_it++)
    // 	  cout << layer_it->first << '\t' << *segment_it << endl;
    //   for (vector<int>::const_iterator label_it = valid_pixel_proposals.begin(); label_it != valid_pixel_proposals.end(); label_it++) {
    //     for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    //       int proposal_surface_id = *label_it / static_cast<int>(pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
    //       cout << proposal_surface_id << '\t';
    //     }
    //     cout << endl;
    //   }
    //   cout << proposal_num_surfaces_ << '\t' << NUM_LAYERS_ << endl;
    //   cout << current_solution_label << '\t' << valid_pixel_proposals[0] << '\t' << valid_pixel_proposals[1] << endl;
    //   exit(1);
    // }
    
    // if (valid_pixel_proposals.size() > 1)
    //   cout << "yes" << endl;
    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      // for (int proposal_index = 0; proposal_index < pixel_proposal.size(); proposal_index++)
      //   cout << pixel_proposal[proposal_index] << endl;
      exit(1);
    }      
    
    proposal_labels_[pixel] = valid_pixel_proposals;
    
    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
        cout << "has no current solution label at pixel: " << pixel << endl;
        exit(1);
      }
    }
  }
  
  //addSegmentLayerProposals(true);
  addIndicatorVariables();
  return true;
}

bool ProposalDesigner::generateSegmentAddingProposal(const int denoted_segment_adding_type)
{
  cout << "generate segment adding proposal" << endl;
  proposal_type_ = "segment_adding_proposal";
  
  int segment_adding_type = denoted_segment_adding_type;
  if (segment_adding_type == -1)
    segment_adding_type = current_solution_num_surfaces_ == 0 ? 0 : rand() % 3 != 0 ? 1 : 1;
  segment_adding_type = current_solution_num_surfaces_ == 0 ? 0 : 1;
  
  vector<double> visible_depths(NUM_PIXELS_, -1);
  vector<int> visible_surface_ids(NUM_PIXELS_, -1);
  vector<int> visible_layer_indices(NUM_PIXELS_, -1);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
        double depth = current_solution_segments_[surface_id].getDepth(pixel);
        visible_depths[pixel] = depth;
        visible_surface_ids[pixel] = surface_id;
	visible_layer_indices[pixel] = layer_index;
        break;
      }
    }
  }
  vector<set<int> > layer_segments(NUM_LAYERS_);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
	layer_segments[layer_index].insert(surface_id);
      }
    }
  }
  
  vector<bool> bad_fitting_pixel_mask_vec(NUM_PIXELS_, true);
  if (segment_adding_type == 1) {
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      int current_solution_label = current_solution_labels_[pixel];
      for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
	int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
	if (surface_id < current_solution_num_surfaces_) {
	  if (current_solution_segments_.at(surface_id).checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, pixel) == true)
	    bad_fitting_pixel_mask_vec[pixel] = false;
	  break;
	}
      }
    }
  }
  
  ImageMask bad_fitting_pixel_mask(bad_fitting_pixel_mask_vec, IMAGE_WIDTH_, IMAGE_HEIGHT_);
  for (int iteration = 0; iteration < 1; iteration++) {
    bad_fitting_pixel_mask.erode();
    bad_fitting_pixel_mask.dilate();
  }
  bad_fitting_pixel_mask -= getInvalidPointMask(point_cloud_, IMAGE_WIDTH_, IMAGE_HEIGHT_);
  //ImageMask bad_fitting_pixel_mask(true, IMAGE_WIDTH_, IMAGE_	HEIGHT_);
  
  vector<double> pixel_weights(NUM_PIXELS_, 1);
  if (segment_adding_type == 1) {
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      if (bad_fitting_pixel_mask.at(pixel) == true)
	pixel_weights[pixel] = current_solution_segments_[visible_surface_ids[pixel]].calcPixelFittingCost(image_, point_cloud_, normals_, pixel, penalties_, pixel_weights_3D_[pixel], visible_layer_indices[pixel] == ROOM_STRUCTURE_LAYER_INDEX_);
      if (checkPointValidity(getPoint(point_cloud_, pixel)) == false)
	pixel_weights[pixel] = 0;
    }
  }
  
  
  Mat bad_fitting_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
  double max_pixel_weight = getMax(pixel_weights);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    if (bad_fitting_pixel_mask.at(pixel)) {
      int gray_value = pixel_weights[pixel] / max_pixel_weight * 255;
      //gray_value = 0;
      bad_fitting_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = Vec3b(gray_value, gray_value, gray_value);
    }
    else
      bad_fitting_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = image_.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_);
  }
  
  imwrite("Test/bad_fitting_pixel_mask_image.bmp", bad_fitting_image);
  
  
  proposal_segments_ = current_solution_segments_;
  
  double sum_pixel_weights = 0;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
    if (bad_fitting_pixel_mask.at(pixel) == true)
      sum_pixel_weights += pixel_weights[pixel];
  const double SUM_FITTED_PIXEL_WEIGHTS_THRESHOLD = sum_pixel_weights * statistics_.fitted_pixel_ratio_threshold;
  
  
  vector<vector<set<int> > > layer_pixel_segment_indices_map(NUM_LAYERS_, vector<set<int> >(NUM_PIXELS_));
  map<int, ImageMask> new_segment_masks;
  int proposal_segment_index = proposal_segments_.size();
  for (int segment_type = 0; segment_type <= 2; segment_type +=	2) {
    if (segment_adding_type == 0 && segment_type == 2)
      break;
    ImageMask fitting_mask = bad_fitting_pixel_mask;
    double sum_fitted_pixel_weights = 0;
    
    while (fitting_mask.getPixels().size() > 0) {
      // if (fitting_mask.getNumPixels() < statistics_.small_segment_num_pixels_threshold)
      //   break;
      //cout << segment_type << '\t' << sum_fitted_pixel_weights << '\t' << SUM_FITTED_PIXEL_WEIGHTS_THRESHOLD << endl;
      //cout << fitting_mask.getPixels().size() << endl;
      Segment segment(image_, point_cloud_, normals_, CAMERA_PARAMETERS_, fitting_mask, statistics_, pixel_weights, segment_type, USE_PANORAMA_);
      
      ImageMask fitted_mask = segment.getMask();
      vector<int> fitted_pixels = fitted_mask.getPixels();
      //cout << fitted_pixels.size() << '\t' << fitting_mask.getPixels().size() << endl;
      //cout << fitted_pixels.size() << '\t' << segment.getVali	dity() << '\t' << fitting_mask.getPixels().size() << endl;
      if (fitted_pixels.size() < statistics_.small_segment_num_pixels_threshold)
	break;
      if (segment.getValidity()) {
	int foremost_empty_layer_index = ROOM_STRUCTURE_LAYER_INDEX_;
	for (vector<int>::const_iterator pixel_it = fitted_pixels.begin(); pixel_it != fitted_pixels.end(); pixel_it++) {
	  int current_solution_label = current_solution_labels_[*pixel_it];
	  for (int layer_index = 0; layer_index < foremost_empty_layer_index + 1; layer_index++) {
	    int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
	    if (surface_id < current_solution_num_surfaces_ && current_solution_segments_[surface_id].getDepth(*pixel_it) > segment.getDepth(*pixel_it) + statistics_.depth_conflict_threshold) {
	      foremost_empty_layer_index = layer_index - 1;
	      break;
	    }
	  }
	  if (foremost_empty_layer_index == -1)
	    break;
	}
	if (segment_adding_type == 0 || foremost_empty_layer_index < ROOM_STRUCTURE_LAYER_INDEX_) {
	  //foremost_empty_layer_index = 0;
	  //layer_surfaces[max(foremost_empty_layer_index, 0)].insert(proposal_segment_index);
	  if (segment_adding_type == 0) {
	    proposal_segments_[proposal_segment_index] = segment;
	    for (vector<int>::const_iterator pixel_it = fitted_pixels.begin(); pixel_it != fitted_pixels.end(); pixel_it++)         
              layer_pixel_segment_indices_map[ROOM_STRUCTURE_LAYER_INDEX_][*pixel_it].insert(proposal_segment_index);
            proposal_segment_index++;
          } else if (foremost_empty_layer_index < ROOM_STRUCTURE_LAYER_INDEX_) {
	    for (int layer_index = 0; layer_index <= 0; layer_index++) {
	      proposal_segments_[proposal_segment_index] = segment;
	      for (vector<int>::const_iterator pixel_it = fitted_pixels.begin(); pixel_it != fitted_pixels.end(); pixel_it++)
                layer_pixel_segment_indices_map[layer_index][*pixel_it].insert(proposal_segment_index);
	      
	      new_segment_masks[proposal_segment_index] = ImageMask(fitted_pixels, IMAGE_WIDTH_, IMAGE_HEIGHT_);
	      proposal_segment_index++;
	    }
	  }
	  for (vector<int>::const_iterator pixel_it = fitted_pixels.begin(); pixel_it != fitted_pixels.end(); pixel_it++)
              sum_fitted_pixel_weights += pixel_weights[*pixel_it];
          if (sum_fitted_pixel_weights > SUM_FITTED_PIXEL_WEIGHTS_THRESHOLD)
	    break;
	}
      }
      
      fitting_mask -= fitted_mask;
    }
  }
  
  cout << "number of new segments: " << proposal_segments_.size() - current_solution_num_surfaces_ << endl;
  if (proposal_segments_.size() == current_solution_num_surfaces_)
    return false;
  
  map<int, map<int, bool> > segment_layer_certainty_map;
  if (segment_adding_type == 0) {
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
      for (set<int>::const_iterator segment_it = layer_segments[layer_index].begin(); segment_it != layer_segments[layer_index].end(); segment_it++)
	segment_layer_certainty_map[*segment_it][layer_index] = true;
    //for (map<int, ImageMask>::const_iterator segment_it = new_segment_masks.begin(); segment_it != new_segment_masks.end(); segment_it++)
    //segment_layer_certainty_map[segment_it->first][0] = true;
  } else {
    segment_layer_certainty_map = calcNewSegmentLayers(IMAGE_WIDTH_, IMAGE_HEIGHT_, proposal_segments_, current_solution_labels_, current_solution_num_surfaces_, new_segment_masks, NUM_LAYERS_, statistics_, USE_PANORAMA_);
    layer_pixel_segment_indices_map.assign(NUM_LAYERS_, vector<set<int> >(NUM_PIXELS_));
    for (map<int, ImageMask>::const_iterator segment_it = new_segment_masks.begin(); segment_it != new_segment_masks.end(); segment_it++) {
      //assert(segment_layer_certainty_map[segment_it->first].size() == 1);
      int layer_index = segment_layer_certainty_map[segment_it->first].begin()->first;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
	if (new_segment_masks[segment_it->first].at(pixel))
	  layer_pixel_segment_indices_map[layer_index][pixel].insert(segment_it->first);
    }
  }
  // for (map<int, map<int, bool> >::const_iterator segment_it = segment_layer_certainty_map.begin(); segment_it != segment_layer_certainty_map.end(); segment_it++)
  //   for (map<int, bool>::const_iterator layer_it = segment_it->second.begin(); layer_it != segment_it->second.end(); layer_it++)
  //     cout << "layer swap: " << segment_it->first << '\t' << layer_it->first << '\t' << layer_it->second << endl;
  // exit(1);
  
  {
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      vector<set<int> > pixel_segment_indices_map = layer_pixel_segment_indices_map[layer_index];
      Mat new_segment_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
      map<int, Vec3b> color_table;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
        if (pixel_segment_indices_map[pixel].size() == 0)
          continue;
        int segment_index = 1;
        for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++)
          if (*segment_it >= current_solution_num_surfaces_)
            segment_index *= (*segment_it + 1);
        if (color_table.count(segment_index) == 0) {
          Vec3b color;
          for (int c = 0; c < 3; c++)
            color[c] = rand() % 256;
          color_table[segment_index] = color;
        }
	
        new_segment_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel  % IMAGE_WIDTH_) = color_table[segment_index];
      }
      imwrite("Test/new_segment_image_fitted_" + to_string(layer_index) + ".bmp", new_segment_image);
    }
  }

  layer_pixel_segment_indices_map = fillLayers(blurred_hsv_image_, point_cloud_, normals_, proposal_segments_, penalties_, statistics_, NUM_LAYERS_, current_solution_labels_, current_solution_num_surfaces_, segment_layer_certainty_map, USE_PANORAMA_, false, false, "segment_adding_" + to_string(proposal_iteration_), layer_pixel_segment_indices_map);
  
  
  if (segment_adding_type == 0) {
    vector<set<int> > pixel_segment_indices_map = layer_pixel_segment_indices_map[ROOM_STRUCTURE_LAYER_INDEX_];
    
    const int NUM_PIXEL_SEGMENTS_THRESHOLD = 2;
    vector<bool> unfitted_pixel_mask(NUM_PIXELS_, false);
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
      if (pixel_segment_indices_map[pixel].size() < NUM_PIXEL_SEGMENTS_THRESHOLD)
        unfitted_pixel_mask[pixel] = true;
    while (true) {
      bool has_change = false;
      vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
        if (unfitted_pixel_mask[pixel] == false)
          continue;
        vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_, USE_PANORAMA_);
        for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
          for (set<int>::const_iterator neighbor_segment_it = pixel_segment_indices_map[*neighbor_pixel_it].begin(); neighbor_segment_it != pixel_segment_indices_map[*neighbor_pixel_it].end(); neighbor_segment_it++) {
            if (proposal_segments_[*neighbor_segment_it].getDepth(pixel) > 0)
              new_pixel_segment_indices_map[pixel].insert(*neighbor_segment_it);
          }
        }
	
        if (new_pixel_segment_indices_map[pixel].size() >= NUM_PIXEL_SEGMENTS_THRESHOLD)
          unfitted_pixel_mask[pixel] = false;
        if (new_pixel_segment_indices_map[pixel].size() != pixel_segment_indices_map[pixel].size())
          has_change = true;
      }
      pixel_segment_indices_map = new_pixel_segment_indices_map;
      if (has_change == false)
        break;
    }
    
    layer_pixel_segment_indices_map[ROOM_STRUCTURE_LAYER_INDEX_] = pixel_segment_indices_map;
  }
  
  proposal_num_surfaces_ = proposal_segments_.size();
  //return generateLayerIndicatorProposal();
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  //int max_num_proposals = 0;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    
    map<int, set<int> > pixel_layer_surfaces_map;
    if (segment_adding_type != 0) {
      for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
	int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
	if (surface_id < current_solution_num_surfaces_)
	  pixel_layer_surfaces_map[layer_index].insert(surface_id);
	else
	  pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);
      }
    }
    
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
      pixel_layer_surfaces_map[layer_index].insert(layer_pixel_segment_indices_map[layer_index][pixel].begin(), layer_pixel_segment_indices_map[layer_index][pixel].end());
    
    if (segment_adding_type == 0 && layer_pixel_segment_indices_map[ROOM_STRUCTURE_LAYER_INDEX_][pixel].size() == 0)
      for (int new_segment_index = current_solution_num_surfaces_; new_segment_index < proposal_num_surfaces_; new_segment_index++)
	pixel_layer_surfaces_map[ROOM_STRUCTURE_LAYER_INDEX_].insert(new_segment_index);
    
    // for (int layer_index = 0; layer_index < ROOM_STRUCTURE_LAYER_INDEX_; layer_index++)
    //   pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);

    
    // if (checkPointValidity(getPoint(point_cloud_, pixel)) == false)
    //   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
    // 	pixel_layer_surfaces_map[layer_index].insert(layer_surfaces[layer_index].begin(), layer_surfaces[layer_index].end());
    
    // } else if (segment_adding_type == 1) {
    //   pixel_layer_surfaces_map[max(foremost_empty_layer_index, 0)].insert(pixel_segment_indices_map[pixel].begin(), pixel_segment_indices_map[pixel].end());
    // } else {
    //   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
    // 	pixel_layer_surfaces_map[layer_index].insert(layer_pixel_segment_indices_map[layer_index][pixel].begin(), layer_pixel_segment_indices_map[layer_index][pixel].end());
    // }
    
    
    // if (pixel_segment_index_map[pixel] != -1)
    //   //      for (int target_layer_index = 0; target_layer_index < NUM_LAYERS_; target_layer_index++)
    //   pixel_layer_surfaces_map[max(foremost_empty_layer_index, 0)].insert(pixel_segment_index_map[pixel]);
    // else
    //   for (int new_segment_index = current_solution_num_surfaces_; new_segment_index < proposal_num_surfaces_; new_segment_index++)
    // 	if (proposal_segments_[new_segment_index].checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, pixel) || bad_fitting_mask[pixel] == true)
    //       pixel_layer_surfaces_map[max(foremost_empty_layer_index, 0)].insert(new_segment_index);
    
    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);
    
    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
	valid_pixel_proposals.push_back(*label_it);
    
    // if (valid_pixel_proposals.size() > 1)
    //   cout << "yes" << endl;
    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      for (int proposal_index = 0; proposal_index < pixel_proposals.size(); proposal_index++) {
	cout << pixel_proposals[proposal_index] << endl;
	cout << proposal_segments_[pixel_proposals[proposal_index] % (proposal_num_surfaces_ + 1)].getDepth(pixel) << endl;
      }
      exit(1);
    }      
    
    proposal_labels_[pixel] = valid_pixel_proposals;
    
    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
	cout << "has no current solution label at pixel: " << pixel << endl;
	exit(1);
      }
    }
    
    
    //   // int new_segment_index = pixel_segment_index_map[pixel];
    //   // if (new_segment_index != -1) {
    //   //   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    //   // 	int surface_id = proposal_label / static_cast<int>(pow(proposal_num_surfaces_ + 1, NUM_LAYERS_  - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
    //   // 	pixel_proposals.push_back(proposal_label + (new_segment_index - surface_id) * pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index));
    //   // 	if (surface_id < proposal_num_surfaces_)
    //   // 	  break;
    //   //   }
    //   // }
    
    
    // //   int foremost_layer_surface_id = proposal_label / static_cast<int>(pow(proposal_num_surfaces_ + 1, NUM_LAYERS_  - 1 - max(foremost_empty_layer_index, 0))) % (proposal_num_surfaces_ + 1);
    // //   for (int new_segment_index = current_solution_num_surfaces_; new_segment_index < proposal_num_surfaces_; new_segment_index++) {
    // // 	if (pixel_segment_index_map[pixel] == -1 || new_segment_index == pixel_segment_index_map[pixel]) {
    // // 	  int new_proposal_label = proposal_label + (new_segment_index - foremost_layer_surface_id) * pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - max(foremost_empty_layer_index, 0));
    // // 	  if (checkLabelValidity(pixel, new_proposal_label, proposal_num_surfaces_, proposal_segments_))
    // //         pixel_proposals.push_back(new_proposal_label);
    // // 	}
    // //   }
    
    // //   if (foremost_empty_layer_index == -1 && foremost_layer_surface_id != proposal_num_surfaces_) {
    // // 	int new_proposal_label = proposal_label + (proposal_num_surfaces_ - foremost_layer_surface_id) * pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - 0);
    // //     if (checkLabelValidity(pixel, new_proposal_label, proposal_num_surfaces_, proposal_segments_))
    // // 	  pixel_proposals.push_back(new_proposal_label);	
    // //   }
    // // } else {
    // //   for (int new_segment_index = current_solution_num_surfaces_; new_segment_index < proposal_num_surfaces_; new_segment_index++)
    // // 	if (pixel_segment_index_map[pixel] == -1 || new_segment_index == pixel_segment_index_map[pixel])
    // // 	  pixel_proposals.push_back(pow(proposal_num_surfaces_ + 1, NUM_LAYERS_) - 1 - proposal_num_surfaces_ + new_segment_index);
    // // }
    
    // proposal_labels_[pixel] = pixel_proposals;
    // current_solution_indices_[pixel] = 0;  
    
    // if (pixel_proposals.size() > max_num_proposals) {
    //   cout << "max number of proposals: " << pixel % IMAGE_WIDTH_ << '\t' << pixel / IMAGE_WIDTH_ << '\t' << pixel_proposals.size() << endl;
    //   max_num_proposals = pixel_proposals.size();
    // }
  }
  //addSegmentLayerProposals(true);
  addIndicatorVariables();
  
  return true;
}

bool ProposalDesigner::generateStructureExpansionProposal(const int denoted_expansion_layer_index, const int denoted_expansion_pixel)
{
  cout << "generate structure expansion proposal" << endl;
  proposal_type_ = "structure_expansion_proposal";
  
  vector<bool> candidate_segment_mask(current_solution_num_surfaces_, true);
  vector<double> visible_depths(NUM_PIXELS_, -1);
  vector<double> background_depths(NUM_PIXELS_, -1);
  vector<int> segment_backmost_layer_index_map(current_solution_num_surfaces_, 0);
  vector<int> visible_segmentation(NUM_PIXELS_, -1);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    bool is_visible = true;
    bool is_background = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
        double depth = current_solution_segments_[surface_id].getDepth(pixel);
        if (is_visible) {
	  visible_depths[pixel] = depth;
	  visible_segmentation[pixel] = surface_id;
          is_visible = false;
        }
	if (layer_index == NUM_LAYERS_ - 1) {
	  background_depths[pixel] = depth;
          candidate_segment_mask[surface_id] = false;
	}
	segment_backmost_layer_index_map[surface_id] = max(segment_backmost_layer_index_map[surface_id], layer_index);
      }
    }
  }
  // for (map<int, Segment>::const_iterator segment_it = current_solution_segments_.begin(); segment_it != current_solution_segments_.end(); segment_it++) {
  //   if (segment_it->second.getType() != 0)
  //     candidate_segment_mask[segment_it->first] = false;
  // }
  
  unique_ptr<StructureFinder> structure_finder(new StructureFinder(IMAGE_WIDTH_, IMAGE_HEIGHT_, current_solution_segments_, candidate_segment_mask, visible_segmentation, visible_depths, background_depths, segment_backmost_layer_index_map, penalties_, statistics_));
  vector<pair<double, vector<int> > > structure_score_surface_ids_pairs = structure_finder->getStructures();
  if (structure_score_surface_ids_pairs.size() == 0)
    return false;
  
  double score_sum = 0;
  for (int pair_index = 0; pair_index < structure_score_surface_ids_pairs.size(); pair_index++)
    score_sum += structure_score_surface_ids_pairs[pair_index].first;
  
  vector<int> structure_surface_ids;
  double selected_score = randomProbability() * score_sum;
  for (int pair_index = 0; pair_index < structure_score_surface_ids_pairs.size(); pair_index++) {
    score_sum += structure_score_surface_ids_pairs[pair_index].first;
    if (score_sum >= selected_score) {
      //cout << pair_index << endl;
      structure_surface_ids = structure_score_surface_ids_pairs[pair_index].second;
     break;
    }
  }
  
  int backmost_layer_index = 0;
  set<int> structure_surfaces;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int surface_id = structure_surface_ids[pixel];
    if (surface_id == -1)
      continue;
    backmost_layer_index = max(backmost_layer_index, segment_backmost_layer_index_map[surface_id]);
  }
  
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    
    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      pixel_layer_surfaces_map[layer_index].insert(surface_id);
      
      if (surface_id < proposal_num_surfaces_ && layer_index <= backmost_layer_index)
	for (int target_layer_index = 0; target_layer_index < layer_index; target_layer_index++)
	  pixel_layer_surfaces_map[target_layer_index].insert(surface_id);
    }
    if (structure_surface_ids[pixel] != -1)
      pixel_layer_surfaces_map[backmost_layer_index].insert(structure_surface_ids[pixel]);
    
    for (int target_layer_index = 0; target_layer_index < backmost_layer_index; target_layer_index++)
      pixel_layer_surfaces_map[target_layer_index].insert(proposal_num_surfaces_);
    
    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);
    
    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
        valid_pixel_proposals.push_back(*label_it);
    
    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      exit(1);
    }      
    
    proposal_labels_[pixel] = valid_pixel_proposals;
    
    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
        cout << "has no current solution label at pixel: " << pixel << endl;
        exit(1);
      }
    }
  }
  
  //addSegmentLayerProposals(false);
  addIndicatorVariables();
  
  return true;
}

bool ProposalDesigner::generateBackwardMergingProposal(const int denoted_target_layer_index)
{
  cout << "generate backward merging proposal" << endl;
  proposal_type_ = "backward_merging_proposal";
  
  vector<set<int> > layer_segments(NUM_LAYERS_);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
        layer_segments[layer_index].insert(surface_id);
      }
    }
  }
  
  map<int, map<int, bool> > segment_layer_certainty_map;
  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
    for (set<int>::const_iterator segment_it = layer_segments[layer_index].begin(); segment_it != layer_segments[layer_index].end(); segment_it++)
      if (layer_index < ROOM_STRUCTURE_LAYER_INDEX_)
	segment_layer_certainty_map[*segment_it][ROOM_STRUCTURE_LAYER_INDEX_] = false;
      else
	segment_layer_certainty_map[*segment_it][ROOM_STRUCTURE_LAYER_INDEX_] = true;
  
  vector<vector<set<int> > > layer_pixel_segment_indices_map = fillLayers(blurred_hsv_image_, point_cloud_, normals_, current_solution_segments_, penalties_, statistics_, NUM_LAYERS_, current_solution_labels_, current_solution_num_surfaces_, segment_layer_certainty_map, USE_PANORAMA_, false, false, "backward_merging_" + to_string(proposal_iteration_));
  
  
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    
    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      pixel_layer_surfaces_map[layer_index].insert(surface_id);
    }
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
      pixel_layer_surfaces_map[layer_index].insert(layer_pixel_segment_indices_map[layer_index][pixel].begin(), layer_pixel_segment_indices_map[layer_index][pixel].end());
    
    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);
    
    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
        valid_pixel_proposals.push_back(*label_it);
    
    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      exit(1);
    }      
    
    proposal_labels_[pixel] = valid_pixel_proposals;
    
    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
        cout << "has no current solution label at pixel: " << pixel << endl;
        exit(1);
      }
    }
  }
  
  //addSegmentLayerProposals(false);
  addIndicatorVariables();
  
  return true;
}

bool ProposalDesigner::generateBoundaryRefinementProposal()
{
  cout << "generate boundary refinement proposal" << endl;
  proposal_type_ = "boundary_refinement_proposal";
  
  vector<double> visible_depths(NUM_PIXELS_, -1);
  vector<double> background_depths(NUM_PIXELS_, -1);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    bool is_visible = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
        double depth = current_solution_segments_[surface_id].getDepth(pixel);
        if (is_visible) {
          visible_depths[pixel] = depth;
          is_visible = false;
        }
	if (layer_index == NUM_LAYERS_ - 1)
	  background_depths[pixel] = depth;
      }
    }
  }
  
  vector<vector<set<int> > > layer_pixel_segment_indices_map(NUM_LAYERS_, vector<set<int> >(NUM_PIXELS_));
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_)
        layer_pixel_segment_indices_map[layer_index][pixel].insert(surface_id);
    }
  }
  
  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    vector<set<int> > pixel_segment_indices_map = layer_pixel_segment_indices_map[layer_index];
    while (true) {
      bool has_change = false;
      
      vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	// if (pixel_segment_indices_map[pixel].size() == 0)
	//   continue;
	vector<int> neighbor_pixels;
	int x = pixel % IMAGE_WIDTH_;
	int y = pixel / IMAGE_WIDTH_;
	if (x > 0)
	  neighbor_pixels.push_back(pixel - 1);
	if (x < IMAGE_WIDTH_ - 1)
	  neighbor_pixels.push_back(pixel + 1);
	if (y > 0)
	  neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
	if (y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
	if (x > 0 && y > 0)
	  neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
	if (x > 0 && y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
	if (x < IMAGE_WIDTH_ - 1 && y > 0)
	  neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
	if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
	for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	  // if (pixel_segment_indices_map[*neighbor_pixel_it].	size() > 0)
	  //   continue;
	  for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
	    if (new_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
	      continue;
	    double segment_depth = current_solution_segments_[*segment_it].getDepth(*neighbor_pixel_it);
	    if (segment_depth < 0)
	      continue;
	    // if (*neighbor_pixel_it == 139 * IMAGE_WIDTH_ + 126) {
	    //   cout << segment_depth << '\t' << visible_depths[*neighbor_pixel_it] << endl;
	    //   exit(1);
	    // }
	    if ((segment_depth > visible_depths[*neighbor_pixel_it] - statistics_.depth_conflict_threshold && segment_depth < background_depths[*neighbor_pixel_it] + statistics_.depth_conflict_threshold) || current_solution_segments_[*segment_it].checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, *neighbor_pixel_it)) {
	      new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
	      has_change = true;
	    }
	    
            // if (layer_index < NUM_LAYERS_ - 1) {
	    //   if (segment_depth > visible_depths[*neighbor_pixel_it] - statistics_.depth_conflict_threshold || current_solution_segments_[*segment_it].checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, *neighbor_pixel_it)) {
	    // 	new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
	    // 	has_change = true;
	    //   }
	    // } else {
	    //   if (segment_depth > visible_depths[*neighbor_pixel_it] - statistics_.depth_conflict_threshold && segment_depth < background_depths[*neighbor_pixel_it]) {
            //     new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
            //     has_change = true;
            //   }
            // }
	  }
	}
      }
      if (has_change == false)
	break;
      pixel_segment_indices_map = new_pixel_segment_indices_map	;
    }
    
    const int NUM_DILATION_ITERATIONS = 2;
    for (int iteration = 0; iteration < NUM_DILATION_ITERATIONS; iteration++) {
      vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	// if (pixel_segment_indices_map[pixel].size() == 0)
	//   continue;
	vector<int> neighbor_pixels;
	int x = pixel % IMAGE_WIDTH_;
	int y = pixel / IMAGE_WIDTH_;
	if (x > 0)
	  neighbor_pixels.push_back(pixel - 1);
	if (x < IMAGE_WIDTH_ - 1)
	  neighbor_pixels.push_back(pixel + 1);
	if (y > 0)
	  neighbor_pixels.push_back(pixel - IMAGE_WIDTH_);
	if (y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(pixel + IMAGE_WIDTH_);
	if (x > 0 && y > 0)
	  neighbor_pixels.push_back(pixel - 1 - IMAGE_WIDTH_);
	if (x > 0 && y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(pixel - 1 + IMAGE_WIDTH_);
	if (x < IMAGE_WIDTH_ - 1 && y > 0)
	  neighbor_pixels.push_back(pixel + 1 - IMAGE_WIDTH_);
	if (x < IMAGE_WIDTH_ - 1 && y < IMAGE_HEIGHT_ - 1)
	  neighbor_pixels.push_back(pixel + 1 + IMAGE_WIDTH_);
	for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	  // if (pixel_segment_indices_map[*neighbor_pixel_it].size() > 0)
	  //   continue;
	  for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
	    if (new_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
	      continue;
	    if (current_solution_segments_[*segment_it].getDepth(*neighbor_pixel_it) > 0)
	      new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
	  }
	}
      }
      pixel_segment_indices_map = new_pixel_segment_indices_map;
    }
    layer_pixel_segment_indices_map[layer_index] = pixel_segment_indices_map;
  }
  
  
  // Mat boundary_refinement_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
  // map<int, Vec3b> color_table;
  // for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
  //   if (layer_pixel_segment_indices_map[2][pixel].count(3) >	0)
  //     boundary_refinement_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = 255;
  // imwrite("Test/boundary_refinement_image.bmp", boundary_refinement_image);
  // exit(1);
  
  
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    
    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int	>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      pixel_layer_surfaces_map[layer_index].insert(surface_id);
    }
    
    for (int layer_index = 0; layer_index < ROOM_STRUCTURE_LAYER_INDEX_; layer_index++) {
      pixel_layer_surfaces_map[layer_index].insert(layer_pixel_segment_indices_map[layer_index][pixel].begin(), layer_pixel_segment_indices_map[layer_index][pixel].end());
      pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);
    }
    
    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);
    
    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
        valid_pixel_proposals.push_back(*label_it);
    
    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      exit(1);
    }      
    
    proposal_labels_[pixel] = valid_pixel_proposals;
    
    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
        cout << "has no current solution label at pixel: " << pixel << endl;
        exit(1);
      }
    }
    
    
    // if (pixel == 2 * IMAGE_WIDTH_ + 108) {
    //   for (vector<int>::const_iterator label_it = valid_pixel_proposals.begin(); label_it != valid_pixel_proposals.end(); label_it++) {
    //  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    //    int proposal_surface_id = *label_it / static_cast<int>(pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (proposal_num_surfaces_ + 1);
    //    cout << proposal_surface_id << '\t';
    //  }
    //  cout << endl;
    //   }
    //   exit(1);
    // }
  }
  
  //addSegmentLayerProposals(false);
  addIndicatorVariables();
  
  return true;
}

bool ProposalDesigner::generateDesiredProposal()
{
  cout << "generate desired proposal" << endl;
  proposal_type_ = "desired_proposal";

  //cout << current_solution_segments_[1].calcPixelFittingCost(blurred_hsv_image_, point_cloud_, normals_, 4642, penalties_, 1, false) << '\t' << current_solution_segments_[4].calcPixelFittingCost(blurred_hsv_image_, point_cloud_, normals_, 4642, penalties_, 1, false) << endl;
  //exit(1);
  
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  vector<int> visible_pixels;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    //proposal_labels_[pixel].push_back(current_solution_label);
    
    int layer_0_surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - 0)) % (current_solution_num_surfaces_ + 1);
    int layer_1_surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - 1)) % (current_solution_num_surfaces_ + 1);
    int layer_2_surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - 2)) % (current_solution_num_surfaces_ + 1);
    int layer_3_surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - 3)) % (current_solution_num_surfaces_ + 1);
    
    int proposal_label = current_solution_label;
    // if (layer_1_surface_id == 3) {
    //   if (current_solution_segments_[5].getDepth(pixel) < current_solution_segments_[4].getDepth(pixel))
    //   proposal_label -= 1;
    //   //visible_pixels.push_back(pixel);      
    // }
    
    //if (layer_2_surface_id != 3 && layer_2_surface_id != current_solution_num_surfaces_)
    //proposal_label += (current_solution_num_surfaces_ - layer_2_surface_id) * pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - 2) + (layer_2_surface_id - layer_3_surface_id);
    
    // if (current_solution_segments_[1].getDepth(pixel) < current_solution_segments_[layer_2_surface_id].getDepth(pixel))
    //   proposal_label += (1 - layer_2_surface_id) * pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - 2);
    
    //if (layer_2_surface_id == 12)
    //proposal_label += (current_solution_num_surfaces_ - layer_2_surface_id) * pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - 2);
    //proposal_label += (12 - layer_3_surface_id) * pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - 3);
    
    if (layer_2_surface_id == 1)
      proposal_label += (current_solution_num_surfaces_ - layer_2_surface_id) * pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - 2);
    proposal_labels_[pixel].push_back(current_solution_label);
    proposal_labels_[pixel].push_back(proposal_label);
  }
  //  proposal_segments_[2].refitSegment(image_, point_cloud_, visible_pixels);
  
  //addSegmentLayerProposals(false);
  addIndicatorVariables();
  
  return true;
}

bool ProposalDesigner::generateSingleProposal()
{
  cout << "generate single proposal" << endl;
  proposal_type_ = "single_proposal";
  
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    proposal_labels_[pixel].push_back(current_solution_label);
  }
  //addSegmentLayerProposals(false);
  addIndicatorVariables();
  
  return true;
}

bool ProposalDesigner::generateRandomMoveProposal()
{
  cout << "generate random move proposal" << endl;
  proposal_type_ = "random_move_proposal";
  
  vector<vector<int> > layer_surface_ids(NUM_LAYERS_, vector<int>(NUM_PIXELS_, current_solution_num_surfaces_));
  vector<vector<double> > layer_depths(NUM_LAYERS_, vector<double>(NUM_PIXELS_, -1));
  vector<int> visible_layer_indices(NUM_PIXELS_);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    bool is_visible = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
        double depth = current_solution_segments_.at(surface_id).getDepth(pixel);
        layer_depths[layer_index][pixel] = depth;
        layer_surface_ids[layer_index][pixel] = surface_id;
	
        if (is_visible)
          visible_layer_indices[pixel] = layer_index;
        is_visible = false;
      }
    }
  }
  
  
  vector<int> segment_layers; // = estimateLayer(blurred_hsv_image_, point_cloud_, normals_, current_solution_segments_, penalties_, statistics_, NUM_LAYERS_, current_solution_labels_, USE_PANORAMA_);
  for (int layer_index = ROOM_STRUCTURE_LAYER_INDEX_; layer_index <= NUM_LAYERS_ + 1; layer_index++) {
    bool is_layer_empty = true;
    for (int segment_id = 0; segment_id < current_solution_num_surfaces_; segment_id++) {
      if (segment_layers[segment_id] == layer_index) {
	is_layer_empty = false;
	break;
      }
    }
    if (is_layer_empty)
      for (int segment_id = 0; segment_id < current_solution_num_surfaces_; segment_id++)
        if (segment_layers[segment_id] < layer_index)
	  segment_layers[segment_id]++;
  }
  map<int, map<int, bool> > segment_layer_certainty_map;
  for (int segment_id = 0; segment_id < current_solution_num_surfaces_; segment_id++) {
    segment_layer_certainty_map[segment_id][min(segment_layers[segment_id], NUM_LAYERS_ - 1)] = true;
    cout << "segment layer map: " << segment_id << '\t' << segment_layers[segment_id] << endl;
  }
  
  vector<vector<set<int> > > layer_pixel_segment_indices_map = fillLayers(blurred_hsv_image_, point_cloud_, normals_, current_solution_segments_, penalties_, statistics_, NUM_LAYERS_, current_solution_labels_, current_solution_num_surfaces_, segment_layer_certainty_map, USE_PANORAMA_, true, true, "random_move_" + to_string(proposal_iteration_));
  
  
  //vector<vector<set<int> > > layer_pixel_segment_indices_map(NUM_LAYERS_, vector<set<int> >(NUM_PIXELS_));
  //set<int> backmost_segments;
  // if (true) {
  //   for (vector<int>::const_iterator segment_it = segment_layers.begin(); segment_it != segment_layers.end(); segment_it++)
  //     cout << segment_it - segment_layers.begin() << '\t' << *segment_it << endl;
  //   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
  //     layer_pixel_segment_indices_map[layer_index].assign(NUM_PIXELS_, set<int>());
  
  //   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
  //     for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //       if (visible_layer_indices[pixel] != layer_index)
  //         continue;
  //       int surface_id = layer_surface_ids[layer_index][pixel];
  //       if (surface_id == current_solution_num_surfaces_)
  //         continue;
  
  // 	if (surface_id == 9)
  //         segment_layers[surface_id] = 1;
  // 	else if (surface_id == 11 || surface_id == 12)
  // 	  segment_layers[surface_id] = 0;
  //       else if (surface_id == 8)
  // 	  segment_layers[surface_id] = 2;
  //       //else
  // 	  //	  segment_layers[surface_id] = 2;
  
  //       if (segment_layers[surface_id] < NUM_LAYERS_)
  //         layer_pixel_segment_indices_map[segment_layers[surface_id]][pixel].insert(surface_id);
  // 	else
  // 	  backmost_segments.insert(surface_id);
  //       // if (layer_index == 1)
  //       //   cout << *layer_pixel_segment_indices_map[layer_index][pixel].begin() << endl;
  //     }
  //   }
  // }
  // if (false) {
  //   map<int, map<int, int> > layer_segment_layer_choices;
  //   layer_segment_layer_choices[1][7] = 2;
  //   layer_segment_layer_choices[2][5] = 1;
  //   layer_segment_layer_choices[2][6] = 1;
  //   layer_segment_layer_choices[2][9] = 1;
  //   layer_segment_layer_choices[3][0] = 3;
  //   layer_segment_layer_choices[3][1] = 3;
  //   layer_segment_layer_choices[3][2] = 3;
  //   layer_segment_layer_choices[3][3] = 3;
  //   layer_segment_layer_choices[3][4] = 3;
  //   layer_segment_layer_choices[3][8] = 3;
  //   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
  //     layer_pixel_segment_indices_map[layer_index].assign(NUM_PIXELS_, set<int>());
  
  //   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
  //     for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  // 	if (visible_layer_indices[pixel] != layer_index)
  // 	  continue;
  // 	int surface_id = layer_surface_ids[layer_index][pixel];
  // 	if (surface_id == current_solution_num_surfaces_)
  // 	  continue;
  // 	if (layer_segment_layer_choices[layer_index][surface_id] < NUM_LAYERS_)
  // 	  layer_pixel_segment_indices_map[layer_segment_layer_choices[layer_index][surface_id]][pixel].insert(surface_id);
  // 	  // if (layer_index == 1)
  // 	//   cout << *layer_pixel_segment_indices_map[layer_index][pixel].begin() << endl;
  //     }
  //   }
  // }
  
  
  // map<int, Vec3b> segment_color_table;
  // for (map<int, Segment>::const_iterator segment_it = current_solution_segments_.begin(); segment_it != current_solution_segments_.end(); segment_it++) {
  //   segment_color_table[segment_it->first] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
  // }
  // segment_color_table[current_solution_num_surfaces_] = Vec3b(0, 0, 0);
  
  
  // if (false) {
  //   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
  //     map<int, Mat> segment_images;
  //     for (map<int, Segment>::const_iterator segment_it = current_solution_segments_.begin(); segment_it != current_solution_segments_.end(); segment_it++) {
  //       segment_images[segment_it->first] = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
  //     }
  //     Mat pixel_growth_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
  //     for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //       Vec3b color(0, 0, 0);
  //       double segment_color_weight = 1.0 / layer_pixel_segment_indices_map[layer_index][pixel].size();
  //       for (set<int>::const_iterator segment_it = layer_pixel_segment_indices_map[layer_index][pixel].begin(); segment_it != layer_pixel_segment_indices_map[layer_index][pixel].end(); segment_it++) {
  //         color += segment_color_table[*segment_it] * segment_color_weight;
  //         if (*segment_it < current_solution_num_surfaces_)
  //           segment_images[*segment_it].at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = segment_color_table[*segment_it];
  //       }
  //       pixel_growth_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = color;
  //     }
  //     imwrite("Test/random_move_intermediate_image_" + to_string(layer_index) + ".bmp", pixel_growth_image);
  
  //     // for (map<int, Segment>::const_iterator segment_it = current_solution_segments_.begin(); segment_it != current_solution_segments_.end(); segment_it++) {
  //     //   imwrite("Test/growed_segment_image_" + to_string(segment_it->first) + "_" + to_string(layer_index) + ".bmp", segment_images[segment_it->first]);
  //     // }
  //   }
  //   //exit(1);
  // }
  
  // for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
  //   vector<set<int> > pixel_segment_indices_map = layer_pixel_segment_indices_map[layer_index];
  //   int NUM_ITERATIONS = 1;
  //   for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
  //     //dilate
  //     {
  // 	vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
  // 	for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  // 	  vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_, USE_PANORAMA_);
  // 	  for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
  // 	    for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
  // 	      if (new_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
  // 		continue;
  // 	      if (current_solution_segments_[*segment_it].getDepth(*neighbor_pixel_it) > 0)
  // 		new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
  // 	    }
  // 	  }
  // 	}
  // 	pixel_segment_indices_map = new_pixel_segment_indices_map;
  //     }
  //   }
  //   layer_pixel_segment_indices_map[layer_index] = pixel_segment_indices_map;
  // }
  // for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
  //   bool empty = true;
  //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //     if (layer_pixel_segment_indices_map[layer_index][pixel].size() > 0) {
  // 	empty = false;
  // 	break;
  //     }
  //   }
  //   if (empty) {
  //     for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
  //       layer_pixel_segment_indices_map[layer_index][pixel].insert(current_solution_num_surfaces_);
  //     continue;
  //   }
  //   vector<double> min_depths(NUM_PIXELS_, 0);
  //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
  //     if (backmost_segments.count(layer_surface_ids[visible_layer_indices[pixel]][pixel]) == 0 && current_solution_segments_.at(layer_surface_ids[visible_layer_indices[pixel]][pixel]).checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, pixel) == true)
  // 	min_depths[pixel] = layer_depths[visible_layer_indices[pixel]][pixel];
  //   vector<double> max_depths(NUM_PIXELS_, 1000000);
  //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //     for (int backward_layer_index = layer_index + 1; backward_layer_index < NUM_LAYERS_; backward_layer_index++) {
  //   	if (layer_depths[backward_layer_index][pixel] > 0 && segment_layers[layer_surface_ids[backward_layer_index][pixel]] > layer_index) {
  //   	  max_depths[pixel] = layer_depths[backward_layer_index][pixel];
  //   	  break;
  //   	}
  //     }
  //   }
  //   vector<int> surface_ids = fillLayerOpenGM(blurred_hsv_image_, point_cloud_, normals_, current_solution_segments_, penalties_, statistics_, layer_pixel_segment_indices_map[layer_index], min_depths, max_depths, layer_index == ROOM_STRUCTURE_LAYER_INDEX_);
  //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
  //     layer_pixel_segment_indices_map[layer_index][pixel].insert(surface_ids[pixel]);
  
  
  //   Mat random_move_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
  //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //     if (surface_ids[pixel] == current_solution_num_surfaces_)
  // 	continue;
  //     Vec3b color = segment_color_table[surface_ids[pixel]];
  //     random_move_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = color;
  //   }
  //   imwrite("Test/random_move_image_" + to_string(layer_index) + ".bmp", random_move_image);
  
  //   Mat random_move_depth_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
  //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //     if (surface_ids[pixel] == current_solution_num_surfaces_)
  //       continue;
  //     double depth = current_solution_segments_[surface_ids[pixel]].getDepth(pixel);
  //     random_move_depth_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = min(300 / depth, 255.0);
  //   }
  //   imwrite("Test/random_move_depth_image_" + to_string(layer_index) + ".bmp", random_move_depth_image);
  // }
  // //exit(1);
  
  
  // for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
  //   //dilate empty pixels
  //   vector<set<int> > pixel_segment_indices_map = layer_pixel_segment_indices_map[layer_index];
  
  //   {
  //     for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  // 	bool on_boundary = false;
  //       vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_, USE_PANORAMA_);
  //       for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
  //         for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
  //           if (pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) == 0) {
  //             on_boundary = true;
  //             break;
  //           }
  //         }
  //         if (on_boundary)
  // 	    break;
  //       }
  // 	if (on_boundary)
  // 	  pixel_segment_indices_map[pixel].insert(current_solution_num_surfaces_);
  //     }
  //   }
  
  //   const int NUM_DILATION_ITERATIONS = 2;
  //   for (int iteration = 0; iteration < NUM_DILATION_ITERATIONS; iteration++) {
  //     vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
  //     for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  // 	// if (pixel_segment_indices_map[pixel].size() == 0)
  // 	//   continue;
  // 	vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_, USE_PANORAMA_);
  // 	for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
  // 	  // if (pixel_segment_indices_map[*neighbor_pixel_it].size() > 0)
  // 	  //   continue;
  // 	  for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
  // 	    if (new_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
  // 	      continue;
  // 	    new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
  // 	  }
  // 	}
  //     }
  //     pixel_segment_indices_map = new_pixel_segment_indices_map;
  //   }
  //   layer_pixel_segment_indices_map[layer_index] = pixel_segment_indices_map;
  // }
  
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;
  
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    
    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      pixel_layer_surfaces_map[layer_index].insert(surface_id);
    }
    
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      pixel_layer_surfaces_map[layer_index].insert(layer_pixel_segment_indices_map[layer_index][pixel].begin(), layer_pixel_segment_indices_map[layer_index][pixel].end());
      
    }
    
    
    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);
    
    // if (pixel == 26303) {
    //   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
    //     for (set<int>::const_iterator surface_it = pixel_layer_surfaces_map[layer_index].begin(); surface_it != pixel_layer_surfaces_map[layer_index].end(); surface_it++)
    //       cout << layer_index << '\t' << *surface_it << endl;
    //   //exit(1);
    // } else
    //   continue;
    
    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
        valid_pixel_proposals.push_back(*label_it);
    
    
    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      exit(1);
    }
    
    proposal_labels_[pixel] = valid_pixel_proposals;
    
    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
        //      cout << convertToProposalLabel(current_solution_indices_[pixel]) << '\t' << *valid_pixel_proposals.begin() << '\t' << proposal_num_surfaces_ << endl;
        cout << "has no current solution label at pixel: " << pixel << endl;
        exit(1);
      }
    }
  }
  
  addIndicatorVariables();
  
  return true;
}

bool ProposalDesigner::generatePixelGrowthProposal()
{
  cout << "generate pixel growth proposal" << endl;
  proposal_type_ = "pixel_growth_proposal";
  
  vector<vector<int> > layer_surface_ids(NUM_LAYERS_, vector<int>(NUM_PIXELS_, current_solution_num_surfaces_));
  vector<vector<double> > layer_depths(NUM_LAYERS_, vector<double>(NUM_PIXELS_, -1));
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    bool is_visible = true;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id < current_solution_num_surfaces_) {
        double depth = current_solution_segments_[surface_id].getDepth(pixel);
	layer_depths[layer_index][pixel] = depth;
	layer_surface_ids[layer_index][pixel] = surface_id;
      }
    }
  }
  
  vector<vector<bool> > surface_frontal_surface_mask(current_solution_num_surfaces_, vector<bool>(current_solution_num_surfaces_, false));
  vector<vector<bool> > surface_backward_surface_mask(current_solution_num_surfaces_, vector<bool>(current_solution_num_surfaces_, false));
  
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      if (surface_id == current_solution_num_surfaces_)
	continue;
      double depth = current_solution_segments_.at(surface_id).getDepth(pixel);
      if (depth <= 0)
	continue;
      for (int other_surface_id = 0; other_surface_id < current_solution_num_surfaces_; other_surface_id++) {
	if (other_surface_id == surface_id)
	  continue;
	double other_depth = current_solution_segments_.at(other_surface_id).getDepth(pixel);
	if (other_depth <= 0)
          continue;
	
	if (other_depth < depth - statistics_.depth_change_smoothness_threshold) {
	  surface_frontal_surface_mask[surface_id][other_surface_id] = true;
	  //if (surface_id == 6 && other_surface_id == 2)
	  //cout << pixel << '\t' << depth << '\t' << other_depth << endl;
	}
        if (other_depth > depth + statistics_.depth_change_smoothness_threshold)
          surface_backward_surface_mask[surface_id][other_surface_id] = true;
      }
    }
  }
  // for (int surface_id_1 = 0; surface_id_1 < current_solution_num_surfaces_; surface_id_1++) {
  //   for (int surface_id_2 = 0; surface_id_2 < current_solution_num_surfaces_; surface_id_2++) {
  //     cout << surface_id_1 << '\t' << surface_id_2 << '\t' << surface_frontal_surface_mask[surface_id_1][surface_id_2] << '\t' << surface_backward_surface_mask[surface_id_1][surface_id_2] << endl;
  //   }
  // }
  
  for (int surface_id_1 = 0; surface_id_1 < current_solution_num_surfaces_; surface_id_1++) {
    for (int surface_id_2 = surface_id_1 + 1; surface_id_2 < current_solution_num_surfaces_; surface_id_2++) {
      if (current_solution_segments_.at(surface_id_1).getSegmentType() > 0 || current_solution_segments_.at(surface_id_2).getSegmentType() > 0)
	continue;
      vector<double> plane_1 = current_solution_segments_.at(surface_id_1).getPlane();
      vector<double> plane_2 = current_solution_segments_.at(surface_id_2).getPlane();
      if (abs(plane_1[3] - plane_2[3]) < statistics_.depth_conflict_threshold && calcAngle(vector<double>(plane_1.begin(), plane_1.begin() + 3), vector<double>(plane_2.begin(), plane_2.begin() + 3)) < statistics_.similar_angle_threshold) {
	surface_frontal_surface_mask[surface_id_1][surface_id_2] = true;
	surface_frontal_surface_mask[surface_id_2][surface_id_1] = true;
        surface_backward_surface_mask[surface_id_1][surface_id_2] = true;
        surface_backward_surface_mask[surface_id_2][surface_id_1] = true;
      }
    }
  }
  
  //cout << surface_frontal_surface_mask[0][1] << '\t' << current_solution_labels_[130] % (current_solution_num_surfaces_ + 1) << endl;
  
  vector<vector<set<int> > > layer_pixel_segment_indices_map(NUM_LAYERS_, vector<set<int> >(NUM_PIXELS_));
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      layer_pixel_segment_indices_map[layer_index][pixel].insert(surface_id);
    }
  }
  
  // cout << current_solution_labels_[115 * IMAGE_WIDTH_ + 67] % (current_solution_num_surfaces_ + 1) << '\t' << calcNorm(getPoint(point_cloud_, 115 * IMAGE_WIDTH_ + 67)) << '\t' << current_solution_segments_[4].getDepth(115 * IMAGE_WIDTH_ + 67) << '\t' << current_solution_segments_[6].getDepth(115 * IMAGE_WIDTH_ + 67) << '\t' << current_solution_segments_[8].getDepth(115 * IMAGE_WIDTH_ + 67) << endl;
  // exit(1);
  
  ////moves to backmost layer
  // if (false)
  // {
  //   map<int, int> segment_depth_conflict_pixel_counter;
  //   map<int, int> segment_smooth_pixel_counter;
  //   map<int, int> segment_boundary_pixel_counter;
  //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //     int surface_id = layer_surface_ids[ROOM_STRUCTURE_LAYER_INDEX_][pixel];
  //     if (surface_id == current_solution_num_surfaces_)
  // 	continue;
  //     vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_, USE_PANORAMA_);
  //     double segment_depth = current_solution_segments_[surface_id].getDepth(pixel);
  //     bool is_on_boundary = false;
  //     bool has_depth_conflict = false;
  //     bool is_smooth = false;
  //     for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
  // 	int neighbor_surface_id = layer_surface_ids[ROOM_STRUCTURE_LAYER_INDEX_][*neighbor_pixel_it];
  // 	if (neighbor_surface_id == surface_id)
  // 	  continue;
  // 	is_on_boundary = true;
  // 	if (neighbor_surface_id == current_solution_num_surfaces_)
  // 	  continue;
  // 	is_smooth = true;
  // 	double neighbor_segment_neighbor_depth = current_solution_segments_[neighbor_surface_id].getDepth(*neighbor_pixel_it);
  // 	// double segment_neighbor_depth = current_solution_segments_[surface_id].getDepth(*neighbor_pixel_it);
  // 	// double neighbor_segment_depth = current_solution_segments_[neighbor_surface_id].getDepth(pixel);
  // 	// double diff_1 = abs(segment_depth - neighbor_segment_depth);
  //       // double diff_2 = abs(segment_neighbor_depth - neighbor_segment_neighbor_depth);
  //       // double diff_middle = (depth_1_1 - depth_2_1) * (depth_1_2 - depth_2_2) <= 0 ? 0 : 1000000;
  //       // double min_diff = min(min(diff_1, diff_2), diff_middle);
  // 	if (segment_depth > neighbor_segment_neighbor_depth + statistics_.depth_conflict_threshold) {
  // 	  has_depth_conflict = true;
  // 	  is_smooth = false;
  // 	  break;
  // 	}
  //     }
  //     if (is_on_boundary)
  // 	segment_boundary_pixel_counter[surface_id]++;
  //     if (has_depth_conflict)
  // 	segment_depth_conflict_pixel_counter[surface_id]++;
  //     if (is_smooth)
  // 	segment_smooth_pixel_counter[surface_id]++;
  //   }
  
  //   const double DEPTH_CONFLICT_PIXEL_THRESHOLD = 0.1;
  //   const double SMOOTH_PIXEL_THRESHOLD = 0.5;
  //   set<int> moved_backward_segments;
  //   for (map<int, int>::const_iterator segment_it = segment_boundary_pixel_counter.begin(); segment_it != segment_boundary_pixel_counter.end(); segment_it++) {
  //     cout << segment_depth_conflict_pixel_counter[segment_it->first] << '\t' << segment_smooth_pixel_counter[segment_it->first] << '\t' << segment_it->second << endl;
  //     if (segment_depth_conflict_pixel_counter[segment_it->first] > segment_it->second * DEPTH_CONFLICT_PIXEL_THRESHOLD && segment_smooth_pixel_counter[segment_it->first] < segment_it->second * SMOOTH_PIXEL_THRESHOLD) {
  // 	moved_backward_segments.insert(segment_it->first);
  //     }
  //   }
  //   //exit(1);
  
  //   for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
  //     int surface_id = layer_surface_ids[ROOM_STRUCTURE_LAYER_INDEX_][pixel];
  //     if (moved_backward_segments.count(surface_id) > 0) {
  // 	//cout << pixel << endl;
  // 	layer_pixel_segment_indices_map[ROOM_STRUCTURE_LAYER_INDEX_ + 1][pixel].insert(surface_id);
  // 	layer_pixel_segment_indices_map[ROOM_STRUCTURE_LAYER_INDEX_][pixel].erase(current_solution_num_surfaces_);
  // 	layer_pixel_segment_indices_map[ROOM_STRUCTURE_LAYER_INDEX_][pixel].insert(current_solution_num_surfaces_);
  //     }
  //   }
  // }
  
  
  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    //erode a bit
    if (false)
      {
        vector<set<int> > pixel_segment_indices_map = layer_pixel_segment_indices_map[layer_index];
        const int NUM_DILATION_ITERATIONS = 2;
        for (int iteration = 0; iteration < NUM_DILATION_ITERATIONS; iteration++) {
	  vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
	  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	    vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_, USE_PANORAMA_);
	    for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
	      bool on_boundary = false;
	      for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
		if (pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) == 0) {
		  on_boundary = true;
		  break;
		}
	      }
	      if (on_boundary)
		new_pixel_segment_indices_map[pixel].erase(*segment_it);
	    }
	  }
          pixel_segment_indices_map = new_pixel_segment_indices_map;
        }
        layer_pixel_segment_indices_map[layer_index] = pixel_segment_indices_map;
      }
  }
  
  for (int layer_index = 0; layer_index <= ROOM_STRUCTURE_LAYER_INDEX_; layer_index++) {
    //grow pixels in current layer
    if (true)
      {
	vector<set<int> > pixel_segment_indices_map = layer_pixel_segment_indices_map[layer_index];
	vector<bool> active_pixel_mask(NUM_PIXELS_, true);
	while (true) {
	  bool has_change = false;
	    
	  vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
	  vector<bool> new_active_pixel_mask(NUM_PIXELS_, false);
	  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	    if (active_pixel_mask[pixel] == false)
	      continue;
	      
	    if (checkPointValidity(getPoint(point_cloud_, pixel)) == false)
	      continue;
	    //if (pixel_segment_indices_map[pixel].count(2) > 0 && pixel == 22769)
	    //exit(1);
	      
	    vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_, USE_PANORAMA_);
	    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
		
	      for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
		//if (*segment_it == 2 && *neighbor_pixel_it == 22370)
		//exit(1);
		if (*segment_it == current_solution_num_surfaces_ || new_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0 || current_solution_segments_.at(*segment_it).getSegmentType() > 0)
		  continue;
		double segment_neighbor_depth = current_solution_segments_[*segment_it].getDepth(*neighbor_pixel_it);
		if (segment_neighbor_depth < 0)
		  continue;
		  
		int neighbor_surface_id = layer_surface_ids[layer_index][*neighbor_pixel_it];
		if (neighbor_surface_id == current_solution_num_surfaces_ || neighbor_surface_id == *segment_it) {
		  new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
		  new_active_pixel_mask[*neighbor_pixel_it] = true;
		  has_change = true;
		  continue;
		}
		if (surface_frontal_surface_mask[*segment_it][neighbor_surface_id] == false)
		  continue;
		  
		double frontal_neighbor_depth = -1;
		for (int frontal_layer_index = layer_index; frontal_layer_index >= 0; frontal_layer_index--) {
		  if (layer_depths[frontal_layer_index][*neighbor_pixel_it] > 0) {
		    frontal_neighbor_depth = layer_depths[frontal_layer_index][*neighbor_pixel_it];
		    break;
		  }
		}
		if (frontal_neighbor_depth < 0) {
		  vector<double> point = getPoint(point_cloud_, *neighbor_pixel_it);
		  if (checkPointValidity(point))
		    frontal_neighbor_depth = calcNorm(point);
		}
		double backward_neighbor_depth = 1000000;
		for (int backward_layer_index = layer_index + 1; backward_layer_index < layer_index; backward_layer_index++) {
		  if (layer_depths[backward_layer_index][*neighbor_pixel_it] > 0) {
		    backward_neighbor_depth = layer_depths[backward_layer_index][*neighbor_pixel_it];
		    break;
		  }
		}
		  
		// if (*neighbor_pixel_it == 16 * IMAGE_WIDTH_ + 198 && *segment_it == 3) {
		//   cout << segment_neighbor_depth << '\t' << frontal_neighbor_depth << endl;
		// 	cout << current_solution_segments_[*segment_it].checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, *neighbor_pixel_it) << endl;
		// 	cout << current_solution_segments_[*segment_it].checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, *neighbor_pixel_it + 1) << endl;
		// 	exit(1);
		// }
		  
		// if (*segment_it == 2 && *neighbor_pixel_it == 22369) {
		//   cout << pixel << '\t' << frontal_neighbor_depth << '\t' << backward_neighbor_depth << '\t' << segment_neighbor_depth << endl;
		//   exit(1);
		// }
		  
		if ((segment_neighbor_depth < frontal_neighbor_depth - statistics_.depth_conflict_threshold || segment_neighbor_depth > backward_neighbor_depth + statistics_.depth_conflict_threshold) && current_solution_segments_[*segment_it].checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, *neighbor_pixel_it) == false && checkPointValidity(getPoint(point_cloud_, *neighbor_pixel_it)))
		  continue;
		  
		  
		vector<double> segment_point = current_solution_segments_[*segment_it].getSegmentPoint(pixel);
		if (checkPointValidity(segment_point) == false)
		  continue;
		//vector<double> neighbor_point = current_solution_segments_[*segment_it].getSegmentPoint(*neighbor_pixel_it);
		// vector<double> plane = current_solution_segments_[neighbor_surface_id].getPlane();
		// double distance = calcPointPlaneDistance(point, plane);
		// double neighbor_distance = calcPointPlaneDistance(neighbor_point, plane);
		  
		  
		double distance = current_solution_segments_[*segment_it].getDepth(pixel) - current_solution_segments_[neighbor_surface_id].getDepth(pixel);
		double neighbor_distance = current_solution_segments_[*segment_it].getDepth(*neighbor_pixel_it) - current_solution_segments_[neighbor_surface_id].getDepth(*neighbor_pixel_it);
		  
		if (distance <= statistics_.depth_change_smoothness_threshold) {
		  if ((distance - statistics_.depth_change_smoothness_threshold) * (neighbor_distance - statistics_.depth_change_smoothness_threshold) <= 0)
		    continue;
		}
		if (distance >= -statistics_.depth_change_smoothness_threshold) {
		  if ((distance + statistics_.depth_change_smoothness_threshold) * (neighbor_distance + statistics_.depth_change_smoothness_threshold) <= 0)
		    continue;
		}
		if (*segment_it == 9 && neighbor_surface_id == 2 && false) { // && pixel == 34 * IMAGE_WIDTH_ + 0 && false) {
		  cout << pixel % IMAGE_WIDTH_ << '\t' << pixel / IMAGE_WIDTH_ << '\t' << distance << '\t' << neighbor_distance << endl;
		  //		      exit(1);
		}
		  
		  
		// if (*segment_it == 2 && *neighbor_pixel_it == 22369)
		// 	exit(1);
		//cout << *neighbor_pixel_it << '\t' << *segment_it << endl;
		new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
		new_active_pixel_mask[*neighbor_pixel_it] = true;
		has_change = true;
	      }
	    }
	  }
	  if (has_change == false)
	    break;
	  pixel_segment_indices_map = new_pixel_segment_indices_map;
	  active_pixel_mask = new_active_pixel_mask;
	}
	  
	if (true) {
	  const int NUM_DILATION_ITERATIONS = 1;
	  for (int iteration = 0; iteration < NUM_DILATION_ITERATIONS; iteration++) {
	    vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
	    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	      vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_, USE_PANORAMA_);
	      for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
		for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
		  if (new_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
		    continue;
		  if (*segment_it == current_solution_num_surfaces_ || current_solution_segments_[*segment_it].getDepth(*neighbor_pixel_it) > 0)
		    new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
		}
	      }
	    }
	    pixel_segment_indices_map = new_pixel_segment_indices_map;
	  }
	}
	  
	layer_pixel_segment_indices_map[layer_index] = pixel_segment_indices_map;
      }
  }
    
  for (int layer_index = 1; layer_index <= ROOM_STRUCTURE_LAYER_INDEX_; layer_index++) {
    //grow pixels in backward layer
    if (true)
      {
	vector<set<int> > pixel_segment_indices_map(NUM_PIXELS_);
	vector<bool> active_pixel_mask(NUM_PIXELS_, false);
	for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	  for (int frontal_layer_index = 0; frontal_layer_index < layer_index; frontal_layer_index++) {
	    if (layer_surface_ids[frontal_layer_index][pixel] < current_solution_num_surfaces_) {
	      pixel_segment_indices_map[pixel].insert(layer_surface_ids[frontal_layer_index][pixel]);
	      active_pixel_mask[pixel] = true;
	    }
	  }
	}
	while (true) {
	  bool has_change = false;
	    
	  vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
	  vector<bool> new_active_pixel_mask(NUM_PIXELS_, false);
	  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	    if (active_pixel_mask[pixel] == false)
	      continue;
	      
	    vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_, USE_PANORAMA_);
	    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	      for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
		if (*segment_it == current_solution_num_surfaces_ || new_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0 || current_solution_segments_.at(*segment_it).getSegmentType() > 0)
		  continue;
		double segment_neighbor_depth = current_solution_segments_[*segment_it].getDepth(*neighbor_pixel_it);
		if (*segment_it == 12 && false) {
		  cout << pixel << '\t' << *neighbor_pixel_it << endl;
		  cout << "segment neighbor depth: " << segment_neighbor_depth << endl;
		}
		  
		if (segment_neighbor_depth < 0)
		  continue;
		  
		  
		int neighbor_surface_id = layer_surface_ids[layer_index][*neighbor_pixel_it];
		if (*segment_it == 12 && false)
		  cout << "neighbor surface id: " << neighbor_surface_id << endl;
		  
		if (neighbor_surface_id < current_solution_num_surfaces_ && surface_backward_surface_mask[*segment_it][neighbor_surface_id] == false)
		  continue;
		  
		double frontal_neighbor_depth = -1;
		for (int frontal_layer_index = layer_index - 1; frontal_layer_index >= 0; frontal_layer_index--) {
		  if (layer_depths[frontal_layer_index][pixel] > 0) {
		    frontal_neighbor_depth = layer_depths[frontal_layer_index][pixel];
		    break;
		  }
		}
		if (frontal_neighbor_depth < 0) {
		  vector<double> point = getPoint(point_cloud_, *neighbor_pixel_it);
		  if (checkPointValidity(point))
		    frontal_neighbor_depth = calcNorm(point);
		}
		if (*segment_it == 12 && false)
		  cout << "frontal neighbor depth: " << frontal_neighbor_depth << endl;
		  
		double backward_neighbor_depth = 1000000;
		for (int backward_layer_index = layer_index; backward_layer_index < NUM_LAYERS_; backward_layer_index++) {
		  if (layer_depths[backward_layer_index][pixel] > 0) {
		    backward_neighbor_depth = layer_depths[backward_layer_index][pixel];
		    break;
		  }
		}
		if (*segment_it == 12 && false)
		  cout << "backward neighbor depth: " << backward_neighbor_depth << endl;
		  
		if ((segment_neighbor_depth < frontal_neighbor_depth - statistics_.depth_conflict_threshold || segment_neighbor_depth > backward_neighbor_depth + statistics_.depth_conflict_threshold) && current_solution_segments_[*segment_it].checkPixelFitting(blurred_hsv_image_, point_cloud_, normals_, *neighbor_pixel_it) == false)
		  continue;
		  
		vector<double> point = current_solution_segments_[*segment_it].getSegmentPoint(pixel);
		//vector<double> neighbor_point = current_solution_segments_[*segment_it].getSegmentPoint(*neighbor_pixel_it);
		if (checkPointValidity(point) && neighbor_surface_id != current_solution_num_surfaces_) {
		  // vector<double> plane = current_solution_segments_[neighbor_surface_id].getPlane();
		  // double distance = calcPointPlaneDistance(point, plane);
		  // double neighbor_distance = calcPointPlaneDistance(neighbor_point, plane);
		  double distance = current_solution_segments_[*segment_it].getDepth(pixel) - current_solution_segments_[neighbor_surface_id].getDepth(pixel);
		  double neighbor_distance = current_solution_segments_[*segment_it].getDepth(*neighbor_pixel_it) - current_solution_segments_[neighbor_surface_id].getDepth(*neighbor_pixel_it);
		    
		  if (*segment_it == 12 && false)
		    cout << "distance: " << distance << '\t' << neighbor_distance << endl;
		  
		  if (distance <= statistics_.depth_change_smoothness_threshold) {
		    if ((distance - statistics_.depth_change_smoothness_threshold) * (neighbor_distance - statistics_.depth_change_smoothness_threshold) <= 0)
		      continue;
		  }
		  if (distance >= -statistics_.depth_change_smoothness_threshold) {
		    if ((distance + statistics_.depth_change_smoothness_threshold) * (neighbor_distance + statistics_.depth_change_smoothness_threshold) <= 0)
		      continue;
		  }
		}
		
		if (*segment_it == 12 && false)
		  cout << "expand" << endl;
		  
		new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
		new_active_pixel_mask[*neighbor_pixel_it] = true;
		has_change = true;
	      }
	    }
	  }
	  if (has_change == false)
	    break;
	  pixel_segment_indices_map = new_pixel_segment_indices_map;
	  active_pixel_mask = new_active_pixel_mask;
	}
	  
	if (true) {
	  const int NUM_DILATION_ITERATIONS = 1;
	  for (int iteration = 0; iteration < NUM_DILATION_ITERATIONS; iteration++) {
	    vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
	    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	      // if (pixel_segment_indices_map[pixel].size() == 0)
	      //   continue;
	      vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_, USE_PANORAMA_);
	      for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
		// if (pixel_segment_indices_map[*neighbor_pixel_it].size() > 0)
		//   continue;
		for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
		  if (new_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
		    continue;
		  if (*segment_it == current_solution_num_surfaces_ || current_solution_segments_[*segment_it].getDepth(*neighbor_pixel_it) > 0)
		    new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
		}
	      }
	    }
	    pixel_segment_indices_map = new_pixel_segment_indices_map;
	  }
	}
	
	for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
	  layer_pixel_segment_indices_map[layer_index][pixel].insert(pixel_segment_indices_map[pixel].begin(), pixel_segment_indices_map[pixel].end());
      }
  }
  
  if (false)
    {
      map<int, Vec3b> segment_color_table;
      for (map<int, Segment>::const_iterator segment_it = current_solution_segments_.begin(); segment_it != current_solution_segments_.end(); segment_it++) {
	segment_color_table[segment_it->first] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
      }
      segment_color_table[current_solution_num_surfaces_] = Vec3b(0, 0, 0);
      for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
	map<int, Mat> segment_images;
	for (map<int, Segment>::const_iterator segment_it = current_solution_segments_.begin(); segment_it != current_solution_segments_.end(); segment_it++) {
	  segment_images[segment_it->first] = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
	}
	Mat pixel_growth_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
	for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	  Vec3b color(0, 0, 0);
	  double segment_color_weight = 1.0 / layer_pixel_segment_indices_map[layer_index][pixel].size();
	  for (set<int>::const_iterator segment_it = layer_pixel_segment_indices_map[layer_index][pixel].begin(); segment_it != layer_pixel_segment_indices_map[layer_index][pixel].end(); segment_it++) {
	    color += segment_color_table[*segment_it] * segment_color_weight;
	    if (*segment_it < current_solution_num_surfaces_)
	      segment_images[*segment_it].at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = segment_color_table[*segment_it];
	  }
	  pixel_growth_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = color;
	}
	imwrite("Test/growed_pixel_growth_image_" + to_string(layer_index) + ".bmp", pixel_growth_image);
	
	for (map<int, Segment>::const_iterator segment_it = current_solution_segments_.begin(); segment_it != current_solution_segments_.end(); segment_it++) {
	  imwrite("Test/growed_segment_image_" + to_string(segment_it->first) + "_" + to_string(layer_index) + ".bmp", segment_images[segment_it->first]);
	}
      }
    }
  
  
  for (int layer_index = 0; layer_index <= ROOM_STRUCTURE_LAYER_INDEX_; layer_index++) {
    //erode pixels in each layer
    if (true)
      {
	vector<set<int> > pixel_segment_indices_map = layer_pixel_segment_indices_map[layer_index];
	vector<bool> active_pixel_mask(NUM_PIXELS_, true);
	while (true) {
	  bool has_change = false;
	    
	  vector<bool> new_active_pixel_mask(NUM_PIXELS_, false);
	  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	    if (active_pixel_mask[pixel] == false)
	      continue;
	      
	    vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_, USE_PANORAMA_);
	    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	      set<int> new_neighbor_segments;
	      for (set<int>::const_iterator neighbor_segment_it = pixel_segment_indices_map[*neighbor_pixel_it].begin(); neighbor_segment_it != pixel_segment_indices_map[*neighbor_pixel_it].end(); neighbor_segment_it++) {
		//if (layer_index != 2 || *neighbor_segment_it != 12 || *neighbor_pixel_it != 27934)
		// continue;
		if (*neighbor_segment_it == current_solution_num_surfaces_ || pixel_segment_indices_map[pixel].size() == 0 || (pixel_segment_indices_map[pixel].count(current_solution_num_surfaces_) > 0 && pixel_segment_indices_map[pixel].size() >= 1) || pixel_segment_indices_map[pixel].count(*neighbor_segment_it) > 0) {
		  new_neighbor_segments.insert(*neighbor_segment_it);
		  continue;
		}
		  
		vector<double> neighbor_segment_neighbor_point = current_solution_segments_[*neighbor_segment_it].getSegmentPoint(*neighbor_pixel_it);
		bool validity = false;
		map<int, double> segment_distance_map;
		map<int, double> segment_neighbor_distance_map;
		if (checkPointValidity(neighbor_segment_neighbor_point)) {
		  for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
		    if (*segment_it == current_solution_num_surfaces_)
		      continue;
		    //		  vector<double> plane = current_solution_segments_[*segment_it].getPlane();
		    //		  double neighbor_distance = calcPointPlaneDistance(neighbor_segment_neighbor_point, plane);
		    double neighbor_distance = current_solution_segments_[*segment_it].getDepth(*neighbor_pixel_it) - current_solution_segments_[*neighbor_segment_it].getDepth(*neighbor_pixel_it);
		      
		    if (abs(neighbor_distance) <= statistics_.depth_change_smoothness_threshold) {
		      validity = true;
		      break;
		    }
		      
		    vector<double> neighbor_segment_point = current_solution_segments_[*neighbor_segment_it].getSegmentPoint(pixel);
		    double distance = checkPointValidity(neighbor_segment_point) ? current_solution_segments_[*segment_it].getDepth(pixel) - current_solution_segments_[*neighbor_segment_it].getDepth(pixel) : neighbor_distance;
		    if (abs(distance) <= statistics_.depth_change_smoothness_threshold || distance * neighbor_distance <= 0) {
		      validity = true;
		      break;
		    }
		    segment_distance_map[*segment_it] = distance;
		    segment_neighbor_distance_map[*segment_it] = neighbor_distance;
		  }
		}
		// if (validity) {
		// 	if (pixel_segment_indices_map[pixel].count(*neighbor_segment_it) == 0 && current_solution_segments_[*neighbor_segment_it].checkPairwiseConvexity(*neighbor_pixel_it, pixel) == false)
		// 	  validity = false;
		// }
		  
		if (validity) {
		  new_neighbor_segments.insert(*neighbor_segment_it);
		  continue;
		}
		
		if (*neighbor_segment_it == 0 && layer_index == 2 && false) {
		  cout << pixel << '\t' << *neighbor_pixel_it << endl;
		  for (map<int, double>::const_iterator segment_it = segment_distance_map.begin(); segment_it != segment_distance_map.end(); segment_it++)
		    cout << segment_it->first << '\t' << segment_it->second << '\t' << segment_neighbor_distance_map[segment_it->first] << endl;
		  //exit(1);
		}
		
		new_active_pixel_mask[*neighbor_pixel_it] = true;
		has_change = true;
	      }
	      pixel_segment_indices_map[*neighbor_pixel_it] = new_neighbor_segments;
	    }
	  }
	  if (has_change == false)
	    break;
	  active_pixel_mask = new_active_pixel_mask;
	}
	// if (layer_index == 2)
	//   exit(1);
	
	if (true) {
	  const int NUM_ITERATIONS = 1;
	  for (int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
	    //erode
	    {
	      vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
	      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
		vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_, USE_PANORAMA_);
		for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
		  bool on_boundary = false;
		  for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
		    if (pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) == 0) {
		      on_boundary = true;
		      break;
		    }
		  }
		  if (on_boundary)
		    new_pixel_segment_indices_map[pixel].erase(*segment_it);
		}
	      }
	      pixel_segment_indices_map = new_pixel_segment_indices_map;
	    }
	    
	    //dilate
	    {
	      vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
	      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
		vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_, USE_PANORAMA_);
		for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
		  for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
		    if (new_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
		      continue;
		    if (*segment_it == current_solution_num_surfaces_ || current_solution_segments_[*segment_it].getDepth(*neighbor_pixel_it) > 0)
		      new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
		  }
		}
	      }
	      pixel_segment_indices_map = new_pixel_segment_indices_map;
	    }
	  }
	}
	
	layer_pixel_segment_indices_map[layer_index] = pixel_segment_indices_map;
      }
  }
    
  for (int layer_index = 0; layer_index <= ROOM_STRUCTURE_LAYER_INDEX_; layer_index++) {
    //moves to frontal layers
    if (true)
      {
	for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	  int current_solution_surface_id = layer_surface_ids[layer_index][pixel];
	  if (current_solution_surface_id == current_solution_num_surfaces_)
	    continue;
	  if (layer_pixel_segment_indices_map[layer_index][pixel].count(current_solution_surface_id) > 0) {
	    bool has_backward_segments = false;
	    if (layer_pixel_segment_indices_map[layer_index][pixel].size() > 1)
	      for (set<int>::const_iterator segment_it = layer_pixel_segment_indices_map[layer_index][pixel].begin(); segment_it != layer_pixel_segment_indices_map[layer_index][pixel].end(); segment_it++) {
		if (*segment_it != current_solution_surface_id && *segment_it != current_solution_num_surfaces_ && current_solution_segments_[*segment_it].getDepth(pixel) > current_solution_segments_[current_solution_surface_id].getDepth(pixel) + statistics_.depth_conflict_threshold) {
		  has_backward_segments = true;
		  break;
		}
	      }
	    if (has_backward_segments == false)
	      continue;
	  }
	  bool exists_in_backward_layers = false;
	  for (int backward_layer_index = layer_index + 1; backward_layer_index < NUM_LAYERS_; backward_layer_index++) {
	    if (layer_pixel_segment_indices_map[backward_layer_index][pixel].count(current_solution_surface_id) > 0) {
	      exists_in_backward_layers = true;
	      break;
	    }
	  }
	  // if (pixel == 27934 && layer_index == 2) {
	  //   cout << layer_surface_ids[layer_index][pixel] << endl;
	  //   for (set<int>::const_iterator segment_it = layer_pixel_segment_indices_map[3][pixel].begin(); segment_it != layer_pixel_segment_indices_map[3][pixel].end(); segment_it++)
	  //     cout << *segment_it << endl;
	  //   assert(exists_in_backward_layers == false);
	  // }
	  if (exists_in_backward_layers == false) {
	    if (layer_index > 0)
	      layer_pixel_segment_indices_map[layer_index - 1][pixel].insert(current_solution_surface_id);
	    else
	      layer_pixel_segment_indices_map[layer_index][pixel].insert(current_solution_surface_id);
	  }
	  //layer_pixel_segment_indices_map[layer_index][pixel].insert(layer_surface_ids[layer_index][pixel]);
	}
      }
  }
    
  {
    //delete outliers in each layer
    const double num_pixels_threshold_ratio = 0.1;
    map<int, map<int, int> > segment_layer_pixel_counter;
    map<int, int> segment_pixel_counter;
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
	int current_solution_surface_id = layer_surface_ids[layer_index][pixel];
	if (current_solution_surface_id < current_solution_num_surfaces_)
	  segment_pixel_counter[current_solution_surface_id]++;
	for (set<int>::const_iterator segment_it = layer_pixel_segment_indices_map[layer_index][pixel].begin(); segment_it != layer_pixel_segment_indices_map[layer_index][pixel].end(); segment_it++)
	  segment_layer_pixel_counter[*segment_it][layer_index]++;
      }
    }
    map<int, set<int> > segment_invalid_layers;
    for (map<int, map<int, int> >::const_iterator segment_it = segment_layer_pixel_counter.begin(); segment_it != segment_layer_pixel_counter.end(); segment_it++)
      for (map<int, int>::const_iterator layer_it = segment_it->second.begin(); layer_it != segment_it->second.end(); layer_it++)
	if (layer_it->second < segment_pixel_counter[segment_it->first] * num_pixels_threshold_ratio)
	  segment_invalid_layers[segment_it->first].insert(layer_it->first);
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	set<int> new_segment_indices;
	for (set<int>::const_iterator segment_it = layer_pixel_segment_indices_map[layer_index][pixel].begin(); segment_it != layer_pixel_segment_indices_map[layer_index][pixel].end(); segment_it++)
	  if (segment_invalid_layers.count(*segment_it) == 0 || segment_invalid_layers[*segment_it].count(layer_index) == 0)
	    new_segment_indices.insert(*segment_it);
	layer_pixel_segment_indices_map[layer_index][pixel] = new_segment_indices;
      }
    }
  }
    
  vector<bool> moved_forward_segment_mask(current_solution_num_surfaces_, false);
  vector<vector<bool> > layer_existing_segment_mask(NUM_LAYERS_, vector<bool>(current_solution_num_surfaces_, false));
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    for (int layer_index = 0; layer_index <= ROOM_STRUCTURE_LAYER_INDEX_; layer_index++) {
      int current_solution_surface_id = layer_surface_ids[layer_index][pixel];
      if (current_solution_surface_id == current_solution_num_surfaces_)          
	continue;
      if (layer_index > 0 && layer_pixel_segment_indices_map[layer_index - 1][pixel].count(current_solution_surface_id) > 0) {
	layer_existing_segment_mask[layer_index - 1][current_solution_surface_id] = true;
	moved_forward_segment_mask[current_solution_surface_id] = true;
      }
      if (layer_index < NUM_LAYERS_ - 1 && layer_pixel_segment_indices_map[layer_index + 1][pixel].count(current_solution_surface_id) > 0)
	layer_existing_segment_mask[layer_index + 1][current_solution_surface_id] = true;
      if (layer_pixel_segment_indices_map[layer_index][pixel].count(current_solution_surface_id) > 0)
	layer_existing_segment_mask[layer_index][current_solution_surface_id] = true;
    }
  }
    
  for (int layer_index = 0; layer_index <= ROOM_STRUCTURE_LAYER_INDEX_; layer_index++) {
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      set<int> new_segment_indices;
      for (set<int>::const_iterator segment_it = layer_pixel_segment_indices_map[layer_index][pixel].begin(); segment_it != layer_pixel_segment_indices_map[layer_index][pixel].end(); segment_it++) {
	if (*segment_it == current_solution_num_surfaces_ || layer_existing_segment_mask[layer_index][*segment_it] == true)
	  new_segment_indices.insert(*segment_it);
      }
      layer_pixel_segment_indices_map[layer_index][pixel] = new_segment_indices;
    }
  }
    
  for (int layer_index = 1; layer_index <= ROOM_STRUCTURE_LAYER_INDEX_; layer_index++) {
    if (true)
      {
	for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	  int current_solution_surface_id = layer_surface_ids[layer_index][pixel];
	  if (current_solution_surface_id == current_solution_num_surfaces_)      
	    continue;
	  if (moved_forward_segment_mask[current_solution_surface_id] == false)
	    continue;
	  layer_pixel_segment_indices_map[layer_index - 1][pixel].insert(current_solution_surface_id);
	}
      }
  }
    
    
  for (int layer_index = 0; layer_index < ROOM_STRUCTURE_LAYER_INDEX_; layer_index++) {
    //add empty surfaces
    if (true)
      {
	for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	  if (layer_surface_ids[layer_index][pixel] == current_solution_num_surfaces_)
	    continue;
	  bool exists_in_other_layers = false;
	  int surface_id = layer_surface_ids[layer_index][pixel];
	  for (int other_layer_index = 0; other_layer_index < NUM_LAYERS_; other_layer_index++) {
	    if (other_layer_index != layer_index && layer_pixel_segment_indices_map[other_layer_index][pixel].count(surface_id) > 0) {
	      exists_in_other_layers = true;
	      break;
	    }
	  }
	  if (exists_in_other_layers)
	    layer_pixel_segment_indices_map[layer_index][pixel].insert(current_solution_num_surfaces_);
	}
      }
  }
  
  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    //dilate in each layer
    if (true)
      {
	vector<set<int> > pixel_segment_indices_map = layer_pixel_segment_indices_map[layer_index];
	const int NUM_DILATION_ITERATIONS = 2;
	for (int iteration = 0; iteration < NUM_DILATION_ITERATIONS; iteration++) {
	  vector<set<int> > new_pixel_segment_indices_map = pixel_segment_indices_map;
	  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
	    // if (pixel_segment_indices_map[pixel].size() == 0)
	    //   continue;
	    vector<int> neighbor_pixels = findNeighbors(pixel, IMAGE_WIDTH_, IMAGE_HEIGHT_, USE_PANORAMA_);
	    for (vector<int>::const_iterator neighbor_pixel_it = neighbor_pixels.begin(); neighbor_pixel_it != neighbor_pixels.end(); neighbor_pixel_it++) {
	      // if (pixel_segment_indices_map[*neighbor_pixel_it].size() > 0)
	      //   continue;
	      for (set<int>::const_iterator segment_it = pixel_segment_indices_map[pixel].begin(); segment_it != pixel_segment_indices_map[pixel].end(); segment_it++) {
		if (new_pixel_segment_indices_map[*neighbor_pixel_it].count(*segment_it) > 0)
		  continue;
		//if (*segment_it == current_solution_num_surfaces_ || current_solution_segments_[*segment_it].getDepth(*neighbor_pixel_it) > 0)
		if (*segment_it != current_solution_num_surfaces_ && current_solution_segments_[*segment_it].checkPixelFitting(image_, point_cloud_, normals_, pixel))
		  new_pixel_segment_indices_map[*neighbor_pixel_it].insert(*segment_it);
	      }
	    }
	  }
	  pixel_segment_indices_map = new_pixel_segment_indices_map;
	}
	layer_pixel_segment_indices_map[layer_index] = pixel_segment_indices_map;
      }
  }
  
  
  map<int, Vec3b> segment_color_table;
  for (map<int, Segment>::const_iterator segment_it = current_solution_segments_.begin(); segment_it != current_solution_segments_.end(); segment_it++) {
    segment_color_table[segment_it->first] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
  }
  segment_color_table[current_solution_num_surfaces_] = Vec3b(0, 0, 0);
  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    map<int, Mat> segment_images;
    for (map<int, Segment>::const_iterator segment_it = current_solution_segments_.begin(); segment_it != current_solution_segments_.end(); segment_it++) {
      segment_images[segment_it->first] = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
    }
    Mat pixel_growth_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC3);
    for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
      Vec3b color(0, 0, 0);
      double segment_color_weight = 1.0 / layer_pixel_segment_indices_map[layer_index][pixel].size();
      for (set<int>::const_iterator segment_it = layer_pixel_segment_indices_map[layer_index][pixel].begin(); segment_it != layer_pixel_segment_indices_map[layer_index][pixel].end(); segment_it++) {
	// if (pixel == 7266)
	//   cout << layer_index << '\t' << *segment_it << endl;
	color += segment_color_table[*segment_it] * segment_color_weight;
	if (*segment_it < current_solution_num_surfaces_)
	  segment_images[*segment_it].at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = segment_color_table[*segment_it];
      }
      pixel_growth_image.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = color;
    }
    imwrite("Test/pixel_growth_image_" + to_string(layer_index) + ".bmp", pixel_growth_image);
    for (map<int, Segment>::const_iterator segment_it = current_solution_segments_.begin(); segment_it != current_solution_segments_.end(); segment_it++) {
      imwrite("Test/segment_image_" + to_string(segment_it->first) + "_" + to_string(layer_index) + ".bmp", segment_images[segment_it->first]);
    }
  }
    
    
  proposal_num_surfaces_ = current_solution_num_surfaces_;
  proposal_segments_ = current_solution_segments_;
    
  proposal_labels_.assign(NUM_PIXELS_, vector<int>());
  current_solution_indices_.assign(NUM_PIXELS_, 0);
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++) {
    int current_solution_label = current_solution_labels_[pixel];
      
    map<int, set<int> > pixel_layer_surfaces_map;
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
      pixel_layer_surfaces_map[layer_index].insert(surface_id);
    }
      
    for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
      pixel_layer_surfaces_map[layer_index].insert(layer_pixel_segment_indices_map[layer_index][pixel].begin(), layer_pixel_segment_indices_map[layer_index][pixel].end());
      // if (pixel_layer_surfaces_map[layer_index].size() != 1)
      // 	cout << pixel_layer_surfaces_map[layer_index].size() << endl;
      //pixel_layer_surfaces_map[layer_index].insert(proposal_num_surfaces_);
    }
      
    //pixel_layer_surfaces_map[1].erase(12);
    //pixel_layer_surfaces_map[2].erase(12);
    vector<int> pixel_proposals = calcPixelProposals(proposal_num_surfaces_, pixel_layer_surfaces_map);
      
    // if (pixel == 26303) {
    //   for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
    //     for (set<int>::const_iterator surface_it = pixel_layer_surfaces_map[layer_index].begin(); surface_it != pixel_layer_surfaces_map[layer_index].end(); surface_it++)
    //       cout << layer_index << '\t' << *surface_it << endl;
    //   //exit(1);
    // } else
    //   continue;
      
    vector<int> valid_pixel_proposals;
    for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
      if (checkLabelValidity(pixel, *label_it, proposal_num_surfaces_, proposal_segments_) == true)
	valid_pixel_proposals.push_back(*label_it);
      
      
    if (valid_pixel_proposals.size() == 0) {
      cout << "empty proposal at pixel: " << pixel << endl;
      exit(1);
    }
      
    proposal_labels_[pixel] = valid_pixel_proposals;
      
    if (current_solution_num_surfaces_ > 0) {
      current_solution_indices_[pixel] = find(valid_pixel_proposals.begin(), valid_pixel_proposals.end(), convertToProposalLabel(current_solution_label)) - valid_pixel_proposals.begin();
      if (current_solution_indices_[pixel] == valid_pixel_proposals.size()) {
	//	cout << convertToProposalLabel(current_solution_indices_[pixel]) << '\t' << *valid_pixel_proposals.begin() << '\t' << proposal_num_surfaces_ << endl;
	cout << "has no current solution label at pixel: " << pixel << endl;
	exit(1);
      }
    }
  }
    
  addIndicatorVariables();
  
  return true;
}

void ProposalDesigner::initializeCurrentSolution()
{
  current_solution_labels_ = vector<int>(NUM_PIXELS_, 0);
  current_solution_num_surfaces_ = 0;
  current_solution_segments_.clear();
}

vector<int> ProposalDesigner::calcPixelProposals(const int num_surfaces, const map<int, set<int> > &pixel_layer_surfaces_map)
{
  vector<int> pixel_proposals(1, 0);
  for (map<int, set<int> >::const_iterator layer_it = pixel_layer_surfaces_map.begin(); layer_it != pixel_layer_surfaces_map.end(); layer_it++) {
    vector<int> new_pixel_proposals;
    for (set<int>::const_iterator segment_it = layer_it->second.begin(); segment_it != layer_it->second.end(); segment_it++)
      for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
	new_pixel_proposals.push_back(*label_it + *segment_it * pow(num_surfaces + 1, NUM_LAYERS_ - 1 - layer_it->first));
    pixel_proposals = new_pixel_proposals;
  }
  return pixel_proposals;
}

int ProposalDesigner::convertToProposalLabel(const int current_solution_label)
{
  int proposal_label = 0;
  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++) {
    int surface_id = current_solution_label / static_cast<int>(pow(current_solution_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index)) % (current_solution_num_surfaces_ + 1);
    if (surface_id == current_solution_num_surfaces_)
      proposal_label += proposal_num_surfaces_ * pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index);
    else
      proposal_label += (surface_id) * pow(proposal_num_surfaces_ + 1, NUM_LAYERS_ - 1 - layer_index);
  }
  return proposal_label;
}

vector<int> ProposalDesigner::getCurrentSolutionIndices()
{
  return current_solution_indices_;
}

bool ProposalDesigner::generateLayerIndicatorProposal()
{
  cout << "number of surfaces: " << proposal_num_surfaces_ << endl;
  vector<int> pixel_labels(proposal_num_surfaces_ + 1);
  for (int surface_id = 0; surface_id < proposal_num_surfaces_ + 1; surface_id++)
    pixel_labels[surface_id] = surface_id;
  proposal_labels_.assign(NUM_PIXELS_ * NUM_LAYERS_, pixel_labels);
  vector<int> layer_labels(NUM_LAYERS_);
  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
    layer_labels[layer_index] = layer_index;
  for (int pixel = 0; pixel < NUM_PIXELS_; pixel++)
    proposal_labels_.push_back(layer_labels);
  for (int layer_index = 0; layer_index < NUM_LAYERS_; layer_index++)
    for (int surface_id = 0; surface_id < proposal_num_surfaces_ + 1; surface_id++)
      proposal_labels_.push_back(vector<int>(1, 1));
  //addIndicatorVariables();
  
  //current_solution_indices_.assign(NUM_PIXELS_, 0);
  return true;
}

// vector<int> calcPixelProposals(const int num_layers, const int num_surfaces, const map<int, set<int> > &pixel_layer_surfaces_map)
// {
//   vector<int> pixel_proposals(1, 0);
//   for (map<int, set<int> >::const_iterator layer_it = pixel_layer_surfaces_map.begin(); layer_it != pixel_layer_surfaces_map.end(); layer_it++) {
//     vector<int> new_pixel_proposals;
//     for (set<int>::const_iterator segment_it = layer_it->second.begin(); segment_it != layer_it->second.end(); segment_it++)
//       for (vector<int>::const_iterator label_it = pixel_proposals.begin(); label_it != pixel_proposals.end(); label_it++)
//         new_pixel_proposals.push_back(*label_it + *segment_it * pow(num_surfaces + 1, num_layers - 1 - layer_it->first));
//     pixel_proposals = new_pixel_proposals;
//   }
//   return pixel_proposals;
// }

void ProposalDesigner::testColorLikelihood()
{
  Mat color_likelihood_image = Mat::zeros(IMAGE_HEIGHT_, IMAGE_WIDTH_, CV_8UC1);
  for (int pixel = 0; pixel < IMAGE_WIDTH_ * IMAGE_HEIGHT_; pixel++) {
    double color_likelihood = current_solution_segments_[current_solution_labels_[pixel] % (current_solution_num_surfaces_ + 1)].predictColorLikelihood(image_.at<Vec3b>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_));
    double max_color_likelihood = current_solution_segments_[current_solution_labels_[pixel] % (current_solution_num_surfaces_ + 1)].getMaxColorLikelihood();
    //assert(color_likelihood <= max_color_likelihood);
    if (color_likelihood > max_color_likelihood + 0.0001)
      cout << pixel << '\t' << color_likelihood << '\t' << max_color_likelihood << '\t' << current_solution_labels_[pixel] % (current_solution_num_surfaces_ + 1) << endl;
    color_likelihood_image.at<uchar>(pixel / IMAGE_WIDTH_, pixel % IMAGE_WIDTH_) = min((max_color_likelihood - color_likelihood) / statistics_.pixel_fitting_color_likelihood_threshold, 1.0) * 255;
  }
  imwrite("Test/color_likelihood_image.bmp", color_likelihood_image);
}
