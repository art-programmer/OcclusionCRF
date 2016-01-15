#ifndef __LayerDepthMap__LayerDepthRepresenter__
#define __LayerDepthMap__LayerDepthRepresenter__

#include <vector>
#include <map>
#include <opencv2/core/core.hpp>
#include <memory>
//#include "ProposalGenerator.h"
#include "Segment.h"

#include <Eigen/Dense>

//using namespace cv;
using namespace std;
//using namespace Eigen;

class LayerDepthRepresenter {
  
 public:
  LayerDepthRepresenter(const cv::Mat &image, const vector<double> &point_cloud, const RepresenterPenalties &penalties, const DataStatistics &statistics, const int scene_index, const cv::Mat &ori_image, const vector<double> &ori_point_cloud, const bool first_time, const int num_layers, const bool use_panorama);
  
  ~LayerDepthRepresenter();
  
  //draw layers to directory file_path
  void drawLayers(const vector<int> &front_layer, const vector<int> &back_layer, const string file_path, const int index = 0);
  
 private:
  const cv::Mat image_;
  const cv::Mat ori_image_;
  //  vector<int> segmentation_;
  const vector<double> point_cloud_;
  const vector<double> ori_point_cloud_;
  vector<double> normals_;
  const int IMAGE_WIDTH_;
  const int IMAGE_HEIGHT_;
  const int NUM_PIXELS_;
  
  const RepresenterPenalties PENALTIES_;
  const DataStatistics STATISTICS_;
  
  const int SCENE_INDEX_;
  const bool FIRST_TIME_;
  const bool USE_PANORAMA_;
  
  //  map<int, vector<double> > surface_models_;
  //map<int, vector<double> > surface_point_clouds_;
  map<int, vector<double> > surface_depths_;
  map<int, int> surface_colors_;
  //map<int, int> surface_pixel_counter_;
  double max_depth_;
  
  //  vector<int> labels_;
  
  vector<bool> ROI_mask_;
  int num_surfaces_;
  int num_layers_;

//unique_ptr<ProposalGenerator> proposal_generator_;

  vector<vector<int> > layers_;

  map<int, set<int> > layer_surfaces_;
  map<int, set<int> > layer_front_surfaces_;
  map<int, set<int> > layer_back_surfaces_;

  vector<double> camera_parameters_;

  double sub_sampling_ratio_;

  double disp_image_numerator_;
  

  void optimizeLayerRepresentation();

  //  void initializeLabels();
  void calcSurfaceInfo();

  //bool readLayers(vector<int> &solution, int &solution_num_surfaces, map<int, Segment> &solution_segments, const int iteration);
  
  void writeLayersMulti(const vector<int> &segmentation, const vector<int> &solution, const string file_path, const string filename = "layer_image");
  void writeRenderingInfo(const vector<int> &solution, const int solution_num_surfaces, const map<int, Segment> &solution_segments);
  void writeRenderingValues(const vector<int> &segmentation, const vector<int> &solution, const string file_path);
  vector<int> readLayersMulti(const vector<int> &segmentation, const string file_path, const string filename);
  void optimizeUsingFusionMove();
  
  void writeSegmentationImage(const vector<int> &segmentation, const string filename);
  void writeDispImage(const vector<int> &segmentation, const string filename);
  //  void drawCost(const vector<int> &solution, const string &filename);

  map<int, vector<double> > calcSurfaceDepthsUsingSpline(const vector<double> &point_cloud, const vector<int> &segmentation, const int image_width);
  map<int, vector<double> > calcSurfaceDepthsUsingPlane(const vector<double> &point_cloud, const vector<int> &segmentation, const int image_width);
  map<int, vector<double> > calcSurfaceDepthsUsingExtrapolation(const vector<double> &point_cloud, const vector<int> &segmentation, const int image_width);


  void writeSubRegions(const vector<set<int> > &sub_regions, const vector<map<int, int> > &sub_region_inpainting_surfaces);
  bool readSubRegions(vector<set<int> > &sub_regions, vector<map<int, int> > &sub_region_inpainting_surfaces);


  double interpolateDepthValue(const vector<double> &depths, const int depth_width, const int depth_height, const double x, const double y) const;
  vector<double> subSampleDepthMap(const vector<double> &depths, const int ori_width, const int ori_height, const int new_width, const int new_height);
  vector<double> subSampleDepthMap(const vector<int> &surface_ids);
  vector<int> subSampleSurfaceIds(const vector<int> &surface_ids, const vector<double> &sub_sampled_depths, const double ratio);
  void subSampleSurfaceIdsDepth(const vector<int> &surface_ids, const vector<double> &depths, vector<int> &new_surface_ids, vector<double> &new_depths, const double ratio);


  void calcSegmentTriangles(const vector<int> &multi_layer_labels, const string file_path);

  vector<int> mergeSetSolutions(const map<int, vector<int> > &set_solutions);
  
  void generateLayerImageHTML(const int scene_index, const map<int, vector<double> > &iteration_statistics_map, const map<int, string> &iteration_proposal_type_map);

  //DataStatistics calcInputStatistics(const bool first_time);
  void upsampleSolution(const vector<int> &solution_labels, const int solution_num_surfaces, const map<int, Segment> &solution_segments, vector<int> &upsampled_solution_labels, int &upsampled_solution_num_surfaces, map<int, Segment> &upsampled_solution_segments);
  void refineSolution(const vector<int> &solution_labels, const int solution_num_surfaces, const map<int, Segment> &solution_segments, vector<int> &refined_solution_labels);
};

void writeLayers(const cv::Mat &image, const vector<double> &point_cloud, const vector<double> &camera_parameters, const int num_layers, const vector<int> &solution, const int solution_num_surfaces, const map<int, Segment> &solution_segments, const int scene_index, const int result_index);
void writeLayers(const cv::Mat &image, const int image_width, const int image_height, const vector<double> &point_cloud, const vector<double> &camera_parameters, const int num_layers, const vector<int> &solution, const int solution_num_surfaces, const map<int, Segment> &solution_segments, const int scene_index, const int result_index, const cv::Mat &ori_image, const vector<double> &ori_point_cloud);
bool readLayers(const int image_width, const int image_height, const vector<double> &camera_parameters, const RepresenterPenalties &penalties, const DataStatistics &statistics, const int num_layers, vector<int> &solution, int &solution_num_surfaces, map<int, Segment> &solution_segments, const int scene_index, const int result_index, const bool use_panorama);


#endif /* defined(__LayerDepthMap__LayerDepthRepresenter__) */
