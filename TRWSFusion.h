#ifndef __LayerDepthMap__TRWSFusion__
#define __LayerDepthMap__TRWSFusion__

#include <stdio.h>
#include <vector>
#include <map>
#include <set>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "DataStructures.h"
#include "TRW_S/MRFEnergy.h"
#include "Segment.h"

using namespace std;

class TRWSFusion
{
 public:
  
  TRWSFusion(const cv::Mat &image, const vector<double> &point_cloud, const vector<double> &normals, const std::vector<double> &pixel_weights_3D, const RepresenterPenalties &penalties, const DataStatistics &statistics, const bool consider_surface_cost = true);
  // Copy constructor
  //TRWSFusion(TRWSFusion &solver);
  
  // Destructor
  ~TRWSFusion();
  
  //    bool Save(char* filename);
  //    bool Load(char* filename);
  
  void reset();
  
  //void addUnaryTerm(const int i, const int label, const double E);
  //void addPairwiseTerm(const int i, const int j, const int label_i, const int label_j, const double E);
  
  void setOriLabels(const vector<int> ori_labels);
  
  vector<int> fuse(const vector<vector<int> > &proposal_labels, const int proposal_num_surfaces, const int proposal_num_layers, const map<int, Segment> &proposal_segments, const vector<int> &previous_solution_indices, const vector<bool> &proposal_ROI_mask = vector<bool>());
  std::vector<double> getEnergyInfo();
  vector<int> getSolution();
  
  void setNoCostMask(const map<long, int> neighbor_pair_label_map);
  
 private:
  const int IMAGE_WIDTH_, IMAGE_HEIGHT_, NUM_PIXELS_;
  const cv::Mat image_;
  cv::Mat blurred_hsv_image_;
  const vector<double> point_cloud_;
  const vector<double> normals_;
  std::vector<double> pixel_weights_3D_;
  const RepresenterPenalties penalties_;
  const DataStatistics statistics_;
  const bool consider_surface_cost_;
  
  int proposal_num_surfaces_;
  int proposal_num_layers_;
  map<int, Segment> proposal_segments_;
  //map<int, vector<double> > proposal_surface_depths_;
  vector<bool> proposal_ROI_mask_;
  vector<int> proposal_distance_to_boundaries_;
  
  
  double energy_;
  double lower_bound_;
  
  vector<int> solution_;
  vector<int> ori_labels_;
  
  double color_diff_var_;
  
  
  MRFEnergy<TypeGeneral> *initializeEnergyFromCalculation();
  MRFEnergy<TypeGeneral> *initializeEnergyFromFile();
  
  double calcDataCost(const int pixel, const int label);
  double calcSmoothnessCost(const int pixel_1, const int pixel_2, const int label_1, const int label_2);
  double calcSmoothnessCostMulti(const int pixel_1, const int pixel_2, const int label_1, const int label_2);
  
  double checkSolutionEnergy(const vector<int> &solution_for_check);
  vector<int> getOptimalSolution();
  
  //void calcDepthSVar();
  
  void calcColorDiffVar();
  double calcColorDiff(const int pixel_1, const int pixel_2);
  
  vector<vector<set<int> > > calcOverlapPixels(const vector<vector<int> > &proposal_labels);
};

#endif /* defined(__LayerDepthMap__TRWSFusion__) */
