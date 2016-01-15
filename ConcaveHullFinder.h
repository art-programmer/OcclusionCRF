#ifndef ConcaveHullFinder_H
#define ConcaveHullFinder_H

#include <vector>
#include <map>
#include <set>
#include <memory>

#include "DataStructures.h"
#include "TRW_S/MRFEnergy.h"
#include "Segment.h"

class ConcaveHullFinder{

 public:
  ConcaveHullFinder(const int image_width, const int image_height, const std::vector<double> &point_cloud, const std::vector<int> &segmentation, const std::map<int, Segment> &segments, const std::vector<bool> &ROI_mask, const RepresenterPenalties penalties, const DataStatistics statistics, const bool consider_background);
  
  ~ConcaveHullFinder();

  std::vector<int> getConcaveHull();
  std::set<int> getConcaveHullSurfaces();
  
 private:
  const std::vector<int> segmentation_;
  const std::vector<double> point_cloud_;
  const int IMAGE_WIDTH_;
  const int IMAGE_HEIGHT_;

  
  //std::map<int, std::vector<double> > surface_point_clouds_;
  //std::map<int, std::vector<double> > surface_depths_;
  //std::map<int, int> segment_direction_map_;  
  //std::vector<double> surface_normals_angles_;
  std::map<int, std::map<int, int> > surface_relations_;
  
  
  const std::vector<bool> ROI_mask_;
  const int NUM_SURFACES_;
  const int NUM_PIXELS_;
  
  const RepresenterPenalties penalties_;
  const DataStatistics statistics_;
  
  std::vector<int> concave_hull_labels_;
  std::set<int> concave_hull_surfaces_;
  
  
  
  void initializeConcaveHullLabels();
  void optimizeConcaveHull();
  double calcSmoothnessCost(int pixel_1, int pixel_2, int surface_1, int surface_2);
  
  void calcConcaveHullBrutally();
  void calcConcaveHullBackground();
};

#endif /* defined(__LayerDepthMap__ConcaveHullFinder__) */
