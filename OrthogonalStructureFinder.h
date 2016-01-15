#ifndef OrthogonalStructureFinder_H
#define OrthogonalStructureFinder_H

#include <vector>
#include <map>
#include <set>
#include <memory>

#include "DataStructures.h"
#include "TRW_S/MRFEnergy.h"
#include "Segment.h"

class OrthogonalStructureFinder{
  
 public:
  OrthogonalStructureFinder(const int image_width, const int image_height, const std::vector<double> &point_cloud, const std::map<int, Segment> &segments, const std::vector<int> &visible_segmentation, const std::vector<bool> &ROI_mask, const DataStatistics statistics, const set<int> &invalid_segments = set<int>());
  
  std::vector<std::pair<double, std::vector<int> > > calcOrthogonalStructures(const int NUM_HORIZONTAL_SURFACES, const int NUM_VERTICAL_SURFACES);
  
 private:
  const std::vector<int> visible_segmentation_;
  const std::vector<double> point_cloud_;
  const int IMAGE_WIDTH_;
  const int IMAGE_HEIGHT_;
  const std::map<int, Segment> segments_;
  
  //std::map<int, std::vector<double> > surface_point_clouds_;
  //std::map<int, std::vector<double> > surface_depths_;
  //std::map<int, int> segment_direction_map_;  
  //std::vector<double> surface_normals_angles_;
  //std::map<int, std::map<int, int> > surface_relations_;
  
  std::map<int, bool> segment_pair_convexity_map_;
  std::map<int, bool> segment_pair_orthogonality_map_;
  std::vector<int> horizontal_segments_;
  std::vector<int> vertical_segments_;
  
  const std::vector<bool> ROI_mask_;
  const int NUM_SURFACES_;
  const int NUM_PIXELS_;
  
  const DataStatistics statistics_;
};

#endif /* defined(__LayerDepthMap__OrthogonalStructureFinder__) */
