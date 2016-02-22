#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>

#include "LayerDepthRepresenter.h"
#include "DataStructures.h"
#include "tests.h"
#include "utils.h"
#include "cv_utils.h"
#include <iomanip>

using namespace cv;
using namespace std;
using namespace cv_utils;

int main(int argc, char *argv[])
{
  //testImageCompletion();
  
  //Mat mask_image = imread("Test/mask_image.bmp");
  // ImageMask mask;
  // mask.readMaskImage(mask_image);
  // mask.resizeByRatio(2, 2);
  // mask.smooth("median", 3);
  // mask.smooth("median", 3);
  // mask.resizeByRatio(2, 2);
  // mask.smooth("median", 3);
  // mask.smooth("median", 3);
  // imwrite("Test/upsampled_mask_image.bmp", mask.drawMaskImage());
  
  // Mat upsampled_mask_image = mask_image.clone();
  // resize(upsampled_mask_image, upsampled_mask_image, Size(upsampled_mask_image.cols * 2, upsampled_mask_image.rows * 2));
  // medianBlur(upsampled_mask_image, upsampled_mask_image, 5);
  // medianBlur(upsampled_mask_image, upsampled_mask_image, 5);
  // resize(upsampled_mask_image, upsampled_mask_image, Size(upsampled_mask_image.cols * 2, upsampled_mask_image.rows * 2));
  // medianBlur(upsampled_mask_image, upsampled_mask_image, 5);
  // medianBlur(upsampled_mask_image, upsampled_mask_image, 5);
  // resize(upsampled_mask_image, upsampled_mask_image, Size(upsampled_mask_image.cols * 2, upsampled_mask_image.rows * 2));
  // medianBlur(upsampled_mask_image, upsampled_mask_image, 5);
  // medianBlur(upsampled_mask_image, upsampled_mask_image, 5);
  // imwrite("Test/upsampled_mask_image.bmp", upsampled_mask_image);
  // exit(1);

  srand(time(0));
  
  vector<double> point_cloud;
  Mat image;
  Mat ori_image;
  vector<double> ori_point_cloud;
  vector<int> segmentation;
  //MatrixXd projection_matrix;
  char scene_type = 'R';
  istringstream scene_index_arg_ss(argv[1]);
  int scene_index = -1;
  scene_index_arg_ss >> scene_index;
  bool first_time = false;
  // if (first_time == true)
  //   rm(result_dir_ss.str().c_str());
  
  // if (argc <= 2)
  //   first_time = true;
  // else {
  //   string first_time_str(argv[2]);
  //   if (first_time_str.compare("true") == 0)
  //     first_time = true;
  // }
  
  
  istringstream num_layers_arg_ss(argv[2]);
  int num_layers;
  num_layers_arg_ss >> num_layers;
  istringstream dataset_index_arg_ss(argv[3]);
  int dataset_index;
  dataset_index_arg_ss >> dataset_index;
  
  
  stringstream result_dir_ss;
  result_dir_ss << "Results/scene_" << scene_index;
  struct stat sb;
  if (stat(result_dir_ss.str().c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode))
    mkdir(result_dir_ss.str().c_str(), 0777);
  stringstream cache_dir_ss;
  cache_dir_ss << "Cache/scene_" << scene_index;
  if (stat(cache_dir_ss.str().c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode))
    mkdir(cache_dir_ss.str().c_str(), 0777);
  
  bool use_panorama = false;
  if (argc >= 7 && string(argv[6]) == "panorama")
    use_panorama = true;
  
  if (argc >= 8 && string(argv[7]) == "clear_cache")
    remove((cache_dir_ss.str() + "/iteration_info.txt").c_str());
  
  if (use_panorama == false) {
    if (dataset_index == 0) {
      if (scene_type == 'R') {
	char *image_filename = new char[100];
	sprintf(image_filename, "Inputs/image_%02d.bmp", scene_index - 1000);
	image = imread(image_filename);
	ori_image = image.clone();

	stringstream point_cloud_filename;
	//point_cloud_filename.width(2);
	//point_cloud_filename.setfill('0');
	point_cloud_filename << "Inputs/point_cloud_" << setw(2) << setfill('0') << to_string(scene_index - 1000) << ".txt";
        point_cloud = loadPointCloud(point_cloud_filename.str());
        //point_cloud = rotatePointCloud(point_cloud, rotation_angle);
        //point_cloud = readPointCloudFromObj("Inputs/cse013_0h.obj", image.cols, image.rows);
        ori_point_cloud = point_cloud;
        // stringstream point_cloud_mesh_filename;
	// point_cloud_mesh_filename << "Results/scene_" << scene_index << "/point_cloud_mesh.ply";
	// savePointCloudAsMesh(point_cloud, point_cloud_mesh_filename.str().c_str());
      }

      zoomScene(image, point_cloud, 1.0 * 200 / image.cols, 1.0 * 200 / image.cols);
      stringstream zoomed_image_filename;
      zoomed_image_filename << "Results/scene_" << scene_index << "/zoomed_image.bmp";
      imwrite(zoomed_image_filename.str(), image);
      
      
    } else {
      istringstream filename_ss(argv[4]);
      istringstream angle_ss(argv[5]);
      int rotation_angle;
      angle_ss >> rotation_angle;
      
      // char *image_filename = new char[100];
      // sprintf(image_filename, "Inputs/image_%02d.bmp", scene_index);
      
      //    image = imread("Inputs/cse013_0h.png");
      stringstream image_filename;
      image_filename << "Inputs/" << filename_ss.str() << "_" << rotation_angle << ".png";
      image = imread(image_filename.str());
      ori_image = image.clone();
      
      stringstream point_cloud_filename;
      point_cloud_filename << "Inputs/" << filename_ss.str() << "_" << rotation_angle << ".obj";
      point_cloud = readPointCloudFromObj(point_cloud_filename.str(), image.cols, image.rows, rotation_angle);
      //point_cloud = rotatePointCloud(point_cloud, rotation_angle);
      //point_cloud = readPointCloudFromObj("Inputs/cse013_0h.obj", image.cols, image.rows);
      ori_point_cloud = point_cloud;
      
      //    zoomScene(image, point_cloud, 0.2, 0.2);
      zoomScene(image, point_cloud, 1.0 * 200 / image.cols, 1.0 * 200 / image.cols);
      //zoomScene(ori_image, ori_point_cloud, 1.0 * (image.cols * static_cast<int>(ori_image.cols / image.cols)) / ori_image.cols, 1.0 * (image.cols * static_cast<int>(ori_image.cols / image.cols)) / ori_image.cols);
      zoomScene(ori_image, ori_point_cloud, 1.0 * image.cols / ori_image.cols, 1.0 * image.cols / ori_image.cols);
      
      
      stringstream zoomed_image_filename;
      zoomed_image_filename << "Results/scene_" << scene_index << "/zoomed_image.bmp";
      imwrite(zoomed_image_filename.str(), image);
      
      stringstream large_image_filename;
      large_image_filename << "Results/scene_" << scene_index << "/large_image.bmp";
      imwrite(large_image_filename.str(), ori_image);
    }
  } else {
    // Mat test_image = imread("Results/scene_0/zoomed_image.bmp");
    // Mat new_test_image(test_image.size(), CV_8UC3);
    // for (int y = 0; y < test_image.rows; y++)
    //   for (int x = 0; x < test_image.cols; x++)
    // 	new_test_image.at<Vec3b>(y, x) = test_image.at<Vec3b>((y * test_image.cols + x) % test_image.rows, (y * test_image.cols + x) / test_image.rows);
    // imwrite("Test/zoomed_image.bmp", new_test_image);
    // exit(1);
    
    stringstream zoomed_image_filename;
    zoomed_image_filename << "Results/scene_" << scene_index << "/zoomed_image.bmp";
    
    stringstream point_cloud_filename;
    point_cloud_filename << "Results/scene_" << scene_index << "/point_cloud";
    if (imread(zoomed_image_filename.str()).empty() || readPointCloud(point_cloud_filename.str(), point_cloud) == false) {
      
      istringstream filename_ss(argv[4]);
      
      // char *image_filename = new char[100];
      // sprintf(image_filename, "Inputs/image_%02d.bmp", scene_index);
      
      //    image = imread("Inputs/cse013_0h.png");
      stringstream ptx_filename;
      ptx_filename << "Inputs/" << filename_ss.str() << ".ptx";
      vector<double> camera_parameters;
      readPtxFile(ptx_filename.str(), image, point_cloud, camera_parameters);
      ori_image = image.clone();
      ori_point_cloud = point_cloud;
      
      //    zoomScene(image, point_cloud, 0.2, 0.2);
      zoomScene(image, point_cloud, 1.0 * 500 / image.cols, 1.0 * 500 / image.cols);
      //zoomScene(ori_image, ori_point_cloud, 1.0 * (image.cols * static_cast<int>(ori_image.cols / image.cols)) / ori_image.cols, 1.0 * (image.cols * static_cast<int>(ori_image.cols / image.cols)) / ori_image.cols);
      
      
      imwrite(zoomed_image_filename.str(), image);
      
      stringstream large_image_filename;
      large_image_filename << "Results/scene_" << scene_index << "/large_image.bmp";
      imwrite(large_image_filename.str(), ori_image);
      
      writePointCloud(point_cloud_filename.str(), point_cloud, image.cols, image.rows);
    } else {
      image = imread(zoomed_image_filename.str());
      ori_image = image.clone();
      ori_point_cloud = point_cloud;
      // cropScene(image, point_cloud, 0, 5, image.cols - 1, image.rows - 1);
      // imwrite(zoomed_image_filename.str(), image);
      // writePointCloud(point_cloud_filename.str(), point_cloud, image.cols, image.rows);
    }
  }
  // vector<double> camera_parameters;
  // camera_parameters.push_back(focal_length);
  // camera_parameters.push_back(cx);
  // camera_parameters.push_back(cy);
  // for (int pixel = 0; pixel < image.cols * image.rows; pixel++) {
  //   int projected_pixel = projectPoint(vector<double>(point_cloud.begin() + pixel * 3, point_cloud.begin() + (pixel + 1) * 3), image.cols, image.rows, camera_parameters, true);
  //   cout << pixel % image.cols << '\t' << pixel / image.cols << '\t' << projected_pixel % image.cols << '\t' << projected_pixel / image.cols << endl;
  // }
  // exit(1);
  
  bool check_guided_image_filtering = false;
  if (check_guided_image_filtering) {
    // {
    //   const int IMAGE_WIDTH = 5;
    //   const int IMAGE_HEIGHT = 5;
    //   const int WINDOW_SIZE = 3;
    //   vector<double> values(25);
    //   for (int i = 0; i < 25; i++)
    // 	values[i] = 1;
    //   vector<double> mask = calcBoxIntegrationMask(values, 5, 5);
      
    //   vector<double> sum_mask = calcBoxIntegrationMask(values, IMAGE_WIDTH, IMAGE_HEIGHT);
    //   vector<double> values2(IMAGE_WIDTH * IMAGE_HEIGHT);
    //   transform(values.begin(), values.end(), values2.begin(), [](const double &x) { return pow(x, 2); });
    //   vector<double> sum2_mask = calcBoxIntegrationMask(values2, IMAGE_WIDTH, IMAGE_HEIGHT);
    //   vector<double> means(IMAGE_WIDTH * IMAGE_HEIGHT, 0);
    //   vector<double> vars(IMAGE_WIDTH * IMAGE_HEIGHT, 0);
    //   for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++) {
    //     int x_1 = pixel % IMAGE_WIDTH - (WINDOW_SIZE - 1) / 2;
    //     int y_1 = pixel / IMAGE_WIDTH - (WINDOW_SIZE - 1) / 2;
    //     int x_2 = pixel % IMAGE_WIDTH + (WINDOW_SIZE - 1) / 2;
    //     int y_2 = pixel / IMAGE_WIDTH + (WINDOW_SIZE - 1) / 2;
    //     int area = (min(pixel % IMAGE_WIDTH + (WINDOW_SIZE - 1) / 2, IMAGE_WIDTH - 1) - max(pixel % IMAGE_WIDTH - (WINDOW_SIZE - 1) / 2, 0) + 1) * (min(pixel / IMAGE_WIDTH + (WINDOW_SIZE - 1) / 2, IMAGE_HEIGHT - 1) - max(pixel / IMAGE_WIDTH - (WINDOW_SIZE - 1) / 2, 0) + 1);
    //     double mean = calcBoxIntegration(sum_mask, IMAGE_WIDTH, IMAGE_HEIGHT, x_1, y_1, x_2, y_2) / area;
    //     double var = calcBoxIntegration(sum2_mask, IMAGE_WIDTH, IMAGE_HEIGHT, x_1, y_1, x_2, y_2) / area - pow(mean, 2);
    //     means[pixel] = mean;
    //     vars[pixel] = var;
    //   }

    //   for (int y = 0; y < 5; y++) {
    // 	for (int x = 0; x < 5; x++)
    // 	  cout << vars[y * 5 + x] << '\t';
    // 	cout << endl;
    //   }

    //   exit(1);
    // }    
    // const int IMAGE_WIDTH = image.cols;
    // const int IMAGE_HEIGHT = image.rows;
    // const int NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;

    // vector<vector<double> > image_values(3, vector<double>(NUM_PIXELS));
    // for (int y = 0; y < IMAGE_HEIGHT; y++) {
    //   for (int x = 0; x < IMAGE_WIDTH; x++) {
    //     int pixel = y * IMAGE_WIDTH + x;
    //     Vec3b image_color = image.at<Vec3b>(y, x);
    //     for (int c = 0; c < 3; c++) {
    //       image_values[c][pixel] = 1.0 * image_color[c] / 256;
    //     }
    //   }
    // }
    // vector<vector<double> > image_means(3);
    // vector<vector<double> > image_vars(3);
    // for (int c = 0; c < 3; c++) {
    //   vector<double> dummy_vars;
    //   calcWindowMeansAndVars(image_values[c], IMAGE_WIDTH, IMAGE_HEIGHT, 21, image_means[c], image_vars[c]);
    // }

    // Mat mean_image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
    // Mat var_image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);
    // for (int y = 0; y < IMAGE_HEIGHT; y++) {
    //   for (int x = 0; x < IMAGE_WIDTH; x++) {
    //     int pixel = y * IMAGE_WIDTH + x;
    // 	Vec3b mean_color;
    // 	double var = 0;
    //     for (int c = 0; c < 3; c++) {
    //       mean_color[c] = max(min(image_means[c][pixel] * 256, 255.0), 0.0);
    // 	  var += image_vars[c][pixel];
    // 	}
    // 	mean_image.at<Vec3b>(y, x) = mean_color;
    // 	var_image.at<uchar>(y, x) = max(min(var * 256, 255.0), 0.0);;
    //   }
    // }
    // imwrite("Test/mean_image.bmp", mean_image);
    // imwrite("Test/var_image.bmp", var_image);
    // exit(1);

    
    Mat test_image = Mat::zeros(100, 100, CV_8UC1);
    for (int y = 50; y < 100; y++)
      for (int x = 0; x < 100; x++)
	test_image.at<uchar>(y, x) = 255;
    Mat output_image;
    guidedFilter(ori_image, ori_image, output_image, 20, 0.01);
    imwrite("Test/guided_filtered_image.bmp", output_image);
    exit(1);
  }
  
  bool check_energy_diff = false;
  if (check_energy_diff) {
    Mat test_image = Mat::zeros(image.rows, image.cols, CV_8UC1);
    ifstream energy_in_str_1("Test/energy_0");
    ifstream energy_in_str_2("Test/energy_1");
    for (int y = 0; y < test_image.rows; y++) {
      for (int x = 0; x < test_image.cols; x++) {
        int pixel = y * test_image.cols + x;
        double energy_1;
        energy_in_str_1 >> energy_1;
        double energy_2;
        energy_in_str_2 >> energy_2;
	//energy_2 = 0;
        if (energy_1 != energy_2)
	  cout << pixel % image.cols << '\t' << pixel / image.cols << '\t' << energy_1 << '\t' << energy_2 << endl;
        test_image.at<uchar>(y, x) = min(max(1.0 * (energy_1 - energy_2) / 1000 * 128 + 128, 0.0), 255.0);
      }
    }
    imwrite("Test/energy_diff_image.bmp", test_image);
    exit(1);
  }
  
  // bool check_blending = false;
  // if (check_blending) {
  //   Mat texture_image = imread("Results/scene_10001/texture_image_3.bmp");
  //   Mat mask = Mat::ones(texture_image.rows, texture_image.cols, texture_image.depth()) * 255;
  //   // for (int y = 0; y < mask.rows / 2; y++)
  //   //   for (int x = 0; x < mask.cols; x++)
  //   //  mask.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
  
  //   Mat result;
  //   //seamlessClone(texture_image, texture_image, mask, Point(texture_image.cols / 2, texture_image.rows / 2), result, MIXED_CLONE);
  //   GaussianBlur(texture_image, result, Size(5, 5), 0, 0);
  //   GaussianBlur(result, result, Size(5, 5), 0, 0);
  //   GaussianBlur(result, result, Size(5, 5), 0, 0);
  //   imshow("texture_image", result);
  //   waitKey();
  //   exit(1);
  // }
  
  if (true) {
    stringstream texture_ori_filename;
    texture_ori_filename << "Results/scene_" << scene_index << "/" << "texture_image_ori.bmp";
    if (imread(texture_ori_filename.str()).empty())
      imwrite(texture_ori_filename.str(), ori_image);
    
    
    stringstream depth_values_filename;
    depth_values_filename << "Results/scene_" << scene_index << "/" << "depth_values_ori";
    if (!ifstream(depth_values_filename.str().c_str())) {
      vector<double> inpainted_point_cloud = inpaintPointCloud(ori_point_cloud, ori_image.cols, ori_image.rows);
      ofstream depth_values_out_str(depth_values_filename.str().c_str());
      depth_values_out_str << ori_image.cols << '\t' << ori_image.rows << endl;
      for (int point_index = 0; point_index < inpainted_point_cloud.size() / 3; point_index++)
        depth_values_out_str << inpainted_point_cloud[point_index * 3 + 2] << endl;
      depth_values_out_str.close();
    }
    
    stringstream point_cloud_ply_filename;
    point_cloud_ply_filename << "Results/scene_" << scene_index << "/point_cloud.ply";
    if (!ifstream(point_cloud_ply_filename.str().c_str()))
      savePointCloudAsPly(image, point_cloud, point_cloud_ply_filename.str().c_str());
    
    stringstream disp_image_filename;
    disp_image_filename << "Results/scene_" << scene_index << "/disp_image.bmp";
    if (imread(disp_image_filename.str()).empty())
      imwrite(disp_image_filename.str(), drawDispImage(point_cloud, image.cols, image.rows));
    
    stringstream ori_disp_image_filename;
    ori_disp_image_filename << "Results/scene_" << scene_index << "/disp_image_ori.bmp";
    if (imread(ori_disp_image_filename.str()).empty())
      imwrite(ori_disp_image_filename.str(), drawDispImage(ori_point_cloud, ori_image.cols, ori_image.rows));
    
    stringstream pixel_weight_image_filename;    
    pixel_weight_image_filename << "Results/scene_" << scene_index << "/pixel_weight_image.bmp";
    if (imread(pixel_weight_image_filename.str()).empty()) {
      vector<double> curvatures = calcCurvatures(point_cloud, image.cols, image.rows, 30);
      Mat pixel_weight_image = Mat::ones(image.rows, image.cols, CV_8UC1) * 255;
      for (int pixel = 0; pixel < image.rows * image.cols; pixel++) {
	//	cout << curvatures[pixel] << endl;
	assert(curvatures[pixel] <= 1.0 / 3);
	pixel_weight_image.at<uchar>(pixel / image.cols, pixel % image.cols) = max(1 - curvatures[pixel] * 10, 0.1) * 255;
        if (checkPointValidity(getPoint(point_cloud, pixel)) == false)
	  pixel_weight_image.at<uchar>(pixel / image.cols, pixel % image.cols) = 0;
      }
      imwrite(pixel_weight_image_filename.str(), pixel_weight_image);
    }

    stringstream image_filename;
    image_filename << "Results/scene_" << scene_index << "/image.bmp";
    //if (imread(image_filename.str()).empty())
      imwrite(image_filename.str(), ori_image);
  }

  stringstream flattened_image_filename;    
  flattened_image_filename << "Results/scene_" << scene_index << "/flattened_image.bmp";
  image = imread(flattened_image_filename.str());
  //cout << image.cols << '\t' << image.rows << endl;
  //  exit(1);
  cout << "scene: " << scene_index << endl;
  //exit(1);
  
  RepresenterPenalties penalties;
  DataStatistics statistics;
  
  
  // penalties.depth_inconsistency_pen = 2000;
  // penalties.normal_inconsistency_pen = 0;
  // penalties.color_inconsistency_pen = 10;
  // penalties.distance_2D_pen = 0;
  // penalties.close_parallel_surface_pen = 0;
  // penalties.layer_empty_pen = 0;
  
  penalties.smoothness_pen = 1000;
  penalties.smoothness_small_constant_pen = 5;
  penalties.smoothness_concave_shape_pen = 1000;
  penalties.smoothness_segment_splitted_pen = 0;
  penalties.smoothness_spurious_empty_pen = 0;
  penalties.smoothness_boundary_pen = 100;
  penalties.smoothness_min_boundary_pen = penalties.smoothness_boundary_pen * 0.0;
  
  
  penalties.surface_pen = 0;
  penalties.layer_pen = 0;
  penalties.surface_splitted_pen = 0;
  penalties.behind_room_structure_surface_pen = penalties.surface_pen;
  
  //penalties.data_cost_depth_change_ratio = 4;
  //penalties.data_cost_angle_diff_ratio = 1;
  //penalties.data_cost_color_likelihood_ratio = 5;
  //penalties.data_cost_non_plane_pen = 0.05;
  
  penalties.large_pen = 10000;
  penalties.huge_pen = 1000000;

  penalties.other_viewpoint_depth_change_pen = penalties.smoothness_pen * 0.2;
  penalties.other_viewpoint_depth_conflict_pen = penalties.huge_pen * 0.2;

  
  penalties.data_term_layer_decrease_ratio = 1;
  penalties.smoothness_term_layer_decrease_ratio = 1; //sqrt(0.5);
  
  penalties.point_plane_distance_weight = 1;
  penalties.point_plane_angle_weight = 0.1;
  penalties.color_likelihood_weight = 0.1;
  penalties.non_plane_weight = 0.1;
  penalties.data_cost_weight = 1000;
  penalties.behind_room_structure_cost_ratio = 0.5;

  penalties.max_depth_change = 0.3;
  penalties.max_depth_diff = 0.2;
  penalties.max_angle_diff = 30 * M_PI / 180;
  penalties.max_color_diff = 30;

  //penalties.smoothness_empty_non_empty_ratio = 0.03 / penalties.max_depth_change;
  
  statistics.pixel_fitting_distance_threshold = dataset_index == 1 ? 0.03 : 0.05;
  statistics.pixel_fitting_angle_threshold = 30 * M_PI / 180;
  statistics.pixel_fitting_color_likelihood_threshold = 20;
  statistics.depth_diff_var = 0.01;
  statistics.similar_angle_threshold = 20 * M_PI / 180;
  // statistics.fitting_color_likelihood_threshold = 0.01;
  // statistics.parallel_angle_threshold = 10 * M_PI / 180;
  statistics.viewpoint_movement = 0.01;
  //statistics.color_diff_threshold = 0.5;
  statistics.depth_conflict_threshold = statistics.pixel_fitting_distance_threshold;
  statistics.depth_change_smoothness_threshold = dataset_index == 1 ? 0.02 : 0.03;
  statistics.bspline_surface_num_pixels_threshold = image.cols * image.rows / 50;
  statistics.background_depth_diff_tolerance = 0.00;
  statistics.num_grams = 20;
  statistics.fitted_pixel_ratio_threshold = 0.95;
  statistics.small_segment_num_pixels_threshold = 10;
  statistics.segment_refitting_common_ratio_threshold = 0.8;
  statistics.color_diff_var = 100;
  
  //statistics.color_diff_var = DataStatistics::calcColorDiffVar(ori_image);
  statistics.pixel_weight_curvature_ratio = 10;
  statistics.min_pixel_weight = 0.1;
  
  LayerDepthRepresenter representer(image, point_cloud, penalties, statistics, scene_index, ori_image, ori_point_cloud, first_time, num_layers, use_panorama);
  
  return 0;
}
