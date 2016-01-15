#include "tests.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <Eigen/Dense>

#include "cv_utils.h"
#include "ImageMask.h"

using namespace cv;
using namespace std;
using namespace Eigen;

void testImageCompletion()
{
  // {
  //   const int IMAGE_WIDTH = 512;
  //   const int IMAGE_HEIGHT = 512;
  //   Mat image = Mat::zeros(IMAGE_WIDTH, IMAGE_HEIGHT, CV_8UC3);
  //   vector<bool> source_mask(IMAGE_WIDTH * IMAGE_HEIGHT, true);
  //   vector<bool> target_mask(IMAGE_WIDTH * IMAGE_HEIGHT, true);
  //   for (int y = 0; y < IMAGE_HEIGHT; y++) {
  //     for (int x = 0; x < IMAGE_WIDTH; x++) {
  // 	if (x + y > IMAGE_WIDTH * 2 / 3 && x + y < IMAGE_WIDTH * 4 / 3)
  // 	  image.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
  // 	if (y > IMAGE_HEIGHT / 4 && y < IMAGE_HEIGHT * 3 / 4 && x > IMAGE_WIDTH / 4 && x < IMAGE_WIDTH * 3 / 4)
  // 	  source_mask[y * IMAGE_WIDTH + x] = false;
  //     }
  //   }
  //   Mat completed_image = cv_utils::completeImage(image, source_mask, target_mask, 5);
  //   imwrite("Test/completed_image.bmp", completed_image);
  //   exit(1);
  // }
  Mat image = imread("Test/image_for_completion.bmp");
  resize(image, image, Size(image.cols / 2, image.rows / 2));
  cv_utils::ImageMask source_mask, target_mask;
  ifstream source_mask_in_str("Test/source_mask");
  source_mask_in_str >> source_mask;
  source_mask.resize(image.cols, image.rows);
  ifstream target_mask_in_str("Test/target_mask");
  target_mask_in_str >> target_mask;
  target_mask.resize(image.cols, image.rows);
  // ifstream unwarp_transform_in_str("Test/unwarp_transform");
  // MatrixXd unwarp_transform;
  // unwarp_transform_in_str >> unwarp_transform;
  imwrite("Test/source_mask.bmp", source_mask.drawMaskImage());
  imwrite("Test/target_mask.bmp", target_mask.drawMaskImage());
  Mat completed_image = cv_utils::completeImage(image, source_mask, target_mask, 5);
  //Mat completed_image = cv_utils::completeImageUsingFusionSpace(image, source_mask, target_mask, 5);
  imwrite("Test/completed_image.bmp", completed_image);
  exit(1);
}
