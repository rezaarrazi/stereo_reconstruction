#include "stereo_dataset.h"


int main()
{
    StereoDataset stereo_dataset;
    // stereo_dataset.TestDataset();
    std::vector<std::string> image_pairs = stereo_dataset.GetImagePair(0);
    std::cout << image_pairs[0] << '\n';
    std::cout << image_pairs[1] << '\n';

    cv::Mat left_image = cv::imread(image_pairs[0], CV_LOAD_IMAGE_COLOR);
    cv::imshow("left_image", left_image);
    cv::waitKey(0);
    return 0;
}








