#include "stereo_dataset.h"
#include "feature_extractor.h"
#include "sparse_matcher.h"


int main()
{

    StereoDataset stereo_dataset;
    // stereo_dataset.TestDataset();
    std::array<std::string, 2> image_pairs = stereo_dataset.GetImagePair(0);
    // std::cout << image_pairs[0] << '\n';
    // std::cout << image_pairs[1] << '\n';

    cv::Mat image0 = cv::imread(image_pairs[0], CV_LOAD_IMAGE_COLOR);
    cv::Mat image1 = cv::imread(image_pairs[1], CV_LOAD_IMAGE_COLOR);

    FeatureExtractor feature_extractor;
    feature_extractor.SetImagePair(image0, image1);
    feature_extractor.ExtractFeaturesORB();

    SparseMatcher sparse_matcher;
    sparse_matcher.MatchSparsely(feature_extractor.GetKeypoints(), feature_extractor.GetFeatures());

    sparse_matcher.DisplayMatchings(image0, image1, feature_extractor.GetKeypoints());

    return 0;

}








