#include "stereo_dataset.h"
#include "feature_extractor.h"
#include "sparse_matcher.h"
#include "camera_pose_estimator.h"


int main()
{

    StereoDataset stereo_dataset;
    stereo_dataset.SetImages(0);

    FeatureExtractor feature_extractor;
    feature_extractor.SetImages(stereo_dataset.GetImages());

    feature_extractor.ExtractFeaturesORB();

    SparseMatcher sparse_matcher;

    sparse_matcher.MatchSparsely(feature_extractor.GetKeypoints(), feature_extractor.GetFeatures());

    // sparse_matcher.DisplayMatchings(stereo_dataset.GetImages(), feature_extractor.GetKeypoints());

    stereo_dataset.SetCalibrations(0);

    CameraPoseEstimator camera_pose_estimator;
    camera_pose_estimator.SetCameraIntrinsics(stereo_dataset.GetCameraIntrinsics());
    camera_pose_estimator.SetMatchedPoints(sparse_matcher.GetMatchedPoints());

    camera_pose_estimator.EstimateCameraPose();

    cv::Mat rotation = camera_pose_estimator.GetRotation();
    cv::Mat translation = camera_pose_estimator.GetTranslation();

    std::cout << rotation << '\n';
    std::cout << translation << '\n';

    return 0;

}








