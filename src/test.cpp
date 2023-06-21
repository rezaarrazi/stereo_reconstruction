#include "stereo_dataset.h"
#include "feature_extractor.h"
#include "sparse_matcher.h"
#include "camera_pose_estimator.h"
#include "dense_matcher.h"


int main()
{

    StereoDataset stereo_dataset;
    stereo_dataset.SetImages(0);

    FeatureExtractor feature_extractor;
    feature_extractor.SetImages(stereo_dataset.GetImages());

    feature_extractor.ExtractFeatures(0);

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

    //expected rotation and translation
    //rotation = (cv::Mat_<double>(3,3) << 9.99930076e-01, -1.93063070e-03,  1.16669094e-02,
    //                                        1.93095338e-03,  9.99998136e-01, -1.63926538e-05,
    //                                        -1.16668560e-02,  3.89197656e-05,  9.99931939e-01);
    //translation = (cv::Mat_<double>(3,1) << -0.99758425,
    //                                        0.01570115,
    //                                        -0.06766929);

    std::cout << rotation << '\n';
    std::cout << translation << '\n';

    DenseMatcher dense_matcher;
    dense_matcher.ComputeDisparityMap(stereo_dataset, rotation, translation);

    stereo_dataset.SetDisparityMaps(0);

    cv::Mat out1, out2;
    cv::resize(dense_matcher.GetDisparityMap(), out1, cv::Size(), 0.5, 0.5);
    cv::imshow("disparity", out1);

    cv::resize(stereo_dataset.GetDisparityMaps()[0], out2, cv::Size(), 0.5, 0.5);
    cv::imshow("disparity_gt", out2);

    cv::waitKey(0);

    return 0;

}








