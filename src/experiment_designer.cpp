#include "experiment_designer.h"


std::array<float, 2> ExperimentDesigner::ComputeRMSE(const cv::Mat& rotation, const cv::Mat& translation) const
{

    std::array<float, 2> rmses = {0.0, 0.0};

    for (std::size_t i = 0; i < 3; i++)
        for (std::size_t j = 0; j < 3; j++)
            rmses[0] += pow(rotation.at<float>(i, j) - rotation_gt_.at<float>(i, j), 2);

    rmses[0] /= 9.0;
    rmses[0] = sqrt(rmses[0]);

    for (std::size_t i = 0; i < 3; i++)
        rmses[1] += pow(translation.at<float>(i, 0) - translation_gt_.at<float>(i, 0), 2);

    rmses[1] /= 3.0;
    rmses[1] = sqrt(rmses[1]);

}


void ExperimentDesigner::CompareKeypointNumber()
{

    // The order is ORB, SIFT, SURF, BRISK
    std::array<std::size_t, 4> keypoint_numbers = {0, 0, 0, 0};

    std::size_t image_pair_number = stereo_dataset_.GetImagePairNumber();

    for (std::size_t i = 0; i < image_pair_number; i++)
    {
        stereo_dataset_.SetImages(i);
        feature_extractor_.SetImages(stereo_dataset_.GetImages());

        for (std::size_t j = 0; j < 4; j++)
        {
            feature_extractor_.ExtractFeatures(j);
            keypoint_numbers[j] += feature_extractor_.GetAverageKeypointNumber();
        }
    }

    for (std::size_t i = 0; i < 4; i++)
        std::cout << feature_extractor_names_[i] << ": " << static_cast<std::size_t>(keypoint_numbers[i] / image_pair_number) << '\n';

}


void ExperimentDesigner::CompareCameraPoseEstimation()
{

    // The order is ORB, SIFT, SURF, BRISK
    std::array<std::array<float, 2>, 4> rmses;

    for (std::size_t i = 0; i < 4; i++)
        for (std::size_t j = 0; j < 2; j++)
            rmses[i][j] = 0.0;

    std::size_t image_pair_number = stereo_dataset_.GetImagePairNumber();

    for (std::size_t i = 0; i < image_pair_number; i++)
    {
        stereo_dataset_.SetImages(i);
        stereo_dataset_.SetCalibrations(i);
        feature_extractor_.SetImages(stereo_dataset_.GetImages());
        camera_pose_estimator_.SetCameraIntrinsics(stereo_dataset_.GetCameraIntrinsics());

        for (std::size_t j = 0; j < 4; j++)
        {
            feature_extractor_.ExtractFeatures(j);
            sparse_matcher_.MatchSparsely(feature_extractor_.GetKeypoints(), feature_extractor_.GetFeatures());
            camera_pose_estimator_.SetMatchedPoints(sparse_matcher_.GetMatchedPoints());
            camera_pose_estimator_.EstimateCameraPose();
            std::array<float, 2> rmse = ComputeRMSE(camera_pose_estimator_.GetRotation(), camera_pose_estimator_.GetTranslation());
            rmses[j][0] += rmse[0];
            rmses[j][1] += rmse[1];
        }
    }

    for (std::size_t i = 0; i < 4; i++)
    {
        rmses[i][0] /= image_pair_number;
        rmses[i][1] /= image_pair_number;
    }

    for (std::size_t i = 0; i < 4; i++)
        std::cout << feature_extractor_names_[i] << ": rotation: " << rmses[i][0] << " translation: " << rmses[i][1] << '\n';

}








