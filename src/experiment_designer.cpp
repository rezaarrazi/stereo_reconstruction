#include "experiment_designer.h"


std::array<double, 2> ExperimentDesigner::ComputeRMSE(const cv::Mat& rotation, const cv::Mat& translation) const
{

    std::array<double, 2> rmses = {0.0, 0.0};

    for (std::size_t i = 0; i < 3; i++)
        for (std::size_t j = 0; j < 3; j++)
            rmses[0] += pow(rotation.at<double>(i, j) - ROTATION_GT_.at<double>(i, j), 2);

    rmses[0] /= 9.0;
    rmses[0] = sqrt(rmses[0]);

    for (std::size_t i = 0; i < 3; i++)
        rmses[1] += pow(translation.at<double>(i, 0) - TRANSLATION_GT_.at<double>(i, 0), 2);

    rmses[1] /= 3.0;
    rmses[1] = sqrt(rmses[1]);

    return rmses;

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
        std::cout << FEATURE_EXTRACTOR_NAMES_[i] << ": " << static_cast<std::size_t>(keypoint_numbers[i] / image_pair_number) << '\n';

}


void ExperimentDesigner::CompareFeatureExtractionAndMatching(std::size_t type, std::size_t keypoint_number, double distance_ratio, double ratio)
{

    // The order is ORB, SIFT, SURF, BRISK
    std::array<std::array<double, 2>, 4> rmses;

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

            if (type == 0)
                sparse_matcher_.MatchSparselyBFSortTop(feature_extractor_.GetKeypoints(), feature_extractor_.GetFeatures(), keypoint_number);
            else if (type == 1)
                sparse_matcher_.MatchSparselyBFMinDistance(feature_extractor_.GetKeypoints(), feature_extractor_.GetFeatures(), distance_ratio);
            else if (type == 2)
                sparse_matcher_.MatchSparselyFLANNBased(feature_extractor_.GetKeypoints(), feature_extractor_.GetFeatures(), ratio);
            
            camera_pose_estimator_.SetMatchedPoints(sparse_matcher_.GetMatchedPoints());
            camera_pose_estimator_.EstimateCameraPose(3);

            std::array<double, 2> rmse = ComputeRMSE(camera_pose_estimator_.GetRotation(), camera_pose_estimator_.GetTranslation());

            rmses[j][0] += rmse[0];
            rmses[j][1] += rmse[1];
        }

    }

    for (std::size_t i = 0; i < 4; i++)
    {
        rmses[i][0] /= image_pair_number;
        rmses[i][1] /= image_pair_number;
    }

    std::cout << SPARSE_MATCHER_NAMES_[type] << '\n';

    for (std::size_t i = 0; i < 4; i++)
        std::cout << FEATURE_EXTRACTOR_NAMES_[i] << ": rotation: " << rmses[i][0] << " translation: " << rmses[i][1] << '\n';

}


void ExperimentDesigner::CompareCameraPoseEstimation()
{

    std::array<std::array<double, 2>, 4> rmses;

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

        feature_extractor_.ExtractFeatures(3);
        sparse_matcher_.MatchSparselyFLANNBased(feature_extractor_.GetKeypoints(), feature_extractor_.GetFeatures(), 0.6);

        camera_pose_estimator_.SetMatchedPoints(sparse_matcher_.GetMatchedPoints());

        for (std::size_t j = 0; j < 4; j++)
        {
            camera_pose_estimator_.EstimateCameraPose(j);

            std::array<double, 2> rmse = ComputeRMSE(camera_pose_estimator_.GetRotation(), camera_pose_estimator_.GetTranslation());

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
        std::cout << CAMERA_POSE_ESTIMATOR_NAMES_[i] << ": rotation: " << rmses[i][0] << " translation: " << rmses[i][1] << '\n';

}


void ExperimentDesigner::CompareDisparityMaps()
{

    stereo_dataset_.SetImages(0);
    stereo_dataset_.SetCalibrations(0);
    stereo_dataset_.SetDisparityMaps(0);

    feature_extractor_.SetImages(stereo_dataset_.GetImages());

    std::array<std::size_t, 2> feature_extractor_types = {1, 3};
    std::array<std::string, 2> feature_extractor_names = {"SIFT", "BRISK"};
    std::array<std::string, 2> dense_matcher_names = {"StereoBM", "StereoSGBM"};

    cv::Mat rotation;
    cv::Mat translation;

    for (std::size_t i = 0; i < 2; i++)
    {
        feature_extractor_.ExtractFeatures(feature_extractor_types[i]);

        if (feature_extractor_types[i] == 1)
            sparse_matcher_.MatchSparselyBFSortTop(feature_extractor_.GetKeypoints(), feature_extractor_.GetFeatures(), 50);
        else
            sparse_matcher_.MatchSparselyFLANNBased(feature_extractor_.GetKeypoints(), feature_extractor_.GetFeatures(), 0.6);
        
        camera_pose_estimator_.SetCameraIntrinsics(stereo_dataset_.GetCameraIntrinsics());

        camera_pose_estimator_.SetMatchedPoints(sparse_matcher_.GetMatchedPoints());

        camera_pose_estimator_.EstimateCameraPose(3);

        rotation = camera_pose_estimator_.GetRotation();
        translation = camera_pose_estimator_.GetTranslation();

        dense_matcher_.RectifyImages(stereo_dataset_, rotation, translation);

        for (std::size_t j = 0; j < 2; j++)
        {
            dense_matcher_.ComputeDisparityMap(stereo_dataset_, rotation, translation, j);
            cv::imwrite("../" + feature_extractor_names[i] + dense_matcher_names[j] + ".png", dense_matcher_.GetColorfulDisparityMap());
        }
    }

    cv::Mat gt;
    cv::applyColorMap(stereo_dataset_.GetDisparityMaps()[0], gt, cv::COLORMAP_JET);
    cv::imwrite("../gt.png", gt);

}








